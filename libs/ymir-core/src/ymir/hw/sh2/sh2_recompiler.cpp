#include <ymir/hw/sh2/sh2_recompiler.hpp>

#include <ymir/hw/sh2/sh2.hpp>

#include <ymir/jit/arm64_assembler.hpp>
#include <ymir/jit/jit_memory.hpp>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstring>
#include <mutex>

#if defined(__clang__)
    #pragma clang diagnostic ignored "-Winvalid-offsetof"
#elif defined(__GNUC__)
    #pragma GCC diagnostic ignored "-Winvalid-offsetof"
#endif

namespace ymir::sh2 {

namespace {

    using namespace jit::arm64;

    // Executable memory is shared by every recompiler in the process. On iOS with TXM only one region can ever be
    // prepared, so each recompiler takes a chunk of a single pool instead of creating its own.
    constexpr size_t kPoolSize = 64u * 1024u * 1024u;
    constexpr size_t kChunkSize = 16u * 1024u * 1024u;

    std::mutex g_poolMutex;
    jit::JitMemory g_pool;
    std::vector<size_t> g_freeChunks; // chunks returned by destroyed recompilers

    // Words reserved for assembling a block before it is copied into executable memory
    constexpr size_t kMaxBlockWords = 4096;

    // Registers used by generated code
    constexpr Reg kThis = X19;   // SH2 *
    constexpr Reg kCycles = X20; // cycles executed so far in this block
    constexpr Reg kBudget = X21; // cycles that may be executed
    constexpr Reg kTmpWb = 9;
    constexpr Reg kT0 = 10, kT1 = 11, kT2 = 12, kT3 = 13, kT4 = 14;

} // namespace

namespace {

    // Accesses a field of the SH2 object. Fields close to the start of the object use an immediate offset; the others
    // are reached through a register offset.
    class FieldAccess {
    public:
        explicit FieldAccess(Assembler &a)
            : m_a(a) {}

        void Load(uint32 sizeLog2, Reg rt, size_t offset) {
            if (Encodable(sizeLog2, offset)) {
                m_a.Ldr(sizeLog2, rt, kThis, static_cast<uint32>(offset));
            } else {
                m_a.MovImm(Sz::X, X15, offset);
                m_a.LdrReg(sizeLog2, rt, kThis, X15, Extend::LSL, false);
            }
        }

        void Store(uint32 sizeLog2, Reg rt, size_t offset) {
            if (Encodable(sizeLog2, offset)) {
                m_a.Str(sizeLog2, rt, kThis, static_cast<uint32>(offset));
            } else {
                m_a.MovImm(Sz::X, X15, offset);
                m_a.StrReg(sizeLog2, rt, kThis, X15, Extend::LSL, false);
            }
        }

    private:
        static bool Encodable(uint32 sizeLog2, size_t offset) {
            return (offset & ((1u << sizeLog2) - 1)) == 0 && (offset >> sizeLog2) < 4096;
        }

        Assembler &m_a;
    };

    // Emits `dst = src & imm`, `dst = src | imm` or `dst = src ^ imm` for any 32-bit immediate
    void EmitLogicalImm(Assembler &a, int op, Reg dst, Reg src, uint32 imm) {
        bool ok = false;
        switch (op) {
        case 0: ok = a.AndImm(Sz::W, dst, src, imm); break;
        case 1: ok = a.OrrImm(Sz::W, dst, src, imm); break;
        case 2: ok = a.EorImm(Sz::W, dst, src, imm); break;
        }
        if (ok) {
            return;
        }
        a.MovImm(Sz::W, kT4, imm);
        switch (op) {
        case 0: a.And(Sz::W, dst, src, kT4); break;
        case 1: a.Orr(Sz::W, dst, src, kT4); break;
        case 2: a.Eor(Sz::W, dst, src, kT4); break;
        }
    }

} // namespace

SH2Recompiler::SH2Recompiler(SH2 &sh2, sys::SH2Bus &bus)
    : m_sh2(sh2)
    , m_bus(bus) {
    std::lock_guard lock{g_poolMutex};
    if (!g_pool.IsInitialized() && !g_pool.Initialize(kPoolSize)) {
        return;
    }
    size_t offset;
    if (!g_freeChunks.empty()) {
        offset = g_freeChunks.back();
        g_freeChunks.pop_back();
    } else {
        offset = g_pool.Allocate(kChunkSize);
        if (offset == SIZE_MAX) {
            return;
        }
    }
    m_chunkOffset = offset;
    m_chunkSize = kChunkSize;
    m_blocks.resize(kBlockTableSize);
    m_usable = true;
}

SH2Recompiler::~SH2Recompiler() {
    if (m_usable) {
        std::lock_guard lock{g_poolMutex};
        g_freeChunks.push_back(m_chunkOffset);
    }
}

void SH2Recompiler::Flush() {
    if (!m_usable) {
        return;
    }
    for (Block &block : m_blocks) {
        block.pc = 1;
        block.fn = nullptr;
    }
    m_chunkUsed = 0;
    ++m_stats.cacheFlushes;
}

void SH2Recompiler::GetPoolRegion(uint8 **base, size_t *size) {
    std::lock_guard lock{g_poolMutex};
    *base = g_pool.IsInitialized() ? g_pool.ExecPtr(0) : nullptr;
    *size = g_pool.IsInitialized() ? g_pool.Capacity() : 0;
}

uint64 SH2Recompiler::StepThunk(SH2 *sh2, uint32 instr) {
    return sh2->JitStep(static_cast<uint16>(instr));
}

uint64 SH2Recompiler::CodeUnchangedThunk(SH2 *sh2, uint32 blockPc, uint32 offset) {
    SH2Recompiler *self = sh2->m_recompiler.get();
    const Block &block = self->m_blocks[SlotFor(blockPc)];
    if (block.pc != blockPc || offset > block.byteLen) {
        return 0;
    }
    const uint8 *src = self->m_bus.GetArrayPointer(blockPc);
    if (src == nullptr) {
        return 0;
    }
    return std::memcmp(src + offset, block.bytes.data() + offset, block.byteLen - offset) == 0 ? 1 : 0;
}

SH2Recompiler::Kind SH2Recompiler::Classify(OpcodeType opcode) {
    switch (opcode) {
    // Instructions with no side effects other than registers and the T bit
    case OpcodeType::NOP:
    case OpcodeType::MOV_R:
    case OpcodeType::MOV_I:
    case OpcodeType::MOVA:
    case OpcodeType::MOVT:
    case OpcodeType::CLRT:
    case OpcodeType::SETT:
    case OpcodeType::EXTUB:
    case OpcodeType::EXTUW:
    case OpcodeType::EXTSB:
    case OpcodeType::EXTSW:
    case OpcodeType::SWAPB:
    case OpcodeType::SWAPW:
    case OpcodeType::XTRCT:
    case OpcodeType::ADD:
    case OpcodeType::ADD_I:
    case OpcodeType::ADDC:
    case OpcodeType::ADDV:
    case OpcodeType::AND_R:
    case OpcodeType::AND_I:
    case OpcodeType::NEG:
    case OpcodeType::NEGC:
    case OpcodeType::NOT:
    case OpcodeType::OR_R:
    case OpcodeType::OR_I:
    case OpcodeType::ROTCL:
    case OpcodeType::ROTCR:
    case OpcodeType::ROTL:
    case OpcodeType::ROTR:
    case OpcodeType::SHAL:
    case OpcodeType::SHAR:
    case OpcodeType::SHLL:
    case OpcodeType::SHLL2:
    case OpcodeType::SHLL8:
    case OpcodeType::SHLL16:
    case OpcodeType::SHLR:
    case OpcodeType::SHLR2:
    case OpcodeType::SHLR8:
    case OpcodeType::SHLR16:
    case OpcodeType::SUB:
    case OpcodeType::SUBC:
    case OpcodeType::SUBV:
    case OpcodeType::XOR_R:
    case OpcodeType::XOR_I:
    case OpcodeType::DT:
    case OpcodeType::DIV0U:
    case OpcodeType::CLRMAC:
    case OpcodeType::CMP_EQ_I:
    case OpcodeType::CMP_EQ_R:
    case OpcodeType::CMP_GE:
    case OpcodeType::CMP_GT:
    case OpcodeType::CMP_HI:
    case OpcodeType::CMP_HS:
    case OpcodeType::CMP_PL:
    case OpcodeType::CMP_PZ:
    case OpcodeType::TST_R:
    case OpcodeType::TST_I: return Kind::Inline;

    // Branches with a delay slot: the instruction after it is part of the same block
    case OpcodeType::BRA:
    case OpcodeType::BRAF:
    case OpcodeType::BSR:
    case OpcodeType::BSRF:
    case OpcodeType::JMP:
    case OpcodeType::JSR:
    case OpcodeType::RTS:
    case OpcodeType::RTE:
    case OpcodeType::BTS:
    case OpcodeType::BFS: return Kind::DelayedBranch;

    case OpcodeType::TRAPA: return Kind::EndAfter;

    // Changes the CPU's execution state in ways the block mechanism doesn't handle
    case OpcodeType::SLEEP: return Kind::Stop;

    // Everything else is executed by the interpreter's handlers
    default: return Kind::Helper;
    }
}

uint64 SH2Recompiler::Run(uint64 cyclesLeft) {
    if (!m_usable) {
        return 0;
    }

    const uint32 pc = m_sh2.PC;
    if ((pc & 1u) != 0) {
        return 0;
    }
    // Only the cached and cache-through areas; the others are not ordinary memory
    const uint32 partition = pc >> 29u;
    if (partition != 0 && partition != 1 && partition != 5) {
        return 0;
    }
    const uint8 *src = m_bus.GetArrayPointer(pc);
    if (src == nullptr) {
        return 0;
    }

    Block &block = m_blocks[SlotFor(pc)];
    if (block.pc != pc || std::memcmp(src, block.bytes.data(), block.byteLen) != 0) {
        if (block.pc == pc) {
            ++m_stats.invalidations;
        }
        Compile(block, pc, src, sys::SH2Bus::PageBytesLeft(pc));
    }
    if (block.fn == nullptr) {
        return 0;
    }

    ++m_stats.blocksExecuted;
    if (m_executor) {
        return m_executor(block.fn, m_sh2, cyclesLeft);
    }
    return block.fn(&m_sh2, cyclesLeft);
}

void SH2Recompiler::Compile(Block &block, uint32 pc, const uint8 *src, uint32 bytesAvailable) {
    struct Insn {
        uint16 word;
        OpcodeType opcode;
        Kind kind;
        uint32 pc;
    };
    std::array<Insn, kMaxInstructions + 1> insns{};
    uint32 count = 0;

    // -- Decode --

    bool inDelaySlot = false;
    bool done = false;
    const uint32 maxBytes = std::min<uint32>(bytesAvailable, kMaxInstructions * 2);
    for (uint32 offset = 0; offset + 2 <= maxBytes && !done; offset += 2) {
        const uint16 word = static_cast<uint16>((src[offset] << 8) | src[offset + 1]);
        Insn insn{.word = word, .opcode = OpcodeType::Illegal, .kind = Kind::Helper, .pc = pc + offset};
        if (inDelaySlot) {
            // Delay slots always go through the interpreter's handlers
            insn.opcode = DecodeTable::s_instance.opcodes[1][word];
            insn.kind = Kind::Helper;
            done = true;
        } else {
            insn.opcode = DecodeTable::s_instance.opcodes[0][word];
            insn.kind = Classify(insn.opcode);
            if (insn.kind == Kind::Stop) {
                break;
            }
            if (insn.kind == Kind::DelayedBranch) {
                inDelaySlot = true;
            } else if (insn.kind == Kind::EndAfter) {
                done = true;
            }
        }
        insns[count++] = insn;
    }
    // A delayed branch whose slot doesn't fit in the same memory page can't be translated
    if (inDelaySlot && !done) {
        --count;
    }

    // The bytes the block depends on, which are compared again every time it runs
    block.pc = pc;
    block.byteLen = std::max<uint32>(count * 2, 2);
    block.byteLen = std::min<uint32>(block.byteLen, bytesAvailable);
    std::memcpy(block.bytes.data(), src, block.byteLen);
    block.fn = nullptr;

    if (count == 0) {
        ++m_stats.blocksRejected;
        return;
    }

    // -- Generate --

    // Offsets of the CPU state that generated code reads and writes
    struct {
        size_t R, PC, SR, MAC, WbReg, IntrPending, IntrAllow;
        uint8 WbNone;
    } const L{offsetof(SH2, R),
              offsetof(SH2, PC),
              offsetof(SH2, SR),
              offsetof(SH2, MAC),
              offsetof(SH2, m_wbReg),
              offsetof(SH2, m_intrFlags) + offsetof(SH2::IntrFlags, pending),
              offsetof(SH2, m_intrFlags) + offsetof(SH2::IntrFlags, allow),
              SH2::kWBRegNone};

    std::array<uint32, kMaxBlockWords> words{};
    Assembler a{words.data(), words.size()};
    FieldAccess field{a};

    Label epilogue;
    std::vector<std::pair<Label, uint32>> budgetExits; // label, PC to store
    budgetExits.reserve(count);

    // Prologue
    a.StpPre(Sz::X, X29, X30, SP, -48);
    a.Stp(Sz::X, X19, X20, SP, 16);
    a.Str(3, X21, SP, 32);
    a.Mov(Sz::X, kThis, X0);
    a.Mov(Sz::X, kBudget, X1);
    a.Mov(Sz::X, kCycles, ZR);
    // Like the interpreter, interrupts are allowed again from the first instruction on
    a.MovImm(Sz::W, kT0, 1);
    field.Store(0, kT0, L.IntrAllow);

    auto loadR = [&](Reg dst, uint32 index) { a.Ldr(2, dst, kThis, static_cast<uint32>(L.R + 4 * index)); };
    auto storeR = [&](Reg src, uint32 index) { a.Str(2, src, kThis, static_cast<uint32>(L.R + 4 * index)); };
    // T lives in bit 0 of SR
    auto readTBit = [&](Reg dst) {
        a.Ldr(2, dst, kThis, static_cast<uint32>(L.SR));
        a.AndImm(Sz::W, dst, dst, 1);
    };
    // Sets T to the low bit of `value` (which must be 0 or 1)
    auto writeT = [&](Reg value) {
        a.Ldr(2, kT3, kThis, static_cast<uint32>(L.SR));
        a.Bfi(Sz::W, kT3, value, 0, 1);
        a.Str(2, kT3, kThis, static_cast<uint32>(L.SR));
    };

    // Whether the register holding the pipeline's writeback state is known to be "none"
    bool wbKnownNone = false;
    // Whether the previous instruction was executed by a handler
    bool prevWasHelper = true;
    // Whether the interrupt allow flag is known to be set (the prologue sets it)
    bool allowSet = true;
    // Whether the PC stored in the CPU is behind the instruction about to be executed (inline instructions leave it)
    bool pcStale = false;

    for (uint32 k = 0; k < count; ++k) {
        const Insn &insn = insns[k];
        const uint16 w = insn.word;
        const uint32 n = (w >> 8) & 0xF;
        const uint32 m = (w >> 4) & 0xF;

        // Stop here if the cycle budget is used up, like the interpreter's loop would
        if (k > 0) {
            budgetExits.emplace_back();
            a.Cmp(Sz::X, kCycles, kBudget);
            a.BCond(Cond::CS, budgetExits.back().first);
            budgetExits.back().second = insn.pc;

            // Interrupts are allowed again from this instruction on (a handler may have blocked them for it)
            if (!allowSet) {
                a.MovImm(Sz::W, kT0, 1);
                field.Store(0, kT0, L.IntrAllow);
                allowSet = true;
            }
        }

        if (insn.kind != Kind::Inline) {
            // Executed by the interpreter's handler, which expects the CPU's PC to point at the instruction
            if (pcStale) {
                a.MovImm(Sz::W, kT0, insn.pc);
                a.Str(2, kT0, kThis, static_cast<uint32>(L.PC));
                pcStale = false;
            }
            a.Mov(Sz::X, X0, kThis);
            a.MovImm(Sz::W, X1, w);
            a.MovImm(Sz::X, X16, reinterpret_cast<uint64>(&SH2Recompiler::StepThunk));
            a.Blr(X16);
            a.Add(Sz::X, kCycles, kCycles, X0);
            ++m_stats.helperInstructions;
            // A store may have rewritten instructions further down this very block. The interpreter would execute the
            // new ones, so leave the block (the next entry recompiles it) if the rest changed
            {
                const auto &mem = DecodeTable::s_instance.mem[w];
                // The interpreter fetches whole longwords: an instruction at an aligned address has already fetched the
                // word after it, so changes to that one are not noticed
                const uint32 firstChecked = k + 1 + ((insn.pc & 2u) == 0 ? 1 : 0);
                if (firstChecked < count && (mem.first.write || mem.second.write)) {
                    a.Mov(Sz::X, X0, kThis);
                    a.MovImm(Sz::W, X1, pc);
                    a.MovImm(Sz::W, X2, firstChecked * 2);
                    a.MovImm(Sz::X, X16, reinterpret_cast<uint64>(&SH2Recompiler::CodeUnchangedThunk));
                    a.Blr(X16);
                    a.Cbz(Sz::X, X0, epilogue);
                }
            }
            allowSet = false;
            wbKnownNone = false;
            prevWasHelper = true;

            if (k + 1 < count) {
                // Leave the block if the instruction didn't simply fall through (a taken branch, an exception or a
                // retried bus access)
                a.Ldr(2, kT0, kThis, static_cast<uint32>(L.PC));
                a.MovImm(Sz::W, kT1, insn.pc + 2);
                a.Cmp(Sz::W, kT0, kT1);
                a.BCond(Cond::NE, epilogue);

                // Leave the block if an interrupt is about to be serviced; otherwise interrupts are allowed again from
                // the next instruction on (the handler may have blocked them for this one)
                field.Load(0, kT0, L.IntrPending);
                field.Load(0, kT1, L.IntrAllow);
                a.And(Sz::W, kT0, kT0, kT1);
                a.Cbnz(Sz::W, kT0, epilogue);
            }
            continue;
        }

        // -- Inline translation --
        ++m_stats.inlineInstructions;
        pcStale = true;

        // Cycle accounting: 1 cycle, plus 1 if the writeback stage still holds one of the source registers
        // (kWBRegNone, 0xFF, never matches a register number)
        std::array<uint32, 2> stallRegs{};
        uint32 numStallRegs = 0;
        auto stall = [&](uint32 a0, uint32 a1 = 0xFFFFFFFFu) {
            stallRegs[0] = a0;
            numStallRegs = 1;
            if (a1 != 0xFFFFFFFFu) {
                stallRegs[1] = a1;
                numStallRegs = 2;
            }
        };

        switch (insn.opcode) {
        case OpcodeType::NOP:
        case OpcodeType::CLRT:
        case OpcodeType::SETT:
        case OpcodeType::DIV0U:
        case OpcodeType::CLRMAC: break;
        case OpcodeType::MOV_R:
        case OpcodeType::EXTUB:
        case OpcodeType::EXTUW:
        case OpcodeType::EXTSB:
        case OpcodeType::EXTSW:
        case OpcodeType::SWAPB:
        case OpcodeType::SWAPW:
        case OpcodeType::XTRCT:
        case OpcodeType::ADD:
        case OpcodeType::ADDC:
        case OpcodeType::ADDV:
        case OpcodeType::AND_R:
        case OpcodeType::NEG:
        case OpcodeType::NEGC:
        case OpcodeType::NOT:
        case OpcodeType::OR_R:
        case OpcodeType::SUB:
        case OpcodeType::SUBC:
        case OpcodeType::SUBV:
        case OpcodeType::XOR_R:
        case OpcodeType::CMP_EQ_R:
        case OpcodeType::CMP_GE:
        case OpcodeType::CMP_GT:
        case OpcodeType::CMP_HI:
        case OpcodeType::CMP_HS:
        case OpcodeType::TST_R: stall(m, n); break;
        case OpcodeType::MOV_I:
        case OpcodeType::MOVT:
        case OpcodeType::ADD_I:
        case OpcodeType::ROTCL:
        case OpcodeType::ROTCR:
        case OpcodeType::ROTL:
        case OpcodeType::ROTR:
        case OpcodeType::SHAL:
        case OpcodeType::SHAR:
        case OpcodeType::SHLL:
        case OpcodeType::SHLL2:
        case OpcodeType::SHLL8:
        case OpcodeType::SHLL16:
        case OpcodeType::SHLR:
        case OpcodeType::SHLR2:
        case OpcodeType::SHLR8:
        case OpcodeType::SHLR16:
        case OpcodeType::DT:
        case OpcodeType::CMP_PL:
        case OpcodeType::CMP_PZ: stall(n); break;
        case OpcodeType::MOVA:
        case OpcodeType::AND_I:
        case OpcodeType::OR_I:
        case OpcodeType::XOR_I:
        case OpcodeType::CMP_EQ_I:
        case OpcodeType::TST_I: stall(0); break;
        default: break;
        }

        // The cycle for the instruction itself
        a.AddImm(Sz::X, kCycles, kCycles, 1);
        if (numStallRegs > 0 && !wbKnownNone) {
            field.Load(0, kTmpWb, L.WbReg);
            a.CmpImm(Sz::W, kTmpWb, stallRegs[0]);
            if (numStallRegs > 1) {
                a.CcmpImm(Sz::W, kTmpWb, stallRegs[1], 4, Cond::NE); // Z=1 if the first one already matched
            }
            a.Cinc(Sz::X, kCycles, kCycles, Cond::EQ);
        }
        if (!wbKnownNone) {
            a.MovImm(Sz::W, kTmpWb, L.WbNone);
            field.Store(0, kTmpWb, L.WbReg);
            wbKnownNone = true;
        }
        prevWasHelper = false;

        const uint32 simm8 = static_cast<uint32>(static_cast<sint32>(static_cast<sint8>(w & 0xFF)));
        const uint32 uimm8 = w & 0xFF;

        switch (insn.opcode) {
        case OpcodeType::NOP: break;
        case OpcodeType::MOV_R:
            loadR(kT0, m);
            storeR(kT0, n);
            break;
        case OpcodeType::MOV_I:
            a.MovImm(Sz::W, kT0, simm8);
            storeR(kT0, n);
            break;
        case OpcodeType::MOVA: {
            const uint32 address = (insn.pc & ~3u) + (uimm8 << 2) + 4u;
            a.MovImm(Sz::W, kT0, address);
            storeR(kT0, 0);
            break;
        }
        case OpcodeType::MOVT:
            readTBit(kT0);
            storeR(kT0, n);
            break;
        case OpcodeType::CLRT:
            a.Ldr(2, kT0, kThis, static_cast<uint32>(L.SR));
            EmitLogicalImm(a, 0, kT0, kT0, ~1u);
            a.Str(2, kT0, kThis, static_cast<uint32>(L.SR));
            break;
        case OpcodeType::SETT:
            a.Ldr(2, kT0, kThis, static_cast<uint32>(L.SR));
            EmitLogicalImm(a, 1, kT0, kT0, 1u);
            a.Str(2, kT0, kThis, static_cast<uint32>(L.SR));
            break;
        case OpcodeType::DIV0U: // clears M, Q and T
            a.Ldr(2, kT0, kThis, static_cast<uint32>(L.SR));
            EmitLogicalImm(a, 0, kT0, kT0, ~0x301u);
            a.Str(2, kT0, kThis, static_cast<uint32>(L.SR));
            break;
        case OpcodeType::CLRMAC:
            // MAC is the 64-bit value right after SR's neighbors: clear both halves through the generic path
            field.Store(3, ZR, L.MAC);
            break;
        case OpcodeType::EXTUB:
            loadR(kT0, m);
            a.Uxtb(kT0, kT0);
            storeR(kT0, n);
            break;
        case OpcodeType::EXTUW:
            loadR(kT0, m);
            a.Uxth(kT0, kT0);
            storeR(kT0, n);
            break;
        case OpcodeType::EXTSB:
            loadR(kT0, m);
            a.Sxtb(Sz::W, kT0, kT0);
            storeR(kT0, n);
            break;
        case OpcodeType::EXTSW:
            loadR(kT0, m);
            a.Sxth(Sz::W, kT0, kT0);
            storeR(kT0, n);
            break;
        case OpcodeType::SWAPB:
            loadR(kT0, m);
            EmitLogicalImm(a, 0, kT1, kT0, 0xFFFF0000u); // upper half unchanged
            a.Ubfx(Sz::W, kT2, kT0, 8, 8);               // old bits 15-8 go to bits 7-0
            a.Orr(Sz::W, kT1, kT1, kT2);
            a.Bfi(Sz::W, kT1, kT0, 8, 8); // old bits 7-0 go to bits 15-8
            storeR(kT1, n);
            break;
        case OpcodeType::SWAPW:
            loadR(kT0, m);
            a.RorImm(Sz::W, kT0, kT0, 16);
            storeR(kT0, n);
            break;
        case OpcodeType::XTRCT:
            loadR(kT0, m);
            loadR(kT1, n);
            a.Extr(Sz::W, kT2, kT0, kT1, 16); // (Rm << 16) | (Rn >> 16)
            storeR(kT2, n);
            break;
        case OpcodeType::ADD:
            loadR(kT0, n);
            loadR(kT1, m);
            a.Add(Sz::W, kT0, kT0, kT1);
            storeR(kT0, n);
            break;
        case OpcodeType::ADD_I:
            loadR(kT0, n);
            a.MovImm(Sz::W, kT1, simm8);
            a.Add(Sz::W, kT0, kT0, kT1);
            storeR(kT0, n);
            break;
        case OpcodeType::ADDC:
            loadR(kT0, n);
            loadR(kT1, m);
            readTBit(kT2);
            a.SubsImm(Sz::W, ZR, kT2, 1); // C = T
            a.Adcs(Sz::W, kT0, kT0, kT1);
            a.Cset(Sz::W, kT2, Cond::CS);
            storeR(kT0, n);
            writeT(kT2);
            break;
        case OpcodeType::ADDV:
            loadR(kT0, n);
            loadR(kT1, m);
            a.Adds(Sz::W, kT0, kT0, kT1);
            a.Cset(Sz::W, kT2, Cond::VS);
            storeR(kT0, n);
            writeT(kT2);
            break;
        case OpcodeType::AND_R:
            loadR(kT0, n);
            loadR(kT1, m);
            a.And(Sz::W, kT0, kT0, kT1);
            storeR(kT0, n);
            break;
        case OpcodeType::AND_I:
            loadR(kT0, 0);
            EmitLogicalImm(a, 0, kT0, kT0, uimm8);
            storeR(kT0, 0);
            break;
        case OpcodeType::NEG:
            loadR(kT0, m);
            a.Neg(Sz::W, kT0, kT0);
            storeR(kT0, n);
            break;
        case OpcodeType::NEGC:
            loadR(kT1, m);
            readTBit(kT2);
            a.MovImm(Sz::W, kT4, 1);
            a.Eor(Sz::W, kT2, kT2, kT4); // C = !T (ARM's carry is "no borrow")
            a.SubsImm(Sz::W, ZR, kT2, 1);
            a.Sbcs(Sz::W, kT0, ZR, kT1);
            a.Cset(Sz::W, kT2, Cond::CC);
            storeR(kT0, n);
            writeT(kT2);
            break;
        case OpcodeType::NOT:
            loadR(kT0, m);
            a.Mvn(Sz::W, kT0, kT0);
            storeR(kT0, n);
            break;
        case OpcodeType::OR_R:
            loadR(kT0, n);
            loadR(kT1, m);
            a.Orr(Sz::W, kT0, kT0, kT1);
            storeR(kT0, n);
            break;
        case OpcodeType::OR_I:
            loadR(kT0, 0);
            EmitLogicalImm(a, 1, kT0, kT0, uimm8);
            storeR(kT0, 0);
            break;
        case OpcodeType::ROTCL:
            loadR(kT0, n);
            readTBit(kT1);
            a.LsrImm(Sz::W, kT2, kT0, 31);       // new T
            a.LslImm(Sz::W, kT0, kT0, 1);
            a.Orr(Sz::W, kT0, kT0, kT1);
            storeR(kT0, n);
            writeT(kT2);
            break;
        case OpcodeType::ROTCR:
            loadR(kT0, n);
            readTBit(kT1);
            a.AndImm(Sz::W, kT2, kT0, 1); // new T
            a.LsrImm(Sz::W, kT0, kT0, 1);
            a.Orr(Sz::W, kT0, kT0, kT1, Shift::LSL, 31);
            storeR(kT0, n);
            writeT(kT2);
            break;
        case OpcodeType::ROTL:
            loadR(kT0, n);
            a.LsrImm(Sz::W, kT2, kT0, 31);
            a.RorImm(Sz::W, kT0, kT0, 31);
            storeR(kT0, n);
            writeT(kT2);
            break;
        case OpcodeType::ROTR:
            loadR(kT0, n);
            a.AndImm(Sz::W, kT2, kT0, 1);
            a.RorImm(Sz::W, kT0, kT0, 1);
            storeR(kT0, n);
            writeT(kT2);
            break;
        case OpcodeType::SHAL:
        case OpcodeType::SHLL:
            loadR(kT0, n);
            a.LsrImm(Sz::W, kT2, kT0, 31);
            a.LslImm(Sz::W, kT0, kT0, 1);
            storeR(kT0, n);
            writeT(kT2);
            break;
        case OpcodeType::SHAR:
            loadR(kT0, n);
            a.AndImm(Sz::W, kT2, kT0, 1);
            a.AsrImm(Sz::W, kT0, kT0, 1);
            storeR(kT0, n);
            writeT(kT2);
            break;
        case OpcodeType::SHLR:
            loadR(kT0, n);
            a.AndImm(Sz::W, kT2, kT0, 1);
            a.LsrImm(Sz::W, kT0, kT0, 1);
            storeR(kT0, n);
            writeT(kT2);
            break;
        case OpcodeType::SHLL2:
            loadR(kT0, n);
            a.LslImm(Sz::W, kT0, kT0, 2);
            storeR(kT0, n);
            break;
        case OpcodeType::SHLL8:
            loadR(kT0, n);
            a.LslImm(Sz::W, kT0, kT0, 8);
            storeR(kT0, n);
            break;
        case OpcodeType::SHLL16:
            loadR(kT0, n);
            a.LslImm(Sz::W, kT0, kT0, 16);
            storeR(kT0, n);
            break;
        case OpcodeType::SHLR2:
            loadR(kT0, n);
            a.LsrImm(Sz::W, kT0, kT0, 2);
            storeR(kT0, n);
            break;
        case OpcodeType::SHLR8:
            loadR(kT0, n);
            a.LsrImm(Sz::W, kT0, kT0, 8);
            storeR(kT0, n);
            break;
        case OpcodeType::SHLR16:
            loadR(kT0, n);
            a.LsrImm(Sz::W, kT0, kT0, 16);
            storeR(kT0, n);
            break;
        case OpcodeType::SUB:
            loadR(kT0, n);
            loadR(kT1, m);
            a.Sub(Sz::W, kT0, kT0, kT1);
            storeR(kT0, n);
            break;
        case OpcodeType::SUBC:
            loadR(kT0, n);
            loadR(kT1, m);
            readTBit(kT2);
            a.MovImm(Sz::W, kT4, 1);
            a.Eor(Sz::W, kT2, kT2, kT4); // C = !T
            a.SubsImm(Sz::W, ZR, kT2, 1);
            a.Sbcs(Sz::W, kT0, kT0, kT1);
            a.Cset(Sz::W, kT2, Cond::CC);
            storeR(kT0, n);
            writeT(kT2);
            break;
        case OpcodeType::SUBV:
            loadR(kT0, n);
            loadR(kT1, m);
            a.Subs(Sz::W, kT0, kT0, kT1);
            a.Cset(Sz::W, kT2, Cond::VS);
            storeR(kT0, n);
            writeT(kT2);
            break;
        case OpcodeType::XOR_R:
            loadR(kT0, n);
            loadR(kT1, m);
            a.Eor(Sz::W, kT0, kT0, kT1);
            storeR(kT0, n);
            break;
        case OpcodeType::XOR_I:
            loadR(kT0, 0);
            EmitLogicalImm(a, 2, kT0, kT0, uimm8);
            storeR(kT0, 0);
            break;
        case OpcodeType::DT:
            loadR(kT0, n);
            a.SubsImm(Sz::W, kT0, kT0, 1);
            a.Cset(Sz::W, kT2, Cond::EQ);
            storeR(kT0, n);
            writeT(kT2);
            break;
        case OpcodeType::CMP_EQ_I:
            loadR(kT0, 0);
            a.MovImm(Sz::W, kT1, simm8);
            a.Cmp(Sz::W, kT0, kT1);
            a.Cset(Sz::W, kT2, Cond::EQ);
            writeT(kT2);
            break;
        case OpcodeType::CMP_EQ_R:
        case OpcodeType::CMP_GE:
        case OpcodeType::CMP_GT:
        case OpcodeType::CMP_HI:
        case OpcodeType::CMP_HS: {
            loadR(kT0, n);
            loadR(kT1, m);
            a.Cmp(Sz::W, kT0, kT1);
            Cond cond = Cond::EQ;
            switch (insn.opcode) {
            case OpcodeType::CMP_GE: cond = Cond::GE; break;
            case OpcodeType::CMP_GT: cond = Cond::GT; break;
            case OpcodeType::CMP_HI: cond = Cond::HI; break;
            case OpcodeType::CMP_HS: cond = Cond::CS; break;
            default: break;
            }
            a.Cset(Sz::W, kT2, cond);
            writeT(kT2);
            break;
        }
        case OpcodeType::CMP_PL:
            loadR(kT0, n);
            a.CmpImm(Sz::W, kT0, 0);
            a.Cset(Sz::W, kT2, Cond::GT);
            writeT(kT2);
            break;
        case OpcodeType::CMP_PZ:
            loadR(kT0, n);
            a.CmpImm(Sz::W, kT0, 0);
            a.Cset(Sz::W, kT2, Cond::GE);
            writeT(kT2);
            break;
        case OpcodeType::TST_R:
            loadR(kT0, n);
            loadR(kT1, m);
            a.Tst(Sz::W, kT0, kT1);
            a.Cset(Sz::W, kT2, Cond::EQ);
            writeT(kT2);
            break;
        case OpcodeType::TST_I:
            loadR(kT0, 0);
            a.MovImm(Sz::W, kT1, uimm8);
            a.Tst(Sz::W, kT0, kT1);
            a.Cset(Sz::W, kT2, Cond::EQ);
            writeT(kT2);
            break;
        default: break;
        }
    }

    // After the last instruction: continue at the next one, unless a handler already did
    const Insn &last = insns[count - 1];
    if (last.kind == Kind::Inline) {
        a.MovImm(Sz::W, kT0, last.pc + 2);
        a.Str(2, kT0, kThis, static_cast<uint32>(L.PC));
    }

    // Epilogue
    a.Bind(epilogue);
    a.Mov(Sz::X, X0, kCycles);
    a.Ldr(3, X21, SP, 32);
    a.Ldp(Sz::X, X19, X20, SP, 16);
    a.LdpPost(Sz::X, X29, X30, SP, 48);
    a.Ret();

    // Exits taken when the cycle budget runs out before an instruction
    for (auto &[label, exitPc] : budgetExits) {
        a.Bind(label);
        a.MovImm(Sz::W, kT0, exitPc);
        a.Str(2, kT0, kThis, static_cast<uint32>(L.PC));
        a.B(epilogue);
    }
    (void)prevWasHelper;

    if (a.Overflowed() || !a.Finalize()) {
        ++m_stats.blocksRejected;
        return;
    }

    // -- Install --

    const size_t sizeBytes = a.Size() * sizeof(uint32);
    if (m_chunkUsed + sizeBytes + 16 > m_chunkSize) {
        Flush();
        // The block table was cleared, so the block being compiled has to be filled in again
        block.pc = pc;
        block.byteLen = std::max<uint32>(count * 2, 2);
        block.byteLen = std::min<uint32>(block.byteLen, bytesAvailable);
        std::memcpy(block.bytes.data(), src, block.byteLen);
    }
    const size_t offset = m_chunkOffset + ((m_chunkUsed + 15) & ~size_t{15});
    m_chunkUsed = (offset - m_chunkOffset) + sizeBytes;

    std::lock_guard lock{g_poolMutex};
    g_pool.BeginWrite();
    std::memcpy(g_pool.WritePtr(offset), words.data(), sizeBytes);
    g_pool.EndWrite(offset, sizeBytes);
    block.fn = reinterpret_cast<BlockFn>(g_pool.ExecPtr(offset));
    ++m_stats.blocksCompiled;
}

} // namespace ymir::sh2
