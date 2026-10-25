#include <ymir/hw/sh2/sh2_recompiler.hpp>

#include <ymir/hw/sh2/sh2.hpp>

#include <ymir/jit/arm64_assembler.hpp>
#include <ymir/jit/jit_memory.hpp>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstring>
#include <deque>
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
    constexpr Reg kCodeBase = X22;  // host address of the block's code in the emulated memory
    constexpr Reg kPageTable = X23; // bus page table
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

    // Loads and stores. Accesses to RAM and ROM are translated inline; anything else is left to the interpreter's handler
    case OpcodeType::MOVB_L:
    case OpcodeType::MOVW_L:
    case OpcodeType::MOVL_L:
    case OpcodeType::MOVB_L0:
    case OpcodeType::MOVW_L0:
    case OpcodeType::MOVL_L0:
    case OpcodeType::MOVB_L4:
    case OpcodeType::MOVW_L4:
    case OpcodeType::MOVL_L4:
    case OpcodeType::MOVB_LG:
    case OpcodeType::MOVW_LG:
    case OpcodeType::MOVL_LG:
    case OpcodeType::MOVB_P:
    case OpcodeType::MOVW_P:
    case OpcodeType::MOVL_P:
    case OpcodeType::MOVB_M:
    case OpcodeType::MOVW_M:
    case OpcodeType::MOVL_M:
    case OpcodeType::MOVB_S:
    case OpcodeType::MOVW_S:
    case OpcodeType::MOVL_S:
    case OpcodeType::MOVB_S0:
    case OpcodeType::MOVW_S0:
    case OpcodeType::MOVL_S0:
    case OpcodeType::MOVB_S4:
    case OpcodeType::MOVW_S4:
    case OpcodeType::MOVL_S4:
    case OpcodeType::MOVB_SG:
    case OpcodeType::MOVW_SG:
    case OpcodeType::MOVL_SG:
    case OpcodeType::MOVW_I:
    case OpcodeType::MOVL_I: return Kind::Mem;

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

    // Conditional branches without a delay slot: execution goes on in the same block if they aren't taken
    case OpcodeType::BT:
    case OpcodeType::BF: return Kind::CondBranch;

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
    if (block.pc != pc || block.hostSrc != src || std::memcmp(src, block.bytes.data(), block.byteLen) != 0) {
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
    block.hostSrc = src;
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
        size_t R, PC, SR, GBR, PR, DelaySlot, DelaySlotTarget, MAC, WbReg, IntrPending, IntrAllow;
        uint8 WbNone;
        uint8 WbPR;
    } const L{offsetof(SH2, R),
              offsetof(SH2, PC),
              offsetof(SH2, SR),
              offsetof(SH2, GBR),
              offsetof(SH2, PR),
              offsetof(SH2, m_delaySlot),
              offsetof(SH2, m_delaySlotTarget),
              offsetof(SH2, MAC),
              offsetof(SH2, m_wbReg),
              offsetof(SH2, m_intrFlags) + offsetof(SH2::IntrFlags, pending),
              offsetof(SH2, m_intrFlags) + offsetof(SH2::IntrFlags, allow),
              SH2::kWBRegNone,
              SH2::kWBRegPR};

    std::array<uint32, kMaxBlockWords> words{};
    Assembler a{words.data(), words.size()};
    FieldAccess field{a};

    Label epilogue;
    std::vector<std::pair<Label, uint32>> budgetExits; // label, PC to store
    budgetExits.reserve(count);

    // Whether the block has loads or stores that are translated inline, and whether any of them writes
    bool hasMem = false;
    for (uint32 k = 0; k < count; ++k) {
        hasMem = hasMem || insns[k].kind == Kind::Mem;
    }
    const sys::SH2Bus::PageLayout pages = m_bus.GetPageLayout();

    // Prologue
    a.StpPre(Sz::X, X29, X30, SP, -64);
    a.Stp(Sz::X, X19, X20, SP, 16);
    a.Stp(Sz::X, X21, X22, SP, 32);
    a.Str(3, X23, SP, 48);
    a.Mov(Sz::X, kThis, X0);
    a.Mov(Sz::X, kBudget, X1);
    a.Mov(Sz::X, kCycles, ZR);
    if (hasMem) {
        a.MovImm(Sz::X, kCodeBase, reinterpret_cast<uint64>(src));
        a.MovImm(Sz::X, kPageTable, reinterpret_cast<uint64>(pages.table));
    }
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

    // The value of the pipeline's writeback register if it is known at this point (kWBRegNone if none), or -1
    int knownWb = -1;
    // Whether the previous instruction was executed by a handler
    bool prevWasHelper = true;
    // Whether the interrupt allow flag is known to be set (the prologue sets it)
    bool allowSet = true;
    // Whether the PC stored in the CPU is behind the instruction about to be executed (inline instructions leave it)
    bool pcStale = false;

    // Emits a call to the interpreter's handler for the instruction `w` at `ipc` (the `k`-th of the block) and the
    // checks that follow it: whether code ahead in the block was overwritten, whether control left the block and
    // whether an interrupt is about to be serviced.
    auto emitHelper = [&](uint32 k, uint32 ipc, uint16 w, bool setPc) {
        if (setPc) {
            a.MovImm(Sz::W, kT0, ipc);
            a.Str(2, kT0, kThis, static_cast<uint32>(L.PC));
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
            const uint32 firstChecked = k + 1 + ((ipc & 2u) == 0 ? 1 : 0);
            if (firstChecked < count && (mem.first.write || mem.second.write)) {
                a.Mov(Sz::X, X0, kThis);
                a.MovImm(Sz::W, X1, pc);
                a.MovImm(Sz::W, X2, firstChecked * 2);
                a.MovImm(Sz::X, X16, reinterpret_cast<uint64>(&SH2Recompiler::CodeUnchangedThunk));
                a.Blr(X16);
                a.Cbz(Sz::X, X0, epilogue);
            }
        }

        if (k + 1 < count) {
            // Leave the block if the instruction didn't simply fall through (a taken branch, an exception or a
            // retried bus access)
            a.Ldr(2, kT0, kThis, static_cast<uint32>(L.PC));
            a.MovImm(Sz::W, kT1, ipc + 2);
            a.Cmp(Sz::W, kT0, kT1);
            a.BCond(Cond::NE, epilogue);

            // Leave the block if an interrupt is about to be serviced; otherwise interrupts are allowed again from
            // the next instruction on (the handler may have blocked them for this one)
            field.Load(0, kT0, L.IntrPending);
            field.Load(0, kT1, L.IntrAllow);
            a.And(Sz::W, kT0, kT0, kT1);
            a.Cbnz(Sz::W, kT0, epilogue);
        }
    };

    // Loads and stores. RAM and ROM (array backed memory in the cached and cache-through areas) are accessed directly;
    // anything else jumps to an out-of-line copy of the helper call, before any state has been changed.
    struct SlowPath {
        Label entry;
        Label join;
        uint32 k;
        uint32 pc;
        uint16 w;
    };
    struct CodeCheck {
        Label entry;
        Label back;
        uint32 pcNext;
        uint32 firstChecked;
    };
    std::deque<SlowPath> slowPaths;
    std::deque<CodeCheck> codeChecks;

    // Adds one cycle to the instruction being translated if the pipeline's writeback stage holds one of `regs`
    auto addStall = [&](std::initializer_list<uint32> regs) {
        if (knownWb >= 0) {
            for (uint32 r : regs) {
                if (static_cast<uint32>(knownWb) == r) {
                    a.AddImm(Sz::X, kCycles, kCycles, 1);
                    break;
                }
            }
            return;
        }
        field.Load(0, kTmpWb, L.WbReg);
        auto it = regs.begin();
        a.CmpImm(Sz::W, kTmpWb, *it);
        for (++it; it != regs.end(); ++it) {
            a.CcmpImm(Sz::W, kTmpWb, *it, 4, Cond::NE);
        }
        a.Cinc(Sz::X, kCycles, kCycles, Cond::EQ);
    };
    auto setWbNone = [&]() {
        if (knownWb != L.WbNone) {
            a.MovImm(Sz::W, kTmpWb, L.WbNone);
            field.Store(0, kTmpWb, L.WbReg);
            knownWb = L.WbNone;
        }
    };

    // Branches. The delay slot instruction that follows a delayed branch is executed by the handler like any other
    // instruction, with the interpreter's delay slot state set up here.
    std::deque<Label> labelPool; // labels must outlive the code that refers to them
    auto EmitBranch = [&](uint32 k, const Insn &insn) {
        const OpcodeType op = insn.opcode;
        const uint16 w = insn.word;
        const uint32 ipc = insn.pc;
        const uint32 m = (w >> 8) & 0xF; // branches use the n/m field at bits 8-11
        const sint32 disp8 = static_cast<sint8>(w & 0xFF) * 2;
        const sint32 disp12 = (static_cast<sint32>((w & 0xFFF) << 20) >> 20) * 2;
        ++m_stats.inlineInstructions;

        auto setUpDelaySlot = [&](Reg targetReg) {
            a.MovImm(Sz::W, kT1, 1);
            field.Store(0, kT1, L.DelaySlot);
            field.Store(2, targetReg, L.DelaySlotTarget);
            field.Store(0, ZR, L.IntrPending);
            a.MovImm(Sz::W, kT1, ipc + 2);
            a.Str(2, kT1, kThis, static_cast<uint32>(L.PC));
        };

        switch (op) {
        case OpcodeType::BT:
        case OpcodeType::BF: {
            setWbNone();
            const uint32 target = ipc + 4 + static_cast<uint32>(disp8);
            Label &notTaken = labelPool.emplace_back();
            a.Ldr(2, kT0, kThis, static_cast<uint32>(L.SR));
            if (op == OpcodeType::BT) {
                a.Tbz(kT0, 0, notTaken);
            } else {
                a.Tbnz(kT0, 0, notTaken);
            }
            a.AddImm(Sz::X, kCycles, kCycles, 3);
            a.MovImm(Sz::W, kT1, target);
            a.Str(2, kT1, kThis, static_cast<uint32>(L.PC));
            a.B(epilogue);
            a.Bind(notTaken);
            a.AddImm(Sz::X, kCycles, kCycles, 1);
            pcStale = true;
            break;
        }
        case OpcodeType::BTS:
        case OpcodeType::BFS: {
            setWbNone();
            const uint32 target = ipc + 4 + static_cast<uint32>(disp8);
            Label &notTaken = labelPool.emplace_back();
            a.AddImm(Sz::X, kCycles, kCycles, 1);
            a.Ldr(2, kT0, kThis, static_cast<uint32>(L.SR));
            if (op == OpcodeType::BTS) {
                a.Tbz(kT0, 0, notTaken);
            } else {
                a.Tbnz(kT0, 0, notTaken);
            }
            a.AddImm(Sz::X, kCycles, kCycles, 1);
            a.MovImm(Sz::W, kT2, target);
            setUpDelaySlot(kT2);
            a.Bind(notTaken);
            a.MovImm(Sz::W, kT1, ipc + 2);
            a.Str(2, kT1, kThis, static_cast<uint32>(L.PC));
            pcStale = false;
            break;
        }
        default: {
            // Unconditional delayed branches: 2 cycles, plus a stall for the register the target comes from
            a.AddImm(Sz::X, kCycles, kCycles, 2);
            switch (op) {
            case OpcodeType::BRA: a.MovImm(Sz::W, kT2, ipc + 4 + static_cast<uint32>(disp12)); break;
            case OpcodeType::BSR:
                addStall({L.WbPR});
                a.MovImm(Sz::W, kT2, ipc + 4 + static_cast<uint32>(disp12));
                a.MovImm(Sz::W, kT3, ipc + 4);
                a.Str(2, kT3, kThis, static_cast<uint32>(L.PR));
                break;
            case OpcodeType::BRAF:
                addStall({m});
                loadR(kT2, m);
                a.MovImm(Sz::W, kT3, ipc + 4);
                a.Add(Sz::W, kT2, kT2, kT3);
                break;
            case OpcodeType::BSRF:
                addStall({m, L.WbPR});
                loadR(kT2, m);
                a.MovImm(Sz::W, kT3, ipc + 4);
                a.Add(Sz::W, kT2, kT2, kT3);
                a.MovImm(Sz::W, kT3, ipc + 4);
                a.Str(2, kT3, kThis, static_cast<uint32>(L.PR));
                break;
            case OpcodeType::JMP:
                addStall({m});
                loadR(kT2, m);
                break;
            case OpcodeType::JSR:
                addStall({m, L.WbPR});
                loadR(kT2, m);
                a.MovImm(Sz::W, kT3, ipc + 4);
                a.Str(2, kT3, kThis, static_cast<uint32>(L.PR));
                break;
            case OpcodeType::RTS:
                addStall({L.WbPR});
                a.Ldr(2, kT2, kThis, static_cast<uint32>(L.PR));
                break;
            default: break;
            }
            setWbNone();
            setUpDelaySlot(kT2);
            pcStale = false;
            break;
        }
        }
        (void)k;
    };

    auto EmitMemory = [&](uint32 k, uint32 ipc, uint16 w) {
        const OpcodeType op = insns[k].opcode;
        const uint32 n = (w >> 8) & 0xF;
        const uint32 m = (w >> 4) & 0xF;

        enum class Mode { Reg, RegPost, RegPre, RegDisp, R0Reg, Gbr, Const };
        bool load = true;
        uint32 sizeLog2 = 0;
        Mode mode = Mode::Reg;
        uint32 baseReg = m;   // register the address is based on
        uint32 dataReg = n;   // loads: destination, stores: source
        uint32 disp = 0;      // displacement or constant address
        std::array<uint32, 2> stall{};
        uint32 numStall = 0;
        uint32 stallTimes = 1;
        auto setStall = [&](uint32 r0, uint32 r1 = 0xFFFFFFFFu) {
            stall[0] = r0;
            numStall = 1;
            if (r1 != 0xFFFFFFFFu) {
                stall[1] = r1;
                numStall = 2;
            }
        };
        const uint32 d4 = w & 0xF;
        const uint32 d8 = w & 0xFF;

        switch (op) {
        case OpcodeType::MOVB_L: sizeLog2 = 0; mode = Mode::Reg; setStall(m); break;
        case OpcodeType::MOVW_L: sizeLog2 = 1; mode = Mode::Reg; setStall(m); break;
        case OpcodeType::MOVL_L: sizeLog2 = 2; mode = Mode::Reg; setStall(m); break;
        case OpcodeType::MOVB_L0: sizeLog2 = 0; mode = Mode::R0Reg; setStall(m, 0); break;
        case OpcodeType::MOVW_L0: sizeLog2 = 1; mode = Mode::R0Reg; setStall(m, 0); break;
        case OpcodeType::MOVL_L0: sizeLog2 = 2; mode = Mode::R0Reg; setStall(m, 0); break;
        case OpcodeType::MOVB_L4: sizeLog2 = 0; mode = Mode::RegDisp; dataReg = 0; disp = d4; break;
        case OpcodeType::MOVW_L4: // the interpreter counts the stall twice
            sizeLog2 = 1; mode = Mode::RegDisp; dataReg = 0; disp = d4 << 1; setStall(m); stallTimes = 2; break;
        case OpcodeType::MOVL_L4: sizeLog2 = 2; mode = Mode::RegDisp; disp = d4 << 2; setStall(m); break;
        case OpcodeType::MOVB_LG: sizeLog2 = 0; mode = Mode::Gbr; dataReg = 0; disp = d8; break;
        case OpcodeType::MOVW_LG: sizeLog2 = 1; mode = Mode::Gbr; dataReg = 0; disp = d8 << 1; break;
        case OpcodeType::MOVL_LG: sizeLog2 = 2; mode = Mode::Gbr; dataReg = 0; disp = d8 << 2; break;
        case OpcodeType::MOVB_P: sizeLog2 = 0; mode = Mode::RegPost; setStall(m); break;
        case OpcodeType::MOVW_P: sizeLog2 = 1; mode = Mode::RegPost; setStall(m); break;
        case OpcodeType::MOVL_P: sizeLog2 = 2; mode = Mode::RegPost; setStall(m); break;
        case OpcodeType::MOVW_I:
            sizeLog2 = 1; mode = Mode::Const; disp = ipc + (d8 << 1) + 4; break;
        case OpcodeType::MOVL_I:
            sizeLog2 = 2; mode = Mode::Const; disp = (ipc & ~3u) + (d8 << 2) + 4; break;
        // Stores: the data register is Rm, the address register is Rn
        case OpcodeType::MOVB_M: load = false; sizeLog2 = 0; mode = Mode::RegPre; dataReg = m; baseReg = n; setStall(m, n); break;
        case OpcodeType::MOVW_M: load = false; sizeLog2 = 1; mode = Mode::RegPre; dataReg = m; baseReg = n; setStall(m, n); break;
        case OpcodeType::MOVL_M: load = false; sizeLog2 = 2; mode = Mode::RegPre; dataReg = m; baseReg = n; setStall(m, n); break;
        case OpcodeType::MOVB_S: load = false; sizeLog2 = 0; mode = Mode::Reg; dataReg = m; baseReg = n; setStall(m, n); break;
        case OpcodeType::MOVW_S: load = false; sizeLog2 = 1; mode = Mode::Reg; dataReg = m; baseReg = n; setStall(m, n); break;
        case OpcodeType::MOVL_S: load = false; sizeLog2 = 2; mode = Mode::Reg; dataReg = m; baseReg = n; setStall(m, n); break;
        case OpcodeType::MOVB_S0: load = false; sizeLog2 = 0; mode = Mode::R0Reg; dataReg = m; baseReg = n; setStall(n, 0); break;
        case OpcodeType::MOVW_S0: load = false; sizeLog2 = 1; mode = Mode::R0Reg; dataReg = m; baseReg = n; setStall(n, 0); break;
        case OpcodeType::MOVL_S0: load = false; sizeLog2 = 2; mode = Mode::R0Reg; dataReg = m; baseReg = n; setStall(n, 0); break;
        case OpcodeType::MOVB_S4: load = false; sizeLog2 = 0; mode = Mode::RegDisp; dataReg = 0; baseReg = m; disp = d4; setStall(m, 0); break;
        case OpcodeType::MOVW_S4: load = false; sizeLog2 = 1; mode = Mode::RegDisp; dataReg = 0; baseReg = m; disp = d4 << 1; setStall(m, 0); break;
        case OpcodeType::MOVL_S4: load = false; sizeLog2 = 2; mode = Mode::RegDisp; dataReg = m; baseReg = n; disp = d4 << 2; setStall(m, n); break;
        case OpcodeType::MOVB_SG: load = false; sizeLog2 = 0; mode = Mode::Gbr; dataReg = 0; disp = d8; setStall(0); break;
        case OpcodeType::MOVW_SG: load = false; sizeLog2 = 1; mode = Mode::Gbr; dataReg = 0; disp = d8 << 1; setStall(0); break;
        case OpcodeType::MOVL_SG: load = false; sizeLog2 = 2; mode = Mode::Gbr; dataReg = 0; disp = d8 << 2; setStall(0); break;
        default: break;
        }

        const uint32 size = 1u << sizeLog2;
        slowPaths.emplace_back();
        SlowPath &slow = slowPaths.back();
        slow.k = k;
        slow.pc = ipc;
        slow.w = w;

        // 1. The address
        switch (mode) {
        case Mode::Reg:
        case Mode::RegPost: loadR(1, baseReg); break;
        case Mode::RegPre:
            loadR(1, baseReg);
            a.SubImm(Sz::W, 1, 1, size);
            break;
        case Mode::RegDisp:
            loadR(1, baseReg);
            if (disp != 0) {
                a.AddImm(Sz::W, 1, 1, disp);
            }
            break;
        case Mode::R0Reg:
            loadR(1, baseReg);
            loadR(2, 0);
            a.Add(Sz::W, 1, 1, 2);
            break;
        case Mode::Gbr:
            a.Ldr(2, 1, kThis, static_cast<uint32>(L.GBR));
            if (disp != 0) {
                a.AddImm(Sz::W, 1, 1, disp);
            }
            break;
        case Mode::Const: a.MovImm(Sz::W, 1, disp); break;
        }

        // 2. Only the cached (0), cache-through (1) and uncached (5) areas map to the bus
        a.LsrImm(Sz::W, 2, 1, 29);
        a.MovImm(Sz::W, 3, 0x23);
        a.Lsrv(Sz::W, 3, 3, 2);
        a.Tbz(3, 0, slow.entry);

        // 3. The bus page must be backed by an array (which also means the access can't be held up)
        a.AndImm(Sz::W, 4, 1, pages.addressMask & ~(size - 1));
        a.LsrImm(Sz::W, 5, 4, pages.pageShift);
        a.Add(Sz::X, 6, kPageTable, 5, Shift::LSL, pages.entryShift);
        a.Ldr(3, 7, 6, pages.offArray);
        a.Cbz(Sz::X, 7, slow.entry);
        if (!load) {
            a.Ldr(0, 8, 6, pages.offArrayWritable);
            a.Cbz(Sz::W, 8, slow.entry);
        }
        a.AndImm(Sz::W, 5, 4, (1u << pages.pageShift) - 1);

        // 4. Cycles: one in the cached area, the page's timing in the cache-through area, plus a stall if the
        // register used for the address was just loaded
        a.Ldr(3, 3, 6, load ? pages.offReadCycles[sizeLog2] : pages.offWriteCycles[sizeLog2]);
        a.CmpImm(Sz::W, 2, 0);
        a.MovImm(Sz::W, 4, 1);
        a.Csel(Sz::X, 3, 4, 3, Cond::EQ);
        a.Add(Sz::X, kCycles, kCycles, 3);
        if (numStall > 0) {
            if (knownWb >= 0) {
                for (uint32 i = 0; i < numStall; ++i) {
                    if (static_cast<uint32>(knownWb) == stall[i]) {
                        a.AddImm(Sz::X, kCycles, kCycles, stallTimes);
                        break;
                    }
                }
            } else {
                field.Load(0, kTmpWb, L.WbReg);
                a.CmpImm(Sz::W, kTmpWb, stall[0]);
                if (numStall > 1) {
                    a.CcmpImm(Sz::W, kTmpWb, stall[1], 4, Cond::NE);
                }
                for (uint32 i = 0; i < stallTimes; ++i) {
                    a.Cinc(Sz::X, kCycles, kCycles, Cond::EQ);
                }
            }
        }

        // 5. The access itself; memory is big endian
        if (load) {
            a.LdrReg(sizeLog2, 8, 7, 5, Extend::UXTW, false);
            switch (sizeLog2) {
            case 0: a.Sxtb(Sz::W, 8, 8); break;
            case 1:
                a.Rev16(Sz::W, 8, 8);
                a.Sxth(Sz::W, 8, 8);
                break;
            default: a.Rev(Sz::W, 8, 8); break;
            }
            storeR(8, dataReg);
            if (mode == Mode::RegPost && dataReg != baseReg) {
                a.AddImm(Sz::W, 2, 1, size);
                storeR(2, baseReg);
            }
        } else {
            loadR(8, dataReg);
            switch (sizeLog2) {
            case 0: break;
            case 1: a.Rev16(Sz::W, 8, 8); break;
            default: a.Rev(Sz::W, 8, 8); break;
            }
            a.StrReg(sizeLog2, 8, 7, 5, Extend::UXTW, false);
            if (mode == Mode::RegPre) {
                storeR(1, baseReg);
            }

        }

        // 6. What the pipeline's writeback stage holds afterwards
        const uint32 newWb = load ? dataReg : static_cast<uint32>(L.WbNone);
        a.MovImm(Sz::W, kTmpWb, newWb);
        field.Store(0, kTmpWb, L.WbReg);
        knownWb = static_cast<int>(newWb);
        if (!load) {
            // A write into this block's own code: if instructions ahead of this one changed, leave the block
            const uint32 firstChecked = k + 1 + ((ipc & 2u) == 0 ? 1 : 0);
            if (firstChecked < count) {
                codeChecks.emplace_back();
                CodeCheck &check = codeChecks.back();
                check.pcNext = ipc + 2;
                check.firstChecked = firstChecked;
                a.Add(Sz::X, 2, 7, 5, Shift::LSL, 0);
                a.Sub(Sz::X, 2, 2, kCodeBase);
                a.CmpImm(Sz::X, 2, count * 2);
                a.BCond(Cond::CC, check.entry);
                a.Bind(check.back);
            }
        }
        ++m_stats.inlineInstructions;
        if (k + 1 == count) {
            // The last instruction of the block: continue at the next one. The out-of-line path leaves the PC alone
            // when the handler had to retry the access, so this happens before the two paths join
            a.MovImm(Sz::W, kT0, ipc + 2);
            a.Str(2, kT0, kThis, static_cast<uint32>(L.PC));
        }
        a.Bind(slow.join);
        pcStale = k + 1 != count;
    };

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

        if (insn.kind == Kind::Mem) {
            EmitMemory(k, insn.pc, w);
            continue;
        }
        if (insn.kind == Kind::CondBranch || (insn.kind == Kind::DelayedBranch && insn.opcode != OpcodeType::RTE)) {
            EmitBranch(k, insn);
            continue;
        }

        if (insn.kind != Kind::Inline) {
            // Executed by the interpreter's handler, which expects the CPU's PC to point at the instruction
            emitHelper(k, insn.pc, w, pcStale);
            pcStale = false;
            allowSet = false;
            knownWb = -1;
            prevWasHelper = true;
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
        if (numStallRegs > 0) {
            if (knownWb >= 0) {
                bool stalls = false;
                for (uint32 i = 0; i < numStallRegs; ++i) {
                    stalls = stalls || static_cast<uint32>(knownWb) == stallRegs[i];
                }
                if (stalls) {
                    a.AddImm(Sz::X, kCycles, kCycles, 1);
                }
            } else {
                field.Load(0, kTmpWb, L.WbReg);
                a.CmpImm(Sz::W, kTmpWb, stallRegs[0]);
                if (numStallRegs > 1) {
                    a.CcmpImm(Sz::W, kTmpWb, stallRegs[1], 4, Cond::NE); // Z=1 if the first one already matched
                }
                a.Cinc(Sz::X, kCycles, kCycles, Cond::EQ);
            }
        }
        if (knownWb != L.WbNone) {
            a.MovImm(Sz::W, kTmpWb, L.WbNone);
            field.Store(0, kTmpWb, L.WbReg);
            knownWb = L.WbNone;
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
    if (last.kind == Kind::Inline || last.kind == Kind::CondBranch) {
        a.MovImm(Sz::W, kT0, last.pc + 2);
        a.Str(2, kT0, kThis, static_cast<uint32>(L.PC));
    }

    // Epilogue
    a.Bind(epilogue);
    a.Mov(Sz::X, X0, kCycles);
    a.Ldr(3, X23, SP, 48);
    a.Ldp(Sz::X, X21, X22, SP, 32);
    a.Ldp(Sz::X, X19, X20, SP, 16);
    a.LdpPost(Sz::X, X29, X30, SP, 64);
    a.Ret();

    // Exits taken when the cycle budget runs out before an instruction
    for (auto &[label, exitPc] : budgetExits) {
        a.Bind(label);
        a.MovImm(Sz::W, kT0, exitPc);
        a.Str(2, kT0, kThis, static_cast<uint32>(L.PC));
        a.B(epilogue);
    }
    (void)prevWasHelper;

    // Out-of-line parts of loads and stores
    for (SlowPath &slow : slowPaths) {
        a.Bind(slow.entry);
        emitHelper(slow.k, slow.pc, slow.w, true);
        a.B(slow.join);
    }
    for (CodeCheck &check : codeChecks) {
        a.Bind(check.entry);
        a.MovImm(Sz::W, kT0, check.pcNext);
        a.Str(2, kT0, kThis, static_cast<uint32>(L.PC));
        a.Mov(Sz::X, X0, kThis);
        a.MovImm(Sz::W, X1, pc);
        a.MovImm(Sz::W, X2, check.firstChecked * 2);
        a.MovImm(Sz::X, X16, reinterpret_cast<uint64>(&SH2Recompiler::CodeUnchangedThunk));
        a.Blr(X16);
        a.Cbz(Sz::X, X0, epilogue);
        a.B(check.back);
    }

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
        block.hostSrc = src;
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
