#include <ymir/hw/sh2/sh2_recompiler.hpp>

#include <ymir/hw/sh2/sh2.hpp>

#include <ymir/jit/arm64_assembler.hpp>
#include <ymir/jit/jit_memory.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <bit>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
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
    // On iOS with TXM the pool can only be prepared once per process, so its size has to be one that is known to work.
    constexpr size_t kPoolSize = 64u * 1024u * 1024u;
    constexpr size_t kBigPoolSize = 256u * 1024u * 1024u;

    std::atomic<uint64> g_statBlocks{0}, g_statFlushes{0}, g_statBytes{0}, g_statInstructions{0}, g_statCompileNs{0},
        g_statSlowestNs{0};

    // Optimizations that can be switched off; changing them bumps the generation, which makes every recompiler discard
    // its code. The environment variables are for experiments.
    std::atomic<bool> g_registerCache{std::getenv("YMIR_JIT_NOCACHE") == nullptr};
    std::atomic<bool> g_blockLinking{std::getenv("YMIR_JIT_NOCHAIN") == nullptr};
    std::atomic<uint32> g_optionsGeneration{1};
    uint32 CacheMinUses() {
        static const uint32 value =
            std::getenv("YMIR_JIT_CACHEMIN") ? static_cast<uint32>(std::atoi(std::getenv("YMIR_JIT_CACHEMIN"))) : 6u;
        return value;
    }

    std::mutex g_poolMutex;
    jit::JitMemory g_pool;
    size_t g_chunkSize = 0;
    std::vector<size_t> g_freeChunks; // chunks returned by destroyed recompilers

    // Words reserved for assembling a block before it is copied into executable memory
    constexpr size_t kMaxBlockWords = 8000;

    // Registers used by generated code
    constexpr Reg kThis = X19;   // SH2 *
    constexpr Reg kCycles = X20; // cycles executed so far in this block
    constexpr Reg kBudget = X21; // cycles that may be executed
    constexpr Reg kCodeBase = X22;  // host address of the block's code in the emulated memory
    constexpr Reg kPageTable = X23; // bus page table
    // Callee-saved registers that hold the most used SH-2 registers while a block runs
    constexpr uint32 kFirstCached = 24;
    constexpr uint32 kLastCached = 28;
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
    if (!g_pool.IsInitialized()) {
        const bool txm = jit::ProbeJit().txmDevice;
        if (!txm && g_pool.Initialize(kBigPoolSize)) {
            // plenty of room
        } else if (!g_pool.Initialize(kPoolSize)) {
            return;
        }
        // Master and slave SH-2 get half each (a little less, so that both fit)
        g_chunkSize = (g_pool.Capacity() / 2 - 64u * 1024u) & ~size_t{0xFFFF};
    }
    size_t offset;
    if (!g_freeChunks.empty()) {
        offset = g_freeChunks.back();
        g_freeChunks.pop_back();
    } else {
        offset = g_pool.Allocate(g_chunkSize);
        if (offset == SIZE_MAX) {
            return;
        }
    }
    m_chunkOffset = offset;
    m_optionsGeneration = g_optionsGeneration.load(std::memory_order_relaxed);
    m_chunkSize = g_chunkSize;
    {
        const size_t bytes = (sizeof(Block) * kBlockTableSize + 4095) & ~size_t{4095};
        Block *table = static_cast<Block *>(std::aligned_alloc(4096, bytes));
        if (table == nullptr) {
            std::lock_guard lock{g_poolMutex};
            g_freeChunks.push_back(m_chunkOffset);
            return;
        }
        for (size_t i = 0; i < kBlockTableSize; ++i) {
            new (&table[i]) Block();
        }
        m_blocks.reset(table);
    }
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
    for (uint32 i = 0; i < kBlockTableSize; ++i) {
        Block &block = m_blocks[i];
        block.pc = 1;
        block.fn = nullptr;
        block.chainEntry = nullptr;
    }
    m_chunkUsed = 0;
    ++m_stats.cacheFlushes;
    g_statFlushes.fetch_add(1, std::memory_order_relaxed);
}

void SH2Recompiler::SetOptions(bool registerCache, bool blockLinking) {
    if (g_registerCache.exchange(registerCache) != registerCache | g_blockLinking.exchange(blockLinking) != blockLinking) {
        g_optionsGeneration.fetch_add(1, std::memory_order_relaxed);
    }
}

SH2Recompiler::GlobalStats SH2Recompiler::GetGlobalStats() {
    return {g_statBlocks.load(std::memory_order_relaxed), g_statFlushes.load(std::memory_order_relaxed),
            g_statBytes.load(std::memory_order_relaxed), g_statInstructions.load(std::memory_order_relaxed),
            g_statCompileNs.load(std::memory_order_relaxed), g_statSlowestNs.load(std::memory_order_relaxed)};
}

void SH2Recompiler::GetBlockTableRegion(void **base, size_t *size) const {
    *base = m_blocks.get();
    *size = (sizeof(Block) * kBlockTableSize + 4095) & ~size_t{4095};
}

void SH2Recompiler::GetPoolRegion(uint8 **base, size_t *size) {
    std::lock_guard lock{g_poolMutex};
    *base = g_pool.IsInitialized() ? g_pool.ExecPtr(0) : nullptr;
    *size = g_pool.IsInitialized() ? g_pool.Capacity() : 0;
}

uint64 SH2Recompiler::StepThunk(SH2 *sh2, uint32 instr) {
    sh2->m_jitImpure = true;
    return sh2->JitStep(static_cast<uint16>(instr));
}

uint64 SH2Recompiler::CodeUnchangedThunk(SH2 *sh2, uint32 blockPc, uint32 offset, uint32 storePc) {
    SH2Recompiler *self = sh2->m_recompiler.get();
    const Block &block = self->m_blocks[SlotFor(blockPc)];
    if (block.pc != blockPc) {
        return 0;
    }
    const uint8 *src = self->m_bus.GetArrayPointer(blockPc);
    if (src == nullptr) {
        return 0;
    }
    const bool unchanged =
        offset >= block.byteLen || std::memcmp(src + offset, block.bytes.data() + offset, block.byteLen - offset) == 0;

    // If the write came from an instruction at an aligned address, the interpreter has already fetched the longword it
    // belongs to, so the instruction after it stays the old one even if that is what was overwritten. Remember it for
    // when the block is left there.
    const uint32 knownLen = block.paddedLen != 0 ? block.paddedLen : block.byteLen;
    if (storePc != 0xFFFFFFFFu && (storePc & 3u) == 0 && storePc >= blockPc && storePc - blockPc + 4 <= knownLen) {
        const uint32 pairOffset = storePc - blockPc + 2;
        if (std::memcmp(src + pairOffset, block.bytes.data() + pairOffset, 2) != 0) {
            const uint8 *b = block.bytes.data() + (storePc - blockPc);
            self->m_prefetch = (static_cast<uint32>(b[0]) << 24) | (static_cast<uint32>(b[1]) << 16) |
                               (static_cast<uint32>(b[2]) << 8) | b[3];
            self->m_prefetchPc = storePc + 2;
            self->m_prefetchValid = true;
        }
    }
    return unchanged ? 1 : 0;
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
    case OpcodeType::TST_I:
    case OpcodeType::MUL:
    case OpcodeType::MULS:
    case OpcodeType::MULU:
    case OpcodeType::DMULS:
    case OpcodeType::DMULU:
    case OpcodeType::DIV0S:
    case OpcodeType::DIV1:
    case OpcodeType::CMP_STR:
    case OpcodeType::LDC_GBR_R:
    case OpcodeType::LDC_VBR_R:
    case OpcodeType::LDS_MACH_R:
    case OpcodeType::LDS_MACL_R:
    case OpcodeType::LDS_PR_R:
    case OpcodeType::STC_GBR_R:
    case OpcodeType::STC_SR_R:
    case OpcodeType::STC_VBR_R:
    case OpcodeType::STS_MACH_R:
    case OpcodeType::STS_MACL_R:
    case OpcodeType::STS_PR_R: return Kind::Inline;

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
    case OpcodeType::MOVL_I:
    case OpcodeType::LDC_GBR_M:
    case OpcodeType::LDC_VBR_M:
    case OpcodeType::LDS_MACH_M:
    case OpcodeType::LDS_MACL_M:
    case OpcodeType::LDS_PR_M:
    case OpcodeType::STC_GBR_M:
    case OpcodeType::STC_SR_M:
    case OpcodeType::STC_VBR_M:
    case OpcodeType::STS_MACH_M:
    case OpcodeType::STS_MACL_M:
    case OpcodeType::STS_PR_M: return Kind::Mem;

    case OpcodeType::MACW:
    case OpcodeType::MACL: return Kind::Mac;

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

    if (m_optionsGeneration != g_optionsGeneration.load(std::memory_order_relaxed)) {
        m_optionsGeneration = g_optionsGeneration.load(std::memory_order_relaxed);
        Flush();
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
    // An instruction in the second half of a longword comes from the prefetch buffer, which holds what was in memory
    // when the longword was fetched. If the code was rewritten since, the interpreter has to run it.
    if ((pc & 2u) != 0 && static_cast<uint16>((src[0] << 8) | src[1]) != static_cast<uint16>(m_sh2.m_fetchedOpcodes)) {
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
    m_prefetchValid = false;
    const uint64 cycles = m_executor ? m_executor(block.fn, m_sh2, cyclesLeft) : block.fn(&m_sh2, cyclesLeft);
    if (cycles != 0 && (m_sh2.PC & 2) != 0) {
        // The interpreter expects the longword holding the next instruction to be in the prefetch buffer
        if (m_prefetchValid && m_sh2.PC == m_prefetchPc) {
            m_sh2.m_fetchedOpcodes = m_prefetch;
        } else {
            m_sh2.JitRefillPipeline();
        }
    }
    return cycles;
}

void SH2Recompiler::Compile(Block &block, uint32 pc, const uint8 *src, uint32 bytesAvailable) {
    const auto started = std::chrono::steady_clock::now();
    CompileBlock(block, pc, src, bytesAvailable);
    const uint64 ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count();
    g_statCompileNs.fetch_add(ns, std::memory_order_relaxed);
    uint64 slowest = g_statSlowestNs.load(std::memory_order_relaxed);
    while (ns > slowest && !g_statSlowestNs.compare_exchange_weak(slowest, ns, std::memory_order_relaxed)) {
    }
}

void SH2Recompiler::CompileBlock(Block &block, uint32 pc, const uint8 *src, uint32 bytesAvailable) {
    struct Insn {
        uint16 word;
        OpcodeType opcode;
        Kind kind;
        uint32 pc;
        // For delay slots: how the instruction decodes when the branch before it isn't taken (BT/S and BF/S)
        OpcodeType normalOpcode = OpcodeType::Illegal;
        Kind normalKind = Kind::Helper;
    };
    // Whether the instruction in a delay slot is translated inline. Register-only instructions and loads and stores
    // are, except for the PC-relative loads, whose address depends on the delay slot's target.
    auto canInlineSlot = [](const Insn &slot) {
        if (slot.normalKind == Kind::Inline) {
            return slot.normalOpcode != OpcodeType::MOVA;
        }
        return slot.normalKind == Kind::Mem && slot.normalOpcode != OpcodeType::MOVW_I &&
               slot.normalOpcode != OpcodeType::MOVL_I;
    };
    std::array<Insn, kMaxInstructions + 1> insns{};
    uint32 count = 0;

    // -- Decode --

    bool inDelaySlot = false;
    bool done = false;
    const uint32 maxBytes = std::min<uint32>(bytesAvailable, kMaxInstructions * 2);
    for (uint32 offset = 0; offset + 2 <= maxBytes && !done; offset += 2) {
        // Code that already has a block of its own is not translated again: this block ends here and continues in it
        if (offset > 0 && !inDelaySlot) {
            const Block &existing = m_blocks[SlotFor(pc + offset)];
            if (existing.pc == pc + offset && existing.fn != nullptr) {
                break;
            }
        }
        const uint16 word = static_cast<uint16>((src[offset] << 8) | src[offset + 1]);
        Insn insn{.word = word, .opcode = OpcodeType::Illegal, .kind = Kind::Helper, .pc = pc + offset};
        if (inDelaySlot) {
            // Delay slots always go through the interpreter's handlers
            insn.opcode = DecodeTable::s_instance.opcodes[1][word];
            insn.kind = Kind::Helper;
            insn.normalOpcode = DecodeTable::s_instance.opcodes[0][word];
            insn.normalKind = Classify(insn.normalOpcode);
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
    auto fillBlock = [&] {
        block.pc = pc;
        block.hostSrc = src;
        block.chainEntry = nullptr;
        block.byteLen = std::max<uint32>(count * 2, 2);
        block.byteLen = std::min<uint32>(block.byteLen, bytesAvailable);
        std::memcpy(block.bytes.data(), src, block.byteLen);
        // Other blocks that jump straight into this one compare it in chunks of 16 bytes
        const uint32 padded = (block.byteLen + 2 + 15) & ~15u; // (at least the instruction after the block)
        if (padded <= bytesAvailable) {
            std::memcpy(block.bytes.data() + block.byteLen, src + block.byteLen, padded - block.byteLen);
            block.paddedLen = padded;
        } else {
            block.paddedLen = 0;
        }
    };
    fillBlock();
    block.fn = nullptr;

    if (count == 0) {
        ++m_stats.blocksRejected;
        return;
    }

    // -- Branches that stay inside the block --
    // Loops are common in hot code; a branch to an instruction of the same block is compiled as a jump
    std::array<sint32, kMaxInstructions + 1> jumpIndex{};
    std::array<bool, kMaxInstructions + 1> isTarget{};
    std::array<bool, kMaxInstructions + 1> isSlot{};
    jumpIndex.fill(-1);
    bool hasLoop = false;
    for (uint32 k = 1; k < count; ++k) {
        isSlot[k] = insns[k - 1].kind == Kind::DelayedBranch;
    }
    for (uint32 k = 0; k < count; ++k) {
        const Insn &insn = insns[k];
        const uint16 w = insn.word;
        sint32 disp = 0;
        switch (insn.opcode) {
        case OpcodeType::BT:
        case OpcodeType::BF:
        case OpcodeType::BTS:
        case OpcodeType::BFS: disp = static_cast<sint8>(w & 0xFF) * 2; break;
        case OpcodeType::BRA: disp = (static_cast<sint32>((w & 0xFFF) << 20) >> 20) * 2; break;
        default: continue;
        }
        if (insn.kind == Kind::DelayedBranch) {
            // The delay slot has to be translated inline too, since that's where the jump is emitted
            if (k + 1 >= count || !canInlineSlot(insns[k + 1])) {
                continue;
            }
        }
        const uint32 target = insn.pc + 4 + static_cast<uint32>(disp);
        if (target < pc || target >= pc + count * 2) {
            continue;
        }
        const uint32 index = (target - pc) / 2;
        if (isSlot[index]) {
            continue;
        }
        jumpIndex[k] = static_cast<sint32>(index);
        isTarget[index] = true;
        hasLoop = true;
    }

    // Loading SR can leave an interrupt pending that is serviced a few instructions later, which only straight-line
    // code is checked for: blocks that do it neither loop nor jump into other blocks
    bool hasLoadSr = false;
    for (uint32 k = 0; k < count; ++k) {
        hasLoadSr = hasLoadSr || insns[k].opcode == OpcodeType::LDC_SR_R || insns[k].opcode == OpcodeType::LDC_SR_M;
    }
    if (hasLoadSr) {
        jumpIndex.fill(-1);
        isTarget.fill(false);
        hasLoop = false;
    }

    // -- Generate --

    // Registers that stay in host registers during the block are chosen after a first pass that counts how often each
    // one is used. Generating is done twice: once to count, once for real.
    std::array<uint32, kMaxBlockWords> words{};
    auto generate = [&](const std::array<sint32, 16> &cacheHost, std::array<uint32, 16> &useCount, size_t &outWords,
                        size_t &outChainWord) -> bool {

    // Offsets of the CPU state that generated code reads and writes
    struct {
        size_t R, PC, SR, GBR, VBR, PR, IntcLevel, DelaySlot, DelaySlotTarget, MAC, WbReg, IntrPending, IntrAllow, Impure;
        uint8 WbNone;
        uint8 WbPR;
    } const L{offsetof(SH2, R),
              offsetof(SH2, PC),
              offsetof(SH2, SR),
              offsetof(SH2, GBR),
              offsetof(SH2, VBR),
              offsetof(SH2, PR),
              offsetof(SH2, INTC) + offsetof(InterruptController, pending) +
                  offsetof(InterruptController::PendingInterruptInfo, level),
              offsetof(SH2, m_delaySlot),
              offsetof(SH2, m_delaySlotTarget),
              offsetof(SH2, MAC),
              offsetof(SH2, m_wbReg),
              offsetof(SH2, m_intrFlags) + offsetof(SH2::IntrFlags, pending),
              offsetof(SH2, m_intrFlags) + offsetof(SH2::IntrFlags, allow),
              offsetof(SH2, m_jitImpure),
              SH2::kWBRegNone,
              SH2::kWBRegPR};

    Assembler a{words.data(), words.size()};
    FieldAccess field{a};

    Label epilogue;
    Label chainLabel;
    std::deque<Label> labelPool; // labels must outlive the code that refers to them
    std::deque<Label> instrLabels(count); // targets of branches inside the block
    std::deque<std::pair<Label, uint32>> budgetExits; // label, PC to store; references to labels must stay valid

    // Whether the block has loads or stores that are translated inline, and whether any of them writes
    bool hasMem = false;
    for (uint32 k = 0; k < count; ++k) {
        hasMem = hasMem || insns[k].kind == Kind::Mem || insns[k].kind == Kind::Mac || insns[k].normalKind == Kind::Mem;
    }
    // Whether any instruction writes to memory in generated code; idle detection has to know
    auto isStore = [](OpcodeType op) {
        switch (op) {
        case OpcodeType::MOVB_S:
        case OpcodeType::MOVW_S:
        case OpcodeType::MOVL_S:
        case OpcodeType::MOVB_M:
        case OpcodeType::MOVW_M:
        case OpcodeType::MOVL_M:
        case OpcodeType::MOVB_S0:
        case OpcodeType::MOVW_S0:
        case OpcodeType::MOVL_S0:
        case OpcodeType::MOVB_S4:
        case OpcodeType::MOVW_S4:
        case OpcodeType::MOVL_S4:
        case OpcodeType::MOVB_SG:
        case OpcodeType::MOVW_SG:
        case OpcodeType::MOVL_SG:
        case OpcodeType::STC_GBR_M:
        case OpcodeType::STC_SR_M:
        case OpcodeType::STC_VBR_M:
        case OpcodeType::STS_MACH_M:
        case OpcodeType::STS_MACL_M:
        case OpcodeType::STS_PR_M: return true;
        default: return false;
        }
    };
    bool hasStore = false;
    for (uint32 k = 0; k < count; ++k) {
        hasStore = hasStore || isStore(insns[k].opcode);
    }
    const sys::SH2Bus::PageLayout pages = m_bus.GetPageLayout();

    // SH-2 registers that live in host registers: the host register, or -1
    auto cached = [&](uint32 index) { return cacheHost[index]; };
    // Writes them back to the CPU state, which the interpreter's handlers and the code that called the block read
    auto flushRegs = [&] {
        for (uint32 r = 0; r < 16; ++r) {
            if (cached(r) >= 0) {
                a.Str(2, static_cast<Reg>(cached(r)), kThis, static_cast<uint32>(L.R + 4 * r));
            }
        }
    };
    auto reloadRegs = [&] {
        for (uint32 r = 0; r < 16; ++r) {
            if (cached(r) >= 0) {
                a.Ldr(2, static_cast<Reg>(cached(r)), kThis, static_cast<uint32>(L.R + 4 * r));
            }
        }
    };

    // Prologue
    a.StpPre(Sz::X, X29, X30, SP, -96);
    a.Stp(Sz::X, X19, X20, SP, 16);
    a.Stp(Sz::X, X21, X22, SP, 32);
    a.Stp(Sz::X, X23, X24, SP, 48);
    a.Stp(Sz::X, X25, X26, SP, 64);
    a.Stp(Sz::X, X27, X28, SP, 80);
    a.Mov(Sz::X, kThis, X0);
    a.Mov(Sz::X, kBudget, X1);
    a.Mov(Sz::X, kCycles, ZR);
    // Blocks that jump here have already set up the registers above
    a.Bind(chainLabel);
    reloadRegs();
    a.MovImm(Sz::X, kCodeBase, reinterpret_cast<uint64>(src));
    if (hasMem) {
        a.MovImm(Sz::X, kPageTable, reinterpret_cast<uint64>(pages.table));
    }
    // Like the interpreter, interrupts are allowed again from the first instruction on
    a.MovImm(Sz::W, kT0, 1);
    field.Store(0, kT0, L.IntrAllow);
    if (hasStore) {
        // kT0 is still 1
        field.Store(0, kT0, L.Impure);
    }

    auto loadR = [&](Reg dst, uint32 index) {
        ++useCount[index];
        if (cached(index) >= 0) {
            a.Mov(Sz::W, dst, static_cast<Reg>(cached(index)));
        } else {
            a.Ldr(2, dst, kThis, static_cast<uint32>(L.R + 4 * index));
        }
    };
    auto storeR = [&](Reg src, uint32 index) {
        ++useCount[index];
        if (cached(index) >= 0) {
            a.Mov(Sz::W, static_cast<Reg>(cached(index)), src);
        } else {
            a.Str(2, src, kThis, static_cast<uint32>(L.R + 4 * index));
        }
    };
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
    // The instruction before which the block has to check for an interrupt that became serviceable
    uint32 intrCheckAt = 0xFFFFFFFFu;

    // Emits a call to the interpreter's handler for the instruction `w` at `ipc` (the `k`-th of the block) and the
    // checks that follow it: whether code ahead in the block was overwritten, whether control left the block and
    // whether an interrupt is about to be serviced.
    auto emitHelper = [&](uint32 k, uint32 ipc, uint16 w, bool setPc) {
        if (setPc) {
            a.MovImm(Sz::W, kT0, ipc);
            a.Str(2, kT0, kThis, static_cast<uint32>(L.PC));
        }
        flushRegs();
        a.Mov(Sz::X, X0, kThis);
        a.MovImm(Sz::W, X1, w);
        a.MovImm(Sz::X, X16, reinterpret_cast<uint64>(&SH2Recompiler::StepThunk));
        a.Blr(X16);
        reloadRegs();
        a.Add(Sz::X, kCycles, kCycles, X0);
        ++m_stats.helperInstructions;
        // A store may have rewritten instructions further down this very block. The interpreter would execute the
        // new ones, so leave the block (the next entry recompiles it) if the rest changed
        {
            const auto &mem = DecodeTable::s_instance.mem[w];
            // The interpreter fetches whole longwords: an instruction at an aligned address has already fetched the
            // word after it, so changes to that one are not noticed
            const uint32 firstChecked = hasLoop ? 0 : k + 1 + ((ipc & 2u) == 0 ? 1 : 0);
            // (a store at the end of the block can still rewrite the instruction after it, which the interpreter has fetched)
            const bool pairBeyondBlock = (ipc & 2u) == 0 && k + 1 == count;
            if ((hasLoop || firstChecked < count || pairBeyondBlock) && (mem.first.write || mem.second.write)) {
                a.Mov(Sz::X, X0, kThis);
                a.MovImm(Sz::W, X1, pc);
                a.MovImm(Sz::W, X2, firstChecked * 2);
                a.MovImm(Sz::W, X3, ipc);
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

    // Continues in the block for `targetPc` without going back to the interpreter's loop, if that block exists and the
    // code in memory still matches it; otherwise leaves this block like the caller's fallback would. Only targets that
    // need no extra bookkeeping qualify: an aligned instruction in the same bus page. Returns false (emitting nothing)
    // if the target doesn't qualify. The PC must already be stored if `pcStored`.
    auto emitChain = [&](uint32 targetPc, bool pcStored, bool checkIntr) -> bool {
        if (!g_blockLinking.load(std::memory_order_relaxed) || hasLoadSr || (targetPc & 3u) != 0 || (targetPc >> 16) != (pc >> 16) || targetPc == pc) {
            return false;
        }
        Label *failLabel = &epilogue;
        if (!pcStored) {
            budgetExits.emplace_back();
            budgetExits.back().second = targetPc;
            failLabel = &budgetExits.back().first;
        }
        Label &fail = *failLabel;
        a.Cmp(Sz::X, kCycles, kBudget);
        a.BCond(Cond::CS, fail);
        if (checkIntr) {
            field.Load(0, kT0, L.IntrPending);
            a.Cbnz(Sz::W, kT0, fail);
        }
        const Block *target = &m_blocks[SlotFor(targetPc)];
        a.MovImm(Sz::X, 2, reinterpret_cast<uint64>(target));
        a.Ldr(2, 3, 2, static_cast<uint32>(offsetof(Block, pc)));
        a.MovImm(Sz::W, 4, targetPc);
        a.Cmp(Sz::W, 3, 4);
        a.BCond(Cond::NE, fail);
        a.Ldr(3, 5, 2, static_cast<uint32>(offsetof(Block, chainEntry)));
        a.Cbz(Sz::X, 5, fail);
        // Same place in memory as when the block was compiled
        a.Ldr(3, 6, 2, static_cast<uint32>(offsetof(Block, hostSrc)));
        a.MovImm(Sz::X, 7, static_cast<uint64>(static_cast<sint64>(static_cast<sint32>(targetPc - pc))));
        a.Add(Sz::X, 7, kCodeBase, 7);
        a.Cmp(Sz::X, 6, 7);
        a.BCond(Cond::NE, fail);
        // Same code as when the block was compiled
        a.Ldr(2, 8, 2, static_cast<uint32>(offsetof(Block, paddedLen)));
        a.AddImm(Sz::X, 9, 2, static_cast<uint32>(offsetof(Block, bytes)));
        Label &loop = labelPool.emplace_back();
        a.Bind(loop);
        a.Ldp(Sz::X, 10, 11, 7, 0);
        a.Ldp(Sz::X, 12, 13, 9, 0);
        a.Cmp(Sz::X, 10, 12);
        a.Ccmp(Sz::X, 11, 13, 0, Cond::EQ);
        a.BCond(Cond::NE, fail);
        a.AddImm(Sz::X, 7, 7, 16);
        a.AddImm(Sz::X, 9, 9, 16);
        a.SubsImm(Sz::W, 8, 8, 16);
        a.BCond(Cond::HI, loop);
        flushRegs();
        a.Br(5);
        ++m_stats.chainSites;
        return true;
    };

    // What the interpreter does when a delay slot instruction is done: continue at the branch target, leave the delay
    // slot and work out whether an interrupt held back during the slot can be taken now
    auto delayEnd = [&] {
        field.Load(2, kT0, L.DelaySlotTarget);
        a.Str(2, kT0, kThis, static_cast<uint32>(L.PC));
        field.Store(0, ZR, L.DelaySlot);
        field.Load(0, kT1, L.IntcLevel);
        a.Ldr(2, kT2, kThis, static_cast<uint32>(L.SR));
        a.Ubfx(Sz::W, kT2, kT2, 4, 4);
        a.Cmp(Sz::W, kT1, kT2);
        a.Cset(Sz::W, kT1, Cond::HI);
        field.Store(0, kT1, L.IntrPending);
    };

    // Loads and stores. RAM and ROM (array backed memory in the cached and cache-through areas) are accessed directly;
    // anything else jumps to an out-of-line copy of the helper call, before any state has been changed.
    struct SlowPath {
        Label entry;
        Label join;
        uint32 k;
        uint32 pc;
        uint16 w;
        bool delaySlot;
    };
    struct CodeCheck {
        Label entry;
        Label back;
        uint32 pcNext;
        uint32 firstChecked;
        bool delaySlot;
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

    // Jumps to an instruction of this block: leaves the block instead if an interrupt is about to be taken or the
    // cycle budget is used up, like the interpreter's loop would before the next instruction
    auto emitJump = [&](uint32 k, sint32 index, uint32 targetPc, bool pcStored) {
        auto exitTo = [&]() -> Label & {
            if (pcStored) {
                return epilogue;
            }
            budgetExits.emplace_back();
            budgetExits.back().second = targetPc;
            return budgetExits.back().first;
        };
        if (k + 1 >= intrCheckAt || pcStored) {
            // (after a delay slot the interrupt flags were just recomputed)
            field.Load(0, kT0, L.IntrPending);
            field.Load(0, kT1, L.IntrAllow);
            a.And(Sz::W, kT0, kT0, kT1);
            a.Cbnz(Sz::W, kT0, exitTo());
        }
        a.Cmp(Sz::X, kCycles, kBudget);
        a.BCond(Cond::CS, exitTo());
        a.B(instrLabels[index]);
    };

    // Branches. The delay slot instruction that follows a delayed branch is executed by the handler like any other
    // instruction, with the interpreter's delay slot state set up here.
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
            if (jumpIndex[k] >= 0) {
                emitJump(k, jumpIndex[k], target, false);
            } else {
                a.MovImm(Sz::W, kT1, target);
                a.Str(2, kT1, kThis, static_cast<uint32>(L.PC));
                if (!emitChain(target, true, true)) {
                    a.B(epilogue);
                }
            }
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

    auto EmitMemory = [&](uint32 k, uint32 ipc, uint16 w, OpcodeType op, bool delaySlot) {
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
        // Control and system register transfers use a CPU register instead of Rn/Rm as the data, add cycles and keep
        // interrupts from being serviced after the instruction
        size_t specialOff = 0;
        bool special = false;
        uint32 extraCycles = 0;
        int wbOverride = -1;
        auto setSpecialLoad = [&](size_t off, uint32 extra, int wb) {
            // lds.l/ldc.l @Rm+, reg (Rm is in bits 8-11)
            special = true;
            specialOff = off;
            extraCycles = extra;
            wbOverride = wb;
            sizeLog2 = 2;
            mode = Mode::RegPost;
            baseReg = n;
            setStall(n);
        };
        auto setSpecialStore = [&](size_t off, uint32 extra, bool alsoPR) {
            // sts.l/stc.l reg, @-Rn
            special = true;
            specialOff = off;
            extraCycles = extra;
            load = false;
            sizeLog2 = 2;
            mode = Mode::RegPre;
            baseReg = n;
            if (alsoPR) {
                setStall(n, L.WbPR);
            } else {
                setStall(n);
            }
        };

        switch (op) {
        case OpcodeType::LDC_GBR_M: setSpecialLoad(L.GBR, 2, -1); break;
        case OpcodeType::LDC_VBR_M: setSpecialLoad(L.VBR, 2, -1); break;
        case OpcodeType::LDS_MACH_M: setSpecialLoad(L.MAC + 4, 0, -1); break;
        case OpcodeType::LDS_MACL_M: setSpecialLoad(L.MAC, 0, -1); break;
        case OpcodeType::LDS_PR_M: setSpecialLoad(L.PR, 0, L.WbPR); break;
        case OpcodeType::STC_GBR_M: setSpecialStore(L.GBR, 2, false); break;
        case OpcodeType::STC_SR_M: setSpecialStore(L.SR, 2, false); break;
        case OpcodeType::STC_VBR_M: setSpecialStore(L.VBR, 2, false); break;
        case OpcodeType::STS_MACH_M: setSpecialStore(L.MAC + 4, 0, false); break;
        case OpcodeType::STS_MACL_M: setSpecialStore(L.MAC, 0, false); break;
        case OpcodeType::STS_PR_M: setSpecialStore(L.PR, 0, true); break;
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
        slow.delaySlot = delaySlot;

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
        if (extraCycles != 0) {
            a.AddImm(Sz::X, kCycles, kCycles, extraCycles);
        }
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
            if (special) {
                field.Store(2, 8, specialOff);
            } else {
                storeR(8, dataReg);
            }
            if (mode == Mode::RegPost && (special || dataReg != baseReg)) {
                a.AddImm(Sz::W, 2, 1, size);
                storeR(2, baseReg);
            }
        } else {
            if (special) {
                field.Load(2, 8, specialOff);
            } else {
                loadR(8, dataReg);
            }
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
        const uint32 newWb = special ? (wbOverride >= 0 ? static_cast<uint32>(wbOverride) : static_cast<uint32>(L.WbNone))
                                     : (load ? dataReg : static_cast<uint32>(L.WbNone));
        a.MovImm(Sz::W, kTmpWb, newWb);
        field.Store(0, kTmpWb, L.WbReg);
        knownWb = static_cast<int>(newWb);
        if (special) {
            field.Store(0, ZR, L.IntrAllow);
        }
        if (!load) {
            // A write into this block's own code: if instructions ahead of this one changed, leave the block
            const uint32 firstChecked = hasLoop ? 0 : k + 1 + ((ipc & 2u) == 0 ? 1 : 0);
            const bool pairBeyondBlock = (ipc & 2u) == 0 && k + 1 == count;
            if (hasLoop || firstChecked < count || pairBeyondBlock) {
                codeChecks.emplace_back();
                CodeCheck &check = codeChecks.back();
                check.pcNext = ipc + 2;
                check.firstChecked = firstChecked;
                check.delaySlot = delaySlot;
                // The write [hp, hp + size) overlaps the code [base, base + len) if hp - base + (size - 1) < len + size - 1
                a.Add(Sz::X, 2, 7, 5, Shift::LSL, 0);
                a.Sub(Sz::X, 2, 2, kCodeBase);
                a.AddImm(Sz::X, 2, 2, size - 1);
                a.CmpImm(Sz::X, 2, count * 2 + 2 + size - 1);
                a.BCond(Cond::CC, check.entry);
                a.Bind(check.back);
            }
        }
        ++m_stats.inlineInstructions;
        if (k + 1 == count && !delaySlot) {
            // The last instruction of the block: continue at the next one. The out-of-line path leaves the PC alone
            // when the handler had to retry the access, so this happens before the two paths join
            a.MovImm(Sz::W, kT0, ipc + 2);
            a.Str(2, kT0, kThis, static_cast<uint32>(L.PC));
        }
        a.Bind(slow.join);
        if (special) {
            allowSet = false;
        }
        pcStale = k + 1 != count && !delaySlot;
    };

    // MAC.W and MAC.L: two reads from RAM, a multiply and an accumulate, with the interpreter's quirks (the second
    // address is read after the first register was advanced, saturation rules, one extra cycle)
    auto EmitMac = [&](uint32 k, uint32 ipc, uint16 w, OpcodeType op) {
        const uint32 n = (w >> 8) & 0xF;
        const uint32 m = (w >> 4) & 0xF;
        const bool isWord = op == OpcodeType::MACW;
        const uint32 sizeLog2 = isWord ? 1 : 2;
        const uint32 size = 1u << sizeLog2;

        slowPaths.emplace_back();
        SlowPath &slow = slowPaths.back();
        slow.k = k;
        slow.pc = ipc;
        slow.w = w;
        slow.delaySlot = false;

        // Addresses: @Rn first, then @Rm, which is the advanced Rn if both are the same register
        loadR(1, n);
        if (m == n) {
            a.AddImm(Sz::W, 2, 1, size);
        } else {
            loadR(2, m);
        }
        a.MovImm(Sz::X, 3, 0); // cycles
        auto access = [&](Reg addr, Reg ptr) {
            a.LsrImm(Sz::W, 4, addr, 29);
            a.MovImm(Sz::W, 9, 0x23);
            a.Lsrv(Sz::W, 9, 9, 4);
            a.Tbz(9, 0, slow.entry);
            a.AndImm(Sz::W, 5, addr, pages.addressMask & ~(size - 1));
            a.LsrImm(Sz::W, 9, 5, pages.pageShift);
            a.Add(Sz::X, 6, kPageTable, 9, Shift::LSL, pages.entryShift);
            a.Ldr(3, ptr, 6, pages.offArray);
            a.Cbz(Sz::X, ptr, slow.entry);
            a.AndImm(Sz::W, 5, 5, (1u << pages.pageShift) - 1);
            a.Add(Sz::X, ptr, ptr, 5);
            a.Ldr(3, 9, 6, pages.offReadCycles[sizeLog2]);
            a.CmpImm(Sz::W, 4, 0);
            a.MovImm(Sz::W, 10, 1);
            a.Csel(Sz::X, 9, 10, 9, Cond::EQ);
            a.Add(Sz::X, 3, 3, 9);
        };
        access(2, 8);
        // (the order of the two accesses doesn't matter for timing, but the first address is the one in w1)
        access(1, 7);
        // 7 = host pointer for @Rn (op2), 8 = host pointer for @Rm (op1)

        a.Add(Sz::X, kCycles, kCycles, 3);
        a.AddImm(Sz::X, kCycles, kCycles, 1);
        addStall({m, n});

        // Operands, sign extended
        a.LdrReg(sizeLog2, 10, 7, ZR, Extend::LSL, false); // op2
        a.LdrReg(sizeLog2, 11, 8, ZR, Extend::LSL, false); // op1
        if (isWord) {
            a.Rev16(Sz::W, 10, 10);
            a.Rev16(Sz::W, 11, 11);
            a.Sxth(Sz::W, 10, 10);
            a.Sxth(Sz::W, 11, 11);
        } else {
            a.Rev(Sz::W, 10, 10);
            a.Rev(Sz::W, 11, 11);
        }

        // Rn and Rm advance, Rn first
        loadR(1, n);
        a.AddImm(Sz::W, 1, 1, size);
        storeR(1, n);
        loadR(2, m);
        a.AddImm(Sz::W, 2, 2, size);
        storeR(2, m);

        Label &noSat = labelPool.emplace_back();
        Label &done = labelPool.emplace_back();
        if (isWord) {
            a.Mul(Sz::W, 12, 10, 11);
            a.Sxtw(12, 12); // the product fits in 32 bits
            a.Ldr(2, 13, kThis, static_cast<uint32>(L.SR));
            Label &plain = labelPool.emplace_back();
            a.Tbz(13, 1, plain);
            // S = 1: 32-bit saturating accumulate into MACL
            field.Load(2, 14, L.MAC);
            a.Sxtw(14, 14);
            a.Add(Sz::X, 14, 14, 12);
            a.Sxtw(4, 14);
            a.Cmp(Sz::X, 14, 4);
            a.BCond(Cond::EQ, noSat);
            a.AsrImm(Sz::X, 5, 14, 63);
            a.MovImm(Sz::W, 4, 0x7FFFFFFFu);
            a.Eor(Sz::W, 4, 4, 5);
            field.Store(2, 4, L.MAC);
            field.Load(2, 5, L.MAC + 4);
            a.OrrImm(Sz::W, 5, 5, 1);
            field.Store(2, 5, L.MAC + 4);
            a.B(done);
            a.Bind(noSat);
            field.Store(2, 14, L.MAC);
            a.B(done);
            a.Bind(plain);
            field.Load(3, 14, L.MAC);
            a.Add(Sz::X, 14, 14, 12);
            field.Store(3, 14, L.MAC);
            a.Bind(done);
        } else {
            a.Smull(12, 10, 11);
            field.Load(3, 14, L.MAC);
            a.Add(Sz::X, 14, 14, 12);
            a.Ldr(2, 13, kThis, static_cast<uint32>(L.SR));
            Label &store = labelPool.emplace_back();
            a.Tbz(13, 1, store);
            // S = 1: saturate to 48 bits
            a.MovImm(Sz::X, 4, 0x0000800000000000ull);
            a.Sub(Sz::X, 4, 14, 4);
            a.MovImm(Sz::X, 5, 0xFFFF000000000000ull);
            a.Cmp(Sz::X, 4, 5);
            a.BCond(Cond::CS, store);
            a.Eor(Sz::W, 5, 10, 11);
            a.MovImm(Sz::X, 14, 0x00007FFFFFFFFFFFull);
            a.MovImm(Sz::X, 4, 0xFFFF800000000000ull);
            a.CmpImm(Sz::W, 5, 0);
            a.Csel(Sz::X, 14, 4, 14, Cond::LT);
            a.Bind(store);
            field.Store(3, 14, L.MAC);
        }

        a.MovImm(Sz::W, kTmpWb, L.WbNone);
        field.Store(0, kTmpWb, L.WbReg);
        knownWb = L.WbNone;
        ++m_stats.inlineInstructions;
        if (k + 1 == count) {
            a.MovImm(Sz::W, kT0, ipc + 2);
            a.Str(2, kT0, kThis, static_cast<uint32>(L.PC));
        }
        a.Bind(slow.join);
        pcStale = k + 1 != count;
    };

    // Translates a register-only instruction into inline code, including its cycle accounting
    auto emitInline = [&](const Insn &insn) {
        const uint16 w = insn.word;
        const uint32 n = (w >> 8) & 0xF;
        const uint32 m = (w >> 4) & 0xF;
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
        case OpcodeType::TST_R:
        case OpcodeType::MUL:
        case OpcodeType::MULS:
        case OpcodeType::MULU:
        case OpcodeType::DMULS:
        case OpcodeType::DMULU:
        case OpcodeType::DIV0S:
        case OpcodeType::DIV1:
        case OpcodeType::CMP_STR: stall(m, n); break;
        // The register of control register transfers is in bits 8-11
        case OpcodeType::LDC_GBR_R:
        case OpcodeType::LDC_VBR_R:
        case OpcodeType::LDS_MACH_R:
        case OpcodeType::LDS_MACL_R:
        case OpcodeType::STC_GBR_R:
        case OpcodeType::STC_SR_R:
        case OpcodeType::STC_VBR_R: stall(n); break;
        case OpcodeType::LDS_PR_R:
        case OpcodeType::STS_PR_R: stall(n, L.WbPR); break;
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
        uint32 extraCycles = 0;
        uint32 newWb = L.WbNone;
        bool clearsAllow = false;
        switch (insn.opcode) {
        case OpcodeType::MUL:
        case OpcodeType::DMULS:
        case OpcodeType::DMULU: extraCycles = 2; break;
        case OpcodeType::STS_MACH_R:
        case OpcodeType::STS_MACL_R: newWb = n; break;
        default: break;
        }
        switch (insn.opcode) {
        case OpcodeType::LDC_GBR_R:
        case OpcodeType::LDC_VBR_R:
        case OpcodeType::LDS_MACH_R:
        case OpcodeType::LDS_MACL_R:
        case OpcodeType::LDS_PR_R:
        case OpcodeType::STC_GBR_R:
        case OpcodeType::STC_SR_R:
        case OpcodeType::STC_VBR_R:
        case OpcodeType::STS_MACH_R:
        case OpcodeType::STS_MACL_R:
        case OpcodeType::STS_PR_R: clearsAllow = true; break;
        default: break;
        }
        a.AddImm(Sz::X, kCycles, kCycles, 1 + extraCycles);
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
        if (knownWb != static_cast<int>(newWb)) {
            a.MovImm(Sz::W, kTmpWb, newWb);
            field.Store(0, kTmpWb, L.WbReg);
            knownWb = static_cast<int>(newWb);
        }
        if (clearsAllow) {
            field.Store(0, ZR, L.IntrAllow);
            allowSet = false;
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
        case OpcodeType::MUL:
            loadR(kT0, m);
            loadR(kT1, n);
            a.Mul(Sz::W, kT0, kT0, kT1);
            field.Store(2, kT0, L.MAC);
            break;
        case OpcodeType::MULS:
            loadR(kT0, m);
            loadR(kT1, n);
            a.Sxth(Sz::W, kT0, kT0);
            a.Sxth(Sz::W, kT1, kT1);
            a.Mul(Sz::W, kT0, kT0, kT1);
            field.Store(2, kT0, L.MAC);
            break;
        case OpcodeType::MULU:
            loadR(kT0, m);
            loadR(kT1, n);
            a.Uxth(kT0, kT0);
            a.Uxth(kT1, kT1);
            a.Mul(Sz::W, kT0, kT0, kT1);
            field.Store(2, kT0, L.MAC);
            break;
        case OpcodeType::DMULS:
        case OpcodeType::DMULU:
            loadR(kT0, m);
            loadR(kT1, n);
            if (insn.opcode == OpcodeType::DMULS) {
                a.Smull(kT2, kT0, kT1);
            } else {
                a.Umull(kT2, kT0, kT1);
            }
            field.Store(3, kT2, L.MAC);
            break;
        case OpcodeType::DIV0S:
            loadR(kT0, m);
            loadR(kT1, n);
            a.LsrImm(Sz::W, kT0, kT0, 31); // M
            a.LsrImm(Sz::W, kT1, kT1, 31); // Q
            a.Eor(Sz::W, kT2, kT0, kT1);   // T
            a.Ldr(2, kT3, kThis, static_cast<uint32>(L.SR));
            a.Bfi(Sz::W, kT3, kT0, 9, 1);
            a.Bfi(Sz::W, kT3, kT1, 8, 1);
            a.Bfi(Sz::W, kT3, kT2, 0, 1);
            a.Str(2, kT3, kThis, static_cast<uint32>(L.SR));
            break;
        case OpcodeType::DIV1: {
            // w1 = Rn, w2 = Rm, w3 = SR, w4 = old Q, w5 = M, w6 = T, w7 = new Q, w8 = value before the add/subtract
            loadR(1, n);
            if (m != n) {
                loadR(2, m);
            }
            a.Ldr(2, 3, kThis, static_cast<uint32>(L.SR));
            a.Ubfx(Sz::W, 4, 3, 8, 1);
            a.Ubfx(Sz::W, 5, 3, 9, 1);
            a.AndImm(Sz::W, 6, 3, 1);
            a.LsrImm(Sz::W, 7, 1, 31);
            a.Orr(Sz::W, 1, 6, 1, Shift::LSL, 1);
            if (m == n) {
                a.Mov(Sz::W, 2, 1); // the interpreter shifts Rn in place before reading Rm, which is the same register
            }
            a.Mov(Sz::W, 8, 1);
            a.Sub(Sz::W, kT0, 1, 2);
            a.Add(Sz::W, kT1, 1, 2);
            a.Cmp(Sz::W, 4, 5);
            a.Csel(Sz::W, 1, kT0, kT1, Cond::EQ);
            a.Cmp(Sz::W, 1, 8);
            a.Cset(Sz::W, kT0, Cond::LS);
            a.Cset(Sz::W, kT1, Cond::CC);
            a.Cset(Sz::W, kT2, Cond::CS);
            a.Cset(Sz::W, kT3, Cond::HI);
            a.CmpImm(Sz::W, 4, 0);
            a.Csel(Sz::W, kT0, kT0, kT2, Cond::NE); // M = 1: old Q ? LS : HS
            a.Csel(Sz::W, kT1, kT1, kT3, Cond::NE); // M = 0: old Q ? LO : HI
            a.CmpImm(Sz::W, 5, 0);
            a.Csel(Sz::W, kT0, kT0, kT1, Cond::NE);
            a.Eor(Sz::W, 7, 7, kT0); // new Q
            a.Cmp(Sz::W, 7, 5);
            a.Cset(Sz::W, 6, Cond::EQ); // new T
            a.Bfi(Sz::W, 3, 7, 8, 1);
            a.Bfi(Sz::W, 3, 6, 0, 1);
            a.Str(2, 3, kThis, static_cast<uint32>(L.SR));
            storeR(1, n);
            break;
        }
        case OpcodeType::CMP_STR:
            loadR(kT0, m);
            loadR(kT1, n);
            a.Eor(Sz::W, kT0, kT0, kT1);
            a.MovImm(Sz::W, kT2, 0x01010101u);
            a.Sub(Sz::W, kT1, kT0, kT2);
            a.Bic(Sz::W, kT1, kT1, kT0);
            a.MovImm(Sz::W, kT2, 0x80808080u);
            a.Tst(Sz::W, kT1, kT2);
            a.Cset(Sz::W, kT2, Cond::NE);
            writeT(kT2);
            break;
        case OpcodeType::LDC_GBR_R: loadR(kT0, n); field.Store(2, kT0, L.GBR); break;
        case OpcodeType::LDC_VBR_R: loadR(kT0, n); field.Store(2, kT0, L.VBR); break;
        case OpcodeType::LDS_MACH_R: loadR(kT0, n); field.Store(2, kT0, L.MAC + 4); break;
        case OpcodeType::LDS_MACL_R: loadR(kT0, n); field.Store(2, kT0, L.MAC); break;
        case OpcodeType::LDS_PR_R: loadR(kT0, n); field.Store(2, kT0, L.PR); break;
        case OpcodeType::STC_GBR_R: field.Load(2, kT0, L.GBR); storeR(kT0, n); break;
        case OpcodeType::STC_SR_R: field.Load(2, kT0, L.SR); storeR(kT0, n); break;
        case OpcodeType::STC_VBR_R: field.Load(2, kT0, L.VBR); storeR(kT0, n); break;
        case OpcodeType::STS_MACH_R: field.Load(2, kT0, L.MAC + 4); storeR(kT0, n); break;
        case OpcodeType::STS_MACL_R: field.Load(2, kT0, L.MAC); storeR(kT0, n); break;
        case OpcodeType::STS_PR_R: field.Load(2, kT0, L.PR); storeR(kT0, n); break;
        default: break;
        }
    };

    for (uint32 k = 0; k < count; ++k) {
        const Insn &insn = insns[k];
        const uint16 w = insn.word;
        const uint32 n = (w >> 8) & 0xF;
        const uint32 m = (w >> 4) & 0xF;

        // An instruction that loads SR can make an interrupt pending while also blocking interrupts for the next
        // instruction. The interrupt is serviced once that one is done, which the checks after handlers don't catch.
        if (k > 0 && k >= intrCheckAt) {
            budgetExits.emplace_back();
            budgetExits.back().second = insn.pc;
            field.Load(0, kT0, L.IntrPending);
            field.Load(0, kT1, L.IntrAllow);
            a.And(Sz::W, kT0, kT0, kT1);
            a.Cbnz(Sz::W, kT0, budgetExits.back().first);
        }

        // Stop here if the cycle budget is used up, like the interpreter's loop would
        if (k > 0) {
            budgetExits.emplace_back();
            a.Cmp(Sz::X, kCycles, kBudget);
            a.BCond(Cond::CS, budgetExits.back().first);
            budgetExits.back().second = insn.pc;
        }

        // Jumps from elsewhere in the block arrive here, after the cycle budget was checked by the jump itself. The
        // state they come from is unknown.
        if (isTarget[k]) {
            a.Bind(instrLabels[k]);
            knownWb = -1;
            pcStale = true;
            allowSet = false;
        }

        // Interrupts are allowed again from this instruction on (a handler may have blocked them for it)
        if (!allowSet && (k > 0 || isTarget[k])) {
            a.MovImm(Sz::W, kT0, 1);
            field.Store(0, kT0, L.IntrAllow);
            allowSet = true;
        }

        // The instruction in the delay slot of a branch translated above: register-only instructions are translated
        // inline too, followed by what the interpreter does when a delay slot ends
        if (k > 0 && insns[k - 1].kind == Kind::DelayedBranch && insns[k - 1].opcode != OpcodeType::RTE &&
            canInlineSlot(insn)) {
            Insn slot = insn;
            slot.opcode = insn.normalOpcode;
            slot.kind = insn.normalKind;
            const bool slotIsMem = slot.kind == Kind::Mem;
            auto emitSlot = [&](bool delay) {
                if (slotIsMem) {
                    EmitMemory(k, slot.pc, slot.word, slot.opcode, delay);
                } else {
                    emitInline(slot);
                }
            };
            const OpcodeType branch = insns[k - 1].opcode;
            // An instruction that blocks interrupts for the next one can leave an interrupt pending right at the delay
            // slot's end that has to be serviced after the instruction at the target; only the interpreter's loop does
            // that, so execution doesn't continue directly at the target in that case
            bool slotBlocksIntr = false;
            switch (slot.opcode) {
            case OpcodeType::LDC_GBR_R:
            case OpcodeType::LDC_VBR_R:
            case OpcodeType::LDS_MACH_R:
            case OpcodeType::LDS_MACL_R:
            case OpcodeType::LDS_PR_R:
            case OpcodeType::STC_GBR_R:
            case OpcodeType::STC_SR_R:
            case OpcodeType::STC_VBR_R:
            case OpcodeType::STS_MACH_R:
            case OpcodeType::STS_MACL_R:
            case OpcodeType::STS_PR_R:
            case OpcodeType::LDC_GBR_M:
            case OpcodeType::LDC_VBR_M:
            case OpcodeType::LDS_MACH_M:
            case OpcodeType::LDS_MACL_M:
            case OpcodeType::LDS_PR_M:
            case OpcodeType::STC_GBR_M:
            case OpcodeType::STC_SR_M:
            case OpcodeType::STC_VBR_M:
            case OpcodeType::STS_MACH_M:
            case OpcodeType::STS_MACL_M:
            case OpcodeType::STS_PR_M: slotBlocksIntr = true; break;
            default: break;
            }
            const sint32 jump = slotBlocksIntr ? -1 : jumpIndex[k - 1];
            const uint32 branchTarget = 0; // only used for exits, which find the PC in the CPU state (pcStored)
            // Where the branch goes, if that is known
            uint32 staticTarget = 0;
            bool hasStaticTarget = !slotBlocksIntr;
            {
                const uint16 bw = insns[k - 1].word;
                const uint32 bpc = insns[k - 1].pc;
                if (branch == OpcodeType::BRA || branch == OpcodeType::BSR) {
                    staticTarget = bpc + 4 + static_cast<uint32>((static_cast<sint32>((bw & 0xFFF) << 20) >> 20) * 2);
                } else if (branch == OpcodeType::BTS || branch == OpcodeType::BFS) {
                    staticTarget = bpc + 4 + static_cast<uint32>(static_cast<sint8>(bw & 0xFF) * 2);
                } else {
                    hasStaticTarget = false;
                }
            }
            if (slotBlocksIntr) {
                hasStaticTarget = false;
            }
            if (branch != OpcodeType::BTS && branch != OpcodeType::BFS) {
                emitSlot(true);
                delayEnd();
                if (jump >= 0) {
                    emitJump(k - 1, jump, branchTarget, true);
                } else if (hasStaticTarget) {
                    emitChain(staticTarget, true, true);
                }
            } else {
                // Whether the branch was taken decides if this is a delay slot
                Label &normal = labelPool.emplace_back();
                Label &finished = labelPool.emplace_back();
                const int wbBefore = knownWb;
                const bool allowBefore = allowSet;
                field.Load(0, kT0, L.DelaySlot);
                a.Cbz(Sz::W, kT0, normal);
                emitSlot(true);
                delayEnd();
                if (jump >= 0) {
                    emitJump(k - 1, jump, branchTarget, true);
                } else if (hasStaticTarget) {
                    emitChain(staticTarget, true, true);
                }
                a.B(finished);
                knownWb = wbBefore;
                allowSet = allowBefore;
                a.Bind(normal);
                emitSlot(false);
                if (!slotIsMem) {
                    // (loads and stores continue at the next instruction by themselves, unless held up)
                    a.MovImm(Sz::W, kT0, insn.pc + 2);
                    a.Str(2, kT0, kThis, static_cast<uint32>(L.PC));
                }
                a.Bind(finished);
            }
            pcStale = false;
            continue;
        }

        if (insn.kind == Kind::Mem) {
            EmitMemory(k, insn.pc, w, insn.opcode, false);
            continue;
        }
        if (insn.kind == Kind::Mac) {
            EmitMac(k, insn.pc, w, insn.opcode);
            continue;
        }
        if (insn.kind == Kind::CondBranch || (insn.kind == Kind::DelayedBranch && insn.opcode != OpcodeType::RTE)) {
            EmitBranch(k, insn);
            continue;
        }

        if (insn.kind != Kind::Inline) {
            // Executed by the interpreter's handler, which expects the CPU's PC to point at the instruction
            emitHelper(k, insn.pc, w, pcStale);
            if (insn.opcode == OpcodeType::LDC_SR_R || insn.opcode == OpcodeType::LDC_SR_M) {
                intrCheckAt = k + 2;
            }
            pcStale = false;
            allowSet = false;
            knownWb = -1;
            prevWasHelper = true;
            continue;
        }

        emitInline(insn);
    }

    // After the last instruction: continue at the next one, unless a handler already did
    const Insn &last = insns[count - 1];
    if (last.kind == Kind::Inline || last.kind == Kind::CondBranch) {
        a.MovImm(Sz::W, kT0, last.pc + 2);
        a.Str(2, kT0, kThis, static_cast<uint32>(L.PC));
        emitChain(last.pc + 2, true, true);
    }

    // Epilogue
    a.Bind(epilogue);
    flushRegs();
    a.Mov(Sz::X, X0, kCycles);
    a.Ldp(Sz::X, X27, X28, SP, 80);
    a.Ldp(Sz::X, X25, X26, SP, 64);
    a.Ldp(Sz::X, X23, X24, SP, 48);
    a.Ldp(Sz::X, X21, X22, SP, 32);
    a.Ldp(Sz::X, X19, X20, SP, 16);
    a.LdpPost(Sz::X, X29, X30, SP, 96);
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
        if (slow.delaySlot) {
            // The handler retries an access that is held up by leaving the delay slot unfinished
            field.Load(0, kT0, L.DelaySlot);
            a.Cbnz(Sz::W, kT0, epilogue);
        }
        a.B(slow.join);
    }
    for (CodeCheck &check : codeChecks) {
        a.Bind(check.entry);
        if (check.delaySlot) {
            delayEnd();
        } else {
            a.MovImm(Sz::W, kT0, check.pcNext);
            a.Str(2, kT0, kThis, static_cast<uint32>(L.PC));
        }
        a.Mov(Sz::X, X0, kThis);
        a.MovImm(Sz::W, X1, pc);
        a.MovImm(Sz::W, X2, check.firstChecked * 2);
        a.MovImm(Sz::W, X3, check.pcNext - 2);
        a.MovImm(Sz::X, X16, reinterpret_cast<uint64>(&SH2Recompiler::CodeUnchangedThunk));
        a.Blr(X16);
        a.Cbz(Sz::X, X0, epilogue);
        a.B(check.back);
    }

    if (a.Overflowed() || !a.Finalize()) {
        if (std::getenv("YMIR_JIT_DEBUG") != nullptr) {
            std::fprintf(stderr, "jit: block at %08X rejected (%s, %zu words)\n", pc,
                         a.Overflowed() ? "overflow" : "branch out of range", a.Size());
        }
        return false;
    }
    outWords = a.Size();
    outChainWord = static_cast<size_t>(chainLabel.position);
    return true;
    };

    std::array<sint32, 16> noCache;
    noCache.fill(-1);
    std::array<uint32, 16> useCount{};
    size_t wordCount = 0;
    size_t chainWord = 0;
    const Stats savedStats = m_stats;
    if (!generate(noCache, useCount, wordCount, chainWord)) {
        m_stats = savedStats;
        ++m_stats.blocksRejected;
        return;
    }
    m_stats = savedStats;
    std::array<sint32, 16> cacheHost;
    cacheHost.fill(-1);
    for (uint32 hostReg = kFirstCached; hostReg <= kLastCached; ++hostReg) {
        uint32 best = 16;
        for (uint32 r = 0; r < 16; ++r) {
            if (cacheHost[r] < 0 && useCount[r] >= (hasLoop ? std::max<uint32>(CacheMinUses() / 2, 2u) : CacheMinUses()) && (best == 16 || useCount[r] > useCount[best])) {
                best = r;
            }
        }
        if (best == 16 || !g_registerCache.load(std::memory_order_relaxed)) {
            break;
        }
        cacheHost[best] = static_cast<sint32>(hostReg);
    }
    std::array<uint32, 16> unusedCount{};
    if (!generate(cacheHost, unusedCount, wordCount, chainWord)) {
        ++m_stats.blocksRejected;
        return;
    }

    // -- Install --

    const size_t sizeBytes = wordCount * sizeof(uint32);
    if (m_chunkUsed + sizeBytes + 16 > m_chunkSize) {
        Flush();
        // The block table was cleared, so the block being compiled has to be filled in again
        fillBlock();
    }
    const size_t offset = m_chunkOffset + ((m_chunkUsed + 15) & ~size_t{15});
    m_chunkUsed = (offset - m_chunkOffset) + sizeBytes;

    std::lock_guard lock{g_poolMutex};
    g_pool.BeginWrite();
    std::memcpy(g_pool.WritePtr(offset), words.data(), sizeBytes);
    g_pool.EndWrite(offset, sizeBytes);
    block.fn = reinterpret_cast<BlockFn>(g_pool.ExecPtr(offset));
    if (block.paddedLen != 0) {
        block.chainEntry = g_pool.ExecPtr(offset) + chainWord * sizeof(uint32);
    }
    ++m_stats.blocksCompiled;
    g_statBlocks.fetch_add(1, std::memory_order_relaxed);
    g_statBytes.fetch_add(sizeBytes, std::memory_order_relaxed);
    g_statInstructions.fetch_add(count, std::memory_order_relaxed);
}

} // namespace ymir::sh2
