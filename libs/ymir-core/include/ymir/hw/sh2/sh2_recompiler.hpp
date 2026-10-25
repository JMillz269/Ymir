#pragma once

/**
@file
@brief Dynamic recompiler (JIT) for the SH-2 CPU, targeting AArch64.

Straight-line sequences of instructions are translated into native blocks. Simple register-only instructions are
translated into inline AArch64 code; every other instruction is executed by calling into the interpreter's own
instruction handler, so its behavior (including bus accesses, wait states and timing) is exactly the same as in the
interpreter. Blocks keep the interpreter's cycle accounting, stop at the same points the interpreter would (cycle budget
and interrupts), and are validated against the code memory every time they are entered.
*/

#include <ymir/core/types.hpp>

#include <ymir/hw/sh2/sh2_decode.hpp>

#include <ymir/sys/bus.hpp>

#include <array>
#include <functional>
#include <vector>

namespace ymir::sh2 {

class SH2;

class SH2Recompiler {
public:
    /// @brief Signature of generated blocks.
    ///
    /// Executes instructions starting at the CPU's PC until the budget of cycles is used up, an interrupt needs to be
    /// serviced, or control flow leaves the block. Returns the number of cycles executed.
    using BlockFn = uint64 (*)(SH2 *sh2, uint64 budget);

    /// @brief Replaces how blocks are invoked. Used by tests to run the generated code in a CPU emulator.
    using Executor = std::function<uint64(BlockFn fn, SH2 &sh2, uint64 budget)>;

    SH2Recompiler(SH2 &sh2, sys::SH2Bus &bus);
    ~SH2Recompiler();

    SH2Recompiler(const SH2Recompiler &) = delete;
    SH2Recompiler &operator=(const SH2Recompiler &) = delete;

    /// @brief Whether executable memory was obtained and blocks can be generated.
    bool IsUsable() const {
        return m_usable;
    }

    /// @brief Runs the block at the CPU's PC, compiling it first if needed.
    /// @param[in] cyclesLeft the number of cycles that may still be executed (at least one)
    /// @return the number of cycles executed, or 0 if the interpreter has to run the next instruction
    uint64 Run(uint64 cyclesLeft);

    /// @brief Discards all generated code.
    void Flush();

    void SetExecutor(Executor executor) {
        m_executor = std::move(executor);
    }

    /// @brief The function generated code calls to execute an instruction that wasn't translated inline.
    static uint64 StepThunk(SH2 *sh2, uint32 instr);

    /// @brief The function generated code calls after an instruction that wrote to memory, to find out if the rest of
    /// the block it is running still matches the code in memory. Returns non-zero if so.
    ///
    /// If the code changed and `storePc` is the address of the instruction that wrote to it (or 0xFFFFFFFF), the
    /// interpreter's prefetch buffer is set up to hold what was in memory before the write, as it would have.
    static uint64 CodeUnchangedThunk(SH2 *sh2, uint32 blockPc, uint32 offset, uint32 storePc);

    /// @brief Retrieves the executable memory region shared by all recompilers. For tests.
    static void GetPoolRegion(uint8 **base, size_t *size);

    struct Stats {
        uint64 blocksCompiled = 0;
        uint64 blocksExecuted = 0;
        uint64 blocksRejected = 0; // PCs where no block could be generated
        uint64 invalidations = 0;  // blocks recompiled because their code changed
        uint64 cacheFlushes = 0;
        uint64 inlineInstructions = 0; // translated inline
        uint64 helperInstructions = 0; // executed through the interpreter's handlers
    };
    const Stats &GetStats() const {
        return m_stats;
    }

private:
    static constexpr uint32 kMaxInstructions = 64;
    static constexpr uint32 kMaxBlockBytes = (kMaxInstructions + 1) * 2;
    static constexpr uint32 kBlockTableSize = 8192;

    struct Block {
        uint32 pc = 1; // odd = empty slot
        uint32 byteLen = 0;
        BlockFn fn = nullptr; // null if no code could be generated for this PC
        const uint8 *hostSrc = nullptr; // where the block's code lives in host memory when it was compiled
        std::array<uint8, kMaxBlockBytes> bytes{};
    };

    enum class Kind { Inline, Mem, CondBranch, Helper, DelayedBranch, EndAfter, Stop };

    static Kind Classify(OpcodeType opcode);
    static uint32 SlotFor(uint32 pc) {
        return ((pc >> 1) ^ (pc >> 14)) & (kBlockTableSize - 1);
    }

    void Compile(Block &block, uint32 pc, const uint8 *src, uint32 bytesAvailable);

    SH2 &m_sh2;
    sys::SH2Bus &m_bus;
    bool m_usable = false;

    // Region of the shared executable memory owned by this recompiler
    size_t m_chunkOffset = 0;
    size_t m_chunkSize = 0;
    size_t m_chunkUsed = 0;

    // The longword the interpreter would have prefetched when a block was left right after a write to its own code
    bool m_prefetchValid = false;
    uint32 m_prefetch = 0;
    uint32 m_prefetchPc = 0; // the PC the longword is meant for

    std::vector<Block> m_blocks;
    Executor m_executor;
    Stats m_stats;
};

} // namespace ymir::sh2
