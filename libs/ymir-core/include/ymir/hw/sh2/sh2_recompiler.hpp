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
#include <atomic>
#include <functional>
#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
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

    /// @brief Retrieves the memory holding this recompiler's block table. For tests.
    void GetBlockTableRegion(void **base, size_t *size) const;

    /// @brief Totals over every recompiler in the process, for showing in the user interface.
    struct GlobalStats {
        uint64 blocksCompiled = 0;
        uint64 cacheFlushes = 0;
        uint64 bytesGenerated = 0;
        uint64 instructionsTranslated = 0;
        uint64 compileNanoseconds = 0;
        uint64 slowestCompileNanoseconds = 0;
        uint64 conflictEvictions = 0; // blocks thrown out of the table by a block for another address
        uint64 invalidations = 0;     // blocks compiled again because the code in memory changed
        uint64 inlineInstructions = 0;
        uint64 helperInstructions = 0; // translated as calls to the interpreter
    };
    static GlobalStats GetGlobalStats();

    /// @brief Turns optimizations on or off for every recompiler; generated code is discarded and compiled again.
    /// Meant for finding out which one is responsible when something is slow or wrong.
    static void SetOptions(bool registerCache, bool blockLinking);

    /// @brief Idle detection: a CPU that spends a whole time slice in recompiled code without writing to memory,
    /// calling the interpreter or changing any register is waiting for something. While both CPUs are waiting, the
    /// emulator lets them run in longer slices. Off by default.
    /// @brief The instructions that most often had to be executed by the interpreter (as a share of those calls),
    /// for finding out what to translate next.
    static std::string HelperReport(size_t top);

    static void SetIdleDetection(bool enable) {
        s_idleDetect.store(enable, std::memory_order_relaxed);
    }
    static bool IdleDetectionEnabled() {
        return s_idleDetect.load(std::memory_order_relaxed);
    }
    /// @brief Number of times a loop that kept reading a timer register was moved on in time, for the statistics.
    static uint64 PollSkips() {
        return s_pollSkips.load(std::memory_order_relaxed);
    }
    /// @brief Number of time slices spent waiting, for the statistics display.
    static uint64 IdleSlices() {
        return s_idleSlices.load(std::memory_order_relaxed);
    }
    /// @brief Share of the emulated time (0 to 1) that ran in longer slices because every CPU was waiting.
    static double IdleTimeShare() {
        const uint64 total = s_totalCycles.load(std::memory_order_relaxed);
        return total == 0 ? 0.0 : static_cast<double>(s_coarseCycles.load(std::memory_order_relaxed)) / total;
    }
    static void NoteSliceCycles(uint64 total, uint64 coarse) {
        s_totalCycles.fetch_add(total, std::memory_order_relaxed);
        s_coarseCycles.fetch_add(coarse, std::memory_order_relaxed);
    }

    struct Stats {
        uint64 blocksCompiled = 0;
        uint64 blocksExecuted = 0;
        uint64 blocksRejected = 0; // PCs where no block could be generated
        uint64 invalidations = 0;  // blocks recompiled because their code changed
        uint64 conflictEvictions = 0; // blocks thrown out of the table by a block for another address
        uint64 cacheFlushes = 0;
        uint64 inlineInstructions = 0; // translated inline
        uint64 helperInstructions = 0; // executed through the interpreter's handlers
        uint64 chainSites = 0;         // places where a block jumps straight into another
    };
    const Stats &GetStats() const {
        return m_stats;
    }

private:
    friend class SH2;
    static inline std::atomic<bool> s_idleDetect{false};
    static inline std::atomic<uint64> s_idleSlices{0};
    static inline std::atomic<uint64> s_pollSkips{0};

    // The CPU state when the interpreter was last asked to read a timer register, to spot code that reads one over and
    // over with nothing else changing
    struct PollState {
        std::array<uint32, 16> R;
        uint32 PC, PR, GBR, VBR, SR;
        uint64 MAC;
        bool operator==(const PollState &) const = default;
    };
    PollState m_pollState{};
    bool m_pollValid = false;
    static inline std::atomic<uint64> s_totalCycles{0};
    static inline std::atomic<uint64> s_coarseCycles{0};

    static constexpr uint32 kMaxInstructions = 32;
    // (rounded up to 16 bytes: blocks are compared in chunks of 16 when one block jumps to another)
    static constexpr uint32 kMaxBlockBytes = ((kMaxInstructions + 1) * 2 + 2 + 15) & ~15u;
    // The table has four ways per set: a block lives in one of the four slots its address maps to. Slot numbers
    // returned by SlotFor are those of the first way; generated code only links to blocks found there.
    static constexpr uint32 kBlockWays = 4;
    static constexpr uint32 kBlockTableSize = 65536;
    static constexpr uint32 kBlockSetBits = 14;
    static_assert((1u << kBlockSetBits) * kBlockWays == kBlockTableSize);

    struct Block {
        uint32 pc = 1; // odd = empty slot
        uint32 byteLen = 0;
        BlockFn fn = nullptr; // null if no code could be generated for this PC
        const uint8 *hostSrc = nullptr; // where the block's code lives in host memory when it was compiled
        const void *chainEntry = nullptr; // where other blocks jump to continue in this one; null if they can't
        uint32 paddedLen = 0;             // number of valid bytes in `bytes` for the jump check (a multiple of 16)
        std::array<uint8, kMaxBlockBytes> bytes{};
    };
    static_assert(std::is_standard_layout_v<Block>);

    struct BlockTableDeleter {
        void operator()(Block *p) const {
            std::free(p);
        }
    };

    enum class Kind { Inline, Mem, Mac, CondBranch, Helper, DelayedBranch, EndAfter, Stop };

    static Kind Classify(OpcodeType opcode);
    static uint32 SlotFor(uint32 pc) {
        return (((pc >> 1) * 0x9E3779B1u) >> (32 - kBlockSetBits)) * kBlockWays;
    }
    // The block for `pc` if there is one in the table
    const Block *FindBlock(uint32 pc) const {
        const Block *first = &m_blocks[SlotFor(pc)];
        for (uint32 way = 0; way < kBlockWays; ++way) {
            if (first[way].pc == pc) {
                return &first[way];
            }
        }
        return nullptr;
    }

    void Compile(Block &block, uint32 pc, const uint8 *src, uint32 bytesAvailable);
    void CompileBlock(Block &block, uint32 pc, const uint8 *src, uint32 bytesAvailable);

    SH2 &m_sh2;
    sys::SH2Bus &m_bus;
    bool m_usable = false;

    // Region of the shared executable memory owned by this recompiler
    size_t m_chunkOffset = 0;
    size_t m_chunkSize = 0;
    size_t m_chunkUsed = 0;

    // The longword the interpreter would have prefetched when a block was left right after a write to its own code
    uint32 m_optionsGeneration = 0;
    uint32 m_victimWay = 0;
    bool m_prefetchValid = false;
    uint32 m_prefetch = 0;
    uint32 m_prefetchPc = 0; // the PC the longword is meant for

    std::unique_ptr<Block[], BlockTableDeleter> m_blocks; // page aligned so that tests can map it
    Executor m_executor;
    Stats m_stats;
};

} // namespace ymir::sh2
