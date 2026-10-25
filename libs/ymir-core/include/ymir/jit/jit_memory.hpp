#pragma once

/**
@file
@brief Executable memory for dynamic recompilers.

Handles the differences between platforms:
- iOS with TXM (iOS 26+ on A15/M2 and newer): memory can't be flipped between writable and executable. A read+execute
  pool is reserved and prepared by the JIT enabler (StikDebug/StosDebug "universal" script) through `brk #0xf00d`, and a
  writable alias of the same pages is created with `vm_remap`. Code is written through the alias and run from the pool.
- iOS without TXM: a debugger-enabled process (CS_DEBUGGED) may map read+write+execute memory.
- macOS (Apple silicon): `MAP_JIT` with per-thread write protection toggling.
- Other AArch64 systems: read+write+execute anonymous memory.

The recompilers only generate AArch64 code, so on any other architecture JIT is reported as unsupported.
*/

#include <ymir/core/types.hpp>

#include <cstddef>
#include <string>

namespace ymir::jit {

/// @brief Describes whether JIT can be used on this device and process, and why not if it can't.
struct JitStatus {
    enum class Reason {
        Available,              ///< Executable memory can be allocated
        UnsupportedArchitecture, ///< The recompilers generate AArch64 code only
        NeedsDebugger,          ///< iOS: a JIT enabler must attach to the process first
        AllocationFailed,       ///< The system refused to provide executable memory
    };

    Reason reason = Reason::UnsupportedArchitecture;
    bool txmDevice = false;       ///< iOS device with TXM, which requires the pool and the debugger handshake
    bool processDebugged = false; ///< The process is being debugged (CS_DEBUGGED on iOS)

    bool Available() const {
        return reason == Reason::Available;
    }

    /// @brief A short description meant for users.
    std::string Message() const;
};

/// @brief Checks the platform and process state without allocating anything.
JitStatus ProbeJit();

/// @brief A block of executable memory with a bump allocator.
///
/// Code is written through `WritePtr()` addresses and executed from the corresponding `ExecPtr()` addresses; the two
/// are the same except in the iOS TXM pool. Call `BeginWrite()` before and `EndWrite()` after modifying code.
class JitMemory {
public:
    JitMemory() = default;
    ~JitMemory();

    JitMemory(const JitMemory &) = delete;
    JitMemory &operator=(const JitMemory &) = delete;

    /// @brief Allocates the executable region. Safe to call again after a failure (e.g. once a debugger has attached).
    /// @param[in] size the size of the region in bytes; rounded up to the page size
    /// @return true if executable memory is available
    bool Initialize(size_t size);

    bool IsInitialized() const {
        return m_exec != nullptr;
    }

    /// @brief The reason the last Initialize failed, or Available if it succeeded.
    JitStatus Status() const {
        return m_status;
    }

    /// @brief Reserves `size` bytes (aligned to 16) in the region.
    /// @return the offset of the reservation, or SIZE_MAX if the region is full
    size_t Allocate(size_t size);

    /// @brief Gives back everything reserved. Previously generated code must not run afterwards.
    void Reset() {
        m_used = 0;
    }

    size_t Used() const {
        return m_used;
    }
    size_t Capacity() const {
        return m_size;
    }

    uint8 *ExecPtr(size_t offset = 0) const {
        return m_exec + offset;
    }
    uint8 *WritePtr(size_t offset = 0) const {
        return m_write + offset;
    }

    /// @brief Makes the code region writable for the calling thread.
    void BeginWrite();

    /// @brief Finishes modifying code: restores protections and flushes the instruction cache for the range.
    void EndWrite(size_t offset, size_t size);

private:
    enum class Mode { None, Rwx, MapJit, PoolTxm, MprotectFlip };

    bool InitializeTxmPool(size_t size);

    uint8 *m_exec = nullptr;
    uint8 *m_write = nullptr;
    size_t m_size = 0;
    size_t m_used = 0;
    Mode m_mode = Mode::None;
    JitStatus m_status{};
};

} // namespace ymir::jit
