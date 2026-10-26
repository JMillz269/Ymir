#pragma once

/**
@file
@brief Coarse timing counters for finding out where the emulator spends its time. Cheap enough to leave on; read and
reset by the frontend's statistics display.
*/

#include <ymir/core/types.hpp>

#include <atomic>
#include <chrono>

namespace ymir::perf {

// Nanoseconds spent, since the frontend last read them
inline std::atomic<uint64> sh2Ns{0};   // SH-2 CPUs and the SCU (emulator thread)
inline std::atomic<uint64> otherNs{0}; // VDP timing, scheduler events (SCSP, 68000, CD block) on the emulator thread
inline std::atomic<uint64> vdp1Ns{0};  // VDP1 command execution (render thread if threaded)
inline std::atomic<uint64> vdp2Ns{0};  // VDP2 line rendering (render thread if threaded)
inline std::atomic<uint64> frames{0};  // frames completed
inline std::atomic<uint64> emuWaitNs{0}; // emulator waiting for the display to ask for the next frame (video sync)
inline std::atomic<uint64> guiWaitNs{0}; // display waiting for the emulator to finish a frame (video sync)

using Clock = std::chrono::steady_clock;

inline uint64 NanosecondsSince(Clock::time_point start) {
    return static_cast<uint64>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

} // namespace ymir::perf
