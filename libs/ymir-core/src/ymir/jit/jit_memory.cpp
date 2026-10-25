#include <ymir/jit/jit_memory.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__aarch64__) || defined(_M_ARM64)
    #define YMIR_JIT_ARCH_SUPPORTED 1
#else
    #define YMIR_JIT_ARCH_SUPPORTED 0
#endif

#if defined(__APPLE__)
    #include <TargetConditionals.h>
    #include <libkern/OSCacheControl.h>
    #include <pthread.h>
    #include <sys/mman.h>
    #include <sys/sysctl.h>
    #include <sys/utsname.h>
    #include <unistd.h>
    #if TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR
        #include <mach/mach.h>
        #include <mach/vm_map.h>
        #define YMIR_JIT_IOS_DEVICE 1
        extern "C" int csops(pid_t pid, unsigned int ops, void *useraddr, size_t usersize);
    #endif
#elif defined(__unix__)
    #include <sys/mman.h>
    #include <unistd.h>
#endif

#ifndef YMIR_JIT_IOS_DEVICE
    #define YMIR_JIT_IOS_DEVICE 0
#endif

namespace ymir::jit {

namespace {

    bool g_testingMode = false;

#if YMIR_JIT_IOS_DEVICE
    constexpr unsigned kCsOpsStatus = 0;
    constexpr unsigned kCsDebugged = 0x10000000;

    // Same rules as StosDebug's ProcessInfo.detectedTXM, using only the hardware model and OS version so it works in a
    // sandbox. Returns the major OS version through `osMajor`.
    bool DeviceHasTXM() {
        struct utsname info;
        if (uname(&info) != 0) {
            return false;
        }
        const char *hw = info.machine;

        int osMajor = 0;
        char version[64] = {};
        size_t versionLen = sizeof(version);
        if (sysctlbyname("kern.osproductversion", version, &versionLen, nullptr, 0) == 0) {
            osMajor = std::atoi(version);
        }

        if (osMajor >= 27) {
            return std::strcmp(hw, "iPad8,11") != 0 && std::strcmp(hw, "iPad8,12") != 0;
        }
        if (osMajor >= 26) {
            const bool isPad = std::strncmp(hw, "iPad", 4) == 0;
            const bool isPhone = std::strncmp(hw, "iPhone", 6) == 0;
            if (!isPad && !isPhone) {
                return false;
            }
            int major = 0, minor = 0;
            if (std::sscanf(hw + (isPad ? 4 : 6), "%d,%d", &major, &minor) != 2) {
                return false;
            }
            double scale = 10.0;
            for (int m = minor; m >= 10; m /= 10) {
                scale *= 10.0;
            }
            const double hwVersion = major + static_cast<double>(minor) / scale;
            return isPad ? (hwVersion >= 14.5) : (hwVersion >= 14.2);
        }
        return false;
    }

    bool ProcessIsDebugged() {
        int flags = 0;
        if (csops(getpid(), kCsOpsStatus, &flags, sizeof(flags)) != 0) {
            return false;
        }
        return (flags & kCsDebugged) != 0;
    }
#endif

    size_t PageSize() {
#if defined(__APPLE__) || defined(__unix__)
        return static_cast<size_t>(getpagesize());
#else
        return 4096;
#endif
    }

    size_t RoundUp(size_t value, size_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }

} // namespace

std::string JitStatus::Message() const {
    switch (reason) {
    case Reason::Available: return txmDevice ? "Available (TXM mode)" : "Available";
    case Reason::UnsupportedArchitecture: return "Not supported on this device";
    case Reason::NeedsDebugger:
        return "A JIT enabler (StikDebug or StosDebug) must be attached to Ymir. Use \"Enable JIT\" and relaunch if "
               "needed.";
    case Reason::AllocationFailed: return "The system did not allow executable memory";
    }
    return {};
}

void SetJitTestingMode(bool enable) {
    g_testingMode = enable;
}

JitStatus ProbeJit() {
    JitStatus status{};
#if !YMIR_JIT_ARCH_SUPPORTED
    status.reason = g_testingMode ? JitStatus::Reason::Available : JitStatus::Reason::UnsupportedArchitecture;
#elif YMIR_JIT_IOS_DEVICE
    status.txmDevice = DeviceHasTXM();
    status.processDebugged = ProcessIsDebugged();
    status.reason = status.processDebugged ? JitStatus::Reason::Available : JitStatus::Reason::NeedsDebugger;
#else
    status.reason = JitStatus::Reason::Available;
#endif
    return status;
}

JitMemory::~JitMemory() {
    if (m_exec == nullptr) {
        return;
    }
#if defined(__APPLE__) || defined(__unix__)
    // The TXM pool must stay mapped: other code may still be running from it and the debugger is gone. It is a one-time
    // allocation for the lifetime of the process.
    if (m_mode != Mode::PoolTxm) {
        munmap(m_exec, m_size);
    }
#endif
}

bool JitMemory::Initialize(size_t size) {
    if (m_exec != nullptr) {
        return true;
    }

    m_status = ProbeJit();
    if (!m_status.Available()) {
        return false;
    }

    size = RoundUp(size, std::max<size_t>(PageSize(), 16384));

#if !YMIR_JIT_ARCH_SUPPORTED
    if (g_testingMode) {
    #if defined(__APPLE__) || defined(__unix__)
        void *memory = mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (memory != MAP_FAILED) {
            m_exec = m_write = static_cast<uint8 *>(memory);
            m_size = size;
            m_mode = Mode::Rwx;
            return true;
        }
    #endif
    }
    return false;
#elif YMIR_JIT_IOS_DEVICE
    if (m_status.txmDevice) {
        if (!InitializeTxmPool(size)) {
            m_status.reason = JitStatus::Reason::AllocationFailed;
            return false;
        }
        return true;
    }

    // Debugged process on a device without TXM: read+write+execute mappings are allowed
    void *memory = mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (memory != MAP_FAILED) {
        m_exec = m_write = static_cast<uint8 *>(memory);
        m_size = size;
        m_mode = Mode::Rwx;
        return true;
    }

    // Otherwise flip between writable and executable with mprotect
    memory = mmap(nullptr, size, PROT_READ | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (memory != MAP_FAILED) {
        m_exec = m_write = static_cast<uint8 *>(memory);
        m_size = size;
        m_mode = Mode::MprotectFlip;
        return true;
    }
    m_status.reason = JitStatus::Reason::AllocationFailed;
    return false;
#elif defined(__APPLE__)
    void *memory = mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0);
    if (memory == MAP_FAILED) {
        m_status.reason = JitStatus::Reason::AllocationFailed;
        return false;
    }
    m_exec = m_write = static_cast<uint8 *>(memory);
    m_size = size;
    m_mode = Mode::MapJit;
    return true;
#elif defined(__unix__)
    void *memory = mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (memory == MAP_FAILED) {
        m_status.reason = JitStatus::Reason::AllocationFailed;
        return false;
    }
    m_exec = m_write = static_cast<uint8 *>(memory);
    m_size = size;
    m_mode = Mode::Rwx;
    return true;
#else
    return false;
#endif
}

bool JitMemory::InitializeTxmPool(size_t size) {
#if YMIR_JIT_IOS_DEVICE
    // 1. Reserve a read+execute region
    void *rx = mmap(nullptr, size, PROT_READ | PROT_EXEC, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (rx == MAP_FAILED) {
        return false;
    }
    // The enabler's prepare step only understands addresses below 2^36
    if (reinterpret_cast<uintptr_t>(rx) + size > (1ull << 36)) {
        std::fprintf(stderr, "[ymir-jit] TXM region %p is above 2^36; the JIT enabler may not prepare it\n", rx);
    }

    // 2. Ask the enabler to prepare the region (universal script: x16 = 1, x0 = address, x1 = length)
    __asm__ volatile("mov x0, %0\n"
                     "mov x1, %1\n"
                     "mov x16, #1\n"
                     "brk #0xf00d"
                     :
                     : "r"(rx), "r"(size)
                     : "x0", "x1", "x16", "memory");

    // 3. Map a writable alias of the same pages
    vm_address_t rw = 0;
    vm_prot_t curProt = 0, maxProt = 0;
    const kern_return_t kr = vm_remap(mach_task_self(), &rw, size, 0, VM_FLAGS_ANYWHERE, mach_task_self(),
                                      reinterpret_cast<vm_address_t>(rx), FALSE, &curProt, &maxProt, VM_INHERIT_DEFAULT);
    if (kr != KERN_SUCCESS) {
        munmap(rx, size);
        return false;
    }
    if (mprotect(reinterpret_cast<void *>(rw), size, PROT_READ | PROT_WRITE) != 0) {
        munmap(rx, size);
        munmap(reinterpret_cast<void *>(rw), size);
        return false;
    }

    m_exec = static_cast<uint8 *>(rx);
    m_write = reinterpret_cast<uint8 *>(rw);
    m_size = size;
    m_mode = Mode::PoolTxm;

    // 4. The region is prepared: let go of the debugger (universal script: x16 = 0). Only one pool may do this per
    // process, since a second handshake after detaching would crash the app.
    __asm__ volatile("mov x16, #0\n"
                     "brk #0xf00d"
                     :
                     :
                     : "x16", "memory");
    return true;
#else
    (void)size;
    return false;
#endif
}

size_t JitMemory::Allocate(size_t size) {
    const size_t offset = RoundUp(m_used, 16);
    const size_t end = offset + RoundUp(size, 16);
    if (m_exec == nullptr || end > m_size) {
        return SIZE_MAX;
    }
    m_used = end;
    return offset;
}

void JitMemory::BeginWrite() {
#if defined(__APPLE__) && !YMIR_JIT_IOS_DEVICE
    if (m_mode == Mode::MapJit) {
        pthread_jit_write_protect_np(0);
    }
#endif
#if YMIR_JIT_IOS_DEVICE
    if (m_mode == Mode::MprotectFlip) {
        mprotect(m_write, m_size, PROT_READ | PROT_WRITE);
    }
#endif
}

void JitMemory::EndWrite(size_t offset, size_t size) {
#if defined(__APPLE__) && !YMIR_JIT_IOS_DEVICE
    if (m_mode == Mode::MapJit) {
        pthread_jit_write_protect_np(1);
    }
#endif
#if YMIR_JIT_IOS_DEVICE
    if (m_mode == Mode::MprotectFlip) {
        mprotect(m_write, m_size, PROT_READ | PROT_EXEC);
    }
#endif
    if (size == 0) {
        return;
    }
#if defined(__APPLE__)
    sys_icache_invalidate(m_exec + offset, size);
#elif defined(__aarch64__) && defined(__GNUC__)
    __builtin___clear_cache(reinterpret_cast<char *>(m_exec + offset), reinterpret_cast<char *>(m_exec + offset + size));
#endif
}

} // namespace ymir::jit
