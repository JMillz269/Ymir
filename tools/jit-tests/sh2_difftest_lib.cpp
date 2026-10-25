// C API used by sh2_difftest.py to run the SH-2 core with the interpreter and with the recompiler. The recompiler's
// AArch64 blocks are executed in the Unicorn CPU emulator by the Python side.
//
// This file pokes at private members of the SH-2 on purpose.

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <memory>
#include <functional>
#include <vector>
#include <mutex>
#include <atomic>
#include <string>
#include <map>
#include <deque>
#include <span>
#include <optional>
#include <variant>
#include <algorithm>
#include <bit>
#include <fmt/format.h>
#include <ymir/core/types.hpp>
#include <ymir/sys/bus.hpp>
#define private public
#define protected public
#include <ymir/hw/sh2/sh2.hpp>
#include <ymir/hw/sh2/sh2_recompiler.hpp>
#undef private
#undef protected

#include <ymir/jit/jit_memory.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

using namespace ymir;

namespace {

constexpr uint32 kRamSize = 1u << 20;

struct Subject {
    sys::SH2Bus bus{};
    sh2::SH2 sh2{bus, true};
    std::array<uint8, kRamSize> wramHigh{};
    std::array<uint8, kRamSize> wramLow{};
    uint64 cycleCounter = 0;
    bool emulateCache = false;
    bool useRecompiler = false;

    Subject() {
        bus.MapArray(0x600'0000, 0x7FF'FFFF, wramHigh, true);
        bus.MapArray(0x020'0000, 0x02F'FFFF, wramLow, true);
        // Distinct timings so that wait state accounting is exercised
        bus.SetAccessCycles(0x600'0000, 0x7FF'FFFF, 3, 2, 4, 3, 5, 4);
        bus.SetAccessCycles(0x020'0000, 0x02F'FFFF, 2, 2, 2, 2, 3, 3);
        sh2.BindGlobalCycleCounter(cycleCounter);
        sh2.BindEmulateCacheOption(emulateCache);
        sh2.BindRecompilerOption(useRecompiler);
        sh2.Reset(true);
    }
};

size_t RoundUpPage(size_t n) {
    return (n + 4095) & ~size_t{4095};
}

} // namespace

extern "C" {

void *t_create(int useRecompiler) {
    void *memory = std::aligned_alloc(4096, RoundUpPage(sizeof(Subject)));
    std::memset(memory, 0, RoundUpPage(sizeof(Subject))); // uninitialized CPU state would differ between instances
    auto *subject = new (memory) Subject();
    subject->useRecompiler = useRecompiler != 0;
    return subject;
}

void t_destroy(void *handle) {
    auto *subject = static_cast<Subject *>(handle);
    subject->~Subject();
    std::free(subject);
}

void t_testing_mode() {
    jit::SetJitTestingMode(true);
}

// Page-aligned range covering the whole subject, which the emulator maps at the same address
void t_object_range(void *handle, uint64 *base, uint64 *size) {
    *base = reinterpret_cast<uint64>(handle);
    *size = RoundUpPage(sizeof(Subject));
}

uint32 t_is_valid_opcode(uint32 word) {
    return sh2::DecodeTable::s_instance.opcodes[0][word & 0xFFFF] != sh2::OpcodeType::Illegal;
}

// 0 = legal, 1 = illegal, for delay slot decoding
uint32 t_opcode_type(uint32 word, uint32 slot) {
    return static_cast<uint32>(sh2::DecodeTable::s_instance.opcodes[slot & 1][word & 0xFFFF]);
}

// 0 = inline, 1 = helper, 2 = delayed branch, 3 = end after, 4 = stop
uint32 t_kind(uint32 word) {
    return static_cast<uint32>(sh2::SH2Recompiler::Classify(sh2::DecodeTable::s_instance.opcodes[0][word & 0xFFFF]));
}

void t_set_mem(void *handle, uint32 address, const uint8 *data, uint32 length) {
    auto *s = static_cast<Subject *>(handle);
    for (uint32 i = 0; i < length; ++i) {
        const uint32 a = address + i;
        if (a >= 0x600'0000 && a < 0x600'0000 + kRamSize) {
            s->wramHigh[a - 0x600'0000] = data[i];
        } else if (a >= 0x020'0000 && a < 0x020'0000 + kRamSize) {
            s->wramLow[a - 0x020'0000] = data[i];
        }
    }
}

void t_get_ram(void *handle, uint8 *high, uint8 *low) {
    auto *s = static_cast<Subject *>(handle);
    std::memcpy(high, s->wramHigh.data(), kRamSize);
    std::memcpy(low, s->wramLow.data(), kRamSize);
}

// State vector layout:
//  0-15 R0-R15, 16 PC, 17 SR, 18 GBR, 19 VBR, 20 PR, 21 MACH, 22 MACL, 23 wbReg, 24 intr pending, 25 intr allow,
//  26 delay slot, 27 delay slot target, 28 sleep
void t_set_state(void *handle, const uint32 *v) {
    auto *s = static_cast<Subject *>(handle);
    auto &c = s->sh2;
    for (int i = 0; i < 16; ++i) {
        c.R[i] = v[i];
    }
    c.PC = v[16];
    c.SR.u32 = v[17];
    c.GBR = v[18];
    c.VBR = v[19];
    c.PR = v[20];
    c.MAC.H = v[21];
    c.MAC.L = v[22];
    c.m_wbReg = static_cast<uint8>(v[23]);
    c.m_intrFlags.pending = v[24] != 0;
    c.m_intrFlags.allow = v[25] != 0;
    c.m_delaySlot = v[26] != 0;
    c.m_delaySlotTarget = v[27];
    c.m_sleep = v[28] != 0;
    c.m_cyclesExecuted = 0;
    // Prefetch as the interpreter would have
    c.RefillPipeline<false>();
}

void t_get_state(void *handle, uint32 *v) {
    auto *s = static_cast<Subject *>(handle);
    auto &c = s->sh2;
    for (int i = 0; i < 16; ++i) {
        v[i] = c.R[i];
    }
    v[16] = c.PC;
    v[17] = c.SR.u32;
    v[18] = c.GBR;
    v[19] = c.VBR;
    v[20] = c.PR;
    v[21] = c.MAC.H;
    v[22] = c.MAC.L;
    v[23] = c.m_wbReg;
    v[24] = c.m_intrFlags.pending;
    v[25] = c.m_intrFlags.allow;
    v[26] = c.m_delaySlot;
    v[27] = c.m_delaySlotTarget;
    v[28] = c.m_sleep;
}

void t_set_intc_level(void *handle, uint32 level) {
    auto *s = static_cast<Subject *>(handle);
    s->sh2.INTC.pending.level = static_cast<uint8>(level);
    s->sh2.INTC.pending.source = sh2::InterruptSource::IRL;
}

uint64 t_advance(void *handle, uint64 cycles, uint64 spillover) {
    auto *s = static_cast<Subject *>(handle);
    const uint64 executed = s->sh2.Advance<false, false>(cycles, spillover);
    s->cycleCounter += executed;
    return executed;
}

typedef uint64 (*ExecutorCallback)(void *fn, void *sh2, uint64 budget);

void t_set_executor(void *handle, ExecutorCallback callback) {
    auto *s = static_cast<Subject *>(handle);
    // The recompiler is created lazily; make sure it exists so that the executor can be installed
    s->sh2.m_recompiler = std::make_shared<sh2::SH2Recompiler>(s->sh2, s->bus);
    if (callback != nullptr) {
        s->sh2.m_recompiler->SetExecutor([callback](sh2::SH2Recompiler::BlockFn fn, sh2::SH2 &cpu, uint64 budget) {
            return callback(reinterpret_cast<void *>(fn), &cpu, budget);
        });
    }
}

void t_flush(void *handle) {
    static_cast<Subject *>(handle)->sh2.FlushRecompiler();
}

uint64 t_step_thunk_address() {
    return reinterpret_cast<uint64>(&sh2::SH2Recompiler::StepThunk);
}

uint64 t_check_thunk_address() {
    return reinterpret_cast<uint64>(&sh2::SH2Recompiler::CodeUnchangedThunk);
}

uint64 t_check_thunk(void *sh2, uint32 blockPc, uint32 offset) {
    return sh2::SH2Recompiler::CodeUnchangedThunk(static_cast<sh2::SH2 *>(sh2), blockPc, offset);
}

uint64 t_step_thunk(void *sh2, uint32 instr) {
    return sh2::SH2Recompiler::StepThunk(static_cast<sh2::SH2 *>(sh2), instr);
}

void t_pool_region(uint64 *base, uint64 *size) {
    uint8 *b = nullptr;
    size_t sz = 0;
    sh2::SH2Recompiler::GetPoolRegion(&b, &sz);
    *base = reinterpret_cast<uint64>(b);
    *size = sz;
}

void t_stats(void *handle, uint64 *out) {
    auto *s = static_cast<Subject *>(handle);
    if (!s->sh2.m_recompiler) {
        std::memset(out, 0, 7 * sizeof(uint64));
        return;
    }
    const auto &st = s->sh2.m_recompiler->GetStats();
    out[0] = st.blocksCompiled;
    out[1] = st.blocksExecuted;
    out[2] = st.blocksRejected;
    out[3] = st.invalidations;
    out[4] = st.cacheFlushes;
    out[5] = st.inlineInstructions;
    out[6] = st.helperInstructions;
}

} // extern "C"
