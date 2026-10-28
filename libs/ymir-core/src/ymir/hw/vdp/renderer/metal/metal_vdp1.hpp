#pragma once

/**
@file
@brief VDP1 drawing on the GPU for the Metal renderer.

Follows the Direct3D 12 renderer: the commands are turned into lines (spans) of pixels on the CPU, which the polygon
drawing compute shader draws in parallel into an intermediate buffer, and the output merger shader then merges that into
the framebuffer memory. The framebuffer memory lives on the GPU; the CPU copy is brought up to date only when something
reads it (see SyncFB).

Internal resolutions above 1x: commands are drawn twice, once into the native framebuffer (which stays authoritative:
the CPU reads it and VDP2 falls back on it) and once with coordinates scaled up into a larger framebuffer, like the
software renderer does.

Not available (yet): deinterlacing and transparent meshes.
*/

#include "../common_hw/common_hw_defs.hpp"
#include "../common_hw/memory_usage_tracker.hpp"

#include "metal_gpu.hpp"

#include <ymir/hw/vdp/renderer/common/vdp1_steppers.hpp>
#include <ymir/hw/vdp/vdp_state.hpp>

#include <ymir/util/bit_ops.hpp>
#include <ymir/util/dirty_bitmap.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstring>
#include <vector>

namespace ymir::vdp {

class MetalVDP1 {
public:
    MetalVDP1(VDPState &state, metal::Gpu &gpu)
        : vdpState(state)
        , gpu(gpu) {
        m_batch.cpuSpanParams.resize(kMaxVDP1Spans);
        m_batch.cpuSpanPrefixSums.resize(kMaxVDP1Spans + 1);
        m_batch.cpuCmdParams.resize(kMaxVDP1Commands);
        m_hrBatch.cpuSpanParams.resize(kMaxVDP1Spans);
        m_hrBatch.cpuSpanPrefixSums.resize(kMaxVDP1Spans + 1);
        m_hrBatch.cpuCmdParams.resize(kMaxVDP1Commands);
        std::memset(&cpuCommonRenderParams, 0, sizeof(cpuCommonRenderParams));
        std::memset(&cpuEraseParams, 0, sizeof(cpuEraseParams));
        std::memset(&cpuPolyDrawParams, 0, sizeof(cpuPolyDrawParams));
    }

    /// @brief Brings the GPU side in line with the CPU side: VRAM, framebuffer memory and registers. Drops everything
    /// that was waiting to be drawn.
    void ResyncFromCPU() {
        m_batch.cpuSpanCount = 0;
        m_batch.cpuCmdCount = 0;
        m_batch.cpuSpanPrefixSums[0] = 0;
        vramTexUsageTracker.Clear();
        currPolyDrawShaderIndex = static_cast<size_t>(-1);
        currOutputMergerShaderIndex = static_cast<size_t>(-1);
        cpuFBRAMWritesList.clear();
        fbramDirty.ClearAll();
        action = Action::None;
        m_hrBatch.cpuSpanCount = 0;
        m_hrBatch.cpuCmdCount = 0;
        m_hrBatch.cpuSpanPrefixSums[0] = 0;
        m_batch.ResetBounds();
        m_hrBatch.ResetBounds();
        m_hrValid.fill(false);
        HRClearDirty(0);
        HRClearDirty(1);
        m_hrPass = false;

        std::memcpy(vramShadow.data(), vdpState.mem1.VRAM.data(), vramShadow.size());
        vramDirty.SetAll();
        VDP1FlushVRAM();
        gpu.Vdp1UploadFBRAM(0, vdpState.mem1.FBRAM.data(), static_cast<uint32>(kVDP1FBRAMSize * 2));
        fbramGpuDirty = false;

        VDP1UpdateCommonRenderParams();
        cpuCommonRenderParams.displayParams.drawFB = vdpState.fbIndex.draw;
    }

    void SetVRes(uint32 vres) {
        VRes = vres;
    }

    // -----------------------------------------------------------------------------------------------------------------
    // Called by the renderer

    /// @brief Notes a write to VDP1 VRAM. The write has been made to the VRAM in the VDP state already.
    void WriteVRAM(uint32 address, uint16 value, uint32 size) {
        // Submit spans if the target address is in use by a textured polygon in the current batch
        if (vramTexUsageTracker.IsInUse(address)) [[unlikely]] {
            VDP1SubmitSpans();
        }
        // The GPU gets its VRAM from the shadow copy, which is one write behind the one in the VDP state: the batch
        // above has to be drawn with the contents from before this write, so this write can only reach the shadow copy
        // after it.
        address &= kVDP1VRAMSize - 1;
        if (size == 1) {
            vramShadow[address] = static_cast<uint8>(value);
        } else {
            util::WriteBE<uint16>(&vramShadow[address & ~1u], value);
        }
        vramDirty.Set(address >> kVRAMDirtyBitmapChunkSizeShift);
    }

    void WriteFB(uint32 address, uint32 size) {
        VDP1UpdateAction(Action::FBRAMWrite);
        for (uint32 i = 0; i < size; ++i) {
            fbramDirty.Set(address + i);
        }
    }

    void WriteReg() {
        VDP1UpdateAction(Action::RegWrite);
    }

    /// @brief Makes the CPU copy of the framebuffer memory match the GPU's, for reads by the CPU.
    void SyncFB() {
        // The writes by the CPU are in the CPU copy already, but would be lost by replacing it with the GPU's copy
        VDP1FlushWork();
        if (fbramGpuDirty) {
            fbramGpuDirty = false;
            gpu.Vdp1DownloadFBRAM(vdpState.mem1.FBRAM.data(), static_cast<uint32>(kVDP1FBRAMSize * 2));
        }
    }

    /// @brief Compares the VRAM on the GPU with the one in the VDP state and logs the differences (for diagnostics).
    void DebugCompareVRAM() {
        VDP1FlushWork();
        std::vector<uint8> gpuVRAM(kVDP1VRAMSize);
        gpu.Vdp1DownloadVRAM(gpuVRAM.data(), static_cast<uint32>(gpuVRAM.size()));
        uint32 diffsGpu = 0, diffsShadow = 0;
        uint32 first = ~0u, last = 0;
        for (uint32 i = 0; i < kVDP1VRAMSize; ++i) {
            if (gpuVRAM[i] != vdpState.mem1.VRAM[i]) {
                if (diffsGpu < 6) {
                    metal::LogMessage("    VRAM %05X: GPU %02X, CPU %02X, shadow %02X", i, gpuVRAM[i],
                                      vdpState.mem1.VRAM[i], vramShadow[i]);
                }
                ++diffsGpu;
                first = std::min(first, i);
                last = i;
            }
            diffsShadow += vramShadow[i] != vdpState.mem1.VRAM[i];
        }
        metal::LogMessage("  VRAM: %u bytes differ between the GPU and the CPU (in %05X..%05X), %u between the shadow "
                          "copy and the CPU",
                          diffsGpu, first, last, diffsShadow);
    }

    /// @brief Submits everything that was waiting to be drawn.
    void FlushWork() {
        VDP1FlushWork();
    }

    void EraseFramebuffer(uint64 cycles) {
        VDP1EraseFramebuffer(cycles);
    }

    void SwapFramebuffer() {
        VDP1UpdateAction(Action::Swap);
        VDP1HRLatch();
    }

    /// @brief Sets the internal resolution scale to use from the next framebuffer swap on (1 = native).
    void SetInternalResolution(uint32 scale) {
        m_hrRequested = std::clamp<uint32>(scale, 1, kHRMaxScale);
    }

    /// @brief The scale VDP1 is currently drawn at in the scaled up framebuffers (1 = they aren't in use).
    uint32 InternalResolution() const {
        return m_hrScale;
    }

    /// @brief Whether the scaled up framebuffer of the given index holds what the native one holds (at a larger scale).
    bool HRFramebufferValid(uint8 fbIndex) const {
        return m_hrScale > 1 && m_hrValid[fbIndex] && HRSupportsTVMR();
    }

    /// @brief The lines (native) of the scaled up framebuffer of the given index that may differ from the native
    /// framebuffer scaled up.
    /// @return whether there are any
    bool HRDirtyLines(uint8 fbIndex, uint32 &first, uint32 &last) const {
        if (m_hrDirtyLast[fbIndex] < m_hrDirtyFirst[fbIndex]) {
            return false;
        }
        first = static_cast<uint32>(m_hrDirtyFirst[fbIndex]);
        last = static_cast<uint32>(m_hrDirtyLast[fbIndex]);
        return true;
    }

    struct Stats {
        uint32 resyncs = 0;          ///< scaled up framebuffers filled from the native ones
        uint32 hrBatches = 0;        ///< batches drawn into scaled up framebuffers
        uint32 hrWriteEntries = 0;   ///< CPU framebuffer writes applied to the scaled up framebuffers
        uint64 nativePixels = 0;     ///< pixels (before clipping to the framebuffer) of the batches drawn at the normal size
        uint64 nativeSpans = 0;
        uint64 hrPixels = 0;         ///< the same for the scaled up framebuffers
        uint64 hrSpans = 0;
    };
    Stats TakeStats() {
        const Stats stats = m_stats;
        m_stats = {};
        return stats;
    }

    void BeginFrame() {
        const VDP1Regs &regs1 = vdpState.regs1;
        const VDP2Regs &regs2 = vdpState.regs2;
        m_doubleV =
            kDeinterlace && regs2.TVMD.LSMDn == InterlaceMode::DoubleDensity && !regs1.dblInterlaceEnable;
    }

    void ExecuteCommand(uint32 cmdAddress, VDP1Command::Control control) {
        VDP1UpdateAction(Action::Draw);

        switch (control.command) {
        case VDP1Command::CommandType::DrawNormalSprite: [[fallthrough]];
        case VDP1Command::CommandType::DrawScaledSprite: [[fallthrough]];
        case VDP1Command::CommandType::DrawDistortedSprite: [[fallthrough]];
        case VDP1Command::CommandType::DrawDistortedSpriteAlt: [[fallthrough]];
        case VDP1Command::CommandType::DrawPolygon: [[fallthrough]];
        case VDP1Command::CommandType::DrawPolylines: [[fallthrough]];
        case VDP1Command::CommandType::DrawPolylinesAlt: [[fallthrough]];
        case VDP1Command::CommandType::DrawLine: {
            // (prepared first as it may need what has been drawn so far)
            const bool replay = m_hrScale > 1 && VDP1HRShouldReplay();
            if (m_hrScale > 1 && !replay) {
                // Drawn at the native resolution only, so the scaled up framebuffer no longer follows it
                m_hrValid[vdpState.fbIndex.draw] = false;
            }
            m_hrPass = false;
            VDP1DrawCommand(cmdAddress, control);
            if (replay) {
                // Draw the command again with scaled up coordinates into the scaled up framebuffer
                m_hrPass = true;
                VDP1DrawCommand(cmdAddress, control);
                m_hrPass = false;
            }
            break;
        }

        case VDP1Command::CommandType::UserClipping: [[fallthrough]];
        case VDP1Command::CommandType::UserClippingAlt: VDP1Cmd_SetUserClipping(cmdAddress); break;
        case VDP1Command::CommandType::SystemClipping: VDP1Cmd_SetSystemClipping(cmdAddress); break;
        case VDP1Command::CommandType::SetLocalCoordinates: VDP1Cmd_SetLocalCoordinates(cmdAddress); break;
        }
    }

private:
    // These enhancements aren't available with the Metal renderer yet
    static constexpr bool kDeinterlace = false;
    static constexpr bool kTransparentMeshes = false;

    VDPState &vdpState;
    metal::Gpu &gpu;

    /// @brief Current action being processed on the VDP1 side.
    enum class Action {
        None,       ///< No action -- just reset or flushed everything
        Erase,      ///< Erasing frame
        Swap,       ///< Swapping frame
        Draw,       ///< Executing drawing commands
        FBRAMWrite, ///< Writing to FBRAM
        RegWrite,   ///< Writing to registers
    };
    Action action = Action::None;

    /// @brief Largest internal resolution scale.
    static constexpr uint32 kHRMaxScale = 4;

    /// @brief The spans and commands waiting to be drawn.
    struct Batch {
        std::vector<VDP1SpanParams> cpuSpanParams;
        std::vector<HLSLuint> cpuSpanPrefixSums;
        std::vector<VDP1CommandParams> cpuCmdParams;
        std::vector<HLSLuint> cpuSpanBlockStarts; ///< see BuildSpanBlockStarts
        uint32 cpuSpanCount = 0;
        uint32 cpuCmdCount = 0;

        /// Area the spans draw to (the pixels, including the ones antialiasing adds)
        sint32 minX = INT32_MAX, minY = INT32_MAX, maxX = INT32_MIN, maxY = INT32_MIN;

        void ResetBounds() {
            minX = minY = INT32_MAX;
            maxX = maxY = INT32_MIN;
        }
    } m_batch;

    /// @brief The same for the scaled up framebuffer (internal resolution above 1x only).
    Batch m_hrBatch;
    /// @brief Whether the commands being added are for the scaled up framebuffer.
    bool m_hrPass = false;
    /// @brief The scale requested through the settings, and the one in use (see VDP1HRLatch).
    uint32 m_hrRequested = 1;
    uint32 m_hrScale = 1;
    /// @brief Whether the GPU resources of the scaled up framebuffers exist (or creating them has failed).
    bool m_hrCreated = false;
    bool m_hrFailed = false;
    /// @brief Whether the scaled up framebuffers are in line with the native ones.
    std::array<bool, 2> m_hrValid{};
    /// @brief Lines (native) of each scaled up framebuffer that drawing may have made differ from the native one
    /// scaled up (none if last < first)
    std::array<sint32, 2> m_hrDirtyFirst{INT32_MAX, INT32_MAX};
    std::array<sint32, 2> m_hrDirtyLast{-1, -1};
    Stats m_stats{};

    void HRClearDirty(uint8 fbIndex) {
        m_hrDirtyFirst[fbIndex] = INT32_MAX;
        m_hrDirtyLast[fbIndex] = -1;
    }

    Batch &ActiveBatch() {
        return m_hrPass ? m_hrBatch : m_batch;
    }

    /// @brief Bit shift for the granularity for VRAM dirty bitmap chunks.
    static constexpr size_t kVRAMDirtyBitmapChunkSizeShift = 8;
    static constexpr size_t kVRAMDirtyBitmapChunkSize = static_cast<size_t>(1) << kVRAMDirtyBitmapChunkSizeShift;
    static constexpr size_t kVRAMDirtyBitmapSize = kVDP1VRAMSize / kVRAMDirtyBitmapChunkSize;

    util::DirtyBitmap<kVRAMDirtyBitmapSize> vramDirty;
    /// @brief The VRAM contents the GPU has or is about to get (see WriteVRAM).
    std::vector<uint8> vramShadow = std::vector<uint8>(kVDP1VRAMSize);
    /// @brief Tracks VRAM usage by textures in a batch of spans.
    MemoryUsageTracker<kVDP1VRAMSize> vramTexUsageTracker;

    util::DirtyBitmap<kVDP1FBRAMSize> fbramDirty;
    std::vector<VDP1FBRAMWrite> cpuFBRAMWritesList;

    /// @brief Whether the GPU may have changed the framebuffer memory since the CPU copy was last brought up to date.
    bool fbramGpuDirty = false;

    VDP1CommonRenderParams cpuCommonRenderParams{};
    VDP1EraseParams cpuEraseParams{};
    VDP1PolyDrawParams cpuPolyDrawParams{};

    /// @brief Currently active polygon drawing shader
    size_t currPolyDrawShaderIndex = static_cast<size_t>(-1);
    /// @brief Currently active output merger shader
    size_t currOutputMergerShaderIndex = static_cast<size_t>(-1);
    /// @brief Whether to double vertical coordinates when drawing VDP1 sprites
    bool m_doubleV = false;

    uint32 VRes = kDefaultResV;

    // -----------------------------------------------------------------------------------------------------------------
    // Internal resolution

    /// @brief Whether the VDP1 mode allows drawing at a higher resolution.
    bool HRSupportsTVMR() const {
        const VDP1RegTVMR &tvmr = vdpState.regs1.tvmr;
        return !tvmr.hdtvEnable && !tvmr.fbRotEnable && !tvmr.pixel8Bits;
    }

    /// @brief Takes the requested scale into use. Called when the framebuffers swap, when nothing is pending.
    void VDP1HRLatch() {
        uint32 scale = m_hrRequested;
        if (scale > 1 && !m_hrCreated && !m_hrFailed) {
            std::string error;
            if (gpu.Vdp1EnableHR(error)) {
                m_hrCreated = true;
            } else {
                m_hrFailed = true;
                metal::LogMessage("Internal resolution not available: %s", error.c_str());
            }
        }
        if (m_hrFailed) {
            scale = 1;
        }
        if (scale != m_hrScale) {
            m_hrScale = scale;
            m_hrValid.fill(false);
            HRClearDirty(0);
            HRClearDirty(1);
        }
    }

    /// @brief Makes sure the scaled up framebuffer of the given index is in line with the native one.
    /// @return whether the framebuffer can be drawn to
    bool HRPrepare(uint8 fbIndex) {
        if (m_hrScale <= 1) {
            return false;
        }
        if (!HRSupportsTVMR()) {
            m_hrValid[fbIndex] = false;
            return false;
        }
        if (!m_hrValid[fbIndex]) {
            // What has been drawn natively so far is what the scaled up framebuffer is made from
            VDP1SubmitSpans();
            VDP1FlushFBRAM();

            VDP1CommonRenderParams params = cpuCommonRenderParams;
            params.displayParams.drawFB = fbIndex;
            params.enhancements.vdp1Scale = m_hrScale - 1;
            metal::Vdp1ResyncPassParams pass{};
            pass.constants = &params;
            pass.constantsSize = sizeof(params);
            gpu.Vdp1ResyncHR(pass);
            m_hrValid[fbIndex] = true;
            HRClearDirty(fbIndex);
            ++m_stats.resyncs;
        }
        return true;
    }

    /// @brief Whether the command about to be drawn natively is to be drawn into the scaled up framebuffer too.
    bool VDP1HRShouldReplay() {
        if (m_doubleV || vdpState.regs1.dblInterlaceEnable) {
            return false;
        }
        if (vdpState.regs2.TVMD.LSMDn == InterlaceMode::DoubleDensity) {
            return false;
        }
        return HRPrepare(vdpState.fbIndex.draw);
    }

    /// @brief Scales up the coordinates of a quad. Native coordinates are inclusive pixel indices, so a quad covers its
    /// pixels' whole area. Each vertex is scaled and pushed outwards by the part of a native pixel that lies beyond the
    /// vertex, proportionally to its position along each axis, so that axis-aligned quads cover exactly the scaled up
    /// pixels and neighbors still tile.
    void HRScaleQuad(CoordS32 &a, CoordS32 &b, CoordS32 &c, CoordS32 &d) const {
        const sint32 s = m_hrScale;
        CoordS32 *pts[4] = {&a, &b, &c, &d};
        sint32 minX = a.x(), maxX = a.x(), minY = a.y(), maxY = a.y();
        for (auto *p : pts) {
            minX = std::min(minX, p->x());
            maxX = std::max(maxX, p->x());
            minY = std::min(minY, p->y());
            maxY = std::max(maxY, p->y());
        }
        const sint32 spanX = maxX - minX;
        const sint32 spanY = maxY - minY;
        for (auto *p : pts) {
            const sint32 ox = spanX == 0 ? (s - 1) / 2 : ((p->x() - minX) * (s - 1)) / spanX;
            const sint32 oy = spanY == 0 ? (s - 1) / 2 : ((p->y() - minY) * (s - 1)) / spanY;
            p->x() = p->x() * s + ox;
            p->y() = p->y() * s + oy;
        }
    }

    /// @brief Scales up the coordinates of a line: it stays one (large) pixel thin and goes through the center of the
    /// native pixels.
    void HRScaleLine(CoordS32 &a, CoordS32 &b) const {
        const sint32 s = m_hrScale;
        a.x() = a.x() * s + s / 2;
        a.y() = a.y() * s + s / 2;
        b.x() = b.x() * s + s / 2;
        b.y() = b.y() * s + s / 2;
    }

    // -----------------------------------------------------------------------------------------------------------------
    // Shader selection

    /// @brief Constructs a polygon drawing shader index from its variant options.
    size_t MakeVDP1PolyDrawShaderIndex(VDP1Command::DrawMode mode) const {
        size_t value = 0;
        bit::deposit_into<0>(value, kTransparentMeshes);
        if (mode.msbOn) {
            // MSB -> mode 3 (MSB)
            bit::deposit_into<1, 2>(value, 3);
        } else if (vdpState.regs1.tvmr.pixel8Bits) {
            // 8-bit sprite data -> mode 0 (Copy) -- shading modes not supported
            bit::deposit_into<1, 2>(value, 0);
        } else if (mode.colorCalcBits == 3) {
            // Half-Transparency -> mode 2 (OIT)
            bit::deposit_into<1, 2>(value, 2);
        } else if (mode.colorCalcBits == 1) {
            // Shadow -> mode 1 (Shift)
            bit::deposit_into<1, 2>(value, 1);
        }
        // Other modes -> mode 0 (Copy)
        return value;
    }

    /// @brief Constructs an output merger shader index from its variant options.
    size_t MakeVDP1OutputMergerShaderIndex(VDP1Command::DrawMode mode) const {
        size_t value = 0;
        bit::deposit_into<0>(value, kTransparentMeshes);
        if (vdpState.regs1.tvmr.pixel8Bits) {
            // 8-bit sprite data -> mode 0 (Copy) -- shading modes not supported
            bit::deposit_into<1, 2>(value, 0);
        } else if (mode.colorCalcBits == 1) {
            // Shadow -> mode 1
            bit::deposit_into<1, 2>(value, 1);
        } else if (mode.colorCalcBits == 3) {
            // Half-Luminance -> mode 2
            bit::deposit_into<1, 2>(value, 2);
        }
        // Other modes -> mode 0
        // MSB doesn't invoke this shader.
        return value;
    }

    static bool IsVDP1PolyDrawShaderOIT(size_t index) {
        return bit::extract<1, 2>(index) == 2;
    }

    static bool IsVDP1PolyDrawShaderMSB(size_t index) {
        return bit::extract<1, 2>(index) == 3;
    }

    // -----------------------------------------------------------------------------------------------------------------
    // Actions

    /// @brief Updates the current VDP1 action, flushing any pending work if necessary.
    void VDP1UpdateAction(Action newAction) {
        if (action != newAction) {
            switch (action) {
            case Action::None: break;
            case Action::Erase: /* nothing to sync */ break;
            case Action::Swap: cpuCommonRenderParams.displayParams.drawFB = vdpState.fbIndex.draw; break;
            case Action::Draw: VDP1SubmitSpans(); break;
            case Action::FBRAMWrite: VDP1FlushFBRAM(); break;
            case Action::RegWrite: VDP1UpdateCommonRenderParams(); break;
            }

            action = newAction;
        }
    }

    void VDP1FlushWork() {
        VDP1UpdateAction(Action::None);
    }

    // -----------------------------------------------------------------------------------------------------------------
    // Memory

    /// @brief Sends the VRAM chunks written to since the last call to the GPU.
    void VDP1FlushVRAM() {
        if (!vramDirty) {
            return;
        }

        size_t pos, count = 0;
        for (pos = vramDirty.FindNext(count); pos < vramDirty.Size(); pos = vramDirty.FindNext(count, pos + count)) {
            const uint32 vramOffset = static_cast<uint32>(pos << kVRAMDirtyBitmapChunkSizeShift);
            const uint32 size = static_cast<uint32>(count << kVRAMDirtyBitmapChunkSizeShift);
            gpu.Vdp1UploadVRAM(vramOffset, &vramShadow[vramOffset], size);
        }
        vramDirty.ClearAll();
    }

    /// @brief Sends the writes made to the framebuffer by the CPU to the GPU.
    void VDP1FlushFBRAM() {
        if (!fbramDirty) {
            return;
        }

        auto &fb = vdpState.mem1.FBRAM[vdpState.fbIndex.draw];

        // Group modified FBRAM writes into 32-bit chunks
        size_t pos, count = 0;
        const uint64 *bitmapData = fbramDirty.GetData();
        for (pos = fbramDirty.FindNextGroup<4>(count); pos < fbramDirty.Size();
             pos = fbramDirty.FindNextGroup<4>(count, pos + count)) {
            const uint32 baseAddress = static_cast<uint32>(pos);

            for (size_t i = 0; i < count; i += 4) {
                VDP1FBRAMWrite &write = cpuFBRAMWritesList.emplace_back();
                write.address = baseAddress + static_cast<uint32>(i);

                const uint8 bits = bitmapData[write.address >> 6u] >> (write.address & 63u);
                write.andMask = 0;
                for (uint32 j = 0; j < 4; ++j) {
                    const uint32 shift = j * 8u;
                    if (((bits >> j) & 1u) == 0u) {
                        write.andMask |= (0xFFu << shift);
                    }
                }
                write.orMask = util::ReadLE<uint32>(&fb[write.address]);
                write.orMask &= ~write.andMask;
            }
        }
        fbramDirty.ClearAll();

        if (cpuFBRAMWritesList.empty()) {
            return;
        }

        struct Constants {
            VDP1CommonRenderParams common;
            HLSLuint writeCount;
        } constants{};
        static_assert(offsetof(Constants, writeCount) == sizeof(VDP1CommonRenderParams));
        constants.common = cpuCommonRenderParams;
        constants.writeCount = static_cast<HLSLuint>(cpuFBRAMWritesList.size());

        metal::Vdp1FBWritePassParams pass{};
        pass.constants = &constants;
        pass.constantsSize = sizeof(constants);
        pass.writes = cpuFBRAMWritesList.data();
        pass.writesSize = static_cast<uint32>(cpuFBRAMWritesList.size() * sizeof(VDP1FBRAMWrite));
        pass.count = constants.writeCount;
        gpu.Vdp1WriteFB(pass);

        // The scaled up framebuffer gets the pixels that were written
        const uint8 drawFB = vdpState.fbIndex.draw;
        if (m_hrScale > 1 && m_hrValid[drawFB]) {
            if (HRSupportsTVMR()) {
                constants.common.enhancements.vdp1Scale = m_hrScale - 1;
                pass.hr = true;
                gpu.Vdp1WriteFB(pass);
                m_stats.hrWriteEntries += pass.count;
            } else {
                m_hrValid[drawFB] = false;
            }
        }

        fbramGpuDirty = true;
        cpuFBRAMWritesList.clear();
    }

    void VDP1EraseFramebuffer(uint64 cycles) {
        VDP1UpdateAction(Action::Erase);
        VDP1HRLatch();

        const VDP1Regs &regs1 = vdpState.regs1;

        if (regs1.eraseX1Latch == regs1.eraseX3Latch || regs1.eraseY1Latch == regs1.eraseY3Latch) {
            // Nothing to erase
            return;
        }

        const VDP2Regs &regs2 = vdpState.regs2;

        // Vertical scale is doubled in double-interlace mode
        const bool doubleDensity = regs2.TVMD.LSMDn == InterlaceMode::DoubleDensity;
        const uint32 scaleV = doubleDensity ? 1u : 0u;

        // Constrain erase area to certain limits based on current resolution
        const uint32 maxH = (regs2.TVMD.HRESOn & 1) ? 428 : 400;
        const uint32 maxV = VRes >> scaleV;

        cpuEraseParams.coords.x1 = std::min<uint32>(regs1.eraseX1Latch, maxH) >> 3u;
        cpuEraseParams.coords.y1 = std::min<uint32>(regs1.eraseY1Latch, maxV);
        cpuEraseParams.coords.x3 = std::min<uint32>(regs1.eraseX3Latch, maxH) >> 3u;
        cpuEraseParams.coords.y3 = std::min<uint32>(regs1.eraseY3Latch, maxV);
        cpuEraseParams.coords.scaleV = scaleV;

        cpuEraseParams.erase.writeValue = bit::byte_swap<uint16>(regs1.eraseWriteValueLatch);
        cpuEraseParams.erase.addressShift = regs1.tvmr.eraseOffsetShift - 8;

        cpuEraseParams.vblank.enable = cycles != 0;
        if (cpuEraseParams.vblank.enable) {
            // Compute last line and pixel that can be drawn with the given cycle budget
            const uint32 lineWidth = (cpuEraseParams.coords.x3 << 3u) - (cpuEraseParams.coords.x1 << 3u);
            if (lineWidth > 0) {
                cpuEraseParams.vblank.maxY = cycles / lineWidth + cpuEraseParams.coords.y1;
                cpuEraseParams.vblank.maxX = cycles % lineWidth + cpuEraseParams.coords.x1;
            } else {
                cpuEraseParams.vblank.maxY = 0;
                cpuEraseParams.vblank.maxX = 0;
            }
        }

        struct Constants {
            VDP1CommonRenderParams common;
            VDP1EraseParams erase;
        } constants{};
        static_assert(offsetof(Constants, erase) == sizeof(VDP1CommonRenderParams));
        constants.common = cpuCommonRenderParams;
        constants.erase = cpuEraseParams;

        // The scaled up framebuffer is erased too, if it's in use and the erase area is one it can follow (it has to be
        // brought in line with the native one first)
        const uint8 eraseFB = vdpState.fbIndex.display;
        const bool hrErase = m_hrScale > 1 && !doubleDensity && regs1.tvmr.eraseOffsetShift == 9 && HRPrepare(eraseFB);
        if (m_hrScale > 1 && !hrErase) {
            m_hrValid[eraseFB] = false;
        }

        metal::Vdp1ErasePassParams pass{};
        pass.constants = &constants;
        pass.constantsSize = sizeof(constants);
        pass.width = (cpuEraseParams.coords.x3 << 3) - (cpuEraseParams.coords.x1 << 3) + 1;
        pass.height = cpuEraseParams.coords.y3 - cpuEraseParams.coords.y1 + 1;
        gpu.Vdp1Erase(pass);

        if (hrErase) {
            if (cycles == 0) {
                // (everything is erased, so the scaled up framebuffer is the native one scaled up again)
                HRClearDirty(eraseFB);
            }
            constants.common.enhancements.vdp1Scale = m_hrScale - 1;
            pass.hr = true;
            pass.width *= m_hrScale;
            pass.height *= m_hrScale;
            gpu.Vdp1Erase(pass);
        }

        fbramGpuDirty = true;
    }

    /// @brief Draws the batches of spans.
    void VDP1SubmitSpans() {
        VDP1SubmitBatch(m_batch, false);
        VDP1SubmitBatch(m_hrBatch, true);
        vramTexUsageTracker.Clear();
    }

    /// @brief Fills the table the polygon drawing shader finds the span of a pixel with: for every group of 64 pixels
    /// (a group of threads), the index of the span holding its first pixel.
    /// @return the number of entries
    static uint32 BuildSpanBlockStarts(Batch &batch) {
        const uint32 numPixels = batch.cpuSpanPrefixSums[batch.cpuSpanCount];
        const uint32 numBlocks = (numPixels + 63) / 64;
        if (batch.cpuSpanBlockStarts.size() < numBlocks) {
            batch.cpuSpanBlockStarts.resize(numBlocks);
        }
        uint32 span = 0;
        for (uint32 block = 0; block < numBlocks; ++block) {
            const uint32 pixel = block * 64;
            while (batch.cpuSpanPrefixSums[span + 1] <= pixel) {
                ++span;
            }
            batch.cpuSpanBlockStarts[block] = span;
        }
        return numBlocks;
    }

    /// @brief Draws a batch of spans, into the scaled up framebuffer if `hr`.
    void VDP1SubmitBatch(Batch &batch, bool hr) {
        if (batch.cpuSpanCount == 0) {
            // No spans to dispatch
            batch.cpuCmdCount = 0;
            return;
        }

        // We should have a shader selected by now
        assert(currPolyDrawShaderIndex != static_cast<size_t>(-1));

        cpuPolyDrawParams.numSpans = batch.cpuSpanCount;

        VDP1FlushVRAM();

        struct Constants {
            VDP1CommonRenderParams common;
            VDP1PolyDrawParams polyDraw;
        } constants{};
        static_assert(offsetof(Constants, polyDraw) == sizeof(VDP1CommonRenderParams));
        constants.common = cpuCommonRenderParams;
        constants.common.enhancements.vdp1Scale = hr ? m_hrScale - 1 : 0;
        constants.polyDraw = cpuPolyDrawParams;
        const uint32 scale = hr ? m_hrScale : 1;

        const VDP1Regs &regs1 = vdpState.regs1;
        const uint32 pixelsPerEntry = regs1.tvmr.pixel8Bits ? 4u : 2u; // each entry is 32 bits

        metal::Vdp1DrawParams pass{};
        pass.constants = &constants;
        pass.constantsSize = sizeof(constants);
        pass.commonSize = sizeof(VDP1CommonRenderParams);
        pass.spans = batch.cpuSpanParams.data();
        pass.spansSize = static_cast<uint32>(sizeof(VDP1SpanParams) * batch.cpuSpanCount);
        pass.prefixSums = batch.cpuSpanPrefixSums.data();
        pass.prefixSumsSize = static_cast<uint32>(sizeof(HLSLuint) * (batch.cpuSpanCount + 1));
        const uint32 numBlocks = BuildSpanBlockStarts(batch);
        pass.spanBlockStarts = batch.cpuSpanBlockStarts.data();
        pass.spanBlockStartsSize = static_cast<uint32>(sizeof(HLSLuint) * numBlocks);
        pass.commands = batch.cpuCmdParams.data();
        pass.commandsSize = static_cast<uint32>(sizeof(VDP1CommandParams) * batch.cpuCmdCount);
        pass.shadingMode = static_cast<uint32>(bit::extract<1, 2>(currPolyDrawShaderIndex));
        pass.mergeMode = static_cast<uint32>(bit::extract<1, 2>(currOutputMergerShaderIndex));
        pass.numPixels = batch.cpuSpanPrefixSums[batch.cpuSpanCount];
        (hr ? m_stats.hrPixels : m_stats.nativePixels) += pass.numPixels;
        (hr ? m_stats.hrSpans : m_stats.nativeSpans) += batch.cpuSpanCount;
        pass.mergeWidth = regs1.tvmr.fbSizeH * scale / pixelsPerEntry;
        pass.mergeHeight = regs1.tvmr.fbSizeV * scale;
        if (hr) {
            // The scaled up framebuffer is large and its pixels don't wrap around, so only merge the area drawn to
            // (pixels, antialiasing adds one more around the line)
            const sint32 fbW = static_cast<sint32>(regs1.tvmr.fbSizeH * scale);
            const sint32 fbH = static_cast<sint32>(regs1.tvmr.fbSizeV * scale);
            const sint32 x0 = std::clamp(batch.minX - 1, 0, fbW - 1);
            const sint32 x1 = std::clamp(batch.maxX + 1, 0, fbW - 1);
            const sint32 y0 = std::clamp(batch.minY - 1, 0, fbH - 1);
            const sint32 y1 = std::clamp(batch.maxY + 1, 0, fbH - 1);
            if (x1 < x0 || y1 < y0) {
                batch.cpuSpanCount = 0;
                batch.cpuCmdCount = 0;
                batch.ResetBounds();
                return;
            }
            const sint32 entryX0 = x0 / static_cast<sint32>(pixelsPerEntry);
            const sint32 entryX1 = x1 / static_cast<sint32>(pixelsPerEntry);
            constants.common.mergeOriginX = entryX0;
            constants.common.mergeOriginY = y0;
            pass.mergeWidth = entryX1 - entryX0 + 1;
            pass.mergeHeight = y1 - y0 + 1;

            // The lines of the native framebuffer these pixels belong to may now differ from the native one
            const uint8 drawFB = cpuCommonRenderParams.displayParams.drawFB;
            m_hrDirtyFirst[drawFB] = std::min(m_hrDirtyFirst[drawFB], y0 / static_cast<sint32>(scale));
            m_hrDirtyLast[drawFB] = std::max(m_hrDirtyLast[drawFB], y1 / static_cast<sint32>(scale));
            ++m_stats.hrBatches;
        }
        pass.hr = hr;
        gpu.Vdp1Draw(pass);

        fbramGpuDirty = true;

        batch.cpuSpanCount = 0;
        batch.cpuCmdCount = 0;
        batch.ResetBounds();
    }

    // -----------------------------------------------------------------------------------------------------------------
    // Commands

    /// @brief Adds the spans of a drawing command, for the native or the scaled up framebuffer (see m_hrPass).
    void VDP1DrawCommand(uint32 cmdAddress, VDP1Command::Control control) {
        switch (control.command) {
        case VDP1Command::CommandType::DrawNormalSprite: VDP1Cmd_DrawNormalSprite(cmdAddress, control); break;
        case VDP1Command::CommandType::DrawScaledSprite: VDP1Cmd_DrawScaledSprite(cmdAddress, control); break;
        case VDP1Command::CommandType::DrawDistortedSprite: [[fallthrough]];
        case VDP1Command::CommandType::DrawDistortedSpriteAlt: VDP1Cmd_DrawDistortedSprite(cmdAddress, control); break;

        case VDP1Command::CommandType::DrawPolygon: VDP1Cmd_DrawPolygon(cmdAddress); break;
        case VDP1Command::CommandType::DrawPolylines: [[fallthrough]];
        case VDP1Command::CommandType::DrawPolylinesAlt: VDP1Cmd_DrawPolylines(cmdAddress); break;
        case VDP1Command::CommandType::DrawLine: VDP1Cmd_DrawLine(cmdAddress); break;
        default: break;
        }
    }

    struct VDP1CommandData {
        VDP1Command::DrawMode mode;
        uint16 color;

        uint32 charAddr;
        VDP1Command::Size size;
    };

    struct VDP1SpanData {
        uint16 cmdIndex;
        VDP1Command::DrawMode mode;
        Color555 gouraud0;
        Color555 gouraud1;

        uint32 charAddr;
        VDP1Command::Size size;
        uint32 texV;
        bool flipH;
        uint32 endCodeIndex;
    };

    void VDP1UpdateCommonRenderParams() {
        VDP1CommonRenderParams &params = cpuCommonRenderParams;
        const VDP1Regs &regs1 = vdpState.regs1;
        const VDP2Regs &regs2 = vdpState.regs2;
        const bool doubleDensity = regs2.TVMD.LSMDn == InterlaceMode::DoubleDensity;

        auto &displayParams = params.displayParams;
        displayParams.fbSizeH = std::countr_zero(regs1.tvmr.fbSizeH) - 9u;
        displayParams.fbSizeV = std::countr_zero(regs1.tvmr.fbSizeV) - 8u;
        displayParams.pixel8Bits = regs1.tvmr.pixel8Bits;
        displayParams.doubleDensity = doubleDensity;
        displayParams.dblInterlaceEnable = regs1.dblInterlaceEnable;
        displayParams.dblInterlaceDrawLine = regs1.dblInterlaceDrawLine;
        displayParams.evenOddCoordSelect = regs1.evenOddCoordSelect;
    }

    void VDP1SelectPolyDrawShader(VDP1Command::DrawMode mode) {
        // Submit existing spans before switching shaders
        const size_t polyDrawIndex = MakeVDP1PolyDrawShaderIndex(mode);
        const size_t outputMergerIndex = MakeVDP1OutputMergerShaderIndex(mode);
        if (currPolyDrawShaderIndex != polyDrawIndex || currOutputMergerShaderIndex != outputMergerIndex) {
            VDP1SubmitSpans();
            currPolyDrawShaderIndex = polyDrawIndex;
            currOutputMergerShaderIndex = outputMergerIndex;
        }
    }

    uint16 VDP1AddCommand(const VDP1CommandData &data, bool textured) {

        Batch &batch = ActiveBatch();

        // If list is full, flush it
        if (batch.cpuCmdCount + 1 >= kMaxVDP1Commands) {
            VDP1SubmitSpans();
        }

        // Switch polygon drawing shader based on the current settings
        VDP1SelectPolyDrawShader(data.mode);

        const uint16 cmdIndex = batch.cpuCmdCount++;
        VDP1CommandParams &cmdParams = batch.cpuCmdParams[cmdIndex];

        const VDP1State &state = vdpState.state1;
        const sint32 doubleV = m_doubleV ? 1 : 0;
        if (m_hrPass) {
            // Clipping areas cover whole native pixels, so they end at the last scaled up pixel of the last one
            const uint32 scale = m_hrScale;
            cmdParams.userClip0.x = state.userClipX0 * scale;
            cmdParams.userClip0.y = state.userClipY0 * scale;
            cmdParams.userClip1.x = (state.userClipX1 + 1u) * scale - 1u;
            cmdParams.userClip1.y = (state.userClipY1 + 1u) * scale - 1u;
            cmdParams.sysClip.h = (state.sysClipH + 1u) * scale - 1u;
            cmdParams.sysClip.v = (state.sysClipV + 1u) * scale - 1u;
        } else {
            cmdParams.userClip0.x = state.userClipX0;
            cmdParams.userClip0.y = (state.userClipY0 << doubleV) | doubleV;
            cmdParams.userClip1.x = state.userClipX1;
            cmdParams.userClip1.y = (state.userClipY1 << doubleV) | doubleV;
            cmdParams.sysClip.h = state.sysClipH;
            cmdParams.sysClip.v = (state.sysClipV << doubleV) | doubleV;
        }

        cmdParams.cmdcolr = data.color;
        cmdParams.cmdpmod = data.mode.u16;
        if (textured) {
            cmdParams.cmdsize = data.size.u16;
            cmdParams.cmdsrca = data.charAddr >> 3u;
        }

        return cmdIndex;
    }

    bool VDP1AddSpan(CoordS32 coord0, CoordS32 coord1, VDP1SpanData &data, bool textured, bool antialias) {
        // Discard if completely out of bounds
        if (coord0.x() < 0 && coord1.x() < 0) {
            return false;
        }
        if (coord0.y() < 0 && coord1.y() < 0) {
            return false;
        }
        const sint32 doubleV = m_doubleV ? 1 : 0;
        const sint32 sysClipH =
            m_hrPass ? (static_cast<sint32>(vdpState.state1.sysClipH) + 1) * static_cast<sint32>(m_hrScale) - 1
                     : vdpState.state1.sysClipH;
        if (coord0.x() > sysClipH && coord1.x() > sysClipH) {
            return false;
        }
        const sint32 sysClipV =
            m_hrPass ? (static_cast<sint32>(vdpState.state1.sysClipV) + 1) * static_cast<sint32>(m_hrScale) - 1
                     : (vdpState.state1.sysClipV << doubleV) | doubleV;
        if (coord0.y() > sysClipV && coord1.y() > sysClipV) {
            return false;
        }

        // Determine span length
        LineStepper line{coord0, coord1};
        const uint32 skip = line.SystemClip(sysClipH, sysClipV);
        const uint32 length = line.Length();

        if (length == 0) {
            // Entire line was clipped
            return false;
        }

        Batch &batch = ActiveBatch();

        const bool isOIT = IsVDP1PolyDrawShaderOIT(currPolyDrawShaderIndex);
        // (spans that are antialiased can produce two fragments for a pixel)
        const uint32 fragLimit = isOIT ? kMaxVDP1OITFragmentsPerDispatch / 2 : kMaxVDP1FragmentsPerDispatch;

        // Submit spans now if the total fragment count would exceed the limit
        if (batch.cpuSpanPrefixSums[batch.cpuSpanCount] + length >= fragLimit) {
            VDP1SubmitSpans();

            // Readd command
            batch.cpuCmdParams[0] = batch.cpuCmdParams[data.cmdIndex];
            batch.cpuCmdCount = 1;
            data.cmdIndex = 0;
        }

        // Append span to list
        VDP1SpanParams &spanParams = batch.cpuSpanParams[batch.cpuSpanCount];

        const auto [x0, y0] = coord0;
        const auto [x1, y1] = coord1;

        spanParams.coord0 = {x0, y0};
        spanParams.coord1 = {x1, y1};
        spanParams.cmdIndex = data.cmdIndex;
        spanParams.skip = skip;
        spanParams.attrs.antialias = antialias;

        if (data.mode.gouraudEnable) {
            spanParams.gouraud0.r = data.gouraud0.r;
            spanParams.gouraud0.g = data.gouraud0.g;
            spanParams.gouraud0.b = data.gouraud0.b;
            spanParams.gouraud1.r = data.gouraud1.r;
            spanParams.gouraud1.g = data.gouraud1.g;
            spanParams.gouraud1.b = data.gouraud1.b;
        }

        spanParams.attrs.textured = textured;
        if (textured) {
            spanParams.attrs.texV = data.texV;
            spanParams.attrs.flipH = data.flipH;
            spanParams.attrs.endCodeIndex = data.endCodeIndex;
        }

        batch.minX = std::min({batch.minX, x0, x1});
        batch.maxX = std::max({batch.maxX, x0, x1});
        batch.minY = std::min({batch.minY, y0, y1});
        batch.maxY = std::max({batch.maxY, y0, y1});

        // Update prefix sum
        HLSLuint &nextSum = batch.cpuSpanPrefixSums[batch.cpuSpanCount + 1];
        const HLSLuint currSum = batch.cpuSpanPrefixSums[batch.cpuSpanCount];
        nextSum = currSum + length;

        // If list is full, flush it
        ++batch.cpuSpanCount;
        if (batch.cpuSpanCount >= batch.cpuSpanParams.size()) {
            VDP1SubmitSpans();

            // Readd command
            batch.cpuCmdParams[0] = batch.cpuCmdParams[data.cmdIndex];
            batch.cpuCmdCount = 1;
            data.cmdIndex = 0;
        }

        // Indicate that the span was drawn
        return true;
    }

    void VDP1PlotTexturedQuad(VDP1SpanData &data, uint32 cmdAddress, VDP1Command::Control control, CoordS32 coordA,
                              CoordS32 coordB, CoordS32 coordC, CoordS32 coordD) {
        QuadStepper quad{coordA, coordB, coordC, coordD};

        if (data.mode.gouraudEnable) {
            const uint32 gouraudTable = static_cast<uint32>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x1C)) << 3u;
            Color555 gouraudA;
            Color555 gouraudB;
            Color555 gouraudC;
            Color555 gouraudD;
            gouraudA.u16 = vdpState.mem1.ReadVRAM<uint16>(gouraudTable + 0u);
            gouraudB.u16 = vdpState.mem1.ReadVRAM<uint16>(gouraudTable + 2u);
            gouraudC.u16 = vdpState.mem1.ReadVRAM<uint16>(gouraudTable + 4u);
            gouraudD.u16 = vdpState.mem1.ReadVRAM<uint16>(gouraudTable + 6u);
            quad.SetupGouraud(gouraudA, gouraudB, gouraudC, gouraudD);
        }

        // A width of zero results in the first fetched texel being used for the entire texture.
        // We simulate this by reducing the texture height to one.
        const uint32 charSizeH = data.size.H * 8;
        const uint32 charSizeV = data.size.V;
        const bool flipV = control.flipV;
        TextureStepper texVStepper{};
        quad.SetupTexture(texVStepper, charSizeH == 0 ? 1 : charSizeV, flipV);

        data.texV = charSizeV; // out of range value to ensure first iteration fetches end codes

        // Optimization for the case where the quad goes outside the system clipping area.
        // Skip rendering the rest of the quad when a line is clipped after plotting at least one line.
        // The first few lines of the quad could also be clipped; that is accounted for by requiring at least one
        // plotted line. The point is to skip the calculations once the quad iterator reaches a point where no more
        // lines can be plotted because they all sit outside the system clip area.
        //
        // This also handles a degenerate case with a bowtie quad sitting outside the corner of the screen with two
        // points poking into the screen area in a configuration similar to this:
        //
        //                       D
        //                        B
        //   +-----------------+
        //   |            A    |
        //   |               C |
        //   |                 |
        //   |                 |
        //   |                 |
        //   +-----------------+
        //
        // In this case, the line gets fully clipped partway through the quad, but comes back into view at the end, so
        // we need to check for two sequences of plotted lines rather than one.
        bool linePlotted = false;
        int plottedSegmentsCount = 0;
        const int plottedSegmentsMax = quad.IsDegenerate() ? 2 : 1;

        // Mark VRAM as in use for this texture
        uint32 texSize = charSizeH * charSizeV;
        switch (data.mode.colorMode) {
        case 0: [[fallthrough]];       // 4 bpp, 16 colors, bank mode
        case 1: texSize >>= 1u; break; // 4 bpp, 16 colors, lookup table mode
        case 2: [[fallthrough]];       // 8 bpp, 64 colors, bank mode
        case 3: [[fallthrough]];       // 8 bpp, 128 colors, bank mode
        case 4: break;                 // 8 bpp, 256 colors, bank mode
        case 5: texSize <<= 1u; break; // 16 bpp, 32768 colors, RGB mode
        }
        vramTexUsageTracker.MarkRange(data.charAddr, texSize);
        if (data.mode.colorMode == 1) {
            // The color lookup table is read from VRAM by the GPU too
            vramTexUsageTracker.MarkRange(static_cast<uint32>(ActiveBatch().cpuCmdParams[data.cmdIndex].cmdcolr) * 8u, 32);
        }

        // TODO: cache this
        auto findEndCodeIndex = [&](uint32 v) -> uint32 {
            if (data.mode.endCodeDisable) {
                return charSizeH;
            }

            int endCodeCount = 0;

            for (uint32 i = 0; i < charSizeH; ++i) {
                const uint32 u = control.flipH ? charSizeH - 1 - i : i;

                const uint32 charIndex = u + v * charSizeH;

                auto processEndCode = [&](bool endCode) -> bool {
                    if (endCode && !data.mode.endCodeDisable) {
                        ++endCodeCount;
                    }
                    return endCodeCount >= 2;
                };

                // Read next texel
                uint32 color;
                switch (data.mode.colorMode) {
                case 0: // 4 bpp, 16 colors, bank mode
                    color = vdpState.mem1.ReadVRAM<uint8>(data.charAddr + (charIndex >> 1));
                    color = (color >> ((~u & 1) * 4)) & 0xF;
                    if (processEndCode(color == 0xF)) {
                        return u;
                    }
                    break;
                case 1: // 4 bpp, 16 colors, lookup table mode
                    color = vdpState.mem1.ReadVRAM<uint8>(data.charAddr + (charIndex >> 1));
                    color = (color >> ((~u & 1) * 4)) & 0xF;
                    if (processEndCode(color == 0xF)) {
                        return u;
                    }
                    break;
                case 2: // 8 bpp, 64 colors, bank mode
                    color = vdpState.mem1.ReadVRAM<uint8>(data.charAddr + charIndex);
                    if (processEndCode(color == 0xFF)) {
                        return u;
                    }
                    break;
                case 3: // 8 bpp, 128 colors, bank mode
                    color = vdpState.mem1.ReadVRAM<uint8>(data.charAddr + charIndex);
                    if (processEndCode(color == 0xFF)) {
                        return u;
                    }
                    break;
                case 4: // 8 bpp, 256 colors, bank mode
                    color = vdpState.mem1.ReadVRAM<uint8>(data.charAddr + charIndex);
                    if (processEndCode(color == 0xFF)) {
                        return u;
                    }
                    break;
                case 5: // 16 bpp, 32768 colors, RGB mode
                    color = vdpState.mem1.ReadVRAM<uint16>(data.charAddr + charIndex * sizeof(uint16));
                    if (processEndCode(color == 0x7FFF)) {
                        return u;
                    }
                    break;
                }
            };
            return charSizeH;
        };

        // Interpolate linearly over edges A-D and B-C
        for (; quad.CanStep(); quad.Step()) {
            // Plot lines between the interpolated points
            const CoordS32 coordL = quad.LeftEdge().Coord();
            const CoordS32 coordR = quad.RightEdge().Coord();

            while (texVStepper.ShouldStepTexel()) {
                texVStepper.StepTexel();
            }
            texVStepper.StepPixel();

            const uint32 newTexV = texVStepper.Value();
            if (newTexV != data.texV) {
                data.texV = newTexV;
                data.endCodeIndex = findEndCodeIndex(newTexV);
            }

            if (data.mode.gouraudEnable) {
                data.gouraud0 = quad.LeftEdge().GouraudValue();
                data.gouraud1 = quad.RightEdge().GouraudValue();
            }

            if (VDP1AddSpan(coordL, coordR, data, true, true)) {
                if (!linePlotted) {
                    linePlotted = true;
                    ++plottedSegmentsCount;
                }
            } else if (plottedSegmentsCount >= plottedSegmentsMax) {
                // No more lines can be drawn past this point
                break;
            } else {
                linePlotted = false;
            }
        }
    }

    void VDP1Cmd_DrawNormalSprite(uint32 cmdAddress, VDP1Command::Control control) {
        if (!vdpState.state2.layerEnabled[0]) {
            return;
        }
        const VDP1State &state = vdpState.state1;

        const VDP1Command::DrawMode mode{.u16 = vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x04)};
        const uint16 color = vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x06);
        const uint32 charAddr = vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x08) << 3u;
        const VDP1Command::Size size{.u16 = vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x0A)};
        const sint32 xa = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x0C)) + state.localCoordX;
        const sint32 ya = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x0E)) + state.localCoordY;
        const uint32 charSizeH = size.H * 8;
        const uint32 charSizeV = size.V;

        const sint32 xb = xa + std::max(charSizeH, 1u) - 1u; // right X
        const sint32 yb = ya + std::max(charSizeV, 1u) - 1u; // bottom Y

        const sint32 doubleV = m_doubleV ? 1 : 0;
        const sint32 yAdd = kDeinterlace ? doubleV : 0;

        CoordS32 coordA{xa, ya << doubleV};
        CoordS32 coordB{xb, ya << doubleV};
        CoordS32 coordC{xb, (yb << doubleV) + yAdd};
        CoordS32 coordD{xa, (yb << doubleV) + yAdd};
        if (m_hrPass) {
            HRScaleQuad(coordA, coordB, coordC, coordD);
        }

        const VDP1CommandData cmdData{
            .mode = mode,
            .color = color,
            .charAddr = charAddr,
            .size = size,
        };
        const uint16 cmdIndex = VDP1AddCommand(cmdData, true);

        VDP1SpanData spanData{
            .cmdIndex = cmdIndex,
            .mode = mode,
            .charAddr = charAddr,
            .size = size,
            .flipH = control.flipH != 0,
        };

        VDP1PlotTexturedQuad(spanData, cmdAddress, control, coordA, coordB, coordC, coordD);
    }

    void VDP1Cmd_DrawScaledSprite(uint32 cmdAddress, VDP1Command::Control control) {
        if (!vdpState.state2.layerEnabled[0]) {
            return;
        }

        const VDP1State &state = vdpState.state1;

        const VDP1Command::DrawMode mode{.u16 = vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x04)};
        const uint16 color = vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x06);
        const uint32 charAddr = vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x08) << 3u;
        const VDP1Command::Size size{.u16 = vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x0A)};
        const sint32 xa = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x0C));
        const sint32 ya = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x0E));

        // Calculated quad coordinates
        sint32 qxa = xa;
        sint32 qya = ya;
        sint32 qxb = xa;
        sint32 qyb = ya;
        sint32 qxc = xa;
        sint32 qyc = ya;
        sint32 qxd = xa;
        sint32 qyd = ya;

        const uint8 zoomPointH = bit::extract<0, 1>(control.zoomPoint);
        const uint8 zoomPointV = bit::extract<2, 3>(control.zoomPoint);

        if (zoomPointH == 0) {
            const sint32 xc = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x14));

            qxb = xc;
            qxc = xc;
        } else {
            const sint32 xb = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x10));

            switch (zoomPointH) {
            case 1:
                qxb += xb;
                qxc += xb;
                break;
            case 2:
                qxa -= xb >> 1;
                qxb += (xb + 1) >> 1;
                qxc += (xb + 1) >> 1;
                qxd -= xb >> 1;
                break;
            case 3:
                qxa -= xb;
                qxd -= xb;
                break;
            }
        }

        if (zoomPointV == 0) {
            const sint32 yc = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x16));

            qyc = yc;
            qyd = yc;
        } else {
            const sint32 yb = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x12));

            switch (zoomPointV) {
            case 1:
                qyc += yb;
                qyd += yb;
                break;
            case 2:
                qya -= yb >> 1;
                qyb -= yb >> 1;
                qyc += (yb + 1) >> 1;
                qyd += (yb + 1) >> 1;
                break;
            case 3:
                qya -= yb;
                qyb -= yb;
                break;
            }
        }

        qxa += state.localCoordX;
        qya += state.localCoordY;
        qxb += state.localCoordX;
        qyb += state.localCoordY;
        qxc += state.localCoordX;
        qyc += state.localCoordY;
        qxd += state.localCoordX;
        qyd += state.localCoordY;

        const sint32 doubleV = m_doubleV ? 1 : 0;
        const sint32 yAdd = kDeinterlace ? doubleV : 0;

        CoordS32 coordA{qxa, qya << doubleV};
        CoordS32 coordB{qxb, qyb << doubleV};
        CoordS32 coordC{qxc, (qyc << doubleV) + yAdd};
        CoordS32 coordD{qxd, (qyd << doubleV) + yAdd};
        if (m_hrPass) {
            HRScaleQuad(coordA, coordB, coordC, coordD);
        }

        const VDP1CommandData cmdData{
            .mode = mode,
            .color = color,
            .charAddr = charAddr,
            .size = size,
        };
        const uint16 cmdIndex = VDP1AddCommand(cmdData, true);

        VDP1SpanData spanData{
            .cmdIndex = cmdIndex,
            .mode = mode,
            .charAddr = charAddr,
            .size = size,
            .flipH = control.flipH != 0,
        };

        VDP1PlotTexturedQuad(spanData, cmdAddress, control, coordA, coordB, coordC, coordD);
    }

    void VDP1Cmd_DrawDistortedSprite(uint32 cmdAddress, VDP1Command::Control control) {
        if (!vdpState.state2.layerEnabled[0]) {
            return;
        }

        const VDP1State &state = vdpState.state1;

        const VDP1Command::DrawMode mode{.u16 = vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x04)};
        const uint16 color = vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x06);
        const uint32 charAddr = vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x08) << 3u;
        const VDP1Command::Size size{.u16 = vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x0A)};
        const sint32 xa = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x0C)) + state.localCoordX;
        const sint32 ya = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x0E)) + state.localCoordY;
        const sint32 xb = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x10)) + state.localCoordX;
        const sint32 yb = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x12)) + state.localCoordY;
        const sint32 xc = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x14)) + state.localCoordX;
        const sint32 yc = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x16)) + state.localCoordY;
        const sint32 xd = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x18)) + state.localCoordX;
        const sint32 yd = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x1A)) + state.localCoordY;

        const bool isRegularRect = (xa == xd) && (xb == xc) && (ya == yb) && (yc == yd);

        const sint32 doubleV = m_doubleV ? 1 : 0;
        const sint32 yAddAB = kDeinterlace && isRegularRect && (ya >= yc) ? doubleV : 0;
        const sint32 yAddCD = kDeinterlace && isRegularRect && (ya < yc) ? doubleV : 0;

        CoordS32 coordA{xa, (ya << doubleV) + yAddAB};
        CoordS32 coordB{xb, (yb << doubleV) + yAddAB};
        CoordS32 coordC{xc, (yc << doubleV) + yAddCD};
        CoordS32 coordD{xd, (yd << doubleV) + yAddCD};
        if (m_hrPass) {
            HRScaleQuad(coordA, coordB, coordC, coordD);
        }

        const VDP1CommandData cmdData{
            .mode = mode,
            .color = color,
            .charAddr = charAddr,
            .size = size,
        };
        const uint16 cmdIndex = VDP1AddCommand(cmdData, true);

        VDP1SpanData spanData{
            .cmdIndex = cmdIndex,
            .mode = mode,
            .charAddr = charAddr,
            .size = size,
            .flipH = control.flipH != 0,
        };

        VDP1PlotTexturedQuad(spanData, cmdAddress, control, coordA, coordB, coordC, coordD);
    }

    void VDP1Cmd_DrawPolygon(uint32 cmdAddress) {
        if (!vdpState.state2.layerEnabled[0]) {
            return;
        }

        const VDP1State &state = vdpState.state1;
        const VDP1Command::DrawMode mode{.u16 = vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x04)};

        const uint16 color = vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x06);
        const sint32 xa = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x0C)) + state.localCoordX;
        const sint32 ya = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x0E)) + state.localCoordY;
        const sint32 xb = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x10)) + state.localCoordX;
        const sint32 yb = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x12)) + state.localCoordY;
        const sint32 xc = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x14)) + state.localCoordX;
        const sint32 yc = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x16)) + state.localCoordY;
        const sint32 xd = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x18)) + state.localCoordX;
        const sint32 yd = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x1A)) + state.localCoordY;
        const uint32 gouraudTable = static_cast<uint32>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x1C)) << 3u;

        const bool isRegularRect = (xa == xd) && (xb == xc) && (ya == yb) && (yc == yd);

        const sint32 doubleV = m_doubleV ? 1 : 0;
        const sint32 yAddAB = kDeinterlace && isRegularRect && (ya >= yc) ? doubleV : 0;
        const sint32 yAddCD = kDeinterlace && isRegularRect && (ya < yc) ? doubleV : 0;

        CoordS32 coordA{xa, (ya << doubleV) + yAddAB};
        CoordS32 coordB{xb, (yb << doubleV) + yAddAB};
        CoordS32 coordC{xc, (yc << doubleV) + yAddCD};
        CoordS32 coordD{xd, (yd << doubleV) + yAddCD};
        if (m_hrPass) {
            HRScaleQuad(coordA, coordB, coordC, coordD);
        }

        Color555 gouraudA;
        Color555 gouraudB;
        Color555 gouraudC;
        Color555 gouraudD;
        if (mode.gouraudEnable) {
            gouraudA.u16 = vdpState.mem1.ReadVRAM<uint16>(gouraudTable + 0u);
            gouraudB.u16 = vdpState.mem1.ReadVRAM<uint16>(gouraudTable + 2u);
            gouraudC.u16 = vdpState.mem1.ReadVRAM<uint16>(gouraudTable + 4u);
            gouraudD.u16 = vdpState.mem1.ReadVRAM<uint16>(gouraudTable + 6u);
        }

        const VDP1CommandData cmdData{
            .mode = mode,
            .color = color,
        };
        const uint16 cmdIndex = VDP1AddCommand(cmdData, false);

        VDP1SpanData spanData{
            .cmdIndex = cmdIndex,
            .mode = mode,
        };

        QuadStepper quad{coordA, coordB, coordC, coordD};

        if (mode.gouraudEnable) {
            quad.SetupGouraud(gouraudA, gouraudB, gouraudC, gouraudD);
        }

        // Optimization for the case where the quad goes outside the system clipping area.
        // Skip rendering the rest of the quad when a line is clipped after plotting at least one line.
        // The first few lines of the quad could also be clipped; that is accounted for by requiring at least one
        // plotted line. The point is to skip the calculations once the quad iterator reaches a point where no more
        // lines can be plotted because they all sit outside the system clip area.
        //
        // This also handles a degenerate case with a bowtie quad sitting outside the corner of the screen with two
        // points poking into the screen area in a configuration similar to this:
        //
        //                       D
        //                        B
        //   +-----------------+
        //   |            A    |
        //   |               C |
        //   |                 |
        //   |                 |
        //   |                 |
        //   +-----------------+
        //
        // In this case, the line gets fully clipped partway through the quad, but comes back into view at the end, so
        // we need to check for two sequences of plotted lines rather than one.
        bool linePlotted = false;
        int plottedSegmentsCount = 0;
        const int plottedSegmentsMax = quad.IsDegenerate() ? 2 : 1;

        // Interpolate linearly over edges A-D and B-C
        for (; quad.CanStep(); quad.Step()) {
            // Plot lines between the interpolated points
            const CoordS32 coordL = quad.LeftEdge().Coord();
            const CoordS32 coordR = quad.RightEdge().Coord();

            if (mode.gouraudEnable) {
                spanData.gouraud0 = quad.LeftEdge().GouraudValue();
                spanData.gouraud1 = quad.RightEdge().GouraudValue();
            }

            if (VDP1AddSpan(coordL, coordR, spanData, false, true)) {
                if (!linePlotted) {
                    linePlotted = true;
                    ++plottedSegmentsCount;
                }
            } else if (plottedSegmentsCount >= plottedSegmentsMax) {
                // No more lines can be drawn past this point
                break;
            } else {
                linePlotted = false;
            }
        }
    }

    void VDP1Cmd_DrawPolylines(uint32 cmdAddress) {
        if (!vdpState.state2.layerEnabled[0]) {
            return;
        }

        const VDP1State &state = vdpState.state1;
        const VDP1Command::DrawMode mode{.u16 = vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x04)};

        const uint16 color = vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x06);
        const sint32 xa = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x0C)) + state.localCoordX;
        const sint32 ya = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x0E)) + state.localCoordY;
        const sint32 xb = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x10)) + state.localCoordX;
        const sint32 yb = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x12)) + state.localCoordY;
        const sint32 xc = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x14)) + state.localCoordX;
        const sint32 yc = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x16)) + state.localCoordY;
        const sint32 xd = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x18)) + state.localCoordX;
        const sint32 yd = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x1A)) + state.localCoordY;
        const uint32 gouraudTable = static_cast<uint32>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x1C)) << 3u;

        const bool isRegularRect = (xa == xd) && (xb == xc) && (ya == yb) && (yc == yd);

        const sint32 doubleV = m_doubleV ? 1 : 0;
        const sint32 yAddAB = kDeinterlace && isRegularRect && (ya >= yc) ? doubleV : 0;
        const sint32 yAddCD = kDeinterlace && isRegularRect && (ya < yc) ? doubleV : 0;

        CoordS32 coordA{xa, (ya << doubleV) + yAddAB};
        CoordS32 coordB{xb, (yb << doubleV) + yAddAB};
        CoordS32 coordC{xc, (yc << doubleV) + yAddCD};
        CoordS32 coordD{xd, (yd << doubleV) + yAddCD};
        if (m_hrPass) {
            HRScaleLine(coordA, coordB);
            HRScaleLine(coordC, coordD);
        }

        Color555 gouraudA;
        Color555 gouraudB;
        Color555 gouraudC;
        Color555 gouraudD;
        if (mode.gouraudEnable) {
            gouraudA.u16 = vdpState.mem1.ReadVRAM<uint16>(gouraudTable + 0u);
            gouraudB.u16 = vdpState.mem1.ReadVRAM<uint16>(gouraudTable + 2u);
            gouraudC.u16 = vdpState.mem1.ReadVRAM<uint16>(gouraudTable + 4u);
            gouraudD.u16 = vdpState.mem1.ReadVRAM<uint16>(gouraudTable + 6u);
        }

        const VDP1CommandData cmdData{
            .mode = mode,
            .color = color,
        };
        const uint16 cmdIndex = VDP1AddCommand(cmdData, false);

        VDP1SpanData spanData{
            .cmdIndex = cmdIndex,
            .mode = mode,
        };

        if (mode.gouraudEnable) {
            spanData.gouraud0 = gouraudA;
            spanData.gouraud1 = gouraudB;
        }
        VDP1AddSpan(coordA, coordB, spanData, false, false);
        if (mode.gouraudEnable) {
            spanData.gouraud0 = gouraudB;
            spanData.gouraud1 = gouraudC;
        }
        VDP1AddSpan(coordB, coordC, spanData, false, false);
        if (mode.gouraudEnable) {
            spanData.gouraud0 = gouraudC;
            spanData.gouraud1 = gouraudD;
        }
        VDP1AddSpan(coordC, coordD, spanData, false, false);
        if (mode.gouraudEnable) {
            spanData.gouraud0 = gouraudD;
            spanData.gouraud1 = gouraudA;
        }
        VDP1AddSpan(coordD, coordA, spanData, false, false);
    }

    void VDP1Cmd_DrawLine(uint32 cmdAddress) {
        if (!vdpState.state2.layerEnabled[0]) {
            return;
        }

        const VDP1State &state = vdpState.state1;
        const VDP1Command::DrawMode mode{.u16 = vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x04)};

        const uint16 color = vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x06);
        const sint32 xa = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x0C)) + state.localCoordX;
        const sint32 ya = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x0E)) + state.localCoordY;
        const sint32 xb = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x10)) + state.localCoordX;
        const sint32 yb = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x12)) + state.localCoordY;

        CoordS32 coordA{xa, ya};
        CoordS32 coordB{xb, yb};
        if (m_hrPass) {
            HRScaleLine(coordA, coordB);
        }

        const VDP1CommandData cmdData{
            .mode = mode,
            .color = color,
        };
        const uint16 cmdIndex = VDP1AddCommand(cmdData, false);

        VDP1SpanData spanData{
            .cmdIndex = cmdIndex,
            .mode = mode,
        };

        if (mode.gouraudEnable) {
            const uint32 gouraudTable = static_cast<uint32>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x1C)) << 3u;
            spanData.gouraud0.u16 = vdpState.mem1.ReadVRAM<uint16>(gouraudTable + 0u);
            spanData.gouraud1.u16 = vdpState.mem1.ReadVRAM<uint16>(gouraudTable + 2u);
        }

        VDP1AddSpan(coordA, coordB, spanData, false, false);
    }

    void VDP1Cmd_SetUserClipping(uint32 cmdAddress) {
        VDP1State &state = vdpState.state1;
        const sint32 doubleV = m_doubleV ? 1 : 0;
        state.userClipX0 = bit::extract<0, 9>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x0C));
        state.userClipX1 = bit::extract<0, 9>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x14));
        state.userClipY0 = bit::extract<0, 8>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x0E));
        state.userClipY1 = bit::extract<0, 8>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x16));
    }

    void VDP1Cmd_SetSystemClipping(uint32 cmdAddress) {
        VDP1State &state = vdpState.state1;
        const sint32 doubleV = m_doubleV ? 1 : 0;
        state.sysClipH = bit::extract<0, 9>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x14));
        state.sysClipV = bit::extract<0, 8>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x16));
    }

    void VDP1Cmd_SetLocalCoordinates(uint32 cmdAddress) {
        VDP1State &state = vdpState.state1;
        state.localCoordX = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x0C));
        state.localCoordY = bit::sign_extend<13>(vdpState.mem1.ReadVRAM<uint16>(cmdAddress + 0x0E));
    }

};

} // namespace ymir::vdp
