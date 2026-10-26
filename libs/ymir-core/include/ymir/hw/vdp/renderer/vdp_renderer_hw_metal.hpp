#pragma once

/**
@file
@brief VDP renderer that draws on the GPU using Metal.

VDP2 (the background layers, the sprite layer made from the VDP1 framebuffer and the final composition) is drawn by the
same compute shaders the Direct3D 12 renderer uses. VDP1 is rendered either by the software renderer this class is
derived from, or, if requested, on the GPU too (polygons, sprites and lines, again with the Direct3D 12 renderer's
shaders). The finished image is read back and delivered through the software renderer's frame callback, so the frontend
needs no changes.
*/

#include <ymir/hw/vdp/renderer/vdp_renderer_sw.hpp>

#include <ymir/util/result.hpp>

#include <memory>

namespace ymir::vdp {

class MetalVDPRenderer final : public SoftwareVDPRenderer {
    MetalVDPRenderer(VDPState &state, const config::VDP2DebugRender &vdp2DebugRenderOptions,
                     const config::VDP2AccessPatternsConfig &vdp2AccessPatternsConfig);

    util::VoidResult<> Initialize(void *device, bool gpuVDP1);

public:
    ~MetalVDPRenderer() override;

    /// @brief Creates a Metal VDP renderer.
    /// @param[in] device an `id<MTLDevice>`
    /// @param[in] gpuVDP1 whether to draw VDP1 on the GPU
    static util::ObjectResult<MetalVDPRenderer> Create(VDPState &state,
                                                       const config::VDP2DebugRender &vdp2DebugRenderOptions,
                                                       const config::VDP2AccessPatternsConfig &vdp2AccessPatternsConfig,
                                                       void *device, bool gpuVDP1 = false);

    bool IsGpuAssisted() const override {
        return true;
    }

    // -------------------------------------------------------------------------
    // Configuration

    void UpdateEnhancements() override;

    // -------------------------------------------------------------------------
    // Basics and save states

    void Reset(bool hard) override;
    void PreSaveStateSync() override;
    void PostLoadStateSync() override;

    // -------------------------------------------------------------------------
    // VDP1 memory and register writes (the software renderer's, unless VDP1 is drawn on the GPU)

    void VDP1WriteVRAM(uint32 address, uint8 value) override;
    void VDP1WriteVRAM(uint32 address, uint16 value) override;
    void VDP1SyncFB() override;
    void VDP1WriteFB(uint32 address, uint8 value) override;
    void VDP1WriteFB(uint32 address, uint16 value) override;
    void VDP1WriteReg(uint32 address, uint16 value) override;

    // -------------------------------------------------------------------------
    // VDP2 memory and register writes

    void VDP2WriteVRAM(uint32 address, uint8 value) override;
    void VDP2WriteVRAM(uint32 address, uint16 value) override;
    void VDP2WriteCRAM(uint32 address, uint8 value) override;
    void VDP2WriteCRAM(uint32 address, uint16 value) override;
    void VDP2WriteReg(uint32 address, uint16 value) override;

    // -------------------------------------------------------------------------
    // Rendering process

    void VDP1EraseFramebuffer(uint64 cycles) override;
    void VDP1SwapFramebuffer() override;
    void VDP1BeginFrame() override;
    void VDP1ExecuteCommand(uint32 cmdAddress, VDP1Command::Control control) override;
    void VDP1EndFrame() override;

    void VDP2SetResolution(uint32 h, uint32 v, bool exclusive) override;
    void VDP2BeginFrame() override;
    void VDP2RenderLine(uint32 y) override;
    void VDP2EndFrame() override;

    // -------------------------------------------------------------------------
    // Debugger

    void UpdateEnabledLayers() override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace ymir::vdp
