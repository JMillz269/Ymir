#pragma once

/**
@file
@brief VDP renderer that draws VDP2 on the GPU using Metal.

VDP1 is rendered by the software renderer this class is derived from. VDP2 (the background layers, the sprite layer made
from the VDP1 framebuffer and the final composition) is drawn by the same compute shaders the Direct3D 12 renderer uses.
The finished image is read back and delivered through the software renderer's frame callback, so the frontend needs no
changes.
*/

#include <ymir/hw/vdp/renderer/vdp_renderer_sw.hpp>

#include <ymir/util/result.hpp>

#include <memory>

namespace ymir::vdp {

class MetalVDPRenderer final : public SoftwareVDPRenderer {
    MetalVDPRenderer(VDPState &state, const config::VDP2DebugRender &vdp2DebugRenderOptions,
                     const config::VDP2AccessPatternsConfig &vdp2AccessPatternsConfig);

    util::VoidResult<> Initialize(void *device);

public:
    ~MetalVDPRenderer() override;

    /// @brief Creates a Metal VDP renderer.
    /// @param[in] device an `id<MTLDevice>`
    static util::ObjectResult<MetalVDPRenderer> Create(VDPState &state,
                                                       const config::VDP2DebugRender &vdp2DebugRenderOptions,
                                                       const config::VDP2AccessPatternsConfig &vdp2AccessPatternsConfig,
                                                       void *device);

    bool IsGpuAssisted() const override {
        return true;
    }

    // -------------------------------------------------------------------------
    // Configuration

    void UpdateEnhancements() override;

    // -------------------------------------------------------------------------
    // Basics and save states

    void Reset(bool hard) override;
    void PostLoadStateSync() override;

    // -------------------------------------------------------------------------
    // VDP2 memory and register writes

    void VDP2WriteVRAM(uint32 address, uint8 value) override;
    void VDP2WriteVRAM(uint32 address, uint16 value) override;
    void VDP2WriteCRAM(uint32 address, uint8 value) override;
    void VDP2WriteCRAM(uint32 address, uint16 value) override;
    void VDP2WriteReg(uint32 address, uint16 value) override;

    // -------------------------------------------------------------------------
    // Rendering process

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
