#pragma once

/**
@file
@brief VDP renderer that draws on the GPU using Metal.

VDP2 (the background layers, the sprite layer made from the VDP1 framebuffer and the final composition) is drawn by the
same compute shaders the Direct3D 12 renderer uses. VDP1 is rendered either by the software renderer this class is
derived from, or, if requested, on the GPU too (polygons, sprites and lines, again with the Direct3D 12 renderer's
shaders). The finished image is read back and delivered through the software renderer's frame callback, so the frontend
needs no changes, unless the frontend gives the renderer a MetalDisplaySink: the image then goes straight into a texture
of the frontend on the GPU, and the frame callback is called with no pixels.
*/

#include <ymir/hw/vdp/renderer/vdp_renderer_sw.hpp>

#include <ymir/util/result.hpp>

#include <memory>

namespace ymir::vdp {

/// @brief Takes the finished images of the Metal renderer on the GPU, for a frontend that displays them with Metal: the
/// renderer copies every image into a texture the sink gives it instead of reading it back to the CPU.
///
/// The renderer calls these on the emulator thread. The objects are passed as `void *` (bridged Objective-C pointers)
/// so that this header stays plain C++.
class MetalDisplaySink {
public:
    virtual ~MetalDisplaySink() = default;

    /// @brief Where to copy an image to.
    struct Target {
        void *texture = nullptr;      ///< `id<MTLTexture>`, RGBA 8-bit, at least (width * scale) x (height * scale)
        void *readyEvent = nullptr;   ///< `id<MTLSharedEvent>` the renderer signals with `readyValue` once it's copied
        uint64 readyValue = 0;
        void *releaseEvent = nullptr; ///< optional `id<MTLSharedEvent>`: the copy waits for it to reach `releaseValue`
        uint64 releaseValue = 0;      ///< (the frontend may still be reading what the texture held)
    };

    /// @brief Gets the texture for the image being finished.
    /// @param[in] width the width of the image (native pixels)
    /// @param[in] height the height of the image (native pixels)
    /// @param[in] scale the internal resolution scale of the image
    /// @param[out] target where to copy the image to
    /// @return `false` to have the image read back and delivered through the frame callback instead
    virtual bool AcquireTarget(uint32 width, uint32 height, uint32 scale, Target &target) = 0;

    /// @brief Notes that the commands copying the image (and signaling `readyValue`) have been committed, so the image
    /// can be displayed.
    virtual void TargetSubmitted(uint64 readyValue) = 0;
};

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

    /// @brief Has the finished images copied into the sink's textures instead of read back (nullptr: read back). Call
    /// on the emulator thread.
    void SetDisplaySink(std::shared_ptr<MetalDisplaySink> sink);

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
