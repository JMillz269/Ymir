#pragma once

/**
@file
@brief The part of the Metal VDP renderer that talks to the GPU.

`MetalVDPRenderer` works out what has to be drawn and what the shaders have to be told, and hands it to this interface.
The implementation lives in metal_gpu.mm so that the rest of the renderer is plain C++.

Everything submitted here executes on the GPU in the order of the calls: a VRAM upload takes effect for the passes
submitted after it and not for those before it. Data passed by pointer is copied during the call.
*/

#include <ymir/core/types.hpp>

#include <memory>
#include <string>

namespace ymir::vdp::metal {

/// @brief Appends a line to the log of the Metal renderer (`metal-vdp-log.txt` in the Documents folder on iOS). The
/// log is started anew every time the app runs and stops growing after a while.
void LogMessage(const char *format, ...) __attribute__((format(printf, 1, 2)));

/// @brief Whether a file or folder with the given name exists in the Documents folder. Used to turn diagnostics on.
bool DiagnosticSwitch(const char *name);

/// @brief Inputs of the sprite and background layer passes.
struct LayerPassParams {
    const void *common = nullptr; ///< VDP2CommonRenderParams (shader constants)
    uint32 commonSize = 0;
    const void *layerParams = nullptr; ///< VDP2LayerRenderParams
    uint32 layerParamsSize = 0;
    const void *cramColors = nullptr; ///< CRAM converted to RGBA8
    uint32 cramColorsSize = 0;
    const void *cramRotCoeffs = nullptr; ///< raw upper half of CRAM
    uint32 cramRotCoeffsSize = 0;
    const void *rotParamBases = nullptr; ///< VDP2RotParamBase for every line, parameters A then B
    uint32 rotParamBasesSize = 0;

    uint32 hres = 0;     ///< width of the display in pixels
    uint32 numLines = 0; ///< number of lines to draw, starting at the line stored in `common`
};

/// @brief Inputs of the composition pass.
struct ComposePassParams {
    const void *common = nullptr; ///< VDP2CommonRenderParams (shader constants)
    uint32 commonSize = 0;
    const void *composeParams = nullptr; ///< VDP2ComposeParams
    uint32 composeParamsSize = 0;
    const void *lnclBack = nullptr; ///< line color and back screen colors (RGBA8, two arrays of lines)
    uint32 lnclBackSize = 0;

    uint32 hres = 0;
    uint32 numLines = 0;
};

/// @brief Inputs of a VDP1 polygon drawing pass: the spans of a batch of commands, followed by the merging of what they
/// drew into the framebuffer.
struct Vdp1DrawParams {
    const void *constants = nullptr; ///< VDP1CommonRenderParams followed by VDP1PolyDrawParams
    uint32 constantsSize = 0;
    uint32 commonSize = 0; ///< size of the VDP1CommonRenderParams part of the constants (the merger only gets that)
    const void *spans = nullptr; ///< VDP1SpanParams
    uint32 spansSize = 0;
    const void *prefixSums = nullptr; ///< number of spans + 1 pixel counts
    uint32 prefixSumsSize = 0;
    const void *commands = nullptr; ///< VDP1CommandParams
    uint32 commandsSize = 0;

    uint32 shadingMode = 0; ///< polygon drawing variant: 0 = copy, 1 = shift (shadow), 2 = OIT, 3 = MSB
    uint32 mergeMode = 0;   ///< output merger variant: 0 = copy, 1 = shift, 2 = OIT (not used by MSB)
    uint32 numPixels = 0;   ///< total number of pixels of the spans
    uint32 mergeWidth = 0;  ///< size of the output merger dispatch in threads (one for every 2 or 4 pixels of a row)
    uint32 mergeHeight = 0;
};

/// @brief Inputs of the VDP1 framebuffer erase pass.
struct Vdp1ErasePassParams {
    const void *constants = nullptr; ///< VDP1CommonRenderParams followed by VDP1EraseParams
    uint32 constantsSize = 0;
    uint32 width = 0;  ///< width of the area to erase in pixels
    uint32 height = 0; ///< height of the area to erase in pixels
};

/// @brief Inputs of the pass that applies CPU writes to the VDP1 framebuffer.
struct Vdp1FBWritePassParams {
    const void *constants = nullptr; ///< VDP1CommonRenderParams followed by the number of writes
    uint32 constantsSize = 0;
    const void *writes = nullptr; ///< VDP1FBRAMWrite
    uint32 writesSize = 0;
    uint32 count = 0;
};

/// @brief A finished frame.
struct FinishedFrame {
    const uint32 *pixels = nullptr; ///< width * height pixels, R G B A bytes, rows packed
    uint32 width = 0;
    uint32 height = 0;
};

class Gpu {
public:
    virtual ~Gpu() = default;

    /// @brief Creates the GPU interface for the given `id<MTLDevice>`.
    /// @param[in] device the Metal device (as a `void *`)
    /// @param[out] error receives a description of the problem if creation fails
    static std::unique_ptr<Gpu> Create(void *device, std::string &error);

    /// @brief Replaces a part of the VDP2 VRAM.
    virtual void UploadVRAM(uint32 offset, const void *data, uint32 size) = 0;

    /// @brief Replaces a part of the VDP1 framebuffer memory the sprite layer is read from. The two framebuffers are
    /// stored one after the other.
    virtual void UploadSpriteFB(uint32 offset, const void *data, uint32 size) = 0;

    /// @brief Creates what the VDP1 passes need. Has to be called before any of the VDP1 functions.
    /// @param[out] error receives a description of the problem if it fails
    virtual bool EnableVdp1(std::string &error) = 0;

    /// @brief Replaces a part of the VDP1 VRAM.
    virtual void Vdp1UploadVRAM(uint32 offset, const void *data, uint32 size) = 0;

    /// @brief Replaces a part of the VDP1 framebuffer memory. The two framebuffers are stored one after the other.
    /// Same memory as `UploadSpriteFB`.
    virtual void Vdp1UploadFBRAM(uint32 offset, const void *data, uint32 size) = 0;

    /// @brief Draws a batch of spans and merges the result into the framebuffer.
    virtual void Vdp1Draw(const Vdp1DrawParams &params) = 0;

    /// @brief Erases the framebuffer.
    virtual void Vdp1Erase(const Vdp1ErasePassParams &params) = 0;

    /// @brief Applies writes made by the CPU to the framebuffer.
    virtual void Vdp1WriteFB(const Vdp1FBWritePassParams &params) = 0;

    /// @brief Waits for everything submitted so far to finish and counts the entries of the intermediate pixel buffers
    /// that were not cleared by the merging (for diagnostics).
    /// @param[out] listHeads receives the number of fragment list heads that are not empty
    /// @param[out] spriteOut receives the number of sprite output entries that are not zero
    /// @param[out] firstHead receives the pixel index of the first list head that is not empty, and its value
    virtual void Vdp1DebugPixelBuffers(uint32 &listHeads, uint32 &spriteOut, uint32 &firstHeadIndex,
                                       uint32 &firstHeadValue) = 0;

    /// @brief Waits for everything submitted so far to finish and copies the VDP1 VRAM (for diagnostics).
    virtual void Vdp1DownloadVRAM(void *dst, uint32 size) = 0;

    /// @brief Waits for everything submitted so far to finish and copies the start of the VDP1 framebuffer memory.
    virtual void Vdp1DownloadFBRAM(void *dst, uint32 size) = 0;

    /// @brief Draws the sprite layer, then the background layers, for the lines in `params`.
    virtual void RenderLayers(const LayerPassParams &params) = 0;

    /// @brief Composes the layers drawn so far into the output image for the lines in `params`.
    virtual void Compose(const ComposePassParams &params) = 0;

    /// @brief Ends the frame: copies the composed image out of the GPU and submits everything.
    /// @param[in] width the width of the image
    /// @param[in] height the height of the image
    /// @return the image of the frame before this one (the pixels stay valid until the next call), or an empty result
    /// if it isn't available
    virtual FinishedFrame FinishFrame(uint32 width, uint32 height) = 0;
};

} // namespace ymir::vdp::metal
