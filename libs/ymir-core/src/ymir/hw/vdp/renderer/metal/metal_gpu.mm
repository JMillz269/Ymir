#include "metal_gpu.hpp"

#include "../common_hw/common_hw_defs.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <cstring>
#include <string>
#include <vector>

namespace ymir::vdp::metal {

namespace {

    // The compute shaders are compiled to Metal libraries at build time (see cmake/CompileShaders.cmake). Names of the
    // shader resources are those of the HLSL variables.
    constexpr const char *kSpriteShaderPath = "src/vdp/cs_vdp2_render_sprite.metallib";
    constexpr const char *kBGShaderPath = "src/vdp/cs_vdp2_render_bgs.metallib";
    constexpr const char *kComposeShaderPath = "src/vdp/cs_vdp2_compose.metallib";
    constexpr const char *kEraseShaderPath = "src/vdp/cs_vdp1_erase.metallib";
    constexpr const char *kFBWriteShaderPath = "src/vdp/cs_vdp1_fbram_write.metallib";
    // (the variants are named after the mesh mode and the shading or merging mode; only the checkerboard mesh mode is
    // used, as transparent meshes aren't available with this renderer)
    constexpr const char *kPolyDrawShaderPaths[4] = {
        "src/vdp/cs_vdp1_polydraw_0_0.metallib",
        "src/vdp/cs_vdp1_polydraw_0_1.metallib",
        "src/vdp/cs_vdp1_polydraw_0_2.metallib",
        "src/vdp/cs_vdp1_polydraw_0_3.metallib",
    };
    constexpr const char *kMergerShaderPaths[3] = {
        "src/vdp/cs_vdp1_output_merger_0_0.metallib",
        "src/vdp/cs_vdp1_output_merger_0_1.metallib",
        "src/vdp/cs_vdp1_output_merger_0_2.metallib",
    };

    /// Texel buffers are emulated by 2D textures this wide (the default width of SPIRV-Cross)
    constexpr NSUInteger kTexelBufferWidth = 4096;

    /// Entries of the buffers with an entry for every pixel of the VDP1 framebuffer (internal sprite output and OIT list
    /// heads); the area a shader can address in them is a multiple of the width of a texel buffer
    constexpr NSUInteger kVDP1PixelBufferEntries = kVDP1FBRAMSize * 2 * 2;

    /// Frames that can be in the GPU's hands at the same time
    constexpr uint32 kNumFrameSlots = 3;

    /// Memory for the data of the passes of a frame
    constexpr NSUInteger kArenaSize = 32u * 1024u * 1024u;

    /// Everything the shaders can be bound to
    enum class Res {
        Unknown,
        Common,
        LayerParams,
        Vram,
        CramColor,
        CramRotCoeff,
        RotParamBases,
        SpriteFB,
        ComposeParams,
        LnclBack,
        LayerOut,
        SpriteAttrs,
        RbgLineColor,
        ColorCalcWindow,
        Composite,

        // VDP1
        Vdp1Spans,
        Vdp1PrefixSums,
        Vdp1Commands,
        Vdp1FBWrites,
        Vdp1SpriteOut,
        Vdp1SpriteOutAtomic,
        Vdp1ListHeads,
        Vdp1ListHeadsAtomic,
        Vdp1Fragments,
        Vdp1Counter,
        Fbram,
    };

    Res ResourceForName(NSString *name) {
        struct Entry {
            const char *name;
            Res res;
        };
        static const Entry kEntries[] = {
            {"g_layerRenderParams", Res::LayerParams},
            {"g_vram", Res::Vram},
            {"g_cramColor", Res::CramColor},
            {"g_cramRotCoeff", Res::CramRotCoeff},
            {"g_rotParamBases", Res::RotParamBases},
            {"g_spriteFB", Res::SpriteFB},
            {"g_composeParams", Res::ComposeParams},
            {"g_lnclBackIn", Res::LnclBack},
            {"g_layerOut", Res::LayerOut},
            {"g_layerIn", Res::LayerOut},
            {"g_spriteAttrsOut", Res::SpriteAttrs},
            {"g_spriteAttrsIn", Res::SpriteAttrs},
            {"g_rbgLineColorOut", Res::RbgLineColor},
            {"g_rbgLineColorIn", Res::RbgLineColor},
            {"g_colorCalcWindowOut", Res::ColorCalcWindow},
            {"g_colorCalcWindowIn", Res::ColorCalcWindow},
            {"g_compositeOut", Res::Composite},
            {"g_spanParams", Res::Vdp1Spans},
            {"g_spanPrefixSums", Res::Vdp1PrefixSums},
            {"g_commandParams", Res::Vdp1Commands},
            {"g_fbramWrites", Res::Vdp1FBWrites},
            {"g_internalSpriteOut", Res::Vdp1SpriteOut},
            {"g_internalSpriteOut_atomic", Res::Vdp1SpriteOutAtomic},
            {"g_listHeads", Res::Vdp1ListHeads},
            {"g_listHeads_atomic", Res::Vdp1ListHeadsAtomic},
            {"g_fragments", Res::Vdp1Fragments},
            {"g_counter", Res::Vdp1Counter},
            {"g_fbramOut", Res::Fbram},
        };
        for (const Entry &entry : kEntries) {
            if ([name isEqualToString:[NSString stringWithUTF8String:entry.name]]) {
                return entry.res;
            }
        }
        // The constant buffer is named after its HLSL cbuffer
        if ([name hasPrefix:@"CommonRenderParams"] || [name hasPrefix:@"type_CommonRenderParams"] ||
            [name hasPrefix:@"RenderParamsBuffer"] || [name hasPrefix:@"type_RenderParamsBuffer"]) {
            return Res::Common;
        }
        return Res::Unknown;
    }

    /// One resource of a shader: what it is and where the shader expects it
    struct Slot {
        Res res;
        bool isTexture;
        NSUInteger index;
    };

    struct Pipeline {
        id<MTLComputePipelineState> state = nil;
        std::vector<Slot> slots;
        bool vdp1 = false; ///< reads the VDP1 VRAM instead of the VDP2 VRAM
    };

    /// Data of a pass, staged in the frame's arena. It stays valid for as long as the contents are the same and the
    /// arena isn't reused.
    struct Blob {
        std::vector<uint8> data;
        uint64 generation = ~0ull;
        NSUInteger offset = 0;
        id<MTLTexture> texture = nil; // view of the data as a texture, for shaders that read it like a texel buffer
    };

    /// Makes the command buffer report to the log if the GPU fails to run it
    void WatchCommandBuffer(id<MTLCommandBuffer> cmdBuf, const char *what) {
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> buf) {
            if (buf.status == MTLCommandBufferStatusError) {
                LogMessage("GPU error in %s: %s", what, [[buf.error description] UTF8String]);
            }
        }];
    }

    NSString *ErrorString(NSError *error) {
        return error != nil ? [error localizedDescription] : @"unknown error";
    }

} // namespace

void LogMessage(const char *format, ...) {
    static std::mutex mutex;
    static FILE *file = nullptr;
    static bool opened = false;
    static uint32 lines = 0;
    std::lock_guard<std::mutex> lock(mutex);
    if (!opened) {
        opened = true;
        NSArray<NSString *> *dirs = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
        if (dirs.count > 0) {
            NSString *path = [dirs[0] stringByAppendingPathComponent:@"metal-vdp-log.txt"];
            file = std::fopen([path fileSystemRepresentation], "w");
        }
    }
    if (file == nullptr || lines >= 3000) {
        return;
    }
    ++lines;
    va_list args;
    va_start(args, format);
    std::vfprintf(file, format, args);
    va_end(args);
    std::fputc('\n', file);
    std::fflush(file);
}

bool DiagnosticSwitch(const char *name) {
    NSArray<NSString *> *dirs = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    if (dirs.count == 0) {
        return false;
    }
    NSString *path = [dirs[0] stringByAppendingPathComponent:[NSString stringWithUTF8String:name]];
    return [[NSFileManager defaultManager] fileExistsAtPath:path];
}

class MetalGpu final : public Gpu {
public:
    ~MetalGpu() override {
        // Wait for the GPU to finish with everything
        if (m_cmdBuf != nil) {
            EndEncoder();
            [m_cmdBuf commit];
            [m_cmdBuf waitUntilCompleted];
            m_cmdBuf = nil;
        }
        if (m_frameStarted) {
            // (this frame was never finished, so nothing will signal its semaphore)
            dispatch_semaphore_signal(m_slots[m_currentSlot].done);
        }
        for (uint32 i = 0; i < kNumFrameSlots; ++i) {
            if (m_slots[i].lastBuf != nil) {
                [m_slots[i].lastBuf waitUntilCompleted];
            }
            if (m_slots[i].done != nil) {
                // A semaphore has to be back at its initial value when it is released
                dispatch_semaphore_wait(m_slots[i].done, DISPATCH_TIME_FOREVER);
                dispatch_semaphore_signal(m_slots[i].done);
            }
        }
    }

    bool Initialize(id<MTLDevice> device, std::string &error) {
        m_device = device;
        m_queue = [device newCommandQueue];
        m_fence = [device newFence];
        if (m_queue == nil) {
            error = "Could not create a Metal command queue";
            return false;
        }

        if (!CreatePipeline(kSpriteShaderPath, m_sprite, error) || !CreatePipeline(kBGShaderPath, m_bgs, error) ||
            !CreatePipeline(kComposeShaderPath, m_compose, error)) {
            return false;
        }

        // Memory the GPU reads the VDP2 VRAM and the VDP1 framebuffers from
        m_vram = [device newBufferWithLength:kVDP2VRAMSize options:MTLResourceStorageModePrivate];
        // (laid out like the Direct3D 12 renderer's: the two framebuffers, then the same for the other fields)
        m_fbram = [device newBufferWithLength:kVDP1FBRAMSize * 2 * 2 * 2 options:MTLResourceStorageModePrivate];
        if (m_vram == nil || m_fbram == nil) {
            error = "Could not allocate GPU memory for VRAM";
            return false;
        }

        // Layer outputs: NBG0-3, RBG0-1, sprite, transparent meshes
        m_layerOut = Create2DTexture(MTLPixelFormatRGBA8Uint, kMaxResH, kMaxResV, 8);
        m_spriteAttrs = Create2DTexture(MTLPixelFormatR16Uint, kMaxResH, kMaxResV, 2);
        m_rbgLineColor = Create2DTexture(MTLPixelFormatRGBA8Uint, kMaxNormalResH, kMaxNormalResV, 2);
        m_colorCalcWindow = Create2DTexture(MTLPixelFormatR8Uint, kMaxResH, kMaxResV, 1);
        m_composite = Create2DTexture(MTLPixelFormatRGBA8Unorm, kMaxResH, kMaxResV, 1);
        if (m_layerOut == nil || m_spriteAttrs == nil || m_rbgLineColor == nil || m_colorCalcWindow == nil ||
            m_composite == nil) {
            error = "Could not allocate GPU textures";
            return false;
        }

        for (uint32 i = 0; i < kNumFrameSlots; ++i) {
            FrameSlot &slot = m_slots[i];
            slot.arena = [device newBufferWithLength:kArenaSize
                                             options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
            slot.readback = [device newBufferWithLength:kMaxResH * kMaxResV * sizeof(uint32)
                                                options:MTLResourceStorageModeShared];
            slot.done = dispatch_semaphore_create(1);
            if (slot.arena == nil || slot.readback == nil) {
                error = "Could not allocate GPU buffers";
                return false;
            }
        }
        return true;
    }

    // -------------------------------------------------------------------------------------------------------------

    void UploadVRAM(uint32 offset, const void *data, uint32 size) override {
        if (size == 0 || offset + size > kVDP2VRAMSize) {
            return;
        }
        UploadTo(m_vram, offset, data, size);
    }

    void UploadSpriteFB(uint32 offset, const void *data, uint32 size) override {
        if (size == 0 || offset + size > kVDP1FBRAMSize * 2) {
            return;
        }
        UploadTo(m_fbram, offset, data, size);
    }

    // -------------------------------------------------------------------------------------------------------------
    // VDP1

    bool EnableVdp1(std::string &error) override {
        if (m_vdp1Enabled) {
            return true;
        }

        for (uint32 i = 0; i < 4; ++i) {
            m_polyDraw[i].vdp1 = true;
            if (!CreatePipeline(kPolyDrawShaderPaths[i], m_polyDraw[i], error)) {
                return false;
            }
        }
        for (uint32 i = 0; i < 3; ++i) {
            m_merger[i].vdp1 = true;
            if (!CreatePipeline(kMergerShaderPaths[i], m_merger[i], error)) {
                return false;
            }
        }
        m_erase.vdp1 = true;
        m_fbWrite.vdp1 = true;
        if (!CreatePipeline(kEraseShaderPath, m_erase, error) || !CreatePipeline(kFBWriteShaderPath, m_fbWrite, error)) {
            return false;
        }

        m_vdp1Vram = [m_device newBufferWithLength:kVDP1VRAMSize options:MTLResourceStorageModePrivate];
        m_spriteOutBuf = [m_device newBufferWithLength:kVDP1PixelBufferEntries * sizeof(uint32)
                                               options:MTLResourceStorageModePrivate];
        m_listHeadsBuf = [m_device newBufferWithLength:kVDP1PixelBufferEntries * sizeof(uint32)
                                               options:MTLResourceStorageModePrivate];
        m_fragments = [m_device newBufferWithLength:kMaxVDP1OITFragmentsPerDispatch * sizeof(VDP1OITFragment)
                                            options:MTLResourceStorageModePrivate];
        m_fbramReadback = [m_device newBufferWithLength:kVDP1FBRAMSize * 2 options:MTLResourceStorageModeShared];
        m_vramReadback = [m_device newBufferWithLength:kVDP1VRAMSize options:MTLResourceStorageModeShared];
        if (m_vramReadback == nil || m_vdp1Vram == nil || m_spriteOutBuf == nil || m_listHeadsBuf == nil || m_fragments == nil ||
            m_fbramReadback == nil) {
            error = "Could not allocate GPU memory for VDP1";
            return false;
        }

        // Shaders access the pixel buffers as texel buffers (and atomically, through the buffers themselves)
        m_spriteOutTex = CreateBufferTexture(m_spriteOutBuf, kVDP1PixelBufferEntries);
        m_listHeadsTex = CreateBufferTexture(m_listHeadsBuf, kVDP1PixelBufferEntries);
        if (m_spriteOutTex == nil || m_listHeadsTex == nil) {
            error = "Could not create GPU textures for VDP1";
            return false;
        }

        // Initial contents: no output pixels, empty fragment lists, empty framebuffers
        id<MTLCommandBuffer> cmdBuf = [m_queue commandBuffer];
        WatchCommandBuffer(cmdBuf, "VDP1 initialization");
        id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
        [blit fillBuffer:m_spriteOutBuf range:NSMakeRange(0, m_spriteOutBuf.length) value:0x00];
        [blit fillBuffer:m_listHeadsBuf range:NSMakeRange(0, m_listHeadsBuf.length) value:0xFF];
        [blit fillBuffer:m_fbram range:NSMakeRange(0, m_fbram.length) value:0x00];
        [blit fillBuffer:m_vdp1Vram range:NSMakeRange(0, m_vdp1Vram.length) value:0x00];
        [blit endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        m_debugClearCheck = DiagnosticSwitch("metal-vdp-compare");
        m_vdp1Enabled = true;
        return true;
    }

    void Vdp1UploadVRAM(uint32 offset, const void *data, uint32 size) override {
        if (size == 0 || offset + size > kVDP1VRAMSize) {
            return;
        }
        UploadTo(m_vdp1Vram, offset, data, size);
    }

    void Vdp1UploadFBRAM(uint32 offset, const void *data, uint32 size) override {
        UploadSpriteFB(offset, data, size);
    }

    void Vdp1Draw(const Vdp1DrawParams &params) override {
        if (params.numPixels == 0 || params.shadingMode > 3 || params.mergeMode > 2) {
            return;
        }
        ++m_logDrawsByMode[params.shadingMode];
        if (m_debugClearCheck && params.shadingMode == 2 && m_traceCalls == 0 && m_logChains == 0) {
            m_traceCalls = 60;
        }
        bool traceThis = false;
        if (m_traceCalls > 0) {
            --m_traceCalls;
            traceThis = true;
            LogMessage("trace: draw shade %u merge %u pixels %u spans %u cmds %u", params.shadingMode, params.mergeMode,
                       params.numPixels, params.spansSize / 48, params.commandsSize / 20);
        }
        if (++m_logDraws <= 5) {
            LogMessage("draw #%u: shade %u merge %u pixels %u spans %u cmds %u", m_logDraws, params.shadingMode,
                params.mergeMode, params.numPixels, params.spansSize / 48, params.commandsSize / 20);
        }
        EnsureFrame();

        Staged staged{};
        uint64 generation;
        do {
            generation = m_arenaGeneration;
            staged.common = StageRaw(m_blobV1Params, params.constants, params.constantsSize);
            staged.v1Spans = StageRaw(m_blobSpans, params.spans, params.spansSize);
            staged.v1PrefixSums = StageRaw(m_blobPrefixSums, params.prefixSums, params.prefixSumsSize,
                                           TexelBufferBytes(params.prefixSumsSize, MTLPixelFormatR32Uint));
            MakeTexelView(m_blobPrefixSums, params.prefixSumsSize, MTLPixelFormatR32Uint);
            staged.v1Commands = StageRaw(m_blobCommands, params.commands, params.commandsSize);
            if (params.shadingMode == 2) {
                // (the counter of fragments starts at zero for every dispatch)
                const uint32 zero = 0;
                staged.v1Counter = StageRaw(m_blobCounter, &zero, sizeof(zero));
            }
        } while (generation != m_arenaGeneration);
        EnsureCompute();

        // Draw the spans
        const Pipeline &draw = m_polyDraw[params.shadingMode];
        [m_encoder setComputePipelineState:draw.state];
        Bind(draw, staged);
        [m_encoder dispatchThreadgroups:MTLSizeMake((params.numPixels + 63) / 64, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        Vdp1Barrier();
        // The drawing wrote the pixel buffers with atomic operations on the buffers, the merger reads and clears them
        // as textures (and the next drawing uses atomics again). Starting a new encoder in between makes sure that
        // each side sees what the other wrote.
        EndEncoder();

        if (params.shadingMode == 2 && m_debugClearCheck && m_logChains < 30) {
            ++m_logChains;
            DebugLogFragmentChains(params);
        }

        if (params.shadingMode != 3) {
            // Merge the output into the framebuffer. The merger gets the common part of the constants only.
            // (MSB doesn't need merging as it is applied to the framebuffer directly)
            Staged mergeStaged{};
            do {
                generation = m_arenaGeneration;
                mergeStaged.common = StageRaw(m_blobMergeCommon, params.constants, params.commonSize);
            } while (generation != m_arenaGeneration);
            EnsureCompute();
            // (the draw above has been recorded already; if the arena was emptied to make room the encoder is a new one)

            const Pipeline &merge = m_merger[params.mergeMode];
            [m_encoder setComputePipelineState:merge.state];
            Bind(merge, mergeStaged);
            [m_encoder dispatchThreadgroups:MTLSizeMake((params.mergeWidth + 7) / 8, (params.mergeHeight + 7) / 8, 1)
                      threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
            Vdp1Barrier();
            EndEncoder();

            if (m_debugClearCheck && m_logClearChecks < 40 && params.shadingMode != 0) {
                ++m_logClearChecks;
                uint32 heads, out, firstIndex, firstValue;
                Vdp1DebugPixelBuffers(heads, out, firstIndex, firstValue);
                LogMessage("after merge (shade %u, merge %u): %u fragment list heads and %u sprite outputs left over "
                           "(first head: pixel %u -> %08X)",
                           params.shadingMode, params.mergeMode, heads, out, firstIndex, firstValue);
                EndEncoder();
            }

            // The merger clears what it has used, but make sure of it: reset the part of the pixel buffer that a
            // pixel of the framebuffer can have an entry in. (A fragment list left behind would be added to the next
            // drawing's lists, which are in the same place.)
            const NSUInteger entries =
                std::min<NSUInteger>(kVDP1PixelBufferEntries, NSUInteger(params.mergeWidth) * 4 * params.mergeHeight);
            id<MTLBuffer> buffer = params.shadingMode == 2 ? m_listHeadsBuf : m_spriteOutBuf;
            id<MTLBlitCommandEncoder> blit = [m_cmdBuf blitCommandEncoder];
            if (m_fenceUpdated) {
                [blit waitForFence:m_fence];
            }
            [blit fillBuffer:buffer range:NSMakeRange(0, entries * sizeof(uint32)) value:params.shadingMode == 2 ? 0xFF : 0x00];
            [blit updateFence:m_fence];
            m_fenceUpdated = true;
            [blit endEncoding];
        }

        if (traceThis) {
            uint32 dp;
            std::memcpy(&dp, params.constants, 4);
            DebugLogSamplePixels("after this draw", (dp >> 7) & 1);
        }
    }

    void Vdp1Erase(const Vdp1ErasePassParams &params) override {
        if (params.width == 0 || params.height == 0) {
            return;
        }
        if (m_traceCalls > 0) {
            --m_traceCalls;
            LogMessage("trace: erase %ux%u", params.width, params.height);
        }
        if (++m_logErases <= 20) {
            LogMessage("erase #%u: %ux%u", m_logErases, params.width, params.height);
        }
        EnsureFrame();

        Staged staged{};
        uint64 generation;
        do {
            generation = m_arenaGeneration;
            staged.common = StageRaw(m_blobV1Params, params.constants, params.constantsSize);
        } while (generation != m_arenaGeneration);
        EnsureCompute();

        [m_encoder setComputePipelineState:m_erase.state];
        Bind(m_erase, staged);
        // (each thread works on two pixels; the shader doesn't depend on the shape of the groups, so use fewer rows per
        // group if the GPU can't run 32x32 threads of this shader at once)
        const NSUInteger rows = std::max<NSUInteger>(1, std::min<NSUInteger>(32, m_erase.state.maxTotalThreadsPerThreadgroup / 32));
        [m_encoder dispatchThreadgroups:MTLSizeMake((params.width + 63) / 64, (params.height + rows - 1) / rows, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, rows, 1)];
        Vdp1Barrier();
    }

    void Vdp1WriteFB(const Vdp1FBWritePassParams &params) override {
        if (params.count == 0) {
            return;
        }
        if (m_traceCalls > 0) {
            --m_traceCalls;
            LogMessage("trace: framebuffer write of %u words", params.count);
        }
        EnsureFrame();

        Staged staged{};
        uint64 generation;
        do {
            generation = m_arenaGeneration;
            staged.common = StageRaw(m_blobV1Params, params.constants, params.constantsSize);
            staged.v1FBWrites = StageRaw(m_blobFBWrites, params.writes, params.writesSize);
        } while (generation != m_arenaGeneration);
        EnsureCompute();

        [m_encoder setComputePipelineState:m_fbWrite.state];
        Bind(m_fbWrite, staged);
        [m_encoder dispatchThreadgroups:MTLSizeMake((params.count + 63) / 64, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        Vdp1Barrier();
    }

    void Vdp1DebugPixelBuffers(uint32 &listHeads, uint32 &spriteOut, uint32 &firstHeadIndex,
                               uint32 &firstHeadValue) override {
        const NSUInteger bytes = kVDP1PixelBufferEntries * sizeof(uint32);
        if (m_debugHeads == nil) {
            m_debugHeads = [m_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            m_debugOut = [m_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        }
        EnsureFrame();
        EndEncoder();
        id<MTLBlitCommandEncoder> blit = [m_cmdBuf blitCommandEncoder];
        [blit copyFromBuffer:m_listHeadsBuf sourceOffset:0 toBuffer:m_debugHeads destinationOffset:0 size:bytes];
        [blit copyFromBuffer:m_spriteOutBuf sourceOffset:0 toBuffer:m_debugOut destinationOffset:0 size:bytes];
        [blit endEncoding];
        DrainArena();

        const uint32 *heads = static_cast<const uint32 *>([m_debugHeads contents]);
        const uint32 *out = static_cast<const uint32 *>([m_debugOut contents]);
        listHeads = 0;
        spriteOut = 0;
        firstHeadIndex = ~0u;
        firstHeadValue = 0;
        for (NSUInteger i = 0; i < kVDP1PixelBufferEntries; ++i) {
            if (heads[i] != 0xFFFFFFFFu) {
                if (listHeads++ == 0) {
                    firstHeadIndex = static_cast<uint32>(i);
                    firstHeadValue = heads[i];
                }
            }
            if (out[i] != 0) {
                ++spriteOut;
            }
        }
    }

    /// Logs the framebuffer under some pixels (for diagnostics)
    void DebugLogSamplePixels(const char *when, uint32 drawFB) {
        EndEncoder();
        id<MTLBlitCommandEncoder> fbBlit = [m_cmdBuf blitCommandEncoder];
        if (m_fenceUpdated) {
            [fbBlit waitForFence:m_fence];
        }
        [fbBlit copyFromBuffer:m_fbram sourceOffset:0 toBuffer:m_fbramReadback destinationOffset:0 size:kVDP1FBRAMSize * 2];
        [fbBlit updateFence:m_fence];
        m_fenceUpdated = true;
        [fbBlit endEncoding];
        DrainArena();
        const uint8 *fb = static_cast<const uint8 *>([m_fbramReadback contents]) + drawFB * kVDP1FBRAMSize;
        char text[300];
        int n = std::snprintf(text, sizeof(text), "  %s, framebuffer %u:", when, drawFB);
        for (uint32 pixel : {9226u, 9227u, 9738u, 13322u}) {
            n += std::snprintf(text + n, sizeof(text) - n, " (%u,%u)=%02X%02X", pixel % 512, pixel / 512, fb[pixel * 2],
                               fb[pixel * 2 + 1]);
        }
        LogMessage("%s", text);
    }

    /// Logs the fragment lists of some pixels (for diagnostics)
    void DebugLogFragmentChains(const Vdp1DrawParams &params) {
        const NSUInteger headBytes = kVDP1PixelBufferEntries * sizeof(uint32);
        const NSUInteger fragBytes = kMaxVDP1OITFragmentsPerDispatch * sizeof(VDP1OITFragment);
        if (m_debugHeads == nil) {
            m_debugHeads = [m_device newBufferWithLength:headBytes options:MTLResourceStorageModeShared];
            m_debugOut = [m_device newBufferWithLength:headBytes options:MTLResourceStorageModeShared];
        }
        if (m_debugFrags == nil) {
            m_debugFrags = [m_device newBufferWithLength:fragBytes options:MTLResourceStorageModeShared];
        }
        EndEncoder();
        id<MTLBlitCommandEncoder> blit = [m_cmdBuf blitCommandEncoder];
        if (m_fenceUpdated) {
            [blit waitForFence:m_fence];
        }
        [blit copyFromBuffer:m_listHeadsBuf sourceOffset:0 toBuffer:m_debugHeads destinationOffset:0 size:headBytes];
        [blit copyFromBuffer:m_fragments sourceOffset:0 toBuffer:m_debugFrags destinationOffset:0 size:fragBytes];
        [blit updateFence:m_fence];
        m_fenceUpdated = true;
        [blit endEncoding];
        DrainArena();

        const uint32 *heads = static_cast<const uint32 *>([m_debugHeads contents]);
        const VDP1OITFragment *frags = static_cast<const VDP1OITFragment *>([m_debugFrags contents]);
        uint32 nonEmpty = 0;
        for (NSUInteger i = 0; i < kVDP1PixelBufferEntries; ++i) {
            nonEmpty += heads[i] != 0xFFFFFFFFu;
        }
        LogMessage("OIT draw #%u: %u pixels with fragments, %u spans, %u pixels", m_logChains, nonEmpty,
                   params.spansSize / 48, params.numPixels);
        {
            // The framebuffer under the pixels, before they are merged
            id<MTLBlitCommandEncoder> fbBlit = [m_cmdBuf blitCommandEncoder];
            [fbBlit copyFromBuffer:m_fbram sourceOffset:0 toBuffer:m_fbramReadback destinationOffset:0 size:kVDP1FBRAMSize * 2];
            [fbBlit endEncoding];
            DrainArena();
            uint32 dp;
            std::memcpy(&dp, params.constants, 4);
            const uint8 *fb = static_cast<const uint8 *>([m_fbramReadback contents]) + ((dp >> 7) & 1) * kVDP1FBRAMSize;
            char text[300];
            int n = std::snprintf(text, sizeof(text), "  framebuffer %u under:", (dp >> 7) & 1);
            for (uint32 pixel : {9226u, 9227u, 9738u, 13322u}) {
                n += std::snprintf(text + n, sizeof(text) - n, " (%u,%u)=%02X%02X", pixel % 512, pixel / 512,
                                   fb[pixel * 2], fb[pixel * 2 + 1]);
            }
            LogMessage("%s", text);
        }
        for (uint32 pixel : {9226u, 9227u, 9738u, 13322u}) {
            char text[400];
            int n = std::snprintf(text, sizeof(text), "  pixel (%u,%u):", pixel % 512, pixel / 512);
            uint32 node = heads[pixel];
            for (uint32 count = 0; node != 0xFFFFFFFFu && count < 8 && n < 350; ++count) {
                if (node >= kMaxVDP1OITFragmentsPerDispatch) {
                    n += std::snprintf(text + n, sizeof(text) - n, " [bad node %08X]", node);
                    break;
                }
                n += std::snprintf(text + n, sizeof(text) - n, " node %u: seq %u color %04X;", node,
                                   frags[node].data >> 16, frags[node].data & 0xFFFF);
                node = frags[node].next;
            }
            LogMessage("%s", text);
        }
    }

    void Vdp1DownloadVRAM(void *dst, uint32 size) override {
        size = std::min<uint32>(size, kVDP1VRAMSize);
        EnsureFrame();
        EndEncoder();
        id<MTLBlitCommandEncoder> blit = [m_cmdBuf blitCommandEncoder];
        [blit copyFromBuffer:m_vdp1Vram sourceOffset:0 toBuffer:m_vramReadback destinationOffset:0 size:size];
        [blit endEncoding];
        DrainArena();
        std::memcpy(dst, [m_vramReadback contents], size);
    }

    void Vdp1DownloadFBRAM(void *dst, uint32 size) override {
        size = std::min<uint32>(size, kVDP1FBRAMSize * 2);
        if (++m_logDownloads <= 30 || m_logDownloads % 200 == 0) {
            LogMessage("FBRAM download #%u", m_logDownloads);
        }
        EnsureFrame();
        EndEncoder();
        id<MTLBlitCommandEncoder> blit = [m_cmdBuf blitCommandEncoder];
        [blit copyFromBuffer:m_fbram sourceOffset:0 toBuffer:m_fbramReadback destinationOffset:0 size:size];
        [blit endEncoding];

        // Submit what has been recorded so far and wait for it
        DrainArena();
        std::memcpy(dst, [m_fbramReadback contents], size);
    }

    void RenderLayers(const LayerPassParams &params) override {
        if (params.numLines == 0) {
            return;
        }
        EnsureFrame();

        // (if the arena fills up while staging, it is emptied and everything has to be staged again)
        Staged staged{};
        uint64 generation;
        do {
            generation = m_arenaGeneration;
            staged.common = Stage(m_blobCommon, params.common, params.commonSize);
            staged.layerParams = Stage(m_blobLayerParams, params.layerParams, params.layerParamsSize);
            staged.cramColor =
                StageTexels(m_blobCramColor, params.cramColors, params.cramColorsSize, MTLPixelFormatRGBA8Uint);
            staged.cramRotCoeff = Stage(m_blobCramRot, params.cramRotCoeffs, params.cramRotCoeffsSize);
            staged.rotParamBases = Stage(m_blobRotBases, params.rotParamBases, params.rotParamBasesSize);
        } while (generation != m_arenaGeneration);
        EnsureCompute();

        // Sprite layer
        [m_encoder setComputePipelineState:m_sprite.state];
        Bind(m_sprite, staged);
        [m_encoder dispatchThreadgroups:MTLSizeMake((params.hres + 31) / 32, params.numLines, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        // Background layers
        [m_encoder setComputePipelineState:m_bgs.state];
        Bind(m_bgs, staged);
        [m_encoder dispatchThreadgroups:MTLSizeMake(params.hres / 32, params.numLines, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 7)];
    }

    void Compose(const ComposePassParams &params) override {
        if (params.numLines == 0) {
            return;
        }
        EnsureFrame();

        Staged staged{};
        uint64 generation;
        do {
            generation = m_arenaGeneration;
            staged.common = Stage(m_blobCommon, params.common, params.commonSize);
            staged.composeParams = Stage(m_blobCompose, params.composeParams, params.composeParamsSize);
            staged.lnclBack =
                StageTexels(m_blobLnclBack, params.lnclBack, params.lnclBackSize, MTLPixelFormatRGBA8Uint);
        } while (generation != m_arenaGeneration);
        EnsureCompute();

        [m_encoder setComputePipelineState:m_compose.state];
        Bind(m_compose, staged);
        [m_encoder dispatchThreadgroups:MTLSizeMake((params.hres + 31) / 32, params.numLines, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    FinishedFrame FinishFrame(uint32 width, uint32 height) override {
        EnsureFrame();
        FrameSlot &slot = m_slots[m_currentSlot];

        width = std::min<uint32>(width, kMaxResH);
        height = std::min<uint32>(height, kMaxResV);
        if (++m_logFrames <= 10 || m_logFrames % 300 == 0) {
            LogMessage("frame %u (%ux%u), draws so far %u (copy %u, shadow %u, half-transparent %u, msb %u), downloads %u",
                       m_logFrames, width, height, m_logDraws, m_logDrawsByMode[0], m_logDrawsByMode[1],
                       m_logDrawsByMode[2], m_logDrawsByMode[3], m_logDownloads);
        }

        // Copy the image out of the GPU
        EndEncoder();
        if (width > 0 && height > 0) {
            id<MTLBlitCommandEncoder> blit = [m_cmdBuf blitCommandEncoder];
            [blit copyFromTexture:m_composite
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(width, height, 1)
                         toBuffer:slot.readback
                destinationOffset:0
           destinationBytesPerRow:width * sizeof(uint32)
         destinationBytesPerImage:width * height * sizeof(uint32)];
            [blit endEncoding];
        }

        dispatch_semaphore_t done = slot.done;
        [m_cmdBuf addCompletedHandler:^(id<MTLCommandBuffer>) {
            dispatch_semaphore_signal(done);
        }];
        [m_cmdBuf commit];
        slot.lastBuf = m_cmdBuf;
        slot.width = width;
        slot.height = height;
        m_cmdBuf = nil;

        // The previous frame has had a whole frame's time to finish
        FinishedFrame result{};
        if (m_hasPrevious) {
            FrameSlot &prev = m_slots[m_previousSlot];
            [prev.lastBuf waitUntilCompleted];
            result.pixels = static_cast<const uint32 *>([prev.readback contents]);
            result.width = prev.width;
            result.height = prev.height;
        }
        m_previousSlot = m_currentSlot;
        m_hasPrevious = true;
        m_currentSlot = (m_currentSlot + 1) % kNumFrameSlots;
        m_frameStarted = false;
        return result;
    }

private:
    struct FrameSlot {
        id<MTLBuffer> arena = nil;
        id<MTLBuffer> readback = nil;
        dispatch_semaphore_t done = nil;
        id<MTLCommandBuffer> lastBuf = nil;
        uint32 width = 0;
        uint32 height = 0;
    };

    /// Where the data of the current pass is in the arena
    struct Staged {
        const Blob *common = nullptr;
        const Blob *layerParams = nullptr;
        const Blob *cramColor = nullptr;
        const Blob *cramRotCoeff = nullptr;
        const Blob *rotParamBases = nullptr;
        const Blob *composeParams = nullptr;
        const Blob *lnclBack = nullptr;
        const Blob *v1Spans = nullptr;
        const Blob *v1PrefixSums = nullptr;
        const Blob *v1Commands = nullptr;
        const Blob *v1FBWrites = nullptr;
        const Blob *v1Counter = nullptr;
    };

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    id<MTLFence> m_fence = nil;
    bool m_fenceUpdated = false;

    Pipeline m_sprite, m_bgs, m_compose;

    // VDP1
    bool m_vdp1Enabled = false;
    uint32 m_logDraws = 0;
    uint32 m_logDownloads = 0;
    uint32 m_logErases = 0;
    uint32 m_logClearChecks = 0;
    bool m_debugClearCheck = false;
    std::array<uint32, 4> m_logDrawsByMode{};
    uint32 m_logFrames = 0;
    std::array<Pipeline, 4> m_polyDraw;
    std::array<Pipeline, 3> m_merger;
    Pipeline m_erase, m_fbWrite;
    id<MTLBuffer> m_vdp1Vram = nil;
    id<MTLBuffer> m_spriteOutBuf = nil;
    id<MTLBuffer> m_listHeadsBuf = nil;
    id<MTLBuffer> m_fragments = nil;
    id<MTLBuffer> m_fbramReadback = nil;
    id<MTLBuffer> m_vramReadback = nil;
    id<MTLBuffer> m_debugHeads = nil;
    id<MTLBuffer> m_debugOut = nil;
    id<MTLBuffer> m_debugFrags = nil;
    uint32 m_logChains = 0;
    uint32 m_traceCalls = 0;
    id<MTLTexture> m_spriteOutTex = nil;
    id<MTLTexture> m_listHeadsTex = nil;

    id<MTLBuffer> m_vram = nil;
    id<MTLBuffer> m_fbram = nil; ///< VDP1 framebuffer memory, shared by the VDP1 passes and the VDP2 sprite layer
    id<MTLTexture> m_layerOut = nil;
    id<MTLTexture> m_spriteAttrs = nil;
    id<MTLTexture> m_rbgLineColor = nil;
    id<MTLTexture> m_colorCalcWindow = nil;
    id<MTLTexture> m_composite = nil;

    std::array<FrameSlot, kNumFrameSlots> m_slots;
    uint32 m_currentSlot = 0;
    uint32 m_previousSlot = 0;
    bool m_hasPrevious = false;
    bool m_frameStarted = false;

    id<MTLCommandBuffer> m_cmdBuf = nil;
    id<MTLComputeCommandEncoder> m_encoder = nil;

    NSUInteger m_arenaOffset = 0;
    uint64 m_arenaGeneration = 0;

    Blob m_blobCommon, m_blobLayerParams, m_blobCramColor, m_blobCramRot, m_blobRotBases, m_blobCompose,
        m_blobLnclBack;
    // (the data of the VDP1 passes is never reused, so these are only places to keep the staging results)
    Blob m_blobV1Params, m_blobSpans, m_blobPrefixSums, m_blobCommands, m_blobFBWrites, m_blobCounter,
        m_blobMergeCommon;

    // -------------------------------------------------------------------------------------------------------------
    // Setup

    id<MTLTexture> Create2DTexture(MTLPixelFormat format, NSUInteger width, NSUInteger height, NSUInteger slices) {
        MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
        desc.textureType = slices > 1 ? MTLTextureType2DArray : MTLTextureType2D;
        desc.pixelFormat = format;
        desc.width = width;
        desc.height = height;
        desc.arrayLength = slices;
        desc.storageMode = MTLStorageModePrivate;
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        return [m_device newTextureWithDescriptor:desc];
    }

    bool CreatePipeline(const char *path, Pipeline &pipeline, std::string &error) {
        auto &fs = g_fsShaders;
        if (!fs.is_file(path)) {
            error = std::string("Embedded shader not found: ") + path;
            return false;
        }
        auto file = fs.open(path);
        dispatch_data_t data = dispatch_data_create(file.begin(), file.size(), nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        NSError *nsError = nil;
        id<MTLLibrary> library = [m_device newLibraryWithData:data error:&nsError];
        if (library == nil) {
            error = std::string("Could not load shader library ") + path + ": " + [ErrorString(nsError) UTF8String];
            return false;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"CSMain"];
        if (function == nil && [library.functionNames count] > 0) {
            function = [library newFunctionWithName:library.functionNames[0]];
        }
        if (function == nil) {
            error = std::string("No compute function in shader library ") + path;
            return false;
        }

        MTLAutoreleasedComputePipelineReflection reflection = nil;
        pipeline.state = [m_device newComputePipelineStateWithFunction:function
                                                               options:MTLPipelineOptionArgumentInfo
                                                            reflection:&reflection
                                                                 error:&nsError];
        if (pipeline.state == nil) {
            error = std::string("Could not create compute pipeline for ") + path + ": " +
                    [ErrorString(nsError) UTF8String];
            return false;
        }

        for (MTLArgument *arg in reflection.arguments) {
            Slot slot{};
            slot.res = ResourceForName(arg.name);
            slot.isTexture = arg.type == MTLArgumentTypeTexture;
            slot.index = arg.index;
            if (arg.type != MTLArgumentTypeBuffer && arg.type != MTLArgumentTypeTexture) {
                continue;
            }
            if (slot.res == Res::Unknown) {
                LogMessage("pipeline %s: resource '%s' not recognized!", path, [arg.name UTF8String]);
                std::fprintf(stderr, "[Ymir] Metal VDP: unrecognized shader resource '%s' in %s\n",
                             [arg.name UTF8String], path);
                continue;
            }
            pipeline.slots.push_back(slot);
        }
        return true;
    }

    // -------------------------------------------------------------------------------------------------------------
    // Frames and command encoding

    void EnsureFrame() {
        if (m_frameStarted) {
            return;
        }
        FrameSlot &slot = m_slots[m_currentSlot];
        // Wait until the GPU is done with the frame that used this slot before
        dispatch_semaphore_wait(slot.done, DISPATCH_TIME_FOREVER);
        m_arenaOffset = 0;
        ++m_arenaGeneration;
        m_cmdBuf = [m_queue commandBuffer];
        WatchCommandBuffer(m_cmdBuf, "frame");
        m_encoder = nil;
        m_frameStarted = true;
    }

    void EndEncoder() {
        if (m_encoder != nil) {
            // Every compute encoder waits for the one before it (see EnsureCompute)
            [m_encoder updateFence:m_fence];
            m_fenceUpdated = true;
            [m_encoder endEncoding];
            m_encoder = nil;
        }
    }

    void EnsureCompute() {
        if (m_encoder == nil) {
            m_encoder = [m_cmdBuf computeCommandEncoder];
            // Metal doesn't necessarily know that a texture made from a buffer and the buffer are the same memory, and
            // runs encoders that don't seem to depend on each other at the same time. The shaders use the same memory
            // both ways (atomics on the buffer, reads and writes through the texture), so make every encoder wait for
            // the previous one.
            if (m_fenceUpdated) {
                [m_encoder waitForFence:m_fence];
            }
        }
    }

    /// Submits the work recorded so far and waits for it, to make room in the arena
    void DrainArena() {
        EndEncoder();
        [m_cmdBuf commit];
        [m_cmdBuf waitUntilCompleted];
        m_cmdBuf = [m_queue commandBuffer];
        WatchCommandBuffer(m_cmdBuf, "frame");
        m_arenaOffset = 0;
        ++m_arenaGeneration;
    }

    NSUInteger Allocate(NSUInteger size) {
        constexpr NSUInteger kAlign = 256;
        size = (size + kAlign - 1) & ~(kAlign - 1);
        if (m_arenaOffset + size > kArenaSize) {
            DrainArena();
        }
        const NSUInteger offset = m_arenaOffset;
        m_arenaOffset += size;
        return offset;
    }

    /// Copies data into the arena, unless the same data is already there
    const Blob *Stage(Blob &blob, const void *data, uint32 size) {
        if (data == nullptr || size == 0) {
            return nullptr;
        }
        if (blob.generation == m_arenaGeneration && blob.data.size() == size &&
            std::memcmp(blob.data.data(), data, size) == 0) {
            return &blob;
        }
        blob.offset = Allocate(size);
        std::memcpy(static_cast<uint8 *>([m_slots[m_currentSlot].arena contents]) + blob.offset, data, size);
        blob.data.assign(static_cast<const uint8 *>(data), static_cast<const uint8 *>(data) + size);
        blob.generation = m_arenaGeneration;
        blob.texture = nil;
        return &blob;
    }

    /// Like Stage, for data the shader reads as a buffer of texels (Buffer<uint4> in HLSL)
    const Blob *StageTexels(Blob &blob, const void *data, uint32 size, MTLPixelFormat format) {
        const Blob *staged = Stage(blob, data, size);
        if (staged == nullptr || blob.texture != nil) {
            return staged;
        }
        // (SPIRV-Cross turns a texel buffer into a 2D texture; a single row holds all the texels)
        const NSUInteger texelSize = 4;
        MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                         width:size / texelSize
                                                                                        height:1
                                                                                     mipmapped:NO];
        desc.storageMode = MTLStorageModeShared;
        desc.cpuCacheMode = MTLCPUCacheModeWriteCombined; // (has to match the buffer)
        desc.usage = MTLTextureUsageShaderRead;
        blob.texture = [m_slots[m_currentSlot].arena newTextureWithDescriptor:desc offset:blob.offset bytesPerRow:size];
        return staged;
    }

    /// Copies data into the arena, without keeping a copy of it or looking for the same data
    const Blob *StageRaw(Blob &blob, const void *data, uint32 size, NSUInteger paddedSize = 0) {
        if (data == nullptr || size == 0) {
            return nullptr;
        }
        blob.offset = Allocate(std::max<NSUInteger>(size, paddedSize));
        std::memcpy(static_cast<uint8 *>([m_slots[m_currentSlot].arena contents]) + blob.offset, data, size);
        blob.generation = m_arenaGeneration;
        blob.texture = nil;
        return &blob;
    }

    /// Bytes per row of a texel buffer with the given number of texels
    NSUInteger TexelBufferRowBytes(NSUInteger texels, MTLPixelFormat format) {
        const NSUInteger width = std::min<NSUInteger>(std::max<NSUInteger>(texels, 1), kTexelBufferWidth);
        const NSUInteger alignment = [m_device minimumLinearTextureAlignmentForPixelFormat:format];
        return (width * 4 + alignment - 1) / alignment * alignment;
    }

    /// Bytes needed to keep the given number of bytes of 32-bit texels in a texture made from a buffer
    NSUInteger TexelBufferBytes(uint32 size, MTLPixelFormat format) {
        const NSUInteger texels = size / 4;
        const NSUInteger rows = std::max<NSUInteger>((texels + kTexelBufferWidth - 1) / kTexelBufferWidth, 1);
        return rows * TexelBufferRowBytes(texels, format);
    }

    /// Makes the staged data readable as a texel buffer of 32-bit texels, which shaders see as a 2D texture
    void MakeTexelView(Blob &blob, uint32 size, MTLPixelFormat format) {
        if (blob.generation != m_arenaGeneration || blob.texture != nil) {
            return;
        }
        const NSUInteger texels = size / 4;
        const NSUInteger rowBytes = TexelBufferRowBytes(texels, format);
        MTLTextureDescriptor *desc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                               width:std::min<NSUInteger>(std::max<NSUInteger>(texels, 1), kTexelBufferWidth)
                                                              height:std::max<NSUInteger>((texels + kTexelBufferWidth - 1) / kTexelBufferWidth, 1)
                                                           mipmapped:NO];
        desc.storageMode = MTLStorageModeShared;
        desc.cpuCacheMode = MTLCPUCacheModeWriteCombined; // (has to match the buffer)
        desc.usage = MTLTextureUsageShaderRead;
        blob.texture = [m_slots[m_currentSlot].arena newTextureWithDescriptor:desc offset:blob.offset bytesPerRow:rowBytes];
    }

    /// Creates the view shaders see a buffer of 32-bit values as a texel buffer through
    id<MTLTexture> CreateBufferTexture(id<MTLBuffer> buffer, NSUInteger entries) {
        MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
                                                                                         width:kTexelBufferWidth
                                                                                        height:entries / kTexelBufferWidth
                                                                                     mipmapped:NO];
        desc.storageMode = MTLStorageModePrivate;
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        return [buffer newTextureWithDescriptor:desc offset:0 bytesPerRow:kTexelBufferWidth * sizeof(uint32)];
    }

    /// Makes what a VDP1 pass wrote visible to the next one
    void Vdp1Barrier() {
        [m_encoder memoryBarrierWithScope:MTLBarrierScopeBuffers | MTLBarrierScopeTextures];
    }

    /// Copies data to a GPU buffer in sequence with the passes
    void UploadTo(id<MTLBuffer> dst, NSUInteger dstOffset, const void *data, uint32 size) {
        EnsureFrame();
        const NSUInteger offset = Allocate(size);
        id<MTLBuffer> arena = m_slots[m_currentSlot].arena;
        std::memcpy(static_cast<uint8 *>([arena contents]) + offset, data, size);
        EndEncoder();
        id<MTLBlitCommandEncoder> blit = [m_cmdBuf blitCommandEncoder];
        [blit copyFromBuffer:arena sourceOffset:offset toBuffer:dst destinationOffset:dstOffset size:size];
        [blit endEncoding];
    }

    void Bind(const Pipeline &pipeline, const Staged &staged) {
        id<MTLBuffer> arena = m_slots[m_currentSlot].arena;
        for (const Slot &slot : pipeline.slots) {
            const Blob *blob = nullptr;
            id<MTLBuffer> buffer = nil;
            NSUInteger offset = 0;
            id<MTLTexture> texture = nil;
            switch (slot.res) {
            case Res::Common: blob = staged.common; break;
            case Res::LayerParams: blob = staged.layerParams; break;
            case Res::CramColor: blob = staged.cramColor; break;
            case Res::CramRotCoeff: blob = staged.cramRotCoeff; break;
            case Res::RotParamBases: blob = staged.rotParamBases; break;
            case Res::ComposeParams: blob = staged.composeParams; break;
            case Res::LnclBack: blob = staged.lnclBack; break;
            case Res::Vram: buffer = pipeline.vdp1 ? m_vdp1Vram : m_vram; break;
            case Res::SpriteFB: buffer = m_fbram; break;
            case Res::Fbram: buffer = m_fbram; break;
            case Res::Vdp1Spans: blob = staged.v1Spans; break;
            case Res::Vdp1PrefixSums: blob = staged.v1PrefixSums; break;
            case Res::Vdp1Commands: blob = staged.v1Commands; break;
            case Res::Vdp1FBWrites: blob = staged.v1FBWrites; break;
            case Res::Vdp1Counter: blob = staged.v1Counter; break;
            case Res::Vdp1SpriteOut: texture = m_spriteOutTex; break;
            case Res::Vdp1SpriteOutAtomic: buffer = m_spriteOutBuf; break;
            case Res::Vdp1ListHeads: texture = m_listHeadsTex; break;
            case Res::Vdp1ListHeadsAtomic: buffer = m_listHeadsBuf; break;
            case Res::Vdp1Fragments: buffer = m_fragments; break;
            case Res::LayerOut: texture = m_layerOut; break;
            case Res::SpriteAttrs: texture = m_spriteAttrs; break;
            case Res::RbgLineColor: texture = m_rbgLineColor; break;
            case Res::ColorCalcWindow: texture = m_colorCalcWindow; break;
            case Res::Composite: texture = m_composite; break;
            case Res::Unknown: break;
            }

            if (blob != nullptr) {
                if (slot.isTexture) {
                    texture = blob->texture;
                } else {
                    buffer = arena;
                    offset = blob->offset;
                }
            }

            if (slot.isTexture) {
                [m_encoder setTexture:texture atIndex:slot.index];
            } else {
                [m_encoder setBuffer:buffer offset:offset atIndex:slot.index];
            }
        }
    }
};

std::unique_ptr<Gpu> Gpu::Create(void *device, std::string &error) {
    if (device == nullptr) {
        error = "No Metal device provided";
        return nullptr;
    }
    auto gpu = std::make_unique<MetalGpu>();
    if (!gpu->Initialize((__bridge id<MTLDevice>)device, error)) {
        return nullptr;
    }
    return gpu;
}

} // namespace ymir::vdp::metal
