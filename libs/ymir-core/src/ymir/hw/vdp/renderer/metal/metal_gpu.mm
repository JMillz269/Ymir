#include "metal_gpu.hpp"

#include "../common_hw/common_hw_defs.hpp"

#include <ymir/util/perf_counters.hpp>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <cstring>
#include <ctime>
#include <pthread.h>
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
    constexpr const char *kHRFBWriteShaderPath = "src/vdp/cs_vdp1_hr_fbram_write.metallib";
    constexpr const char *kHRResyncShaderPath = "src/vdp/cs_vdp1_hr_resync.metallib";
    constexpr const char *kHRDownsampleShaderPath = "src/vdp/cs_vdp1_hr_downsample.metallib";
    constexpr const char *kHRComposeShaderPath = "src/vdp/cs_vdp2_compose_hr.metallib";
    constexpr const char *kHRReplicateShaderPath = "src/vdp/cs_vdp2_hr_replicate.metallib";
    /// Largest internal resolution scale
    constexpr uint32 kMaxHRScale = 4;
    // (the variants are named after the mesh mode and the shading or merging mode: [0] checkerboard meshes,
    // [1] transparent meshes)
    // (the polygon drawing variants used find the span of a pixel through a table, see Vdp1DrawParams)
    /// Pixels drawn by a thread group of the polygon drawing shaders: 64 threads drawing 4 pixels each
    /// (POLYSPEC_PIXELS_PER_THREAD of the variants below)
    constexpr uint32 kPolyDrawPixelsPerGroup = 64 * 4;

    constexpr const char *kPolyDrawShaderPaths[2][4] = {
        {
            "src/vdp/cs_vdp1_polydraw_0_0_st.metallib",
            "src/vdp/cs_vdp1_polydraw_0_1_st.metallib",
            "src/vdp/cs_vdp1_polydraw_0_2_st.metallib",
            "src/vdp/cs_vdp1_polydraw_0_3_st.metallib",
        },
        {
            "src/vdp/cs_vdp1_polydraw_1_0_st.metallib",
            "src/vdp/cs_vdp1_polydraw_1_1_st.metallib",
            "src/vdp/cs_vdp1_polydraw_1_2_st.metallib",
            "src/vdp/cs_vdp1_polydraw_1_3_st.metallib",
        },
    };
    constexpr const char *kMergerShaderPaths[2][3] = {
        {
            "src/vdp/cs_vdp1_output_merger_0_0.metallib",
            "src/vdp/cs_vdp1_output_merger_0_1.metallib",
            "src/vdp/cs_vdp1_output_merger_0_2.metallib",
        },
        {
            "src/vdp/cs_vdp1_output_merger_1_0.metallib",
            "src/vdp/cs_vdp1_output_merger_1_1.metallib",
            "src/vdp/cs_vdp1_output_merger_1_2.metallib",
        },
    };

    /// Texel buffers are emulated by 2D textures this wide (the default width of SPIRV-Cross)
    constexpr NSUInteger kTexelBufferWidth = 4096;

    /// Entries of the buffers with an entry for every pixel of the VDP1 framebuffer (internal sprite output and OIT list
    /// heads): the framebuffer and the transparent mesh one, each with its alternate field at the native size, or the
    /// scaled up framebuffer and mesh one; the area a shader can address in them is a multiple of the width of a texel
    /// buffer
    constexpr NSUInteger kVDP1PixelBufferEntries =
        std::max<NSUInteger>(kVDP1FBRAMSize * 2 * 2, 512 * 256 * kMaxHRScale * kMaxHRScale * 2);

    /// Pixels in the buffers the finished images are copied to: the largest image, scaled up by the highest internal
    /// resolution if that is larger
    constexpr NSUInteger kReadbackPixels =
        std::max<NSUInteger>(kMaxResH * kMaxResV, kMaxNormalResH * kMaxNormalResV * kMaxHRScale * kMaxHRScale);

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
        Vdp1SpanBlockStarts,
        Vdp1Commands,
        Vdp1FBWrites,
        Vdp1SpriteOut,
        Vdp1SpriteOutAtomic,
        Vdp1ListHeads,
        Vdp1ListHeadsAtomic,
        Vdp1Fragments,
        Vdp1Counter,
        Fbram,
        FbramNative,
        FbramHR,
        SpriteFBNative,
        CompositeNative,
        HrStats,
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
            {"g_spanBlockStarts", Res::Vdp1SpanBlockStarts},
            {"g_commandParams", Res::Vdp1Commands},
            {"g_fbramWrites", Res::Vdp1FBWrites},
            {"g_internalSpriteOut", Res::Vdp1SpriteOut},
            {"g_internalSpriteOut_atomic", Res::Vdp1SpriteOutAtomic},
            {"g_listHeads", Res::Vdp1ListHeads},
            {"g_listHeads_atomic", Res::Vdp1ListHeadsAtomic},
            {"g_fragments", Res::Vdp1Fragments},
            {"g_counter", Res::Vdp1Counter},
            {"g_fbramOut", Res::Fbram},
            {"g_fbramNative", Res::FbramNative},
            {"g_fbramHRIn", Res::FbramHR},
            {"g_spriteFBNative", Res::SpriteFBNative},
            {"g_compositeNative", Res::CompositeNative},
            {"g_hrStats", Res::HrStats},
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

/// Number of MetalGpu objects alive, for the log (a renderer is created for every game and settings change)
std::atomic<int> g_liveGpus{0};

class MetalGpu final : public Gpu {
public:
    MetalGpu() {
        ++g_liveGpus;
    }

    ~MetalGpu() override {
        struct LogOnExit {
            id<MTLDevice> device;
            ~LogOnExit() {
                const int live = --g_liveGpus;
                LogMessage("Metal renderer released (%d still alive); Metal memory in use: %.0f MB", live,
                           device != nil ? [device currentAllocatedSize] / 1048576.0 : 0.0);
            }
        } logOnExit{m_device};
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
        if (DiagnosticSwitch("metal-vdp-profile") && [device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
            id<MTLCounterSet> timestamps = nil;
            for (id<MTLCounterSet> set in device.counterSets) {
                if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
                    timestamps = set;
                }
            }
            if (timestamps != nil) {
                MTLCounterSampleBufferDescriptor *descriptor = [[MTLCounterSampleBufferDescriptor alloc] init];
                descriptor.counterSet = timestamps;
                descriptor.storageMode = MTLStorageModeShared;
                descriptor.sampleCount = kNumFrameSlots * kProfileSamplesPerSlot;
                NSError *profileError = nil;
                m_profileBuf = [device newCounterSampleBufferWithDescriptor:descriptor error:&profileError];
                LogMessage("profiling the passes: %s", m_profileBuf != nil ? "on" : "could not create the sample buffer");
            } else {
                LogMessage("profiling the passes: the GPU has no timestamp counters");
            }
        }
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
            // (large enough for the image scaled up by the highest internal resolution too)
            slot.readback = [device newBufferWithLength:kReadbackPixels * sizeof(uint32)
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

        for (uint32 mesh = 0; mesh < 2; ++mesh) {
            for (uint32 i = 0; i < 4; ++i) {
                m_polyDraw[mesh][i].vdp1 = true;
                if (!CreatePipeline(kPolyDrawShaderPaths[mesh][i], m_polyDraw[mesh][i], error)) {
                    return false;
                }
            }
            for (uint32 i = 0; i < 3; ++i) {
                m_merger[mesh][i].vdp1 = true;
                if (!CreatePipeline(kMergerShaderPaths[mesh][i], m_merger[mesh][i], error)) {
                    return false;
                }
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
        // (anywhere in the framebuffer memory: the framebuffers, their alternate fields and the transparent meshes)
        if (size == 0 || m_fbram == nil || offset + size > m_fbram.length) {
            return;
        }
        UploadTo(m_fbram, offset, data, size);
    }

    void Vdp1ClearFBRAM(bool hr, uint32 offset, uint32 size) override {
        id<MTLBuffer> buffer = hr ? m_fbramHR : m_fbram;
        if (buffer == nil || size == 0 || offset + size > buffer.length) {
            return;
        }
        EnsureFrame();
        EndEncoder();
        id<MTLBlitCommandEncoder> blit = [m_cmdBuf blitCommandEncoder];
        [blit fillBuffer:buffer range:NSMakeRange(offset, size) value:0x00];
        [blit endEncoding];
    }

    bool Vdp1EnableHR(std::string &error) override {
        if (m_fbramHR != nil) {
            return true;
        }
        m_hrFbWrite.vdp1 = true;
        m_hrResync.vdp1 = true;
        m_hrDownsample.vdp1 = true;
        if (!CreatePipeline(kHRFBWriteShaderPath, m_hrFbWrite, error) ||
            !CreatePipeline(kHRResyncShaderPath, m_hrResync, error) ||
            !CreatePipeline(kHRDownsampleShaderPath, m_hrDownsample, error) ||
            !CreatePipeline(kHRComposeShaderPath, m_hrCompose, error) ||
            !CreatePipeline(kHRReplicateShaderPath, m_hrReplicate, error)) {
            return false;
        }
        m_hrStatsBuf = [m_device newBufferWithLength:16 options:MTLResourceStorageModeShared];
        if (m_hrStatsBuf != nil) {
            std::memset([m_hrStatsBuf contents], 0, 16);
        }
        m_compositeHR = Create2DTexture(MTLPixelFormatRGBA8Unorm, kMaxNormalResH * kMaxHRScale,
                                        kMaxNormalResV * kMaxHRScale, 1);
        if (m_compositeHR == nil) {
            error = "Could not allocate the GPU textures for the VDP2 internal resolution output";
            return false;
        }
        // (two framebuffers of 512x256 16-bit pixels, scaled up, then their transparent mesh framebuffers)
        const NSUInteger bytes = kVDP1FBRAMSize * 2 * 2 * kMaxHRScale * kMaxHRScale;
        m_fbramHR = [m_device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        if (m_fbramHR == nil) {
            error = "Could not allocate GPU memory for the VDP1 internal resolution framebuffers";
            return false;
        }
        EnsureFrame();
        EndEncoder();
        id<MTLBlitCommandEncoder> blit = [m_cmdBuf blitCommandEncoder];
        [blit fillBuffer:m_fbramHR range:NSMakeRange(0, m_fbramHR.length) value:0x00];
        [blit endEncoding];
        return true;
    }

    void Vdp1ResyncHR(const Vdp1ResyncPassParams &params) override {
        if (m_fbramHR == nil) {
            return;
        }
        EnsureFrame();
        Staged staged{};
        uint64 generation;
        do {
            generation = m_arenaGeneration;
            staged.common = StageRaw(m_blobV1Params, params.constants, params.constantsSize);
        } while (generation != m_arenaGeneration);
        Profile(4);
        EnsureCompute();

        m_bindHR = true;
        [m_encoder setComputePipelineState:m_hrResync.state];
        Bind(m_hrResync, staged);
        m_bindHR = false;
        [m_encoder dispatchThreadgroups:MTLSizeMake((512 * 256 / 2 + 63) / 64, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        Vdp1Barrier();
    }

    void Vdp1DownsampleHR(const Vdp1ResyncPassParams &params) override {
        if (m_fbramHR == nil) {
            return;
        }
        EnsureFrame();
        Staged staged{};
        uint64 generation;
        do {
            generation = m_arenaGeneration;
            staged.common = StageRaw(m_blobV1Params, params.constants, params.constantsSize);
        } while (generation != m_arenaGeneration);
        Profile(4);
        EnsureCompute();

        // (the output is the native framebuffer, the input the scaled up one bound by its own name)
        [m_encoder setComputePipelineState:m_hrDownsample.state];
        Bind(m_hrDownsample, staged);
        [m_encoder dispatchThreadgroups:MTLSizeMake((512 * 256 / 2 + 63) / 64, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        Vdp1Barrier();
    }

    void Vdp1DownloadHRFBRAM(void *dst, uint32 size) override {
        if (m_fbramHR == nil) {
            return;
        }
        size = std::min<uint32>(size, static_cast<uint32>(m_fbramHR.length));
        if (m_hrReadback == nil) {
            // (only made when it's needed, which is for diagnostics)
            m_hrReadback = [m_device newBufferWithLength:m_fbramHR.length options:MTLResourceStorageModeShared];
            if (m_hrReadback == nil) {
                return;
            }
        }
        EnsureFrame();
        EndEncoder();
        id<MTLBlitCommandEncoder> blit = [m_cmdBuf blitCommandEncoder];
        [blit copyFromBuffer:m_fbramHR sourceOffset:0 toBuffer:m_hrReadback destinationOffset:0 size:size];
        [blit endEncoding];
        DrainArena();
        std::memcpy(dst, [m_hrReadback contents], size);
    }

    void Vdp1Draw(const Vdp1DrawParams &params) override {
        if (params.hr && m_fbramHR == nil) {
            return;
        }
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
            staged.v1SpanBlockStarts =
                StageRaw(m_blobSpanBlockStarts, params.spanBlockStarts, params.spanBlockStartsSize,
                         TexelBufferBytes(params.spanBlockStartsSize, MTLPixelFormatR32Uint));
            MakeTexelView(m_blobSpanBlockStarts, params.spanBlockStartsSize, MTLPixelFormatR32Uint);
            staged.v1Commands = StageRaw(m_blobCommands, params.commands, params.commandsSize);
            if (params.shadingMode == 2) {
                // (the counter of fragments starts at zero for every dispatch)
                const uint32 zero = 0;
                staged.v1Counter = StageRaw(m_blobCounter, &zero, sizeof(zero));
            }
        } while (generation != m_arenaGeneration);
        Profile(params.hr ? (params.shadingMode == 2 ? 10 : 2) : 0);
        EnsureCompute();

        // Draw the spans
        const Pipeline &draw = m_polyDraw[params.transparentMesh ? 1 : 0][params.shadingMode];
        [m_encoder setComputePipelineState:draw.state];
        m_bindHR = params.hr;
        Bind(draw, staged);
        m_bindHR = false;
        const NSUInteger groups = (params.numPixels + kPolyDrawPixelsPerGroup - 1) / kPolyDrawPixelsPerGroup;
        [m_encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
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
            Profile(params.hr ? (params.shadingMode == 2 ? 11 : 3) : 1);
            EnsureCompute();
            // (the draw above has been recorded already; if the arena was emptied to make room the encoder is a new one)

            const Pipeline &merge = m_merger[params.transparentMesh ? 1 : 0][params.mergeMode];
            [m_encoder setComputePipelineState:merge.state];
            m_bindHR = params.hr;
            Bind(merge, mergeStaged);
            m_bindHR = false;
            [m_encoder dispatchThreadgroups:MTLSizeMake((params.mergeWidth + 7) / 8, (params.mergeHeight + 7) / 8,
                                                        std::max<uint32>(params.mergeFields, 1))
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
        }

        if (traceThis) {
            uint32 dp;
            std::memcpy(&dp, params.constants, 4);
            DebugLogSamplePixels("after this draw", (dp >> 7) & 1);
        }
    }

    void Vdp1Erase(const Vdp1ErasePassParams &params) override {
        if (params.width == 0 || params.height == 0 || (params.hr && m_fbramHR == nil)) {
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
        Profile(4);
        EnsureCompute();

        [m_encoder setComputePipelineState:m_erase.state];
        m_bindHR = params.hr;
        Bind(m_erase, staged);
        m_bindHR = false;
        // (each thread works on two pixels; the shader doesn't depend on the shape of the groups, so use fewer rows per
        // group if the GPU can't run 32x32 threads of this shader at once)
        const NSUInteger rows = std::max<NSUInteger>(1, std::min<NSUInteger>(32, m_erase.state.maxTotalThreadsPerThreadgroup / 32));
        [m_encoder dispatchThreadgroups:MTLSizeMake((params.width + 63) / 64, (params.height + rows - 1) / rows, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, rows, 1)];
        Vdp1Barrier();
    }

    void Vdp1WriteFB(const Vdp1FBWritePassParams &params) override {
        if (params.count == 0 || (params.hr && m_fbramHR == nil)) {
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
        Profile(4);
        EnsureCompute();

        const Pipeline &write = params.hr ? m_hrFbWrite : m_fbWrite;
        [m_encoder setComputePipelineState:write.state];
        m_bindHR = params.hr;
        Bind(write, staged);
        m_bindHR = false;
        // (the scaled up variant has a thread for each of the two pixels of an entry)
        const uint32 threads = params.hr ? params.count * 2 : params.count;
        [m_encoder dispatchThreadgroups:MTLSizeMake((threads + 63) / 64, 1, 1)
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
        RenderLayersPass(params);
    }

    /// Draws the sprite layer and the background layers
    void RenderLayersPass(const LayerPassParams &params) {
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
        Profile(5);
        EnsureCompute();

        // Sprite layer (and the transparent mesh layer as the second slice)
        [m_encoder setComputePipelineState:m_sprite.state];
        Bind(m_sprite, staged);
        [m_encoder dispatchThreadgroups:MTLSizeMake((params.hres + 31) / 32, params.numLines, params.meshLayer ? 2 : 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        // Background layers
        [m_encoder setComputePipelineState:m_bgs.state];
        Bind(m_bgs, staged);
        [m_encoder dispatchThreadgroups:MTLSizeMake(params.hres / 32, params.numLines, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 7)];
    }

    void Compose(const ComposePassParams &params) override {
        ComposePass(params, false);
    }

    void ComposeHR(const LayerPassParams &layers, const ComposePassParams &compose, uint32 scale,
                   uint32 enhancementsOffset, uint32 startYOffset, uint32 planeStartY,
                   uint32 planeNumLines) override {
        if (m_compositeHR == nil || layers.numLines == 0 || compose.numLines == 0 || scale < 2 || scale > kMaxHRScale ||
            enhancementsOffset + sizeof(uint32) > layers.commonSize ||
            startYOffset + sizeof(uint32) > layers.commonSize) {
            return;
        }
        EnsureFrame();

        std::vector<uint8> common(static_cast<const uint8 *>(layers.common),
                                  static_cast<const uint8 *>(layers.common) + layers.commonSize);
        uint32 enhancements;
        std::memcpy(&enhancements, &common[enhancementsOffset], sizeof(enhancements));
        enhancements &= ~((7u << 2) | (15u << 5));
        enhancements |= (scale - 1) << 2;

        // The lines composed again (planeStartY onwards) are composed in full below; the others are filled with the
        // native image
        std::memcpy(&common[enhancementsOffset], &enhancements, sizeof(enhancements));
        uint32 chunkStartY;
        std::memcpy(&chunkStartY, &common[startYOffset], sizeof(chunkStartY));
        const uint32 chunkEndY = chunkStartY + compose.numLines;
        const uint32 planeEndY = planeStartY + planeNumLines;
        auto fill = [&](uint32 startY, uint32 endY) {
            if (endY <= startY) {
                return;
            }
            std::memcpy(&common[startYOffset], &startY, sizeof(startY));
            Staged staged{};
            uint64 generation;
            do {
                generation = m_arenaGeneration;
                staged.common = Stage(m_blobCommon, common.data(), layers.commonSize);
            } while (generation != m_arenaGeneration);
            Profile(7);
            EnsureCompute();

            m_bindHR = true;
            [m_encoder setComputePipelineState:m_hrReplicate.state];
            Bind(m_hrReplicate, staged);
            m_bindHR = false;
            [m_encoder dispatchThreadgroups:MTLSizeMake((compose.hres * scale + 7) / 8,
                                                        ((endY - startY) * scale + 7) / 8, 1)
                      threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        };
        if (planeNumLines == 0) {
            fill(chunkStartY, chunkEndY);
            return;
        }
        fill(chunkStartY, planeStartY);
        fill(planeEndY, chunkEndY);

        // Compose every native pixel of the lines, drawing the sprite layer of its sub-pixels from the scaled up VDP1
        // framebuffers (one thread each, going through its sub-pixels)
        std::memcpy(&common[startYOffset], &planeStartY, sizeof(planeStartY));
        Staged staged{};
        uint64 generation;
        do {
            generation = m_arenaGeneration;
            staged.layerParams = Stage(m_blobLayerParams, layers.layerParams, layers.layerParamsSize);
            staged.cramColor =
                StageTexels(m_blobCramColor, layers.cramColors, layers.cramColorsSize, MTLPixelFormatRGBA8Uint);
            staged.cramRotCoeff = Stage(m_blobCramRot, layers.cramRotCoeffs, layers.cramRotCoeffsSize);
            staged.rotParamBases = Stage(m_blobRotBases, layers.rotParamBases, layers.rotParamBasesSize);
            staged.composeParams = Stage(m_blobCompose, compose.composeParams, compose.composeParamsSize);
            staged.lnclBack = StageTexels(m_blobLnclBack, compose.lnclBack, compose.lnclBackSize, MTLPixelFormatRGBA8Uint);
            staged.common = Stage(m_blobCommon, common.data(), layers.commonSize);
        } while (generation != m_arenaGeneration);

        Profile(9);
        EnsureCompute();
        m_bindHR = true;
        [m_encoder setComputePipelineState:m_hrCompose.state];
        Bind(m_hrCompose, staged);
        [m_encoder dispatchThreadgroups:MTLSizeMake((compose.hres + 31) / 32, planeNumLines, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        m_bindHR = false;
    }

    void ComposePass(const ComposePassParams &params, bool hr) {
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
        Profile(6);
        EnsureCompute();

        const Pipeline &compose = hr ? m_hrCompose : m_compose;
        [m_encoder setComputePipelineState:compose.state];
        m_bindHR = hr;
        Bind(compose, staged);
        m_bindHR = false;
        [m_encoder dispatchThreadgroups:MTLSizeMake((params.hres + 31) / 32, params.numLines, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    FinishedFrame FinishFrame(uint32 width, uint32 height, uint32 scale) override {
        SampleFrameTimes();
        EnsureFrame();
        FrameSlot &slot = m_slots[m_currentSlot];

        width = std::min<uint32>(width, kMaxResH);
        height = std::min<uint32>(height, kMaxResV);
        if (scale > 1) {
            scale = std::min<uint32>(scale, kMaxHRScale);
            width = std::min<uint32>(width, kMaxNormalResH);
            height = std::min<uint32>(height, kMaxNormalResV);
            if (m_compositeHR == nil) {
                scale = 1;
            }
        } else {
            scale = 1;
        }
        if (m_logFrames > 0 && (m_logFrames + 1) % 300 == 0) {
            const uint64_t micros = m_gpuMicros.exchange(0);
            const uint64_t maxMicros = m_gpuMaxMicros.exchange(0);
            LogMessage("last 300 frames: gpu time average %.2f ms, worst %.2f ms; the emulation waited for the gpu %.2f ms "
                       "per frame (scale %u); Metal memory in use %.0f MB (%d renderers alive)",
                       micros / 300.0 / 1000.0, maxMicros / 1000.0, m_waitMicros / 300.0 / 1000.0, scale,
                       [m_device currentAllocatedSize] / 1048576.0, g_liveGpus.load());
            m_waitMicros = 0;
            LogFrameTimes();
            if (m_hrStatsBuf != nil) {
                uint32 *counts = static_cast<uint32 *>([m_hrStatsBuf contents]);
                const uint32 sampled = counts[0] + counts[1];
                if (sampled > 0) {
                    LogMessage("scaled up image: of the %u sampled pixels (1 in 64) that were looked at in the last 300 "
                               "frames, %.1f%% differ from the native sprite data",
                               sampled, 100.0 * counts[1] / sampled);
                }
                counts[0] = counts[1] = 0;
            }
            if (m_profileBuf != nil) {
                char text[768];
                int n = std::snprintf(text, sizeof(text), "last 300 frames, ms per frame by pass (encoders per frame):");
                for (uint32 kind = 0; kind < kProfileKinds && n > 0 && n < static_cast<int>(sizeof(text)); ++kind) {
                    const uint64_t kindMicros = m_profileMicros[kind].exchange(0);
                    const uint64_t kindCount = m_profileCount[kind].exchange(0);
                    n += std::snprintf(text + n, sizeof(text) - n, " %s %.2f (%.1f);", kProfileNames[kind],
                                       kindMicros / 300.0 / 1000.0, kindCount / 300.0);
                }
                LogMessage("%s", text);
            }
        }
        if (++m_logFrames <= 10 || m_logFrames % 300 == 0) {
            LogMessage("frame %u (%ux%u), draws so far %u (copy %u, shadow %u, half-transparent %u, msb %u), downloads %u",
                       m_logFrames, width, height, m_logDraws, m_logDrawsByMode[0], m_logDrawsByMode[1],
                       m_logDrawsByMode[2], m_logDrawsByMode[3], m_logDownloads);
        }

        // Hand the image to the display on the GPU if it takes it, or copy it out of the GPU
        EndEncoder();
        ymir::vdp::MetalDisplaySink::Target target{};
        const bool toSink = m_displaySink != nullptr && width > 0 && height > 0 &&
                            m_displaySink->AcquireTarget(width, height, scale, target) && target.texture != nullptr;
        if (toSink) {
            const uint32 outWidth = width * scale;
            const uint32 outHeight = height * scale;
            if (target.releaseEvent != nullptr) {
                // (the display may still be reading the image the texture held)
                [m_cmdBuf encodeWaitForEvent:(__bridge id<MTLSharedEvent>)target.releaseEvent value:target.releaseValue];
            }
            id<MTLBlitCommandEncoder> blit = [m_cmdBuf blitCommandEncoder];
            [blit copyFromTexture:(scale > 1 ? m_compositeHR : m_composite)
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(outWidth, outHeight, 1)
                        toTexture:(__bridge id<MTLTexture>)target.texture
                 destinationSlice:0
                 destinationLevel:0
                destinationOrigin:MTLOriginMake(0, 0, 0)];
            [blit endEncoding];
            if (target.readyEvent != nullptr) {
                [m_cmdBuf encodeSignalEvent:(__bridge id<MTLSharedEvent>)target.readyEvent value:target.readyValue];
            }
        } else if (width > 0 && height > 0) {
            id<MTLBlitCommandEncoder> blit = [m_cmdBuf blitCommandEncoder];
            const uint32 outWidth = width * scale;
            const uint32 outHeight = height * scale;
            [blit copyFromTexture:(scale > 1 ? m_compositeHR : m_composite)
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(outWidth, outHeight, 1)
                         toBuffer:slot.readback
                destinationOffset:0
           destinationBytesPerRow:outWidth * sizeof(uint32)
         destinationBytesPerImage:outWidth * outHeight * sizeof(uint32)];
            [blit endEncoding];
        }

        dispatch_semaphore_t done = slot.done;
        std::atomic<uint64_t> *gpuMicros = &m_gpuMicros;
        std::atomic<uint64_t> *gpuMaxMicros = &m_gpuMaxMicros;
        id<MTLCounterSampleBuffer> profileBuf = m_profileBuf;
        std::vector<std::pair<uint32, uint32>> profileEncoders;
        if (profileBuf != nil && !m_profileDrained) {
            profileEncoders = m_profileEncoders;
        }
        m_profileEncoders.clear();
        m_profileDrained = false;
        auto *profileMicros = &m_profileMicros;
        auto *profileCount = &m_profileCount;
        [m_cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            if (profileBuf != nil && !profileEncoders.empty()) {
                const NSUInteger first = profileEncoders.front().second;
                const NSUInteger last = profileEncoders.back().second + 2;
                NSData *data = [profileBuf resolveCounterRange:NSMakeRange(first, last - first)];
                if (data != nil && data.length >= (last - first) * sizeof(MTLCounterResultTimestamp)) {
                    const auto *stamps = static_cast<const MTLCounterResultTimestamp *>(data.bytes);
                    // (the time between the first and the last timestamp is what the whole of the encoders took)
                    uint64_t minStamp = UINT64_MAX, maxStamp = 0;
                    for (const auto &enc : profileEncoders) {
                        const uint64_t a = stamps[enc.second - first].timestamp;
                        const uint64_t b = stamps[enc.second - first + 1].timestamp;
                        if (a != MTLCounterErrorValue && b != MTLCounterErrorValue && b >= a) {
                            minStamp = std::min(minStamp, a);
                            maxStamp = std::max(maxStamp, b);
                        }
                    }
                    const double seconds = buffer.GPUEndTime - buffer.GPUStartTime;
                    if (maxStamp > minStamp && seconds > 0) {
                        const double secondsPerTick = seconds / static_cast<double>(maxStamp - minStamp);
                        for (const auto &enc : profileEncoders) {
                            const uint64_t a = stamps[enc.second - first].timestamp;
                            const uint64_t b = stamps[enc.second - first + 1].timestamp;
                            if (a != MTLCounterErrorValue && b != MTLCounterErrorValue && b >= a) {
                                (*profileMicros)[enc.first].fetch_add(
                                    static_cast<uint64_t>((b - a) * secondsPerTick * 1e6), std::memory_order_relaxed);
                                (*profileCount)[enc.first].fetch_add(1, std::memory_order_relaxed);
                            }
                        }
                    }
                }
            }
            // (how long the GPU spent on the frame)
            const double seconds = buffer.GPUEndTime - buffer.GPUStartTime;
            if (seconds > 0) {
                const uint64_t micros = static_cast<uint64_t>(seconds * 1e6);
                gpuMicros->fetch_add(micros, std::memory_order_relaxed);
                uint64_t seen = gpuMaxMicros->load(std::memory_order_relaxed);
                while (micros > seen && !gpuMaxMicros->compare_exchange_weak(seen, micros)) {
                }
            }
            dispatch_semaphore_signal(done);
        }];
        [m_cmdBuf commit];
        slot.lastBuf = m_cmdBuf;
        slot.width = width;
        slot.height = height;
        slot.scale = scale;
        m_cmdBuf = nil;

        FinishedFrame result{};
        if (toSink) {
            // (the display waits for the copy on the GPU; nothing to wait for here)
            m_displaySink->TargetSubmitted(target.readyValue);
            result.displayed = true;
            result.width = width;
            result.height = height;
            result.scale = scale;
            m_hasPrevious = false;
            m_currentSlot = (m_currentSlot + 1) % kNumFrameSlots;
            m_frameStarted = false;
            return result;
        }

        // The previous frame has had a whole frame's time to finish
        if (m_hasPrevious) {
            FrameSlot &prev = m_slots[m_previousSlot];
            const auto waitStart = std::chrono::steady_clock::now();
            [prev.lastBuf waitUntilCompleted];
            m_waitMicros += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - waitStart)
                                .count();
            result.pixels = static_cast<const uint32 *>([prev.readback contents]);
            result.width = prev.width;
            result.height = prev.height;
            result.scale = prev.scale;
        }
        m_previousSlot = m_currentSlot;
        m_hasPrevious = true;
        m_currentSlot = (m_currentSlot + 1) % kNumFrameSlots;
        m_frameStarted = false;
        return result;
    }

    void SetDisplaySink(std::shared_ptr<ymir::vdp::MetalDisplaySink> sink) override {
        m_displaySink = std::move(sink);
        // (a frame read back before has nothing to do with the next one read back)
        m_hasPrevious = false;
        LogMessage("finished images %s", m_displaySink ? "go straight to the display" : "are read back to the CPU");
    }

private:
    std::shared_ptr<ymir::vdp::MetalDisplaySink> m_displaySink;

    struct FrameSlot {
        id<MTLBuffer> arena = nil;
        id<MTLBuffer> readback = nil;
        dispatch_semaphore_t done = nil;
        id<MTLCommandBuffer> lastBuf = nil;
        uint32 width = 0;
        uint32 height = 0;
        uint32 scale = 1;
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
        const Blob *v1SpanBlockStarts = nullptr;
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

    // Frame times (for the log): the time between frames, the CPU time of the thread that finishes the frames (the
    // emulation thread) and of the whole process, summed since the last report, and the worst thermal state seen
    struct FrameTimes {
        uint32 frames = 0;
        double intervalMs = 0.0, worstIntervalMs = 0.0;
        uint32 slowFrames = 0; ///< frames that took longer than a 60 Hz frame (with some slack)
        double threadCpuMs = 0.0, worstThreadCpuMs = 0.0;
        double processCpuMs = 0.0;
        int worstThermal = 0;
    } m_frameTimes;
    bool m_frameTimesStarted = false;
    pthread_t m_frameThread{};
    std::chrono::steady_clock::time_point m_lastFrameTime{};
    double m_lastThreadCpuMs = 0.0;
    double m_lastProcessCpuMs = 0.0;
    // The frontend's wait counters when the frame times were last logged (they only ever grow)
    uint64_t m_lastEmuWaitNs = 0;
    uint64_t m_lastGuiWaitNs = 0;
    uint64_t m_lastSH2Ns = 0;
    std::array<std::array<Pipeline, 4>, 2> m_polyDraw;
    std::array<std::array<Pipeline, 3>, 2> m_merger;
    Pipeline m_erase, m_fbWrite, m_hrFbWrite, m_hrResync, m_hrDownsample, m_hrCompose, m_hrReplicate;
    id<MTLTexture> m_compositeHR = nil; ///< scaled up output image
    id<MTLBuffer> m_fbramHR = nil; ///< scaled up VDP1 framebuffers (see Vdp1EnableHR)
    id<MTLBuffer> m_hrReadback = nil;
    id<MTLBuffer> m_hrStatsBuf = nil; ///< two counters the composition of the scaled up image adds to (see its shader)
    // Profiling (a folder named metal-vdp-profile in the Documents folder turns it on): the time the GPU spends on each
    // kind of pass, from timestamps taken at the start and the end of the compute encoders
    static constexpr uint32 kProfileKinds = 12;
    static constexpr uint32 kProfileSamplesPerSlot = 256; // two for each encoder
    inline static const char *const kProfileNames[kProfileKinds] = {
        "vdp1 draw",       "vdp1 merge",     "vdp1 draw (scaled up)", "vdp1 merge (scaled up)", "vdp1 erase and writes",
        "vdp2 layers",     "vdp2 compose",   "scaled up fill",        "(unused)",
        "scaled up sprites and compose",
        "vdp1 half-transparent draw (scaled up)", "vdp1 half-transparent merge (scaled up)",
    };
    id<MTLCounterSampleBuffer> m_profileBuf = nil;
    uint32 m_profileKind = 0;
    std::vector<std::pair<uint32, uint32>> m_profileEncoders; // kind and first sample index (within the buffer)
    bool m_profileDrained = false;
    std::array<std::atomic<uint64_t>, kProfileKinds> m_profileMicros{};
    std::array<std::atomic<uint64_t>, kProfileKinds> m_profileCount{};
    int64_t m_waitMicros = 0;
    std::atomic<uint64_t> m_gpuMicros{0};
    std::atomic<uint64_t> m_gpuMaxMicros{0};
    bool m_bindHR = false;         ///< whether VDP1 passes being recorded work on the scaled up framebuffers
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
    Blob m_blobV1Params, m_blobSpans, m_blobPrefixSums, m_blobSpanBlockStarts, m_blobCommands, m_blobFBWrites,
        m_blobCounter, m_blobMergeCommon;

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
    // Frame time statistics (for the log)

    static double CpuClockMs(clockid_t clock) {
        timespec ts{};
        if (clock_gettime(clock, &ts) != 0) {
            return 0.0;
        }
        return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
    }

    /// Notes the times of the frame that is being finished (called at the start of every FinishFrame)
    void SampleFrameTimes() {
        const auto now = std::chrono::steady_clock::now();
        const double threadCpu = CpuClockMs(CLOCK_THREAD_CPUTIME_ID);
        const double processCpu = CpuClockMs(CLOCK_PROCESS_CPUTIME_ID);
        const pthread_t self = pthread_self();
        // (the thread CPU time only adds up while the same thread finishes the frames)
        if (m_frameTimesStarted && pthread_equal(self, m_frameThread)) {
            FrameTimes &t = m_frameTimes;
            const double interval = std::chrono::duration<double, std::milli>(now - m_lastFrameTime).count();
            const double threadCpuDelta = threadCpu - m_lastThreadCpuMs;
            ++t.frames;
            t.intervalMs += interval;
            t.worstIntervalMs = std::max(t.worstIntervalMs, interval);
            if (interval > 1000.0 / 60.0 + 2.0) {
                ++t.slowFrames;
            }
            t.threadCpuMs += threadCpuDelta;
            t.worstThreadCpuMs = std::max(t.worstThreadCpuMs, threadCpuDelta);
            t.processCpuMs += processCpu - m_lastProcessCpuMs;
            if (t.frames % 30 == 1) {
                t.worstThermal = std::max(t.worstThermal, static_cast<int>([[NSProcessInfo processInfo] thermalState]));
            }
        }
        m_frameTimesStarted = true;
        m_frameThread = self;
        m_lastFrameTime = now;
        m_lastThreadCpuMs = threadCpu;
        m_lastProcessCpuMs = processCpu;
    }

    /// Logs the frame times since the last report and starts over
    void LogFrameTimes() {
        FrameTimes &t = m_frameTimes;
        if (t.frames == 0) {
            return;
        }
        NSProcessInfo *info = [NSProcessInfo processInfo];
        static const char *const kThermalNames[] = {"nominal", "fair", "serious", "critical"};
        const int thermal = std::clamp(std::max(t.worstThermal, static_cast<int>(info.thermalState)), 0, 3);
        const double n = t.frames;
        // How long the emulator waited for the display to take the next frame, and the other way around (video sync),
        // and the time spent in the SH-2s and the SCU
        const uint64_t emuWait = ymir::perf::emuWaitNs.load(std::memory_order_relaxed);
        const uint64_t guiWait = ymir::perf::guiWaitNs.load(std::memory_order_relaxed);
        const uint64_t sh2 = ymir::perf::sh2Ns.load(std::memory_order_relaxed);
        LogMessage("last %u frames, per frame: emulator waited for the display %.2f ms, display waited for the emulator "
                   "%.2f ms, SH-2s and SCU %.2f ms",
                   t.frames, (emuWait - m_lastEmuWaitNs) / n / 1.0e6, (guiWait - m_lastGuiWaitNs) / n / 1.0e6,
                   (sh2 - m_lastSH2Ns) / n / 1.0e6);
        m_lastEmuWaitNs = emuWait;
        m_lastGuiWaitNs = guiWait;
        m_lastSH2Ns = sh2;
        LogMessage("last %u frames: %.1f fps (frame interval average %.2f ms, worst %.2f ms, %u frames slower than "
                   "60 Hz); emulation thread CPU time per frame average %.2f ms, worst %.2f ms; whole app CPU time per "
                   "frame %.2f ms; thermal state %s%s",
                   t.frames, 1000.0 * n / t.intervalMs, t.intervalMs / n, t.worstIntervalMs, t.slowFrames,
                   t.threadCpuMs / n, t.worstThreadCpuMs, t.processCpuMs / n, kThermalNames[thermal],
                   info.lowPowerModeEnabled ? ", Low Power Mode on" : "");
        t = {};
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

    /// Notes the kind of the passes about to be recorded; when profiling each kind gets encoders of its own
    void Profile(uint32 kind) {
        if (m_profileBuf != nil && kind != m_profileKind) {
            EndEncoder();
        }
        m_profileKind = kind;
    }

    void EnsureCompute() {
        if (m_encoder == nil) {
            if (m_profileBuf != nil && m_profileEncoders.size() < kProfileSamplesPerSlot / 2) {
                const uint32 index = m_currentSlot * kProfileSamplesPerSlot +
                                     static_cast<uint32>(m_profileEncoders.size()) * 2;
                MTLComputePassDescriptor *descriptor = [MTLComputePassDescriptor computePassDescriptor];
                descriptor.sampleBufferAttachments[0].sampleBuffer = m_profileBuf;
                descriptor.sampleBufferAttachments[0].startOfEncoderSampleIndex = index;
                descriptor.sampleBufferAttachments[0].endOfEncoderSampleIndex = index + 1;
                m_encoder = [m_cmdBuf computeCommandEncoderWithDescriptor:descriptor];
                m_profileEncoders.emplace_back(m_profileKind, index);
            } else {
                m_encoder = [m_cmdBuf computeCommandEncoder];
            }
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
        m_profileDrained = true;
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
            case Res::SpriteFB: buffer = m_bindHR ? m_fbramHR : m_fbram; break;
            case Res::Fbram: buffer = (m_bindHR && pipeline.vdp1) ? m_fbramHR : m_fbram; break;
            case Res::FbramNative: buffer = m_fbram; break;
            case Res::FbramHR: buffer = m_fbramHR; break;
            case Res::SpriteFBNative: buffer = m_fbram; break;
            case Res::CompositeNative: texture = m_composite; break;
            case Res::HrStats: buffer = m_hrStatsBuf; break;
            case Res::Vdp1Spans: blob = staged.v1Spans; break;
            case Res::Vdp1PrefixSums: blob = staged.v1PrefixSums; break;
            case Res::Vdp1SpanBlockStarts: blob = staged.v1SpanBlockStarts; break;
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
            case Res::Composite: texture = m_bindHR ? m_compositeHR : m_composite; break;
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
    LogMessage("Metal renderer created (%d alive); Metal memory in use: %.0f MB", g_liveGpus.load(),
               [(__bridge id<MTLDevice>)device currentAllocatedSize] / 1048576.0);
    return gpu;
}

} // namespace ymir::vdp::metal
