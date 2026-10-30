#include "gfx_context_impl_metal.hpp"
#include "gfx_context_spec_metal.hpp"

#include "gfx_texture_id_manager.hpp"

#include <ymir/gpu/shaders/gpu_shaders.hpp>

#include <ymir/util/string.hpp>

#include <ymir/hw/vdp/renderer/vdp_renderer_hw_metal.hpp>

#include <backends/imgui_impl_metal.h>
#include <backends/imgui_impl_sdl3.h>

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <TargetConditionals.h>

// MetalFX upscaling is only used on iOS
#if TARGET_OS_IOS && __has_include(<MetalFX/MetalFX.h>)
    #import <MetalFX/MetalFX.h>
    #define YMIR_HAS_METALFX 1
#else
    #define YMIR_HAS_METALFX 0
#endif

#include <SDL3/SDL.h>
#include <fmt/format.h>

#include <cmrc/cmrc.hpp>
CMRC_DECLARE(ymir_sdl3_shaders);

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <cmath>
#include <deque>
#include <mutex>

using namespace ymir::gpu;

namespace app::gfx {

static MTLPixelFormat ToMetalValue(PixelFormat format) {
    switch (format) {
    case PixelFormat::Unknown: return MTLPixelFormatInvalid;
    case PixelFormat::R8G8B8A8_UNORM: return MTLPixelFormatRGBA8Unorm;
    case PixelFormat::R8G8B8X8_UNORM: return MTLPixelFormatRGBA8Unorm; // Note: A instead of X
    case PixelFormat::B8G8R8A8_UNORM: return MTLPixelFormatBGRA8Unorm;
    case PixelFormat::B8G8R8X8_UNORM: return MTLPixelFormatBGRA8Unorm; // Note: A instead of X
    }
    return MTLPixelFormatInvalid;
}

static const char *MetalPixelFormatName(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatInvalid: return "INVALID";
    case MTLPixelFormatRGBA8Unorm: return "RGBA8Unorm";
    case MTLPixelFormatBGRA8Unorm: return "BGRA8Unorm";
    default: return "<unhandled>";
    }
}

struct alignas(16) Float4 {
    float x, y, z, w;
};

struct Float3 {
    float x, y, z;
};

struct Float2 {
    float x, y;
};

struct Vertex {
    Float3 position;
    Float2 uv;
};

// Constants of the post-processing shader below. Must match the layout of Params in the shader source.
struct alignas(16) PostConstants {
    Float4 srcRect; // normalized
    Float4 dstRect; // pixels
    Float2 renderTargetSize;
    Float2 contentSize; // size of the source region, in texels
    float sharpness;
    float scanlines;
    float mask;
    float sharpK; // > 0 enables sharp pixel scaling; the number of destination pixels per source pixel
};

static const char *kPostProcessShaderSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct Params {
    float4 srcRect;
    float4 dstRect;
    float2 rtSize;
    float2 contentSize;
    float sharpness;
    float scanlines;
    float mask;
    float sharpK;
};

struct VOut {
    float4 position [[position]];
    float2 uv;
};

vertex VOut ymir_post_vs(uint vid [[vertex_id]], constant Params &p [[buffer(0)]]) {
    float2 corner = float2(float(vid & 1u), float(vid >> 1));
    float2 px = p.dstRect.xy + corner * p.dstRect.zw;
    VOut o;
    o.position = float4(px.x / p.rtSize.x * 2.0 - 1.0, 1.0 - px.y / p.rtSize.y * 2.0, 0.0, 1.0);
    o.uv = p.srcRect.xy + corner * p.srcRect.zw;
    return o;
}

fragment float4 ymir_post_fs(VOut in [[stage_in]], texture2d<float> tex [[texture(0)]], sampler smp [[sampler(0)]],
                             constant Params &p [[buffer(0)]]) {
    float2 uv = in.uv;
    float2 srcPos = (uv - p.srcRect.xy) / p.srcRect.zw; // 0..1 across the source region

    if (p.sharpK > 0.0) {
        // Keep the middle of every source pixel flat and only blend across the border to the next one
        float2 pos = srcPos * p.contentSize;
        float2 base = floor(pos);
        float2 f = clamp((fract(pos) - 0.5) * p.sharpK + 0.5, 0.0, 1.0);
        uv = p.srcRect.xy + ((base + f) / p.contentSize) * p.srcRect.zw;
    }

    float3 c = tex.sample(smp, uv).rgb;

    if (p.sharpness > 0.0) {
        float2 texel = 1.0 / float2(tex.get_width(), tex.get_height());
        float3 blur = tex.sample(smp, uv + float2(texel.x, 0.0)).rgb + tex.sample(smp, uv - float2(texel.x, 0.0)).rgb +
                      tex.sample(smp, uv + float2(0.0, texel.y)).rgb + tex.sample(smp, uv - float2(0.0, texel.y)).rgb;
        blur *= 0.25;
        c = clamp(c + (c - blur) * p.sharpness * 2.0, 0.0, 1.0);
    }

    if (p.scanlines > 0.0) {
        // Bright in the middle of each source row, darker toward the gap between rows
        float y = fract(srcPos.y * p.contentSize.y);
        float beam = 1.0 - pow(abs(y * 2.0 - 1.0), 3.0);
        float w = mix(1.0, 0.25 + 0.75 * beam, p.scanlines);
        c *= w * mix(1.0, 1.2, p.scanlines); // brighten a little to make up for the dark gaps
    }

    if (p.mask > 0.0) {
        // Aperture grille: red, green and blue stripes, three pixels per triad
        uint col = uint(in.position.x) % 3u;
        const float dim = 0.55;
        float3 stripe = float3(col == 0u ? 1.0 : dim, col == 1u ? 1.0 : dim, col == 2u ? 1.0 : dim);
        stripe /= (1.0 + 2.0 * dim) / 3.0; // keep the overall brightness
        c = mix(c, c * stripe, p.mask);
    }

    return float4(clamp(c, 0.0, 1.0), 1.0);
}
)MSL";

struct alignas(16) DrawTextureConstants {
    Float4 srcRect;
    Float4 dstRect;
    Float2 renderTargetSize;
    Float2 rotPivot;
    float rotAngle;
    float _pad[3];
};

static MTLVertexDescriptor *CreateQuadVertexDescriptor() {
    MTLVertexDescriptor *vtxDesc = [MTLVertexDescriptor vertexDescriptor];
    vtxDesc.attributes[0].format = MTLVertexFormatFloat3;
    vtxDesc.attributes[0].offset = offsetof(Vertex, position);
    vtxDesc.attributes[0].bufferIndex = 17;

    vtxDesc.attributes[1].format = MTLVertexFormatFloat2;
    vtxDesc.attributes[1].offset = offsetof(Vertex, uv);
    vtxDesc.attributes[1].bufferIndex = 17;

    vtxDesc.layouts[17].stride = sizeof(Vertex);
    vtxDesc.layouts[17].stepFunction = MTLVertexStepFunctionPerVertex;
    return vtxDesc;
}

// -----------------------------------------------------------------------------
// Images handed over by the Metal VDP renderer on the GPU

/// Three textures the Metal VDP renderer copies its finished images into in turn (see ymir::vdp::MetalDisplaySink),
/// which the display draws from without the images going through the CPU. Two shared events order the two sides on the
/// GPU: the renderer signals the number of each image once it's copied, and the display signals the number of the image
/// it drew once it's done with it; the renderer only waits for that before writing over an image the display drew.
class MetalDisplayFrames final : public ymir::vdp::MetalDisplaySink {
public:
    static constexpr uint32 kCount = 3;
    // Large enough for the largest image: 704x512, or 352x256 at the highest internal resolution (4x)
    static constexpr uint32 kWidth = 352 * 4;
    static constexpr uint32 kHeight = 256 * 4;

    explicit MetalDisplayFrames(id<MTLDevice> device)
        : m_device(device) {
        m_readyEvent = [device newSharedEvent];
        m_releaseEvent = [device newSharedEvent];
    }

    /// An image the display can draw
    struct Frame {
        id<MTLTexture> texture = nil;
        uint32 slot = 0;
        uint64 number = 0;
        uint32 width = 0;
        uint32 height = 0;
        uint32 scale = 1;
    };

    bool AcquireTarget(uint32 width, uint32 height, uint32 scale, Target &target) override {
        if (m_readyEvent == nil || m_releaseEvent == nil || width * scale > kWidth || height * scale > kHeight) {
            return false;
        }
        std::lock_guard lock{m_mutex};
        const uint64 number = m_nextNumber;
        Slot &slot = m_slots[number % kCount];
        if (slot.texture == nil) {
            MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                            width:kWidth
                                                                                           height:kHeight
                                                                                        mipmapped:NO];
            desc.usage = MTLTextureUsageShaderRead;
            desc.storageMode = MTLStorageModePrivate;
            slot.texture = [m_device newTextureWithDescriptor:desc];
            if (slot.texture == nil) {
                return false;
            }
            slot.texture.label = @"[Ymir] Metal VDP renderer image";
        }
        ++m_nextNumber;

        target = {};
        if (slot.drawn) {
            // The display drew the image this texture holds: wait until it's done with it
            target.releaseEvent = (__bridge void *)m_releaseEvent;
            target.releaseValue = slot.number;
        }
        slot.number = number;
        slot.width = width;
        slot.height = height;
        slot.scale = scale;
        slot.drawn = false;
        target.texture = (__bridge void *)slot.texture;
        target.readyEvent = (__bridge void *)m_readyEvent;
        target.readyValue = number;
        return true;
    }

    void TargetSubmitted(uint64 readyValue) override {
        std::lock_guard lock{m_mutex};
        m_submitted = std::max(m_submitted, readyValue);
    }

    /// Gets the latest image handed over, if any, and notes that the display draws it. The caller waits for the ready
    /// event to reach its number before reading it and signals the release event with its number when done.
    bool AcquireLatest(Frame &frame) {
        std::lock_guard lock{m_mutex};
        if (m_submitted == 0) {
            return false;
        }
        const uint32 index = static_cast<uint32>(m_submitted % kCount);
        Slot &slot = m_slots[index];
        if (slot.number != m_submitted || slot.texture == nil) {
            return false;
        }
        slot.drawn = true;
        frame = {slot.texture, index, slot.number, slot.width, slot.height, slot.scale};
        return true;
    }

    /// Stops offering the images handed over so far (when the renderer is replaced)
    void Reset() {
        std::lock_guard lock{m_mutex};
        m_submitted = 0;
    }

    id<MTLSharedEvent> ReadyEvent() const {
        return m_readyEvent;
    }

    id<MTLSharedEvent> ReleaseEvent() const {
        return m_releaseEvent;
    }

private:
    struct Slot {
        id<MTLTexture> texture = nil;
        uint64 number = 0;
        uint32 width = 0;
        uint32 height = 0;
        uint32 scale = 1;
        bool drawn = false; // the display drew this image (and signals the release event with its number when done)
    };

    id<MTLDevice> m_device = nil;
    id<MTLSharedEvent> m_readyEvent = nil;
    id<MTLSharedEvent> m_releaseEvent = nil;
    std::mutex m_mutex;
    std::array<Slot, kCount> m_slots{};
    uint64 m_nextNumber = 1;
    uint64 m_submitted = 0; // the latest image whose copy has been committed
};

// -----------------------------------------------------------------------------

struct MetalGraphicsContext::Impl {
    explicit Impl(const MetalGraphicsContextSpec &spec)
        : spec(spec) {}

    static constexpr size_t kFrameCount = 3;

    MetalGraphicsContextSpec spec;

    SDL_MetalView metalView = nullptr;
    CAMetalLayer *metalLayer = nil;

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> cmdQueue = nil;

    dispatch_semaphore_t frameSemaphore = nullptr;
    uint64 currentFrameNumber = 0;
    size_t frameIndex = 0;

    VertexShader vertexShader;
    PixelShader pixelShader;
    id<MTLFunction> vsFunc = nil;
    id<MTLFunction> psFunc = nil;

    struct RenderTargetPipeline {
        id<MTLRenderPipelineState> pipelineState = nil;
    };
    std::unordered_map<MTLPixelFormat, RenderTargetPipeline> renderTargetPipelines;

    id<MTLRenderPipelineState> pipelineStateFrame = nil;

    id<MTLBuffer> vertexBuffer = nil;
    id<MTLSamplerState> smpNearest = nil;
    id<MTLSamplerState> smpLinear = nil;

    DrawTextureConstants drawTextureConstants;

    // Images of the Metal VDP renderer, handed over on the GPU, with the IDs of their textures and the number of the
    // one acquired for the frame being drawn (0 if none)
    std::shared_ptr<MetalDisplayFrames> displayFrames;
    std::array<TextureID, MetalDisplayFrames::kCount> displayFrameIDs{
        kInvalidTextureID, kInvalidTextureID, kInvalidTextureID};
    uint64 displayFrameDrawn = 0;

    id<CAMetalDrawable> currentDrawable = nil;
    id<MTLCommandBuffer> cmdListFrame = nil;
    id<MTLRenderCommandEncoder> renderEncoderFrame = nil;
    MTLRenderPassDescriptor *mainRenderPassDesc = nil;

    MTLClearColor clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
    bool shouldClear = true;

    uint32 framebufferWidth = 0;
    uint32 framebufferHeight = 0;
    MTLViewport viewport{};
    MTLScissorRect scissorRect{};

    PresentMode presentMode = PresentMode::VSync;
    double frameRateHint = 0.0; // frames per second the app presents; 0 = unrestricted

    // -- Post processing filters --

    PostProcessParams postProcess;
    id<MTLLibrary> postLibrary = nil;
    bool postLibraryFailed = false;
    std::unordered_map<MTLPixelFormat, id<MTLRenderPipelineState>> postPipelines;

    id<MTLRenderPipelineState> GetPostPipeline(MTLPixelFormat format) {
        auto it = postPipelines.find(format);
        if (it != postPipelines.end()) {
            return it->second;
        }
        if (postLibraryFailed) {
            return nil;
        }
        if (postLibrary == nil) {
            NSError *error = nil;
            postLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:kPostProcessShaderSource]
                                               options:nil
                                                 error:&error];
            if (postLibrary == nil) {
                postLibraryFailed = true;
                return nil;
            }
        }
        MTLRenderPipelineDescriptor *desc = [MTLRenderPipelineDescriptor new];
        desc.label = @"[Ymir-GCtx] Post-process pipeline";
        desc.vertexFunction = [postLibrary newFunctionWithName:@"ymir_post_vs"];
        desc.fragmentFunction = [postLibrary newFunctionWithName:@"ymir_post_fs"];
        desc.colorAttachments[0].pixelFormat = format;
        NSError *error = nil;
        id<MTLRenderPipelineState> pso = [device newRenderPipelineStateWithDescriptor:desc error:&error];
        if (pso == nil) {
            postLibraryFailed = true;
            return nil;
        }
        postPipelines[format] = pso;
        return pso;
    }

    // -- MetalFX upscaling --

    UpscaleMode upscaleMode = UpscaleMode::Off;

    id<MTLTexture> fxInput = nil;  // copy of the emulated frame, exactly as large as the frame
    id<MTLTexture> fxOutput = nil; // upscaled frame
    id<MTLTexture> fxDepth = nil;  // temporal scaler only: constant depth
    id<MTLTexture> fxMotion = nil; // temporal scaler only: no motion
    id fxScaler = nil;             // id<MTLFXSpatialScaler> or id<MTLFXTemporalScaler>
    UpscaleMode fxScalerMode = UpscaleMode::Off;
    MTLPixelFormat fxFormat = MTLPixelFormatInvalid;
    uint32 fxInWidth = 0, fxInHeight = 0, fxOutWidth = 0, fxOutHeight = 0;
    bool fxNeedsClear = false;
    bool fxResetHistory = true;

    bool SupportsUpscaling() const {
#if YMIR_HAS_METALFX
        if (@available(iOS 16.0, *)) {
            return [MTLFXSpatialScalerDescriptor supportsDevice:device] &&
                   [MTLFXTemporalScalerDescriptor supportsDevice:device];
        }
#endif
        return false;
    }

    void ReleaseUpscaler() {
        fxInput = nil;
        fxOutput = nil;
        fxDepth = nil;
        fxMotion = nil;
        fxScaler = nil;
        fxScalerMode = UpscaleMode::Off;
        fxFormat = MTLPixelFormatInvalid;
        fxInWidth = fxInHeight = fxOutWidth = fxOutHeight = 0;
    }

#if YMIR_HAS_METALFX
    id<MTLTexture> NewPrivateTexture(MTLPixelFormat format, uint32 width, uint32 height, MTLTextureUsage usage,
                                     NSString *label) API_AVAILABLE(ios(16.0)) {
        MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                        width:width
                                                                                       height:height
                                                                                    mipmapped:NO];
        desc.usage = usage;
        desc.storageMode = MTLStorageModePrivate;
        id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
        texture.label = label;
        return texture;
    }

    /// Makes sure that the scaler and its textures exist for the given sizes and the current upscale mode.
    bool EnsureUpscaler(MTLPixelFormat format, uint32 inW, uint32 inH, uint32 outW, uint32 outH)
        API_AVAILABLE(ios(16.0)) {
        if (fxScaler != nil && fxScalerMode == upscaleMode && fxFormat == format && fxInWidth == inW &&
            fxInHeight == inH && fxOutWidth == outW && fxOutHeight == outH) {
            return true;
        }
        ReleaseUpscaler();

        fxInput = NewPrivateTexture(format, inW, inH, MTLTextureUsageShaderRead, @"[Ymir-GCtx] MetalFX input");
        fxOutput = NewPrivateTexture(format, outW, outH,
                                     MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                                     @"[Ymir-GCtx] MetalFX output");
        if (fxInput == nil || fxOutput == nil) {
            ReleaseUpscaler();
            return false;
        }

        if (upscaleMode == UpscaleMode::Temporal) {
            MTLFXTemporalScalerDescriptor *desc = [MTLFXTemporalScalerDescriptor new];
            desc.inputWidth = inW;
            desc.inputHeight = inH;
            desc.outputWidth = outW;
            desc.outputHeight = outH;
            desc.colorTextureFormat = format;
            desc.depthTextureFormat = MTLPixelFormatDepth32Float;
            desc.motionTextureFormat = MTLPixelFormatRG16Float;
            desc.outputTextureFormat = format;
            desc.autoExposureEnabled = NO;
            fxScaler = [desc newTemporalScalerWithDevice:device];
            if (fxScaler != nil) {
                fxDepth = NewPrivateTexture(MTLPixelFormatDepth32Float, inW, inH,
                                            MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead,
                                            @"[Ymir-GCtx] MetalFX depth");
                fxMotion = NewPrivateTexture(MTLPixelFormatRG16Float, inW, inH,
                                             MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead,
                                             @"[Ymir-GCtx] MetalFX motion");
                if (fxDepth == nil || fxMotion == nil) {
                    fxScaler = nil;
                } else {
                    fxScalerMode = UpscaleMode::Temporal;
                    fxNeedsClear = true;
                    fxResetHistory = true;
                }
            }
        }
        if (fxScaler == nil) {
            // Spatial mode, or a fallback if the temporal scaler could not be created
            MTLFXSpatialScalerDescriptor *desc = [MTLFXSpatialScalerDescriptor new];
            desc.inputWidth = inW;
            desc.inputHeight = inH;
            desc.outputWidth = outW;
            desc.outputHeight = outH;
            desc.colorTextureFormat = format;
            desc.outputTextureFormat = format;
            desc.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;
            fxScaler = [desc newSpatialScalerWithDevice:device];
            fxScalerMode = UpscaleMode::Spatial;
        }
        if (fxScaler == nil) {
            ReleaseUpscaler();
            return false;
        }

        fxFormat = format;
        fxInWidth = inW;
        fxInHeight = inH;
        fxOutWidth = outW;
        fxOutHeight = outH;
        return true;
    }
#endif

    /// Encodes the upscaling of the given region of a texture into the command buffer.
    /// Returns the texture with the upscaled image, or nil if it can't be upscaled.
    id<MTLTexture> EncodeUpscale(id<MTLCommandBuffer> cmdBuf, id<MTLTexture> src, const FRect &srcRect,
                                 uint32 factor) {
#if YMIR_HAS_METALFX
        if (@available(iOS 16.0, *)) {
            const uint32 inW = (uint32)srcRect.w;
            const uint32 inH = (uint32)srcRect.h;
            if (inW == 0 || inH == 0 || srcRect.x < 0 || srcRect.y < 0 || srcRect.x + inW > src.width ||
                srcRect.y + inH > src.height) {
                return nil;
            }
            if (!EnsureUpscaler(src.pixelFormat, inW, inH, inW * factor, inH * factor)) {
                // Unsupported on this device or for this texture format: stop trying
                upscaleMode = UpscaleMode::Off;
                return nil;
            }

            // Copy the frame so that the scaler gets a texture of exactly the frame's size
            id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
            [blit copyFromTexture:src
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake((NSUInteger)srcRect.x, (NSUInteger)srcRect.y, 0)
                       sourceSize:MTLSizeMake(inW, inH, 1)
                        toTexture:fxInput
                 destinationSlice:0
                 destinationLevel:0
                destinationOrigin:MTLOriginMake(0, 0, 0)];
            [blit endEncoding];

            if (fxScalerMode == UpscaleMode::Temporal) {
                if (fxNeedsClear) {
                    // The Saturn's video has no depth or motion information: use constant depth and zero motion
                    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
                    pass.colorAttachments[0].texture = fxMotion;
                    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                    pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
                    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                    pass.depthAttachment.texture = fxDepth;
                    pass.depthAttachment.loadAction = MTLLoadActionClear;
                    pass.depthAttachment.clearDepth = 1.0;
                    pass.depthAttachment.storeAction = MTLStoreActionStore;
                    [[cmdBuf renderCommandEncoderWithDescriptor:pass] endEncoding];
                    fxNeedsClear = false;
                }

                id<MTLFXTemporalScaler> scaler = (id<MTLFXTemporalScaler>)fxScaler;
                scaler.colorTexture = fxInput;
                scaler.depthTexture = fxDepth;
                scaler.motionTexture = fxMotion;
                scaler.outputTexture = fxOutput;
                scaler.inputContentWidth = inW;
                scaler.inputContentHeight = inH;
                scaler.jitterOffsetX = 0.0f;
                scaler.jitterOffsetY = 0.0f;
                scaler.preExposure = 1.0f;
                scaler.reset = fxResetHistory;
                fxResetHistory = NO;
                [scaler encodeToCommandBuffer:cmdBuf];
            } else {
                id<MTLFXSpatialScaler> scaler = (id<MTLFXSpatialScaler>)fxScaler;
                scaler.colorTexture = fxInput;
                scaler.outputTexture = fxOutput;
                scaler.inputContentWidth = inW;
                scaler.inputContentHeight = inH;
                [scaler encodeToCommandBuffer:cmdBuf];
            }
            return fxOutput;
        }
#endif
        return nil;
    }

    struct TextureInstance {
        Texture2DSpec spec;
        id<MTLTexture> resource = nil;
        std::array<id<MTLBuffer>, kFrameCount> stagingBuffers{};
        std::array<void *, kFrameCount> stagingBuffersData{};

        size_t rowPitch = 0;
        size_t uploadBufferSize = 0;
    };

    struct TextureToDelete {
        id<MTLTexture> texture = nil;
        std::array<id<MTLBuffer>, kFrameCount> stagingBuffers{};
        uint64 targetFrameNumber = 0;

        void Destroy() {
            for (size_t i = 0; i < kFrameCount; ++i) {
                stagingBuffers[i] = nil;
            }
            texture = nil;
        }
    };

    std::unordered_map<TextureID, TextureInstance> textures;
    std::deque<TextureToDelete> texturesToDelete;

    TextureIDManager texIDMgr;

    // -------------------------------------------------------------------------

    util::VoidResult<> Init() {
        if (spec.window == nullptr) {
            return util::ErrorMessage{"No window provided to Metal specification"};
        }

        @autoreleasepool {
            if (spec.device != nullptr) {
                device = (__bridge id<MTLDevice>)spec.device;
            } else {
                device = MTLCreateSystemDefaultDevice();
            }

            if (device == nil) {
                return util::ErrorMessage{"Failed to create Metal device"};
            }
            displayFrames = std::make_shared<MetalDisplayFrames>(device);

            if (![device supportsFamily:static_cast<MTLGPUFamily>(spec.featureLevel)]) {
                return util::ErrorMessage{"Metal device does not support requested feature level"};
            }

            metalView = SDL_Metal_CreateView(spec.window);
            if (metalView == nullptr) {
                return util::ErrorMessage{fmt::format("Failed to create SDL Metal view: {}", SDL_GetError())};
            }

            metalLayer = (__bridge CAMetalLayer *)SDL_Metal_GetLayer(metalView);
            if (metalLayer == nil) {
                return util::ErrorMessage{"Failed to get CAMetalLayer from SDL Metal view"};
            }

            metalLayer.device = device;
            metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
            metalLayer.framebufferOnly = YES;
#if TARGET_OS_OSX
            metalLayer.displaySyncEnabled = (presentMode != PresentMode::NoSync);
#endif // iOS drawables are always paced by the display

            int pxW = 0, pxH = 0;
            SDL_GetWindowSizeInPixels(spec.window, &pxW, &pxH);
            framebufferWidth = (uint32)std::max(1, pxW);
            framebufferHeight = (uint32)std::max(1, pxH);
            metalLayer.drawableSize = CGSizeMake(framebufferWidth, framebufferHeight);

            viewport = {
                .originX = 0.0,
                .originY = 0.0,
                .width = (double)framebufferWidth,
                .height = (double)framebufferHeight,
                .znear = 0.0,
                .zfar = 1.0,
            };
            scissorRect = {
                .x = 0,
                .y = 0,
                .width = framebufferWidth,
                .height = framebufferHeight,
            };
            drawTextureConstants.renderTargetSize = {
                (float)framebufferWidth,
                (float)framebufferHeight,
            };

            cmdQueue = [device newCommandQueue];
            if (cmdQueue == nil) {
                return util::ErrorMessage{"Failed to create Metal command queue"};
            }
            cmdQueue.label = @"[Ymir-GCtx] Command queue";

            // Create synchronization objects
            frameSemaphore = dispatch_semaphore_create(kFrameCount);

            // Create nearest neighbor and linear samplers
            {
                // Nearest neighbor
                MTLSamplerDescriptor *nearestSamplerDesc = [MTLSamplerDescriptor new];
                nearestSamplerDesc.label = @"[Ymir-GCtx] Nearest sampler";
                nearestSamplerDesc.minFilter = MTLSamplerMinMagFilterNearest;
                nearestSamplerDesc.magFilter = MTLSamplerMinMagFilterNearest;
                nearestSamplerDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
                nearestSamplerDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
                smpNearest = [device newSamplerStateWithDescriptor:nearestSamplerDesc];

                // Linear
                MTLSamplerDescriptor *linearSamplerDesc = [MTLSamplerDescriptor new];
                linearSamplerDesc.label = @"[Ymir-GCtx] Linear sampler";
                linearSamplerDesc.minFilter = MTLSamplerMinMagFilterLinear;
                linearSamplerDesc.magFilter = MTLSamplerMinMagFilterLinear;
                linearSamplerDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
                linearSamplerDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
                smpLinear = [device newSamplerStateWithDescriptor:linearSamplerDesc];
            }

            // Load shaders
            auto fs = cmrc::ymir_sdl3_shaders::get_filesystem();
            auto loadShader = [&](const char *path) -> util::ValueResult<std::vector<char>> {
                if (!fs.is_file(path)) {
                    return util::ErrorMessage{fmt::format("Shader resource not found: {}", path)};
                }
                auto shaderFile = fs.open(path);
                return std::vector<char>{shaderFile.begin(), shaderFile.end()};
            };

            // Load vertex shader
            auto vsBytecodeResult = loadShader("gctx/quad_vs.metallib");
            if (!vsBytecodeResult) {
                return util::ErrorMessage{
                    fmt::format("Could not load vertex shader: {}", vsBytecodeResult.Error().message)};
            }
            vertexShader.format = ShaderBytecodeFormat::MetalLib;
            vertexShader.bytecode = vsBytecodeResult.Value();
            vertexShader.entrypoint = "VSMain";
            if (auto res = ValidateShader(vertexShader); !res) {
                return util::ErrorMessage{fmt::format("Vertex shader validation failed: {}", res.Error().message)};
            }

            // Load pixel shader
            auto psBytecodeResult = loadShader("gctx/quad_ps.metallib");
            if (!psBytecodeResult) {
                return util::ErrorMessage{
                    fmt::format("Could not load pixel shader: {}", psBytecodeResult.Error().message)};
            }
            pixelShader.format = ShaderBytecodeFormat::MetalLib;
            pixelShader.bytecode = psBytecodeResult.Value();
            pixelShader.entrypoint = "PSMain";
            if (auto res = ValidateShader(pixelShader); !res) {
                return util::ErrorMessage{fmt::format("Pixel shader validation failed: {}", res.Error().message)};
            }

            dispatch_data_t vsData = dispatch_data_create(vertexShader.bytecode.data(), vertexShader.bytecode.size(),
                                                          dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            NSError *vsErr = nil;
            id<MTLLibrary> vsLib = [device newLibraryWithData:vsData error:&vsErr];
            if (vsLib == nil) {
                return util::ErrorMessage{fmt::format("Failed to load Metal vertex library: {}",
                                                      vsErr ? [vsErr.localizedDescription UTF8String] : "Unknown")};
            }
            vsFunc = [vsLib newFunctionWithName:@"VSMain"];
            if (vsFunc == nil) {
                return util::ErrorMessage{"Could not find entrypoint VSMain in vertex shader"};
            }

            dispatch_data_t psData = dispatch_data_create(pixelShader.bytecode.data(), pixelShader.bytecode.size(),
                                                          dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            NSError *psErr = nil;
            id<MTLLibrary> psLib = [device newLibraryWithData:psData error:&psErr];
            if (psLib == nil) {
                return util::ErrorMessage{fmt::format("Failed to load Metal pixel library: {}",
                                                      psErr ? [psErr.localizedDescription UTF8String] : "Unknown")};
            }
            psFunc = [psLib newFunctionWithName:@"PSMain"];
            if (psFunc == nil) {
                return util::ErrorMessage{"Could not find entrypoint PSMain in pixel shader"};
            }

            // Create the graphics pipeline state object (PSO)
            auto psoResult = CreateQuadPipeline(MTLPixelFormatBGRA8Unorm);
            if (!psoResult) {
                return psoResult.Error();
            }
            pipelineStateFrame = psoResult.Value();

            // Create the vertex buffer
            {
                // Define the geometry for a quad
                Vertex vertices[] = {
                    {{0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
                    {{1.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
                    {{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
                    {{1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
                };
                vertexBuffer = [device newBufferWithBytes:vertices
                                                   length:sizeof(vertices)
                                                  options:MTLResourceStorageModeShared];
                vertexBuffer.label = @"[Ymir-GCtx] Vertex buffer";
            }

            mainRenderPassDesc = [MTLRenderPassDescriptor new];
            mainRenderPassDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
            mainRenderPassDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
            mainRenderPassDesc.colorAttachments[0].clearColor = clearColor;
        }

        return {};
    }

    util::ValueResult<id<MTLRenderPipelineState>> CreateQuadPipeline(MTLPixelFormat pixelFormat) {
        const char *name = MetalPixelFormatName(pixelFormat);
        MTLRenderPipelineDescriptor *psoDesc = [MTLRenderPipelineDescriptor new];
        psoDesc.label = [NSString stringWithUTF8String:fmt::format("[Ymir-GCtx] Quad pipeline ({})", name).c_str()];
        psoDesc.vertexFunction = vsFunc;
        psoDesc.fragmentFunction = psFunc;
        psoDesc.vertexDescriptor = CreateQuadVertexDescriptor();
        psoDesc.colorAttachments[0].pixelFormat = pixelFormat;

        NSError *psoErr = nil;
        id<MTLRenderPipelineState> pso = [device newRenderPipelineStateWithDescriptor:psoDesc error:&psoErr];
        if (pso == nil) {
            return util::ErrorMessage{fmt::format("Failed to create pipeline state for [{}], details: {}", name,
                                                  psoErr ? [psoErr.localizedDescription UTF8String] : "Unknown")};
        }
        return pso;
    }

    void Shutdown() {
        @autoreleasepool {
            ReleaseDisplayFrame();
            WaitForGPU();
            if (displayFrames) {
                // (the renderer must never wait for a display that is gone)
                displayFrames->ReleaseEvent().signaledValue = UINT64_MAX;
                displayFrames->Reset();
            }
            DeletePendingTextures(true);

            for (auto &[texID, tex] : textures) {
                tex.resource = nil;
                for (size_t i = 0; i < kFrameCount; ++i) {
                    tex.stagingBuffers[i] = nil;
                    tex.stagingBuffersData[i] = nullptr;
                }
            }
            textures.clear();

            renderTargetPipelines.clear();
            pipelineStateFrame = nil;
            vsFunc = nil;
            psFunc = nil;
            vertexBuffer = nil;
            smpNearest = nil;
            smpLinear = nil;
            mainRenderPassDesc = nil;
            renderEncoderFrame = nil;
            cmdListFrame = nil;
            currentDrawable = nil;
            cmdQueue = nil;
            metalLayer = nil;

            if (metalView != nullptr) {
                SDL_Metal_DestroyView(metalView);
                metalView = nullptr;
            }

            device = nil;
            frameSemaphore = nullptr;
        }
    }

    bool IsInitialized() const {
        return device != nil;
    }

    void WaitForGPU() {
        if (renderEncoderFrame != nil) {
            [renderEncoderFrame endEncoding];
            renderEncoderFrame = nil;
        }
        if (cmdListFrame != nil) {
            [cmdListFrame commit];
            [cmdListFrame waitUntilCompleted];
            cmdListFrame = nil;
        } else if (cmdQueue != nil) {
            id<MTLCommandBuffer> flushBuf = [cmdQueue commandBuffer];
            if (flushBuf != nil) {
                [flushBuf commit];
                [flushBuf waitUntilCompleted];
            }
        }
        if (currentDrawable != nil) {
            currentDrawable = nil;
            dispatch_semaphore_signal(frameSemaphore);
        }
    }

    util::VoidResult<> ResizeFramebuffer(uint32 width, uint32 height) {
        int pxW = 0, pxH = 0;
        if (spec.window != nullptr) {
            SDL_GetWindowSizeInPixels(spec.window, &pxW, &pxH);
        }
        framebufferWidth = (pxW > 0) ? (uint32)pxW : std::max(1u, width);
        framebufferHeight = (pxH > 0) ? (uint32)pxH : std::max(1u, height);

        @autoreleasepool {
            if (metalLayer != nil) {
                metalLayer.drawableSize = CGSizeMake(framebufferWidth, framebufferHeight);
            }
        }

        // Update viewport and scissor rects
        viewport = {
            .originX = 0.0,
            .originY = 0.0,
            .width = (double)framebufferWidth,
            .height = (double)framebufferHeight,
            .znear = 0.0,
            .zfar = 1.0,
        };
        scissorRect = {
            .x = 0,
            .y = 0,
            .width = framebufferWidth,
            .height = framebufferHeight,
        };
        drawTextureConstants.renderTargetSize = {
            (float)framebufferWidth,
            (float)framebufferHeight,
        };

        return {};
    }

    util::VoidResult<> BeginFrame() {
        if (currentDrawable != nil) {
            return {};
        }

        if (metalLayer == nil || cmdQueue == nil) {
            return util::ErrorMessage{"Metal graphics context is not initialized"};
        }

        int pxW = 0, pxH = 0;
        if (spec.window != nullptr) {
            SDL_GetWindowSizeInPixels(spec.window, &pxW, &pxH);
            if (pxW > 0 && pxH > 0 && ((uint32)pxW != framebufferWidth || (uint32)pxH != framebufferHeight)) {
                ResizeFramebuffer((uint32)pxW, (uint32)pxH);
            }
        }

        dispatch_semaphore_wait(frameSemaphore, DISPATCH_TIME_FOREVER);

        currentDrawable = [metalLayer nextDrawable];
        if (currentDrawable == nil) {
            dispatch_semaphore_signal(frameSemaphore);
            return util::ErrorMessage{"Failed to acquire next Metal drawable"};
        }

        mainRenderPassDesc.colorAttachments[0].texture = currentDrawable.texture;
        mainRenderPassDesc.colorAttachments[0].loadAction = shouldClear ? MTLLoadActionClear : MTLLoadActionLoad;
        mainRenderPassDesc.colorAttachments[0].clearColor = clearColor;
        mainRenderPassDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

        cmdListFrame = [cmdQueue commandBuffer];
        if (cmdListFrame == nil) {
            dispatch_semaphore_signal(frameSemaphore);
            currentDrawable = nil;
            return util::ErrorMessage{"Failed to create Metal command buffer"};
        }

        return {};
    }

    util::ValueResult<PresentResult> Present() {
        if (currentDrawable != nil && renderEncoderFrame == nil) {
            EnsureMainRenderEncoder();
        }
        if (renderEncoderFrame != nil) {
            [renderEncoderFrame endEncoding];
            renderEncoderFrame = nil;
        }
        // (an image of the renderer acquired for this frame is released with it at the latest)
        ReleaseDisplayFrame();

        @autoreleasepool {
            if (currentDrawable != nil && cmdListFrame != nil) {
                if (frameRateHint > 0.0) {
                    // Lets a variable refresh rate (ProMotion) display settle at the rate frames are produced at
                    // instead of running at its maximum rate
                    [cmdListFrame presentDrawable:currentDrawable afterMinimumDuration:1.0 / frameRateHint];
                } else {
                    [cmdListFrame presentDrawable:currentDrawable];
                }

                dispatch_semaphore_t sem = frameSemaphore;
                [cmdListFrame addCompletedHandler:^(id<MTLCommandBuffer>) {
                  dispatch_semaphore_signal(sem);
                }];

                [cmdListFrame commit];
            } else if (currentDrawable != nil) {
                dispatch_semaphore_signal(frameSemaphore);
            }
        }

        currentDrawable = nil;
        cmdListFrame = nil;
        shouldClear = true;

        frameIndex = (frameIndex + 1) % kFrameCount;
        ++currentFrameNumber;
        DeletePendingTextures(false);

        return PresentResult::Ok;
    }

    void DeletePendingTextures(bool force) {
        while (!texturesToDelete.empty() &&
               (force || texturesToDelete.front().targetFrameNumber <= currentFrameNumber)) {
            texturesToDelete.front().Destroy();
            texturesToDelete.pop_front();
        }
    }

    void SubmitTextureForDeletion(TextureInstance &texture) {
        TextureToDelete &toDelete = texturesToDelete.emplace_back();
        toDelete.texture = texture.resource;
        toDelete.stagingBuffers = texture.stagingBuffers;
        toDelete.targetFrameNumber = currentFrameNumber + kFrameCount;
    }

    util::ValueResult<TextureInstance> CreateTexture(const Texture2DSpec &textureSpec) {
        const MTLPixelFormat mtlFormat = ToMetalValue(textureSpec.format);
        if (mtlFormat == MTLPixelFormatInvalid) {
            return util::ErrorMessage{"Unsupported pixel format"};
        }

        TextureInstance instance;
        instance.spec = textureSpec;

        @autoreleasepool {
            MTLTextureDescriptor *texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:mtlFormat
                                                                                               width:textureSpec.width
                                                                                              height:textureSpec.height
                                                                                           mipmapped:NO];
            texDesc.usage = MTLTextureUsageShaderRead;
            if (textureSpec.access == TextureAccess::RenderTarget) {
                texDesc.usage |= MTLTextureUsageRenderTarget;
            }
            texDesc.storageMode = MTLStorageModePrivate;

            instance.resource = [device newTextureWithDescriptor:texDesc];
            if (instance.resource == nil) {
                return util::ErrorMessage{"Failed to create Metal texture"};
            }
            if (!textureSpec.name.empty()) {
                instance.resource.label =
                    [NSString stringWithUTF8String:fmt::format("{} texture", textureSpec.name).c_str()];
            }

            const size_t bytesPerPixel = PixelFormatUnitSize(textureSpec.format);
            instance.rowPitch = textureSpec.width * bytesPerPixel;
            instance.uploadBufferSize = instance.rowPitch * textureSpec.height;

            // Staging buffers are used for asynchronous GPU texture uploads via the blit encoder.
            // We store one buffer per frame to enable parallel updates.
            if (instance.uploadBufferSize > 0) {
                for (size_t i = 0; i < kFrameCount; ++i) {
                    instance.stagingBuffers[i] = [device newBufferWithLength:instance.uploadBufferSize
                                                                     options:MTLResourceStorageModeShared];
                    if (instance.stagingBuffers[i] == nil) {
                        return util::ErrorMessage{"Failed to create Metal staging buffer"};
                    }
                    if (!textureSpec.name.empty()) {
                        instance.stagingBuffers[i].label = [NSString
                            stringWithUTF8String:fmt::format("{} staging buffer #{}", textureSpec.name, i).c_str()];
                    }
                    instance.stagingBuffersData[i] = [instance.stagingBuffers[i] contents];
                }
            }
        }

        return instance;
    }

    util::VoidResult<> ResizeTexture(TextureID textureID, uint32 width, uint32 height) {
        auto it = textures.find(textureID);
        if (it == textures.end()) {
            return util::ErrorMessage{"Texture does not exist"};
        }
        TextureInstance &texture = it->second;

        // First, try creating new texture using the existing texture's
        // specifications
        Texture2DSpec newSpec = texture.spec;
        newSpec.width = width;
        newSpec.height = height;

        auto createResult = CreateTexture(newSpec);
        if (!createResult) {
            return createResult.Error();
        }

        // Now that we've succeeded, mark the previous texture for deletion and
        // replace it
        SubmitTextureForDeletion(texture);
        texture = createResult.Value();
        return {};
    }

    void DestroyTexture(TextureID textureID) {
        auto it = textures.find(textureID);
        if (it == textures.end()) {
            return;
        }
        SubmitTextureForDeletion(it->second);
        textures.erase(it);
    }

    bool IsTextureValid(TextureID textureID) const {
        auto it = textures.find(textureID);
        if (it == textures.end()) {
            return false;
        }
        const auto &texture = it->second;
        return texture.resource != nil;
    }

    TextureInstance *GetTexture(TextureID textureID) {
        auto it = textures.find(textureID);
        if (it == textures.end()) {
            return nullptr;
        }
        return &it->second;
    }

    util::VoidResult<> UpdateTexture(TextureID textureID, const IRect *rect,
                                     const std::function<void(void *data, size_t pitch)> &fnUpdate) {
        auto it = textures.find(textureID);
        if (it == textures.end()) {
            return util::ErrorMessage{"Invalid texture ID"};
        }
        TextureInstance &texture = it->second;

        void *stagingBufferData = texture.stagingBuffersData[frameIndex];
        id<MTLBuffer> stagingBuffer = texture.stagingBuffers[frameIndex];
        if (stagingBufferData == nullptr || stagingBuffer == nil) {
            return util::ErrorMessage{"Invalid staging buffer pointer"};
        }

        // Copy data to staging buffer
        fnUpdate(stagingBufferData, texture.rowPitch);

        // Copy buffer to texture
        @autoreleasepool {
            const size_t bytesPerPixel = PixelFormatUnitSize(texture.spec.format);
            const uint32 copyX = (rect != nullptr) ? (uint32)rect->x : 0;
            const uint32 copyY = (rect != nullptr) ? (uint32)rect->y : 0;
            const uint32 copyW = (rect != nullptr) ? (uint32)rect->w : texture.spec.width;
            const uint32 copyH = (rect != nullptr) ? (uint32)rect->h : texture.spec.height;

            const size_t srcOffset = copyY * texture.rowPitch + copyX * bytesPerPixel;
            const MTLOrigin dstOrigin = MTLOriginMake(copyX, copyY, 0);
            const MTLSize copySize = MTLSizeMake(copyW, copyH, 1);

            if (renderEncoderFrame != nil) {
                [renderEncoderFrame endEncoding];
                renderEncoderFrame = nil;
            }

            if (cmdListFrame != nil) {
                id<MTLBlitCommandEncoder> blitEncoder = [cmdListFrame blitCommandEncoder];
                blitEncoder.label = @"[Ymir-GCtx] UpdateTexture blit";
                [blitEncoder copyFromBuffer:stagingBuffer
                               sourceOffset:srcOffset
                          sourceBytesPerRow:texture.rowPitch
                        sourceBytesPerImage:texture.uploadBufferSize
                                 sourceSize:copySize
                                  toTexture:texture.resource
                           destinationSlice:0
                           destinationLevel:0
                          destinationOrigin:dstOrigin];
                [blitEncoder endEncoding];
            } else if (cmdQueue != nil) {
                id<MTLCommandBuffer> opCmdBuf = [cmdQueue commandBuffer];
                opCmdBuf.label = @"[Ymir-GCtx] Operations command buffer";
                id<MTLBlitCommandEncoder> blitEncoder = [opCmdBuf blitCommandEncoder];
                blitEncoder.label = @"[Ymir-GCtx] UpdateTexture blit";
                [blitEncoder copyFromBuffer:stagingBuffer
                               sourceOffset:srcOffset
                          sourceBytesPerRow:texture.rowPitch
                        sourceBytesPerImage:texture.uploadBufferSize
                                 sourceSize:copySize
                                  toTexture:texture.resource
                           destinationSlice:0
                           destinationLevel:0
                          destinationOrigin:dstOrigin];
                [blitEncoder endEncoding];
                [opCmdBuf commit];
                [opCmdBuf waitUntilCompleted];
            }
        }

        return {};
    }

    util::PointerResult<RenderTargetPipeline> GetRenderTargetPipeline(const TextureInstance &texture) {
        const MTLPixelFormat mtlFormat = texture.resource.pixelFormat;
        auto it = renderTargetPipelines.find(mtlFormat);
        if (it != renderTargetPipelines.end()) {
            return &it->second;
        }

        RenderTargetPipeline &pipeline = renderTargetPipelines[mtlFormat];
        auto psoResult = CreateQuadPipeline(mtlFormat);
        if (!psoResult) {
            return psoResult.Error();
        }
        pipeline.pipelineState = psoResult.Value();

        return &pipeline;
    }

    util::VoidResult<> RenderToTexture(TextureID src, TextureID dst, const FRect &srcRect, const FRect &dstRect) {
        TextureInstance *dstTexture = GetTexture(dst);
        if (dstTexture == nullptr) {
            return util::ErrorMessage{"Invalid destination texture"};
        }
        if (dstTexture->spec.access != TextureAccess::RenderTarget) {
            return util::ErrorMessage{"Destination texture is not a valid render target"};
        }
        auto psoResult = GetRenderTargetPipeline(*dstTexture);
        if (!psoResult) {
            return util::ErrorMessage{fmt::format("Could not get render target PSO: {}", psoResult.Error().message)};
        }
        RenderTargetPipeline *pipeline = psoResult.Value();

        TextureInstance *srcTexture = GetTexture(src);
        if (srcTexture == nullptr) {
            return util::ErrorMessage{"Invalid source texture"};
        }

        if (renderEncoderFrame != nil) {
            [renderEncoderFrame endEncoding];
            renderEncoderFrame = nil;
        }

        @autoreleasepool {
            MTLRenderPassDescriptor *rttPassDesc = [MTLRenderPassDescriptor renderPassDescriptor];
            rttPassDesc.colorAttachments[0].texture = dstTexture->resource;
            rttPassDesc.colorAttachments[0].loadAction = MTLLoadActionLoad;
            rttPassDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

            // Drawn in the frame's command buffer if there is one, ahead of what is drawn on screen; otherwise in a
            // command buffer of its own, which is waited for
            const bool ownCmdBuf = cmdListFrame == nil;
            id<MTLCommandBuffer> rttCmdBuf = ownCmdBuf ? [cmdQueue commandBuffer] : cmdListFrame;

            // Optionally upscale the source with MetalFX first, then scale the result into the destination
            id<MTLTexture> srcResource = srcTexture->resource;
            FRect effSrcRect = srcRect;
            float srcTexWidth = (float)srcTexture->spec.width;
            float srcTexHeight = (float)srcTexture->spec.height;
            bool useNearest = srcTexture->spec.filterMode == TextureFilterMode::Nearest;
            if (upscaleMode != UpscaleMode::Off && srcRect.w >= 1.0f && srcRect.h >= 1.0f && dstRect.w > srcRect.w) {
                const uint32 factor = std::clamp<uint32>((uint32)std::lround(dstRect.w / srcRect.w), 2u, 3u);
                if (id<MTLTexture> upscaled = EncodeUpscale(rttCmdBuf, srcTexture->resource, srcRect, factor)) {
                    srcResource = upscaled;
                    effSrcRect = {.x = 0.0f, .y = 0.0f, .w = (float)upscaled.width, .h = (float)upscaled.height};
                    srcTexWidth = (float)upscaled.width;
                    srcTexHeight = (float)upscaled.height;
                    useNearest = false;
                }
            }

            id<MTLRenderCommandEncoder> rttEncoder = [rttCmdBuf renderCommandEncoderWithDescriptor:rttPassDesc];

            // Use the filtering shader if any filter is on and it could be built
            id<MTLRenderPipelineState> postPipeline =
                postProcess.AnyActive() ? GetPostPipeline(dstTexture->resource.pixelFormat) : nil;
            if (postPipeline != nil) {
                PostConstants post{};
                post.srcRect = {effSrcRect.x / srcTexWidth, effSrcRect.y / srcTexHeight, effSrcRect.w / srcTexWidth,
                                effSrcRect.h / srcTexHeight};
                post.dstRect = {dstRect.x, dstRect.y, dstRect.w, dstRect.h};
                post.renderTargetSize = {(float)dstTexture->spec.width, (float)dstTexture->spec.height};
                post.contentSize = {effSrcRect.w, effSrcRect.h};
                post.sharpness = std::clamp(postProcess.sharpness, 0.0f, 1.0f);
                post.scanlines = std::clamp(postProcess.scanlines, 0.0f, 1.0f);
                post.mask = std::clamp(postProcess.mask, 0.0f, 1.0f);
                post.sharpK = postProcess.sharpPixels ? std::max(1.0f, dstRect.w / std::max(effSrcRect.w, 1.0f)) : 0.0f;

                [rttEncoder setRenderPipelineState:postPipeline];
                [rttEncoder setVertexBytes:&post length:sizeof(post) atIndex:0];
                [rttEncoder setFragmentBytes:&post length:sizeof(post) atIndex:0];
                [rttEncoder setFragmentTexture:srcResource atIndex:0];
                // Sharp pixel scaling needs bilinear filtering to blend the borders
                [rttEncoder setFragmentSamplerState:((useNearest && !postProcess.sharpPixels) ? smpNearest : smpLinear)
                                            atIndex:0];
                [rttEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
                [rttEncoder endEncoding];
                if (ownCmdBuf) {
                    [rttCmdBuf commit];
                    [rttCmdBuf waitUntilCompleted];
                }
                return {};
            }

            [rttEncoder setRenderPipelineState:pipeline->pipelineState];
            [rttEncoder setVertexBuffer:vertexBuffer offset:0 atIndex:17];

            // Update constants with source UVs, destination area (in pixels), and
            // render target size
            DrawTextureConstants consts{};
            consts.srcRect = {
                effSrcRect.x / srcTexWidth,
                effSrcRect.y / srcTexHeight,
                effSrcRect.w / srcTexWidth,
                effSrcRect.h / srcTexHeight,
            };
            consts.dstRect = {dstRect.x, dstRect.y, dstRect.w, dstRect.h};
            consts.renderTargetSize = {(float)dstTexture->spec.width, (float)dstTexture->spec.height};
            consts.rotPivot = {dstRect.w * 0.5f, dstRect.h * 0.5f};
            consts.rotAngle = 0.0f;

            [rttEncoder setVertexBytes:&consts length:sizeof(consts) atIndex:0];
            [rttEncoder setFragmentTexture:srcResource atIndex:0];
            [rttEncoder setFragmentSamplerState:(useNearest ? smpNearest : smpLinear) atIndex:0];

            // Draw rectangle
            [rttEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
            [rttEncoder endEncoding];
            if (ownCmdBuf) {
                [rttCmdBuf commit];
                [rttCmdBuf waitUntilCompleted];
            }
        }

        return {};
    }

    util::VoidResult<> DrawTextureRotated(TextureID textureID, const FRect &srcRect, const FRect &dstRect,
                                          double rotAngle, const FPoint2D *rotPivot) {
        TextureInstance *instance = GetTexture(textureID);
        if (instance == nullptr) {
            return util::ErrorMessage{"Invalid texture"};
        }

        auto encoderResult = EnsureMainRenderEncoder();
        if (!encoderResult) {
            return encoderResult;
        }

        // Select sampler based on texture filtering mode
        id<MTLSamplerState> smpDesc =
            (instance->spec.filterMode == TextureFilterMode::Nearest) ? smpNearest : smpLinear;

        // Update constants with source UVs, destination area (in pixels), rotation
        // angle and pivot point
        drawTextureConstants.srcRect = {
            srcRect.x / instance->spec.width,
            srcRect.y / instance->spec.height,
            srcRect.w / instance->spec.width,
            srcRect.h / instance->spec.height,
        };
        drawTextureConstants.dstRect = {
            dstRect.x,
            dstRect.y,
            dstRect.w,
            dstRect.h,
        };
        drawTextureConstants.renderTargetSize = {
            (float)framebufferWidth,
            (float)framebufferHeight,
        };
        if (rotPivot == nullptr) {
            drawTextureConstants.rotPivot.x = dstRect.w * 0.5f;
            drawTextureConstants.rotPivot.y = dstRect.h * 0.5f;
        } else {
            drawTextureConstants.rotPivot.x = rotPivot->x;
            drawTextureConstants.rotPivot.y = rotPivot->y;
        }
        drawTextureConstants.rotAngle = rotAngle;

        // Draw rectangle
        @autoreleasepool {
            [renderEncoderFrame setRenderPipelineState:pipelineStateFrame];
            [renderEncoderFrame setVertexBuffer:vertexBuffer offset:0 atIndex:17];

            [renderEncoderFrame setVertexBytes:&drawTextureConstants length:sizeof(drawTextureConstants) atIndex:0];
            [renderEncoderFrame setFragmentTexture:instance->resource atIndex:0];
            [renderEncoderFrame setFragmentSamplerState:smpDesc atIndex:0];

            [renderEncoderFrame drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        }

        return {};
    }

    // -------------------------------------------------------------------------
    // Images of the Metal VDP renderer

    /// Raises the release event to the given number unless it's there already (the renderer waits for it)
    void SignalReleaseOnCPU(uint64 number) {
        id<MTLSharedEvent> event = displayFrames->ReleaseEvent();
        if (event.signaledValue < number) {
            event.signaledValue = number;
        }
    }

    /// The texture ID under which the display draws the image texture of the given slot
    TextureID DisplayFrameTextureID(const MetalDisplayFrames::Frame &frame) {
        TextureID &id = displayFrameIDs[frame.slot];
        auto it = id != kInvalidTextureID ? textures.find(id) : textures.end();
        if (it == textures.end() || it->second.resource != frame.texture) {
            if (id == kInvalidTextureID) {
                id = texIDMgr.GetNextTextureID();
            }
            TextureInstance instance{};
            instance.spec = {
                .width = MetalDisplayFrames::kWidth,
                .height = MetalDisplayFrames::kHeight,
                .format = PixelFormat::R8G8B8A8_UNORM,
                .access = TextureAccess::Static,
                .filterMode = TextureFilterMode::Nearest,
                .name = "[Ymir] Metal VDP renderer image",
            };
            instance.resource = frame.texture;
            textures[id] = std::move(instance);
        }
        return id;
    }

    /// Gets the latest image of the Metal VDP renderer for the frame being drawn; the frame's command buffer waits on
    /// the GPU for the image to be ready
    std::optional<DisplayTextureSpec> AcquireDisplayFrame() {
        if (!displayFrames) {
            return std::nullopt;
        }
        ReleaseDisplayFrame();
        MetalDisplayFrames::Frame frame{};
        if (!displayFrames->AcquireLatest(frame)) {
            return std::nullopt;
        }
        if (!BeginFrame() || cmdListFrame == nil) {
            // Can't draw it this time; the renderer may write over it
            SignalReleaseOnCPU(frame.number);
            return std::nullopt;
        }
        if (renderEncoderFrame != nil) {
            [renderEncoderFrame endEncoding];
            renderEncoderFrame = nil;
        }
        [cmdListFrame encodeWaitForEvent:displayFrames->ReadyEvent() value:frame.number];
        displayFrameDrawn = frame.number;
        return DisplayTextureSpec{
            .id = DisplayFrameTextureID(frame),
            .width = frame.width,
            .height = frame.height,
            .scale = frame.scale,
        };
    }

    /// Lets the renderer write over the image acquired for this frame once the GPU is done drawing it
    void ReleaseDisplayFrame() {
        if (displayFrameDrawn == 0 || !displayFrames) {
            return;
        }
        if (renderEncoderFrame != nil) {
            [renderEncoderFrame endEncoding];
            renderEncoderFrame = nil;
        }
        if (cmdListFrame != nil) {
            [cmdListFrame encodeSignalEvent:displayFrames->ReleaseEvent() value:displayFrameDrawn];
        } else {
            SignalReleaseOnCPU(displayFrameDrawn);
        }
        displayFrameDrawn = 0;
    }

    /// Copies the latest image of the Metal VDP renderer to memory, rows packed (for screenshots)
    util::ValueResult<size_t> DownloadDisplayFrame(void *buffer, size_t size) {
        MetalDisplayFrames::Frame frame{};
        if (!displayFrames || !displayFrames->AcquireLatest(frame)) {
            return util::ErrorMessage{"No image to download"};
        }
        const uint32 w = frame.width * frame.scale;
        const uint32 h = frame.height * frame.scale;
        const size_t bytes = size_t(w) * h * sizeof(uint32);
        @autoreleasepool {
            id<MTLBuffer> readback = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cmdBuf = [cmdQueue commandBuffer];
            if (readback == nil || cmdBuf == nil) {
                SignalReleaseOnCPU(frame.number);
                return util::ErrorMessage{"Could not allocate the download buffer"};
            }
            [cmdBuf encodeWaitForEvent:displayFrames->ReadyEvent() value:frame.number];
            id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
            [blit copyFromTexture:frame.texture
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:MTLOriginMake(0, 0, 0)
                          sourceSize:MTLSizeMake(w, h, 1)
                            toBuffer:readback
                   destinationOffset:0
              destinationBytesPerRow:w * sizeof(uint32)
            destinationBytesPerImage:bytes];
            [blit endEncoding];
            [cmdBuf encodeSignalEvent:displayFrames->ReleaseEvent() value:frame.number];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
            const size_t copySize = std::min(size, bytes);
            std::memcpy(buffer, [readback contents], copySize);
            return copySize;
        }
    }

    void ClearScreen(gfx::ColorRGBA color) {
        clearColor = MTLClearColorMake(color.r, color.g, color.b, color.a);
        shouldClear = true;
        if (mainRenderPassDesc != nil) {
            mainRenderPassDesc.colorAttachments[0].clearColor = clearColor;
            mainRenderPassDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
        }
    }

    util::VoidResult<> EnsureMainRenderEncoder() {
        if (renderEncoderFrame != nil) {
            return {};
        }

        auto beginRes = BeginFrame();
        if (!beginRes) {
            return beginRes;
        }

        renderEncoderFrame = [cmdListFrame renderCommandEncoderWithDescriptor:mainRenderPassDesc];
        if (renderEncoderFrame == nil) {
            return util::ErrorMessage{"Failed to create Metal render command encoder"};
        }

        return {};
    }
};

// -----------------------------------------------------------------------------

MetalGraphicsContext::MetalGraphicsContext(const MetalGraphicsContextSpec &spec)
    : IGraphicsContext(kBackend)
    , m_impl(std::make_unique<Impl>(spec)) {}

MetalGraphicsContext::~MetalGraphicsContext() {
    Shutdown();
}

util::ObjectResult<MetalGraphicsContext> MetalGraphicsContext::Create(const MetalGraphicsContextSpec &spec) {
    auto context = std::make_unique<MetalGraphicsContext>(spec);
    auto result = context->Initialize();
    if (!result) {
        return result.Error();
    }
    return std::move(context);
}

util::VoidResult<> MetalGraphicsContext::Initialize() {
    return m_impl->Init();
}

void MetalGraphicsContext::Shutdown() {
    if (m_impl->IsInitialized()) {
        m_impl->WaitForGPU();
        ImGuiShutdown();
        m_impl->Shutdown();
    }
}

bool MetalGraphicsContext::IsInitialized() const {
    return m_impl->IsInitialized();
}

util::VoidResult<> MetalGraphicsContext::ResizeFramebuffer(uint32 width, uint32 height) {
    return m_impl->ResizeFramebuffer(width, height);
}

void MetalGraphicsContext::ClearScreen(gfx::ColorRGBA color) {
    m_impl->ClearScreen(color);
}

bool MetalGraphicsContext::ImGuiInit() {
    if (m_imguiInitialized) {
        return true;
    }

    m_imguiInitialized = ImGui_ImplMetal_Init(m_impl->device) && //
                         ImGui_ImplSDL3_InitForMetal(m_impl->spec.window);

    return m_imguiInitialized;
}

void MetalGraphicsContext::ImGuiShutdown() {
    if (m_imguiInitialized) {
        ImGui_ImplSDL3_Shutdown();
        ImGui_ImplMetal_Shutdown();
        m_imguiInitialized = false;
    }
}

void MetalGraphicsContext::ImGuiNewFrame() {
    if (m_imguiInitialized) {
        m_impl->BeginFrame();
        ImGui_ImplMetal_NewFrame(m_impl->mainRenderPassDesc);
        ImGui_ImplSDL3_NewFrame();
    }
}

void MetalGraphicsContext::ImGuiRenderFrame() {
    if (m_imguiInitialized) {
        auto res = m_impl->EnsureMainRenderEncoder();
        if (!res) {
            return;
        }

        ImDrawData *drawData = ImGui::GetDrawData();
        ImGui_ImplMetal_RenderDrawData(drawData, m_impl->cmdListFrame, m_impl->renderEncoderFrame);
    }
}

util::ValueResult<TextureID> MetalGraphicsContext::CreateTexture(const Texture2DSpec &spec) {
    auto result = m_impl->CreateTexture(spec);
    if (!result) {
        return result.Error();
    }

    const TextureID textureID = m_impl->texIDMgr.GetNextTextureID();
    m_impl->textures[textureID] = std::move(result.Value());

    return textureID;
}

void MetalGraphicsContext::DestroyTexture(TextureID textureID) {
    m_impl->DestroyTexture(textureID);
}

bool MetalGraphicsContext::IsTextureValid(TextureID textureID) const {
    return m_impl->IsTextureValid(textureID);
}

ImTextureID MetalGraphicsContext::GetImGuiTextureID(TextureID textureID) const {
    // ImTextureIDs for Metal are the (__bridge void *) pointer to the MTLTexture
    // resource
    Impl::TextureInstance *instance = m_impl->GetTexture(textureID);
    return instance ? reinterpret_cast<ImTextureID>((__bridge void *)instance->resource) : 0;
}

util::VoidResult<> MetalGraphicsContext::ResizeTexture(TextureID textureID, uint32 width, uint32 height) {
    return m_impl->ResizeTexture(textureID, width, height);
}

util::VoidResult<> MetalGraphicsContext::UpdateTexture(TextureID textureID, const IRect *rect,
                                                       const std::function<void(void *data, size_t pitch)> &fnUpdate) {
    return m_impl->UpdateTexture(textureID, rect, fnUpdate);
}

util::VoidResult<> MetalGraphicsContext::RenderToTexture(TextureID src, TextureID dst, const FRect &srcRect,
                                                         const FRect &dstRect) {
    return m_impl->RenderToTexture(src, dst, srcRect, dstRect);
}

util::VoidResult<> MetalGraphicsContext::DrawTextureRotated(TextureID textureID, const FRect &srcRect,
                                                            const FRect &dstRect, double rotAngle,
                                                            const FPoint2D *rotPivot) {
    return m_impl->DrawTextureRotated(textureID, srcRect, dstRect, rotAngle, rotPivot);
}

std::optional<DisplayTextureSpec> MetalGraphicsContext::AcquireCurrentDisplayOutputTexture() {
    return m_impl->AcquireDisplayFrame();
}

void MetalGraphicsContext::ReleaseCurrentDisplayOutputTexture() {
    m_impl->ReleaseDisplayFrame();
}

void MetalGraphicsContext::ResetDisplayOutputTextures() {
    if (m_impl->displayFrames) {
        m_impl->displayFrames->Reset();
    }
}

util::ValueResult<size_t> MetalGraphicsContext::DownloadDisplayOutputTexture(void *buffer, size_t size) {
    return m_impl->DownloadDisplayFrame(buffer, size);
}

std::shared_ptr<ymir::vdp::MetalDisplaySink> MetalGraphicsContext::GetDisplaySink() const {
    return m_impl->displayFrames;
}

util::VoidResult<> MetalGraphicsContext::SetPresentMode(PresentMode mode) {
    m_impl->presentMode = mode;
#if TARGET_OS_OSX
    if (m_impl->metalLayer != nil) {
        m_impl->metalLayer.displaySyncEnabled = (mode != PresentMode::NoSync);
    }
#endif
    return {};
}

void MetalGraphicsContext::SetPostProcess(const PostProcessParams &params) {
    m_impl->postProcess = params;
}

bool MetalGraphicsContext::SupportsUpscaling() const {
    return m_impl->SupportsUpscaling();
}

void MetalGraphicsContext::SetFrameRateHint(double framesPerSecond) {
    m_impl->frameRateHint = framesPerSecond;
}

void MetalGraphicsContext::SetUpscaleMode(UpscaleMode mode) {
    if (mode != UpscaleMode::Off && !m_impl->SupportsUpscaling()) {
        mode = UpscaleMode::Off;
    }
    if (m_impl->upscaleMode != mode) {
        m_impl->upscaleMode = mode;
        m_impl->ReleaseUpscaler(); // the scaler is recreated on the next frame
    }
}

util::ValueResult<PresentResult> MetalGraphicsContext::Present() {
    return m_impl->Present();
}

void *MetalGraphicsContext::GetDevice() const {
    return m_impl->device != nil ? (__bridge void *)m_impl->device : nullptr;
}

} // namespace app::gfx
