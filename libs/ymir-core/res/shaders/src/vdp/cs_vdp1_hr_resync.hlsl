#include "vdp1_defs.hlsli"
#include "vdp1_common_params.hlsli"

#include "util/bit_ops.hlsli"

// Fills the scaled up (internal resolution) framebuffer of the draw framebuffer index from the native framebuffer:
// every native pixel is replicated over the scaled up pixels it covers. One thread per two native pixels.

cbuffer CommonRenderParamsBuffer : register(b0) {
    CommonRenderParams g_commonParams;
}

ByteAddressBuffer g_fbramNative : register(t1);

RWByteAddressBuffer g_fbramOut : register(u1);

static const uint hrScale = BitExtract(g_commonParams.enhancements, 2, 3) + 1u;
static const uint drawFB = BitExtract(g_commonParams.displayParams, 7, 1);
static const bool transparentMeshes = BitTest(g_commonParams.enhancements, 1);

// Replicates the two native pixels of a word over the scaled up pixels they cover. The two pixels cover 2 * scale
// pixels of every row, starting at an even pixel (the first pixel is at an even column), so the row is written in whole
// 32-bit words.
void Upscale(uint word, uint nx, uint ny, uint base) {
    const uint raw0 = BitExtract(word, 0, 16);
    const uint raw1 = BitExtract(word, 16, 16);
    const uint rowPixels = 512u * hrScale;
    for (uint dy = 0; dy < hrScale; ++dy) {
        const uint rowAddress = base + ((ny * hrScale + dy) * rowPixels + nx * hrScale) * 2u;
        for (uint m = 0; m < hrScale; ++m) {
            const uint lo = (m * 2u) < hrScale ? raw0 : raw1;
            const uint hi = (m * 2u + 1u) < hrScale ? raw0 : raw1;
            g_fbramOut.Store(rowAddress + m * 4u, lo | (hi << 16u));
        }
    }
}

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= 512u * 256u / 2u) {
        return;
    }

    const uint nx = (id.x * 2u) & 511u;
    const uint ny = (id.x * 2u) >> 9u;
    const uint base = drawFB * kVDP1FBSize * hrScale * hrScale;
    Upscale(g_fbramNative.Load(drawFB * kVDP1FBSize + id.x * 4u), nx, ny, base);
    if (transparentMeshes) {
        // (the native mesh framebuffers are at the start of the second half of the native framebuffer memory, the
        // scaled up ones follow the two scaled up framebuffers)
        Upscale(g_fbramNative.Load(kVDP1FBRAMSize * 2 + drawFB * kVDP1FBSize + id.x * 4u), nx, ny,
                base + kVDP1FBRAMSize * hrScale * hrScale);
    }
}
