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

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= 512u * 256u / 2u) {
        return;
    }

    const uint word = g_fbramNative.Load(drawFB * kVDP1FBSize + id.x * 4u);
    const uint nx = (id.x * 2u) & 511u;
    const uint ny = (id.x * 2u) >> 9u;

    const uint rowPixels = 512u * hrScale;
    const uint base = drawFB * kVDP1FBSize * hrScale * hrScale;
    for (uint p = 0; p < 2; ++p) {
        const uint raw = BitExtract(word, p * 16u, 16u);
        for (uint dy = 0; dy < hrScale; ++dy) {
            for (uint dx = 0; dx < hrScale; ++dx) {
                const uint address = base + ((ny * hrScale + dy) * rowPixels + (nx + p) * hrScale + dx) * 2u;
                const uint shift = (address & 2u) * 8u;
                uint dummy;
                g_fbramOut.InterlockedAnd(address & ~3u, ~(0xFFFFu << shift), dummy);
                g_fbramOut.InterlockedOr(address & ~3u, raw << shift, dummy);
            }
        }
    }
}
