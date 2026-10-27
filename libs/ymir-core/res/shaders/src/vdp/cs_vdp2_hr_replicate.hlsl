#include "vdp2_common_params.hlsli"

#include "util/bit_ops.hlsli"

// Internal resolution: fills the lines of the scaled up output image from the native image, every pixel of which covers
// scale x scale pixels of the scaled up one. Pixels whose sprite data differs in the scaled up VDP1 framebuffer are
// composed again afterwards.

cbuffer CommonRenderParams : register(b0) {
    CommonRenderParams g_commonParams;
}

Texture2D<float4> g_compositeNative : register(t1);

RWTexture2D<float4> g_compositeOut : register(u1);

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    const uint hrScale = BitExtract(g_commonParams.enhancements, 2, 3) + 1u;
    const uint2 hrPos = uint2(id.x, id.y + g_commonParams.startY * hrScale);
    g_compositeOut[hrPos] = g_compositeNative[hrPos / hrScale];
}
