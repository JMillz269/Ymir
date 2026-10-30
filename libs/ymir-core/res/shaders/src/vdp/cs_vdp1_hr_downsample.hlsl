#include "vdp1_defs.hlsli"
#include "vdp1_common_params.hlsli"

#include "util/bit_ops.hlsli"

// Fills the native framebuffer of the draw framebuffer index from the scaled up (internal resolution) one when only the
// scaled up one is drawn (Skip Native Drawing): every native pixel takes the first scaled up pixel it covers, which is
// what the composition of the scaled up image compares the other scaled up pixels with. One thread per two native
// pixels (the inverse of cs_vdp1_hr_resync).

cbuffer CommonRenderParamsBuffer : register(b0) {
    CommonRenderParams g_commonParams;
}

ByteAddressBuffer g_fbramHRIn : register(t1);

RWByteAddressBuffer g_fbramOut : register(u1);

static const uint hrScale = BitExtract(g_commonParams.enhancements, 2, 3) + 1u;
static const uint drawFB = BitExtract(g_commonParams.displayParams, 7, 1);

uint ReadHRPixel(uint address) {
    const uint word = g_fbramHRIn.Load(address & ~3u);
    return (address & 2u) != 0u ? BitExtract(word, 16, 16) : BitExtract(word, 0, 16);
}

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= 512u * 256u / 2u) {
        return;
    }

    const uint nx = (id.x * 2u) & 511u;
    const uint ny = (id.x * 2u) >> 9u;
    const uint rowPixels = 512u * hrScale;
    const uint rowAddress = drawFB * kVDP1FBSize * hrScale * hrScale + ny * hrScale * rowPixels * 2u;
    const uint raw0 = ReadHRPixel(rowAddress + nx * hrScale * 2u);
    const uint raw1 = ReadHRPixel(rowAddress + (nx + 1u) * hrScale * 2u);
    g_fbramOut.Store(drawFB * kVDP1FBSize + id.x * 4u, raw0 | (raw1 << 16u));
}
