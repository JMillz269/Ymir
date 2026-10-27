#include "vdp1_defs.hlsli"
#include "vdp1_common_params.hlsli"
#include "vdp1_write_params.hlsli"

#include "util/bit_ops.hlsli"

// Applies CPU writes to the VDP1 framebuffer to the scaled up (internal resolution) framebuffer: every pixel that was
// written is read back from the native framebuffer (where the writes have been applied already) and replicated over the
// scaled up pixels it covers.

struct FBRAMWriteParams {
    uint writeCount;
};

cbuffer CommonRenderParamsBuffer : register(b0) {
    CommonRenderParams g_commonParams;
    FBRAMWriteParams g_writeParams;
}

StructuredBuffer<FBRAMWrite> g_fbramWrites : register(t1);
ByteAddressBuffer g_fbramNative : register(t2);

RWByteAddressBuffer g_fbramOut : register(u1);

static const uint hrScale = BitExtract(g_commonParams.enhancements, 2, 3) + 1u;
static const uint drawFB = BitExtract(g_commonParams.displayParams, 7, 1);

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    const uint index = id.x >> 1u;
    const uint px = id.x & 1u;
    if (index >= g_writeParams.writeCount) {
        return;
    }

    const FBRAMWrite write = g_fbramWrites[index];
    const uint mask = BitExtract(write.andMask, px * 16u, 16u);
    if (mask == 0xFFFFu) {
        // This pixel wasn't written
        return;
    }

    // Raw (big-endian) 16 bits of the pixel in the native framebuffer
    const uint nativeAddress = write.address + px * 2u;
    const uint raw = BitExtract(g_fbramNative.Load(nativeAddress & ~3u) , (nativeAddress & 2u) * 8u, 16u);

    const uint pixelIndex = nativeAddress >> 1u;
    const uint nx = pixelIndex & 511u;
    const uint ny = pixelIndex >> 9u;
    if (ny >= 256u) {
        return;
    }

    const uint rowPixels = 512u * hrScale;
    const uint base = drawFB * kVDP1FBSize * hrScale * hrScale;
    for (uint dy = 0; dy < hrScale; ++dy) {
        for (uint dx = 0; dx < hrScale; ++dx) {
            const uint address = base + ((ny * hrScale + dy) * rowPixels + nx * hrScale + dx) * 2u;
            const uint shift = (address & 2u) * 8u;
            uint dummy;
            g_fbramOut.InterlockedAnd(address & ~3u, ~(0xFFFFu << shift), dummy);
            g_fbramOut.InterlockedOr(address & ~3u, raw << shift, dummy);
        }
    }
}
