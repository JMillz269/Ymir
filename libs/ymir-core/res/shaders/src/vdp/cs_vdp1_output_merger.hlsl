#include "vdp1_defs.hlsli"
#include "vdp1_common_params.hlsli"

#include "util/bit_ops.hlsli"
#include "util/data_ops.hlsli"

// Shader specialization macros:
// - POLYSPEC_TRANSPARENT_MESH:
//     0 = disabled (checkerboard)
//     1 = enabled (50% alpha); processes both normal and mesh pixels simultaneously
// - POLYSPEC_MERGE_MODE:
//     0 = Copy (Replace, Half-Luminance)
//     1 = Right-shift (Shadow)
//     2 = OIT (Half-Transparency)
//
// Implementation notes:
// - Works on 32-bit units at a time
// - The Z ID selects the field to draw for deinterlaced rendering (0=main, 1=alternate)

#define POLYSPEC_SHADING_MODE_COPY  0
#define POLYSPEC_SHADING_MODE_SHIFT 1
#define POLYSPEC_SHADING_MODE_OIT   2

// Modify these to adjust IntelliSense highlighting
#ifdef __INTELLISENSE__
#define POLYSPEC_TRANSPARENT_MESH 1
#define POLYSPEC_MERGE_MODE       0
#endif

// TODO: figure out how shadow and transparent pixels on the transparent mesh layer should be rendered
// - also broken on the software renderer

cbuffer RenderParamsBuffer : register(b0) {
    CommonRenderParams g_commonParams;
}

#if POLYSPEC_MERGE_MODE == POLYSPEC_SHADING_MODE_OIT
StructuredBuffer<OITFragment> g_fragments : register(t1);

RWByteAddressBuffer g_fbramOut : register(u1);
RWBuffer<uint> g_listHeads : register(u2);
#else
RWByteAddressBuffer g_fbramOut : register(u1);
RWBuffer<uint> g_internalSpriteOut : register(u2);
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Parameters

static const uint hrScale = BitExtract(g_commonParams.enhancements, 2, 3) + 1u;
static const uint2 fbSize = uint2(
    512u << BitExtract(g_commonParams.displayParams, 0, 1),
    256u << BitExtract(g_commonParams.displayParams, 1, 1)
) * hrScale;
static const uint fbBytes = kVDP1FBSize * hrScale * hrScale;
static const uint fbramBytes = fbBytes * 2;
static const bool pixel8Bits = BitTest(g_commonParams.displayParams, 2);
static const uint drawFB = BitExtract(g_commonParams.displayParams, 7, 1);

static const uint fbOffset = drawFB * fbBytes;

// The transparent mesh outputs and framebuffers come after the alternate field's at the native size, and right after
// the main ones in the scaled up passes (which never draw both fields)
static const uint meshPlane = hrScale > 1 ? 1u : 2u;
static const uint meshFBOffset = fbramBytes * meshPlane;

// ---------------------------------------------------------------------------------------------------------------------
// Mergers

#if POLYSPEC_MERGE_MODE == POLYSPEC_SHADING_MODE_COPY
// ----------------------------------------------------------------------------
// Copy (Replace, Half-Luminance)

void Merge8(uint2 pos, uint field) {
    const uint2 inPos = uint2(pos.x * 4, pos.y);
    const uint inOffset = inPos.x + inPos.y * fbSize.x + field * fbSize.x * fbSize.y;

    // Read internal outputs
    const uint out0 = g_internalSpriteOut[inOffset + 0];
    const uint out1 = g_internalSpriteOut[inOffset + 1];
    const uint out2 = g_internalSpriteOut[inOffset + 2];
    const uint out3 = g_internalSpriteOut[inOffset + 3];

    // Get sequence numbers
    const uint seqNum0 = BitExtract(out0, 16, 16);
    const uint seqNum1 = BitExtract(out1, 16, 16);
    const uint seqNum2 = BitExtract(out2, 16, 16);
    const uint seqNum3 = BitExtract(out3, 16, 16);
    bool hasPixel0 = seqNum0 != 0;
    bool hasPixel1 = seqNum1 != 0;
    bool hasPixel2 = seqNum2 != 0;
    bool hasPixel3 = seqNum3 != 0;
#if POLYSPEC_TRANSPARENT_MESH
    const uint inMeshOffset = inOffset + fbSize.x * fbSize.y * meshPlane;
    const uint meshOut0 = g_internalSpriteOut[inMeshOffset + 0];
    const uint meshOut1 = g_internalSpriteOut[inMeshOffset + 1];
    const uint meshOut2 = g_internalSpriteOut[inMeshOffset + 2];
    const uint meshOut3 = g_internalSpriteOut[inMeshOffset + 3];
    const uint meshSeqNum0 = BitExtract(meshOut0, 16, 16);
    const uint meshSeqNum1 = BitExtract(meshOut1, 16, 16);
    const uint meshSeqNum2 = BitExtract(meshOut2, 16, 16);
    const uint meshSeqNum3 = BitExtract(meshOut3, 16, 16);
    hasPixel0 = hasPixel0 || meshSeqNum0 != 0;
    hasPixel1 = hasPixel1 || meshSeqNum1 != 0;
    hasPixel2 = hasPixel2 || meshSeqNum2 != 0;
    hasPixel3 = hasPixel3 || meshSeqNum3 != 0;
#endif
    if (!hasPixel0 && !hasPixel1 && !hasPixel2 && !hasPixel3) {
        // Nothing written to these pixels
        return;
    }

    // Clear internal outputs
    g_internalSpriteOut[inOffset + 0] = 0;
    g_internalSpriteOut[inOffset + 1] = 0;
    g_internalSpriteOut[inOffset + 2] = 0;
    g_internalSpriteOut[inOffset + 3] = 0;
#if POLYSPEC_TRANSPARENT_MESH
    g_internalSpriteOut[inMeshOffset + 0] = 0;
    g_internalSpriteOut[inMeshOffset + 1] = 0;
    g_internalSpriteOut[inMeshOffset + 2] = 0;
    g_internalSpriteOut[inMeshOffset + 3] = 0;
#endif

    const uint outOffset = inPos.x + inPos.y * fbSize.x + field * fbramBytes;
    uint fbramValue = g_fbramOut.Load(outOffset + fbOffset);
    if (seqNum0 != 0) {
        fbramValue &= ~0xFFu;
        fbramValue |= BitExtract(out0, 0, 8);
    }
    if (seqNum1 != 0) {
        fbramValue &= ~0xFF00u;
        fbramValue |= BitExtract(out1, 0, 8) << 8u;
    }
    if (seqNum2 != 0) {
        fbramValue &= ~0xFF0000u;
        fbramValue |= BitExtract(out2, 0, 8) << 16u;
    }
    if (seqNum3 != 0) {
        fbramValue &= ~0xFF000000u;
        fbramValue |= BitExtract(out3, 0, 8) << 24u;
    }
    g_fbramOut.Store(outOffset + fbOffset, fbramValue);
#if POLYSPEC_TRANSPARENT_MESH
    const uint outMeshOffset = outOffset + meshFBOffset;
    fbramValue = g_fbramOut.Load(outMeshOffset + fbOffset);
    if (hasPixel0) {
        fbramValue &= ~0xFFu;
        if (meshSeqNum0 > seqNum0) {
            fbramValue |= BitExtract(meshOut0, 0, 8);
        }
    }
    if (hasPixel1) {
        fbramValue &= ~0xFF00u;
        if (meshSeqNum1 > seqNum1) {
            fbramValue |= BitExtract(meshOut1, 0, 8) << 8u;
        }
    }
    if (hasPixel2) {
        fbramValue &= ~0xFF0000u;
        if (meshSeqNum2 > seqNum2) {
            fbramValue |= BitExtract(meshOut2, 0, 8) << 16u;
        }
    }
    if (hasPixel3) {
        fbramValue &= ~0xFF000000u;
        if (meshSeqNum3 > seqNum3) {
            fbramValue |= BitExtract(meshOut3, 0, 8) << 24u;
        }
    }
    g_fbramOut.Store(outMeshOffset + fbOffset, fbramValue);
#endif
}

void Merge16(uint2 pos, uint field) {
    const uint2 inPos = uint2(pos.x * 2, pos.y);
    const uint inOffset = inPos.x + inPos.y * fbSize.x + field * fbSize.x * fbSize.y;

    // Read internal outputs
    const uint out0 = g_internalSpriteOut[inOffset + 0];
    const uint out1 = g_internalSpriteOut[inOffset + 1];

    // Get sequence numbers
    const uint seqNum0 = BitExtract(out0, 16, 16);
    const uint seqNum1 = BitExtract(out1, 16, 16);
    bool hasPixel0 = seqNum0 != 0;
    bool hasPixel1 = seqNum1 != 0;
#if POLYSPEC_TRANSPARENT_MESH
    const uint inMeshOffset = inOffset + fbSize.x * fbSize.y * meshPlane;
    const uint meshOut0 = g_internalSpriteOut[inMeshOffset + 0];
    const uint meshOut1 = g_internalSpriteOut[inMeshOffset + 1];
    const uint meshSeqNum0 = BitExtract(meshOut0, 16, 16);
    const uint meshSeqNum1 = BitExtract(meshOut1, 16, 16);
    hasPixel0 = hasPixel0 || meshSeqNum0 != 0;
    hasPixel1 = hasPixel1 || meshSeqNum1 != 0;
#endif
    if (!hasPixel0 && !hasPixel1) {
        // Nothing written to these pixels
        return;
    }

    // Clear internal outputs
    g_internalSpriteOut[inOffset + 0] = 0;
    g_internalSpriteOut[inOffset + 1] = 0;
#if POLYSPEC_TRANSPARENT_MESH
    g_internalSpriteOut[inMeshOffset + 0] = 0;
    g_internalSpriteOut[inMeshOffset + 1] = 0;
#endif

    const uint outOffset = (inPos.x + inPos.y * fbSize.x) * 2 + field * fbramBytes;
    uint fbramValue = g_fbramOut.Load(outOffset + fbOffset);
    if (seqNum0 != 0) {
        fbramValue &= ~0xFFFFu;
        fbramValue |= ByteSwap16(out0);
    }
    if (seqNum1 != 0) {
        fbramValue &= ~0xFFFF0000u;
        fbramValue |= ByteSwap16(out1) << 16u;
    }
    g_fbramOut.Store(outOffset + fbOffset, fbramValue);
#if POLYSPEC_TRANSPARENT_MESH
    const uint outMeshOffset = outOffset + meshFBOffset;
    fbramValue = g_fbramOut.Load(outMeshOffset + fbOffset);
    if (hasPixel0) {
        fbramValue &= ~0xFFFFu;
        if (meshSeqNum0 > seqNum0) {
            fbramValue |= ByteSwap16(meshOut0);
        }
    }
    if (hasPixel1) {
        fbramValue &= ~0xFFFF0000u;
        if (meshSeqNum1 > seqNum1) {
            fbramValue |= ByteSwap16(meshOut1) << 16u;
        }
    }
    g_fbramOut.Store(outMeshOffset + fbOffset, fbramValue);
#endif
}

#elif POLYSPEC_MERGE_MODE == POLYSPEC_SHADING_MODE_SHIFT
// ----------------------------------------------------------------------------
// Right-shift (Shadow)

void Merge16Single(uint2 pos, uint field) {
    const uint2 inPos = uint2(pos.x * 2, pos.y);
    const uint inOffset = inPos.x + inPos.y * fbSize.x + field * fbSize.x * fbSize.y;

    // Read and clear internal outputs
    const uint shift0 = min(g_internalSpriteOut[inOffset + 0], 5);
    const uint shift1 = min(g_internalSpriteOut[inOffset + 1], 5);
    if (shift0 == 0 && shift1 == 0) {
        // Nothing written to these pixels
        return;
    }
    g_internalSpriteOut[inOffset + 0] = 0;
    g_internalSpriteOut[inOffset + 1] = 0;

    const uint outOffset = (inPos.x + inPos.y * fbSize.x) * 2 + field * fbramBytes;
    uint fbramValue = g_fbramOut.Load(outOffset + fbOffset);
    if (shift0 != 0) {
        uint4 color = Uint16ToColor555(ByteSwap16(fbramValue));
        if (color.a != 0u) {
            color.rgb >>= shift0;
            fbramValue &= ~0xFFFFu;
            fbramValue |= ByteSwap16(Color555ToUint16(color));
        }
    }
    if (shift1 != 0) {
        uint4 color = Uint16ToColor555(ByteSwap16(fbramValue >> 16u));
        if (color.a != 0u) {
            color.rgb >>= shift1;
            fbramValue &= ~0xFFFF0000u;
            fbramValue |= ByteSwap16(Color555ToUint16(color)) << 16u;
        }
    }
    g_fbramOut.Store(outOffset + fbOffset, fbramValue);
}

void Merge8(uint2 pos, uint field) {
    // Shadow does not apply to 8-bit mode.
}

#if POLYSPEC_TRANSPARENT_MESH
// Like the software renderer: a shadow mesh pixel stores the sprite pixel under it, halved if it's opaque, in the mesh
// framebuffer, and a shadow sprite pixel removes the mesh pixel under it. (The counts don't say which came last; the mesh
// pixels are taken as the later ones.)
void MergeMesh16(uint2 pos, uint field, uint2 shifts) {
    const uint2 inPos = uint2(pos.x * 2, pos.y);
    const uint inOffset = inPos.x + inPos.y * fbSize.x + field * fbSize.x * fbSize.y;
    const uint meshInOffset = inOffset + fbSize.x * fbSize.y * meshPlane;

    const uint shift0 = shifts[0];
    const uint shift1 = shifts[1];
    const uint meshShift0 = g_internalSpriteOut[meshInOffset + 0];
    const uint meshShift1 = g_internalSpriteOut[meshInOffset + 1];
    if (meshShift0 == 0 && meshShift1 == 0 && shift0 == 0 && shift1 == 0) {
        return;
    }
    g_internalSpriteOut[meshInOffset + 0] = 0;
    g_internalSpriteOut[meshInOffset + 1] = 0;

    // (the sprite pixels have been merged already)
    const uint outOffset = (inPos.x + inPos.y * fbSize.x) * 2 + field * fbramBytes;
    const uint spriteValue = g_fbramOut.Load(outOffset + fbOffset);
    const uint meshAddress = outOffset + meshFBOffset + fbOffset;
    uint meshValue = g_fbramOut.Load(meshAddress);
    [unroll]
    for (uint i = 0; i < 2; ++i) {
        const uint meshShift = i == 0 ? meshShift0 : meshShift1;
        const uint shift = i == 0 ? shift0 : shift1;
        if (meshShift == 0 && shift == 0) {
            continue;
        }
        uint value = 0;
        if (meshShift != 0) {
            uint4 color = Uint16ToColor555(ByteSwap16(spriteValue >> (i * 16u)));
            if (color.a != 0u) {
                color.rgb >>= 1u;
            }
            value = ByteSwap16(Color555ToUint16(color));
        }
        meshValue &= ~(0xFFFFu << (i * 16u));
        meshValue |= value << (i * 16u);
    }
    g_fbramOut.Store(meshAddress, meshValue);
}
#endif

void Merge16(uint2 pos, uint field) {
#if POLYSPEC_TRANSPARENT_MESH
    // (Merge16Single clears the counts of the sprite pixels)
    const uint inOffset = pos.x * 2 + pos.y * fbSize.x + field * fbSize.x * fbSize.y;
    const uint2 shifts = uint2(g_internalSpriteOut[inOffset + 0], g_internalSpriteOut[inOffset + 1]);
    Merge16Single(pos, field);
    MergeMesh16(pos, field, shifts);
#else
    Merge16Single(pos, field);
#endif
}

#elif POLYSPEC_MERGE_MODE == POLYSPEC_SHADING_MODE_OIT
// ----------------------------------------------------------------------------
// OIT (Half-Transparency)

uint4 HalfTransparentBlendOne(uint4 finalColor, uint fragmentData) {
    const uint4 fragColor = Uint16ToColor555(fragmentData);
    if (finalColor.a != 0) {
        finalColor.rgb = (finalColor.rgb + fragColor.rgb) >> 1u;
        return finalColor;
    }
    return fragColor;
}

uint HalfTransparentBlend(uint baseColor, uint listHead) {
    // Early exit if nothing was written to the pixel
    if (listHead == 0xFFFFFFFF) {
        return baseColor;
    }

    uint4 finalColor = Uint16ToColor555(baseColor);

    // Most pixels have a single fragment
    const OITFragment firstFrag = g_fragments[listHead];
    if (firstFrag.next == 0xFFFFFFFF) {
        return Color555ToUint16(HalfTransparentBlendOne(finalColor, firstFrag.data));
    }

    // Blend the fragments from the oldest to the latest, as the pixels would have been drawn: by sequence number, and
    // by slot (which follows the order of the pixels in the span) when they have the same one. The fragments are not
    // copied and sorted, which needs an array per thread that the GPU keeps in slow memory; each round looks through
    // the list for the next fragment instead.
    // TODO: what if the cap is exceeded?
    uint lastSeq = 0;
    uint lastNode = 0;
    for (uint round = 0; round < 32; ++round) {
        uint bestNode = 0xFFFFFFFF;
        uint bestSeq = 0;
        uint bestData = 0;
        uint curr = listHead;
        for (uint visited = 0; curr != 0xFFFFFFFF && visited < 32; ++visited) {
            const OITFragment frag = g_fragments[curr];
            const uint seq = BitExtract(frag.data, 16, 16);
            const bool after = seq > lastSeq || (seq == lastSeq && curr > lastNode);
            const bool better = bestNode == 0xFFFFFFFF || seq < bestSeq || (seq == bestSeq && curr < bestNode);
            if (after && better) {
                bestNode = curr;
                bestSeq = seq;
                bestData = frag.data;
            }
            curr = frag.next;
        }
        if (bestNode == 0xFFFFFFFF) {
            break;
        }
        finalColor = HalfTransparentBlendOne(finalColor, bestData);
        lastSeq = bestSeq;
        lastNode = bestNode;
    }

    return Color555ToUint16(finalColor);
}

// The sequence number of the latest fragment of the list (0 if empty)
uint LatestSeq(uint listHead) {
    uint latest = 0;
    uint curr = listHead;
    for (uint visited = 0; curr != 0xFFFFFFFF && visited < 32; ++visited) {
        const OITFragment frag = g_fragments[curr];
        latest = max(latest, BitExtract(frag.data, 16, 16));
        curr = frag.next;
    }
    return latest;
}

void Merge16Single(uint2 pos, uint field) {
    const uint2 inPos = uint2(pos.x * 2, pos.y);
    const uint inOffset = inPos.x + inPos.y * fbSize.x + field * fbSize.x * fbSize.y;

    const uint2 heads = uint2(
        g_listHeads[inOffset + 0],
        g_listHeads[inOffset + 1]
    );

    // Early exit if nothing was written to either pixel
    if (all(heads == 0xFFFFFFFF)) {
        return;
    }

    // Clear heads
    g_listHeads[inOffset + 0] = 0xFFFFFFFF;
    g_listHeads[inOffset + 1] = 0xFFFFFFFF;

    // Get base FBRAM value
    const uint fbramAddress = (inPos.x + inPos.y * fbSize.x) * 2 + fbOffset + field * fbramBytes;
    uint fbramValue = g_fbramOut.Load(fbramAddress);

    // Modify
    fbramValue =
        (ByteSwap16(HalfTransparentBlend(ByteSwap16(fbramValue >> 16u), heads[1])) << 16u) |
         ByteSwap16(HalfTransparentBlend(ByteSwap16(fbramValue), heads[0]));

    // Write back
    g_fbramOut.Store(fbramAddress, fbramValue);
}

void Merge8(uint2 pos, uint field) {
    // Half-Transparency does not apply to 8-bit mode.
}

#if POLYSPEC_TRANSPARENT_MESH
// Like the software renderer: a half-transparent mesh pixel is blended with the sprite pixel under it and stored in the
// mesh framebuffer (so only the latest one counts), and a sprite pixel removes the mesh pixel under it
void MergeMesh16(uint2 pos, uint field, uint2 latestSprite) {
    const uint2 inPos = uint2(pos.x * 2, pos.y);
    const uint meshInOffset = inPos.x + inPos.y * fbSize.x + (field + meshPlane) * fbSize.x * fbSize.y;
    const uint2 heads = uint2(g_listHeads[meshInOffset + 0], g_listHeads[meshInOffset + 1]);
    if (all(heads == 0xFFFFFFFF) && all(latestSprite == 0)) {
        return;
    }
    g_listHeads[meshInOffset + 0] = 0xFFFFFFFF;
    g_listHeads[meshInOffset + 1] = 0xFFFFFFFF;

    // (the sprite pixels have been merged already)
    const uint outOffset = (inPos.x + inPos.y * fbSize.x) * 2 + field * fbramBytes;
    const uint spriteValue = g_fbramOut.Load(outOffset + fbOffset);
    const uint meshAddress = outOffset + meshFBOffset + fbOffset;
    uint meshValue = g_fbramOut.Load(meshAddress);
    [unroll]
    for (uint i = 0; i < 2; ++i) {
        // The latest mesh fragment
        uint latestMesh = 0;
        uint latestData = 0;
        uint curr = heads[i];
        for (uint visited = 0; curr != 0xFFFFFFFF && visited < 32; ++visited) {
            const OITFragment frag = g_fragments[curr];
            const uint seq = BitExtract(frag.data, 16, 16);
            if (seq >= latestMesh) {
                latestMesh = seq;
                latestData = frag.data;
            }
            curr = frag.next;
        }
        if (latestMesh == 0 && latestSprite[i] == 0) {
            continue;
        }
        uint value = 0;
        if (latestMesh > latestSprite[i]) {
            const uint4 sprite = Uint16ToColor555(ByteSwap16(spriteValue >> (i * 16u)));
            value = ByteSwap16(Color555ToUint16(HalfTransparentBlendOne(sprite, latestData)));
        }
        meshValue &= ~(0xFFFFu << (i * 16u));
        meshValue |= value << (i * 16u);
    }
    g_fbramOut.Store(meshAddress, meshValue);
}
#endif

void Merge16(uint2 pos, uint field) {
#if POLYSPEC_TRANSPARENT_MESH
    const uint inOffset = pos.x * 2 + pos.y * fbSize.x + field * fbSize.x * fbSize.y;
    const uint2 latest = uint2(LatestSeq(g_listHeads[inOffset + 0]), LatestSeq(g_listHeads[inOffset + 1]));
    Merge16Single(pos, field);
    MergeMesh16(pos, field, latest);
#else
    Merge16Single(pos, field);
#endif
}

#endif

// ---------------------------------------------------------------------------------------------------------------------
// Entrypoint

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    const uint2 pos = id.xy + g_commonParams.mergeOrigin;
    if (pixel8Bits) {
        Merge8(pos, id.z);
    } else {
        Merge16(pos, id.z);
    }
}
