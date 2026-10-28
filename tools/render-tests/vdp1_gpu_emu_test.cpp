// Checks the algorithm of the GPU VDP1 path without a GPU.
//
// The compute shaders (cs_vdp1_polydraw, cs_vdp1_output_merger, cs_vdp1_erase, cs_vdp1_fbram_write) are ported to C++
// below and plugged into MetalVDP1 in place of a GPU. The same scene is drawn by the software renderer and the
// framebuffers are compared. Threads of the polygon drawing shader run in random order, like on a GPU.
//
// Usage: vdp1_gpu_emu_test [seed] [scale]
//
// With a scale above 1 the scaled up framebuffers (internal resolution) are compared with those of the software renderer.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// (the scaled up framebuffers of the software renderer are what the GPU's are compared with)
#define private public
#include <ymir/hw/vdp/renderer/vdp_renderer_sw.hpp>
#undef private

#include "ymir/hw/vdp/renderer/metal/metal_vdp1.hpp"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <vector>

using namespace ymir;
using namespace ymir::vdp;

namespace ymir::vdp::metal {
void LogMessage(const char *format, ...) {
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    std::fputc('\n', stderr);
    va_end(args);
}
} // namespace ymir::vdp::metal

namespace {

uint32 BitExtract(uint32 value, uint32 offset, uint32 length) {
    return (value >> offset) & ((1u << length) - 1u);
}
bool BitTest(uint32 value, uint32 bit) {
    return ((value >> bit) & 1) != 0;
}
uint32 ByteSwap16(uint32 v) {
    return ((v >> 8) & 0xFF) | ((v << 8) & 0xFF00);
}

// Port of DivFloor (float division corrected to the exact quotient)
static int DivFloor(int a, int b) {
    int q = int(float(a) / float(b));
    int r = a - q * b;
    if (r < 0) {
        --q;
        r += b;
        if (r < 0) {
            --q;
            r += b;
        }
    } else if (r >= b) {
        ++q;
        r -= b;
        if (r >= b) {
            ++q;
        }
    }
    return q;
}

struct Color555x {
    uint32 r, g, b, a;
};
Color555x Uint16ToColor555(uint32 raw) {
    return {BitExtract(raw, 0, 5), BitExtract(raw, 5, 5), BitExtract(raw, 10, 5), BitExtract(raw, 15, 1)};
}
uint32 Color555ToUint16(Color555x c) {
    return c.r | (c.g << 5) | (c.b << 10) | (c.a << 15);
}

struct TextureStepperH {
    int num, den, accum, value, inc, baseAccum, baseValue;
    void Setup(uint32 length, int start, int end, bool hss = false, int hssSelect = 0) {
        if (hss) {
            start >>= 1;
            end >>= 1;
        }
        const int delta = end - start;
        const uint32 absDelta = std::abs(delta);
        value = start;
        inc = delta >= 0 ? +1 : -1;
        if (hss) {
            value <<= 1;
            value |= hssSelect;
            inc <<= 1;
        }
        num = absDelta;
        den = length;
        if (length <= absDelta) {
            ++num;
            accum = absDelta - (length << 1);
            if (delta >= 0) {
                ++accum;
            }
        } else {
            --den;
            accum = length - (length << 1);
            if (delta < 0) {
                ++accum;
            }
        }
        num <<= 1;
        den <<= 1;
        baseAccum = accum;
        baseValue = value;
    }
    uint32 Value() const {
        return value;
    }
    void SetPixel(uint32 step) {
        accum = baseAccum + num * step;
        value = baseValue;
        if (accum >= 0) {
            const int count = DivFloor(accum, den) + 1;
            value += inc * count;
            accum -= den * count;
        }
    }
};

struct GouraudChannelH {
    int num, den, accum, value, intInc, fracInc, baseValue, baseAccum;
    void Setup(uint32 length, int start, int end) {
        const int delta = end - start;
        const uint32 absDelta = std::abs(delta);
        value = start;
        intInc = 0;
        fracInc = delta >= 0 ? +1 : -1;
        num = absDelta;
        den = length;
        if (length <= absDelta) {
            ++num;
            accum = absDelta - (length << 1);
            if (delta >= 0) {
                ++accum;
            }
        } else {
            --den;
            accum = -int(length);
            if (delta < 0) {
                ++accum;
            }
        }
        num <<= 1;
        den <<= 1;
        if (den != 0) {
            if (accum >= 0) {
                const int count = DivFloor(accum, den) + 1;
                value += fracInc * count;
                accum -= den * count;
            }
            if (num >= den) {
                const int count = DivFloor(num, den);
                intInc += fracInc * count;
                num -= den * count;
            }
        }
        accum = ~accum;
        baseValue = value;
        baseAccum = accum;
    }
    void Skip(int steps) {
        value += intInc * steps;
        accum -= num * steps;
        if (den != 0 && accum < 0) {
            const int count = DivFloor(den - 1 - accum, den);
            value += fracInc * count;
            accum += den * count;
        }
    }
    uint32 Blend(int color) const {
        return std::clamp(value + color - 16, 0, 31);
    }
};

struct LineStepperH {
    int num, den, accum, accumTarget;
    int majIncX, majIncY, minIncX, minIncY;
    int posX, posY, startX, startY;
    uint32 dmaj, step;
    int aaIncX, aaIncY;

    void Setup(int c1x, int c1y, int c2x, int c2y, bool antiAlias) {
        posX = c1x;
        posY = c1y;
        startX = c1x;
        startY = c1y;
        int dx = c2x - c1x, dy = c2y - c1y;
        int adx = std::abs(dx), ady = std::abs(dy);
        dmaj = std::max(adx, ady);
        step = 0;
        const bool xMajor = adx >= ady;
        if (xMajor) {
            majIncX = dx >= 0 ? +1 : -1;
            majIncY = 0;
            minIncX = 0;
            minIncY = dy >= 0 ? +1 : -1;
        } else {
            majIncX = 0;
            majIncY = dy >= 0 ? +1 : -1;
            minIncX = dx >= 0 ? +1 : -1;
            minIncY = 0;
            std::swap(dx, dy);
            std::swap(adx, ady);
        }
        num = ady << 1;
        den = adx << 1;
        accum = adx + 1;
        accumTarget = 0;
        if (!antiAlias && dx < 0) {
            ++accumTarget;
        }
        accum += num;
        posX -= majIncX;
        posY -= majIncY;
        aaIncX = aaIncY = 0;
        if (antiAlias) {
            --accum;
            --accumTarget;
            const bool samesign = (c1x > c2x) == (c1y > c2y);
            if (xMajor) {
                aaIncX = samesign ? 0 : -majIncX;
                aaIncY = samesign ? -minIncY : 0;
            } else {
                aaIncX = samesign ? 0 : -minIncX;
                aaIncY = samesign ? -majIncY : 0;
            }
        }
    }
    void SetStep(uint32 targetStep) {
        targetStep = std::min(targetStep, dmaj);
        const int stepDelta = targetStep + 1 - step;
        if (stepDelta == 0) {
            return;
        }
        step = targetStep + 1;
        posX += majIncX * stepDelta;
        posY += majIncY * stepDelta;
        accum -= num * stepDelta;
        if (den != 0) {
            const int count = DivFloor(accumTarget - accum + den, den);
            accum += den * count;
            posX += minIncX * count;
            posY += minIncY * count;
        }
    }
    bool NeedsAA() const {
        return step > 1 && accum - den + num > accumTarget;
    }
};

struct EmuGpu final : metal::Gpu {
    std::vector<uint8> vram = std::vector<uint8>(kVDP1VRAMSize);
    std::vector<uint8> fbram = std::vector<uint8>(kVDP1FBRAMSize * 2 * 2 * 2);
    std::vector<uint8> fbramHR = std::vector<uint8>(kVDP1FBRAMSize * 32);
    uint8 *cur = nullptr; // framebuffer memory of the pass being run
    uint32 hrScale = 1;   // scale of the pass being run
    std::vector<uint32> spriteOut = std::vector<uint32>(1u << 21);
    std::vector<uint32> listHeads = std::vector<uint32>(1u << 21, 0xFFFFFFFF);
    struct Frag {
        uint32 data, next;
    };
    std::vector<Frag> frags = std::vector<Frag>(kMaxVDP1OITFragmentsPerDispatch);
    std::mt19937 rng{1};

    bool EnableVdp1(std::string &) override {
        return true;
    }
    void Vdp1UploadVRAM(uint32 o, const void *d, uint32 s) override {
        std::memcpy(&vram[o], d, s);
    }
    void Vdp1UploadFBRAM(uint32 o, const void *d, uint32 s) override {
        std::memcpy(&fbram[o], d, s);
    }
    void UploadVRAM(uint32, const void *, uint32) override {}
    void UploadSpriteFB(uint32, const void *, uint32) override {}
    void RenderLayers(const metal::LayerPassParams &) override {}
    void Compose(const metal::ComposePassParams &) override {}
    void ComposeHR(const metal::LayerPassParams &, const metal::ComposePassParams &, uint32, uint32, uint32, uint32,
                   uint32) override {}
    metal::FinishedFrame FinishFrame(uint32, uint32, uint32) override {
        return {};
    }
    void Vdp1DebugPixelBuffers(uint32 &, uint32 &, uint32 &, uint32 &) override {}
    void Vdp1DownloadVRAM(void *dst, uint32 size) override {
        std::memcpy(dst, vram.data(), size);
    }
    void Vdp1DownloadFBRAM(void *dst, uint32 size) override {
        std::memcpy(dst, fbram.data(), size);
    }

    uint32 FbLoad(uint32 address) const {
        uint32 v;
        std::memcpy(&v, &cur[address & ~3u], 4);
        return v;
    }
    void FbStore(uint32 address, uint32 v) {
        std::memcpy(&cur[address & ~3u], &v, 4);
    }
    void Select(bool hr, const void *constants) {
        VDP1CommonRenderParams common;
        std::memcpy(&common, constants, sizeof(common));
        uint32 enh;
        std::memcpy(&enh, &common.enhancements, 4);
        hrScale = hr ? BitExtract(enh, 2, 3) + 1 : 1;
        cur = hr ? fbramHR.data() : fbram.data();
    }

    bool Vdp1EnableHR(std::string &) override {
        return true;
    }
    void Vdp1DownloadHRFBRAM(void *dst, uint32 size) override {
        std::memcpy(dst, fbramHR.data(), size);
    }
    void Vdp1ResyncHR(const metal::Vdp1ResyncPassParams &p) override {
        Select(true, p.constants);
        VDP1CommonRenderParams common;
        std::memcpy(&common, p.constants, sizeof(common));
        uint32 dp;
        std::memcpy(&dp, &common.displayParams, 4);
        const uint32 drawFB = BitExtract(dp, 7, 1);
        for (uint32 id = 0; id < 512 * 256 / 2; ++id) {
            uint32 word;
            std::memcpy(&word, &fbram[drawFB * 256 * 1024 + id * 4], 4);
            const uint32 nx = (id * 2) & 511, ny = (id * 2) >> 9;
            const uint32 raw0 = BitExtract(word, 0, 16), raw1 = BitExtract(word, 16, 16);
            for (uint32 dy = 0; dy < hrScale; ++dy) {
                const uint32 rowAddress = drawFB * 256 * 1024 * hrScale * hrScale +
                                          ((ny * hrScale + dy) * 512 * hrScale + nx * hrScale) * 2;
                for (uint32 m = 0; m < hrScale; ++m) {
                    const uint32 lo = (m * 2) < hrScale ? raw0 : raw1;
                    const uint32 hi = (m * 2 + 1) < hrScale ? raw0 : raw1;
                    FbStore(rowAddress + m * 4, lo | (hi << 16));
                }
            }
        }
    }

    uint32 ReadVRAM8(uint32 a) const {
        return vram[a & 0x7FFFF];
    }
    uint32 ReadVRAM16(uint32 a) const {
        a &= 0x7FFFE;
        return (uint32(vram[a]) << 8) | vram[a + 1];
    }

    // ---- Erase ----
    void Vdp1Erase(const metal::Vdp1ErasePassParams &p) override {
        VDP1CommonRenderParams common;
        VDP1EraseParams erase;
        std::memcpy(&common, p.constants, sizeof(common));
        std::memcpy(&erase, (const uint8 *)p.constants + sizeof(common), sizeof(erase));
        uint32 dp, ev, coords, vb;
        std::memcpy(&dp, &common.displayParams, 4);
        std::memcpy(&coords, &erase.coords, 4);
        std::memcpy(&ev, &erase.erase, 4);
        std::memcpy(&vb, &erase.vblank, 4);
        Select(p.hr, p.constants);
        const uint32 drawFB = BitExtract(dp, 7, 1);
        const uint32 drawFBOffset = (drawFB ^ 1) * 256 * 1024 * hrScale * hrScale;
        const bool vblankErase = BitTest(vb, 0);
        const uint32 vblankEraseMaxY = BitExtract(vb, 1, 9);
        const uint32 vblankEraseMaxX = BitExtract(vb, 10, 10);
        const uint32 addressShift = BitExtract(ev, 16, 1) + 8u;
        const uint32 eraseScaleV = BitExtract(coords, 31, 1);
        const uint32 eraseX1 = BitExtract(coords, 0, 6) << 3u;
        const uint32 eraseY1 = BitExtract(coords, 6, 9) << eraseScaleV;
        const uint32 eraseX3 = BitExtract(coords, 15, 7) << 3u;
        const uint32 eraseY3 = BitExtract(coords, 22, 9) << eraseScaleV;
        for (uint32 idy = 0; idy < 32 * ((p.height + 31) / 32); ++idy) {
            for (uint32 idx = 0; idx < 32 * ((p.width + 63) / 64); ++idx) {
                if (hrScale > 1) {
                    const uint32 hx = idx * 2 + eraseX1 * hrScale, hy = idy + eraseY1 * hrScale;
                    if (hx >= eraseX3 * hrScale || hy >= (eraseY3 + 1) * hrScale) {
                        continue;
                    }
                    if (vblankErase) {
                        const uint32 nx = hx / hrScale, ny = hy / hrScale;
                        if (ny > vblankEraseMaxY || (ny == vblankEraseMaxY && nx > vblankEraseMaxX)) {
                            continue;
                        }
                    }
                    const uint32 wv = BitExtract(ev, 0, 16);
                    FbStore(drawFBOffset + (hy * 512u * hrScale + hx) * 2, (wv << 16) | wv);
                    continue;
                }
                const uint32 posx = idx * 2 + eraseX1, posy = idy + eraseY1;
                if (posx >= eraseX3 + 1 || posy > eraseY3 + 1) {
                    continue;
                }
                bool partialWrite = false;
                if (vblankErase) {
                    if (posy > vblankEraseMaxY) {
                        continue;
                    }
                    if (posy == vblankEraseMaxY && posx > vblankEraseMaxX) {
                        continue;
                    }
                    partialWrite = posy == vblankEraseMaxY && posx + 1 > vblankEraseMaxX;
                }
                const uint32 address = drawFBOffset + ((posy << addressShift) + posx) * 2;
                const uint32 writeValue = BitExtract(ev, 0, 16);
                uint32 value;
                if (partialWrite) {
                    value = FbLoad(address);
                    value &= ~0xFFFFu;
                    value |= writeValue;
                } else {
                    value = (writeValue << 16u) | writeValue;
                }
                FbStore(address, value);
            }
        }
    }

    void Vdp1WriteFB(const metal::Vdp1FBWritePassParams &p) override {
        VDP1CommonRenderParams common;
        std::memcpy(&common, p.constants, sizeof(common));
        uint32 dp;
        std::memcpy(&dp, &common.displayParams, 4);
        const uint32 drawFBOffset = BitExtract(dp, 7, 1) * 256 * 1024;
        const VDP1FBRAMWrite *w = (const VDP1FBRAMWrite *)p.writes;
        if (p.hr) {
            Select(true, p.constants);
            const uint32 drawFB = BitExtract(dp, 7, 1);
            for (uint32 id = 0; id < p.count * 2; ++id) {
                const uint32 index = id >> 1, px = id & 1;
                const uint32 mask = BitExtract(w[index].andMask, px * 16, 16);
                if (mask == 0xFFFF) {
                    continue;
                }
                const uint32 nativeAddress = w[index].address + px * 2;
                uint32 word;
                std::memcpy(&word, &fbram[drawFB * 256 * 1024 + (nativeAddress & ~3u)], 4);
                const uint32 raw = BitExtract(word, (nativeAddress & 2) * 8, 16);
                const uint32 pixelIndex = nativeAddress >> 1;
                const uint32 nx = pixelIndex & 511, ny = pixelIndex >> 9;
                if (ny >= 256) {
                    continue;
                }
                for (uint32 dy = 0; dy < hrScale; ++dy) {
                    for (uint32 dx = 0; dx < hrScale; ++dx) {
                        const uint32 address = drawFB * 256 * 1024 * hrScale * hrScale +
                                               ((ny * hrScale + dy) * 512 * hrScale + nx * hrScale + dx) * 2;
                        const uint32 shift = (address & 2) * 8;
                        uint32 v = FbLoad(address & ~3u);
                        v &= ~(0xFFFFu << shift);
                        v |= raw << shift;
                        FbStore(address & ~3u, v);
                    }
                }
            }
            return;
        }
        Select(false, p.constants);
        for (uint32 i = 0; i < p.count; ++i) {
            const uint32 address = w[i].address + drawFBOffset;
            uint32 value = FbLoad(address);
            value &= w[i].andMask;
            value |= w[i].orMask;
            FbStore(address, value);
        }
    }

    // ---- Polygon drawing ----
    uint32 numSpans;
    const VDP1SpanParams *spans;
    const uint32 *prefix;
    const uint32 *blockStarts;
    uint32 tableMismatches = 0;
    const VDP1CommandParams *cmds;
    uint32 dpBits;
    uint32 shading;

    void WriteOutput(int cx, int cy, uint32 cmdIndex, uint32 value, uint32 fragmentSlot) {
        const VDP1CommandParams &cp = cmds[cmdIndex];
        uint32 sysClipRaw, uc0, uc1, pmodcolr;
        std::memcpy(&sysClipRaw, &cp.sysClip, 4);
        std::memcpy(&uc0, &cp.userClip0, 4);
        std::memcpy(&uc1, &cp.userClip1, 4);
        const uint32 cmdpmod = cp.cmdpmod, cmdcolr = cp.cmdcolr;
        pmodcolr = cmdpmod | (cmdcolr << 16);
        const uint32 sysX = BitExtract(sysClipRaw, 0, 16), sysY = BitExtract(sysClipRaw, 16, 16);
        if (cx < 0 || cy < 0 || uint32(cx) > sysX || uint32(cy) > sysY) {
            return;
        }
        if (BitTest(pmodcolr, 10)) {
            const bool clippingMode = BitTest(pmodcolr, 9);
            const uint32 u0x = BitExtract(uc0, 0, 16), u0y = BitExtract(uc0, 16, 16);
            const uint32 u1x = BitExtract(uc1, 0, 16), u1y = BitExtract(uc1, 16, 16);
            if (((uint32(cx) < u0x || uint32(cy) < u0y || uint32(cx) > u1x || uint32(cy) > u1y)) != clippingMode) {
                return;
            }
        }
        const bool meshEnable = BitTest(pmodcolr, 8);
        if (meshEnable && (((cx / int(hrScale)) ^ (cy / int(hrScale))) & 1)) {
            return;
        }
        if (hrScale > 1 && (cx >= int(512 * hrScale) || cy >= int(256 * hrScale))) {
            return;
        }
        const bool doubleDensity = BitTest(dpBits, 3);
        const bool dblInterlaceEnable = BitTest(dpBits, 4);
        const uint32 dblInterlaceDrawLine = BitExtract(dpBits, 5, 1);
        const uint32 fbW = (512u << BitExtract(dpBits, 0, 1)) * hrScale;
        const uint32 fbOffsetFB = BitExtract(dpBits, 7, 1) * 256 * 1024 * hrScale * hrScale;
        if (doubleDensity && dblInterlaceEnable && (uint32(cy) & 1) != dblInterlaceDrawLine) {
            return;
        }
        if (dblInterlaceEnable) {
            cy >>= 1;
        }
        uint32 outOffset = cy * fbW + cx;
        const bool pixel8Bits = BitTest(dpBits, 2);
        if (shading == 0) {
            spriteOut[outOffset] = std::max(spriteOut[outOffset], value);
        } else if (shading == 1) {
            spriteOut[outOffset] += 1;
        } else if (shading == 2) {
            const uint32 node = fragmentSlot;
            Frag f;
            f.data = value;
            f.next = listHeads[outOffset];
            listHeads[outOffset] = node;
            frags[node] = f;
        } else {
            if (pixel8Bits) {
                outOffset &= ~1u;
            } else {
                outOffset <<= 1u;
            }
            uint32 a = outOffset + fbOffsetFB;
            uint32 v = ByteSwap16(0x8000) << ((a & 2) * 8);
            FbStore(a & ~3u, FbLoad(a & ~3u) | v);
        }
    }
    uint32 nextNode = 0;

    void PolyThread(uint32 id) {
        // GetSpanIndex
        if (id >= prefix[numSpans]) {
            return;
        }
        uint32 lb = 0, ub = numSpans, spanIndex = 0xFFFFFFFF;
        while (lb != ub) {
            const uint32 mid = (lb + ub) >> 1u;
            const uint32 value = prefix[mid];
            if (id == value) {
                spanIndex = mid;
                break;
            }
            if (id > value) {
                lb = mid + 1u;
            } else {
                ub = mid;
            }
        }
        if (spanIndex == 0xFFFFFFFF) {
            spanIndex = lb - 1u;
        }
        // The shader finds the span through the table of the spans of the first pixels of groups of 64
        {
            uint32 span = blockStarts[id >> 6];
            while (prefix[span + 1] <= id) {
                ++span;
            }
            if (span != spanIndex) {
                std::printf("span table mismatch at pixel %u: %u, search %u\n", id, span, spanIndex);
                ++tableMismatches;
            }
        }

        const VDP1SpanParams &span = spans[spanIndex];
        const uint32 spanStep = id - prefix[spanIndex] + span.skip;
        const uint32 cmdIndex = span.cmdIndex;
        const VDP1CommandParams &cp = cmds[cmdIndex];
        const uint32 pmodcolr = cp.cmdpmod | (uint32(cp.cmdcolr) << 16);
        const uint32 cmdsizesrca = cp.cmdsize | (uint32(cp.cmdsrca) << 16);
        uint32 attrs;
        std::memcpy(&attrs, &span.attrs, 4);
        const bool antialias = BitTest(attrs, 0);
        const bool textured = BitTest(attrs, 1);
        const uint32 cmdcolr = BitExtract(pmodcolr, 16, 16);
        const bool pixel8Bits = BitTest(dpBits, 2);

        LineStepperH ls;
        ls.Setup(span.coord0.x, span.coord0.y, span.coord1.x, span.coord1.y, antialias);
        ls.SetStep(spanStep);

        uint32 spriteData;
        if (textured) {
            const uint32 charSizeH = std::max(BitExtract(cmdsizesrca, 8, 6) << 3, 1u);
            const bool flipH = BitTest(attrs, 2);
            const uint32 colorMode = BitExtract(pmodcolr, 3, 3);
            const bool transparentPixelDisable = BitTest(pmodcolr, 6);
            const bool endCodesEnabled = !BitTest(pmodcolr, 7);
            const bool useHighSpeedShrink = BitTest(pmodcolr, 12) && ls.dmaj < charSizeH - 1;
            const bool evenOddCoordSelect = BitTest(dpBits, 6);
            int uStart = 0, uEnd = charSizeH - 1;
            if (flipH) {
                std::swap(uStart, uEnd);
            }
            TextureStepperH us;
            us.Setup(ls.dmaj + 1, uStart, uEnd, useHighSpeedShrink, evenOddCoordSelect);
            us.SetPixel(spanStep);
            uint32 endCodeIndex = 0;
            bool checkEndCodes;
            if (endCodesEnabled && !useHighSpeedShrink) {
                endCodeIndex = BitExtract(attrs, 13, 10);
                checkEndCodes = endCodeIndex < charSizeH;
            } else {
                checkEndCodes = false;
            }
            const uint32 texU = us.Value();
            if (checkEndCodes && (flipH ? (texU <= endCodeIndex) : (texU >= endCodeIndex))) {
                return;
            }
            const uint32 texV = BitExtract(attrs, 3, 10);
            const uint32 charAddr = BitExtract(cmdsizesrca, 16, 16) << 3u;
            uint32 color = 0;
            bool transparent = false, hasEndCode = false;
            const uint32 charIndex = texU + texV * charSizeH;
            switch (colorMode) {
            case 0:
                color = ReadVRAM8(charAddr + (charIndex >> 1));
                color = (color >> ((~texU & 1) * 4)) & 0xF;
                hasEndCode = color == 0xF;
                transparent = color == 0x0;
                color |= cmdcolr & 0xFFF0;
                break;
            case 1:
                color = ReadVRAM8(charAddr + (charIndex >> 1));
                color = (color >> ((~texU & 1) * 4)) & 0xF;
                hasEndCode = color == 0xF;
                transparent = color == 0x0;
                color = ReadVRAM16(color * 2 + cmdcolr * 8);
                break;
            case 2:
                color = ReadVRAM8(charAddr + charIndex);
                transparent = color == 0x00;
                hasEndCode = color == 0xFF;
                color &= 0x3F;
                color |= cmdcolr & 0xFFC0;
                break;
            case 3:
                color = ReadVRAM8(charAddr + charIndex);
                transparent = color == 0x00;
                hasEndCode = color == 0xFF;
                color &= 0x7F;
                color |= cmdcolr & 0xFF80;
                break;
            case 4:
                color = ReadVRAM8(charAddr + charIndex);
                transparent = color == 0x00;
                hasEndCode = color == 0xFF;
                color |= cmdcolr & 0xFF00;
                break;
            case 5:
                color = ReadVRAM16((charAddr & ~0xFu) + charIndex * 2);
                transparent = !BitTest(color, 15);
                hasEndCode = color == 0x7FFF;
                break;
            }
            spriteData = color;
            if ((hasEndCode && endCodesEnabled) || (transparent && !transparentPixelDisable)) {
                return;
            }
        } else {
            spriteData = cmdcolr;
            if (pixel8Bits) {
                spriteData &= 0xFFu;
            }
        }

        uint32 value = 0;
        if (shading == 0 || shading == 2) {
            const uint32 shadingMode = BitExtract(pmodcolr, 0, 2);
            const bool gouraudEnable = BitTest(pmodcolr, 2);
            if (!pixel8Bits && (gouraudEnable || shadingMode == 2)) {
                Color555x src = Uint16ToColor555(spriteData);
                if (gouraudEnable) {
                    GouraudChannelH r, g, b;
                    r.Setup(ls.dmaj + 1, span.gouraud0.r, span.gouraud1.r);
                    g.Setup(ls.dmaj + 1, span.gouraud0.g, span.gouraud1.g);
                    b.Setup(ls.dmaj + 1, span.gouraud0.b, span.gouraud1.b);
                    if (int(spanStep) > 0) {
                        r.Skip(spanStep);
                        g.Skip(spanStep);
                        b.Skip(spanStep);
                    }
                    src.r = r.Blend(src.r);
                    src.g = g.Blend(src.g);
                    src.b = b.Blend(src.b);
                }
                if (shadingMode == 2) {
                    src.r >>= 1;
                    src.g >>= 1;
                    src.b >>= 1;
                }
                spriteData = Color555ToUint16(src);
            }
            value = spriteData | ((spanIndex + 1u) << 16u);
        }
        WriteOutput(ls.posX, ls.posY, cmdIndex, value, id * 2);
        if (antialias && ls.NeedsAA()) {
            WriteOutput(ls.posX + ls.aaIncX, ls.posY + ls.aaIncY, cmdIndex, value, id * 2 + 1);
        }
    }

    // ---- Output merger ----
    static Color555x HalfBlendOne(Color555x finalColor, uint32 data) {
        Color555x fc = Uint16ToColor555(data);
        if (finalColor.a != 0) {
            finalColor.r = (finalColor.r + fc.r) >> 1u;
            finalColor.g = (finalColor.g + fc.g) >> 1u;
            finalColor.b = (finalColor.b + fc.b) >> 1u;
            return finalColor;
        }
        return fc;
    }
    uint32 HalfTransparentBlend(uint32 baseColor, uint32 listHead) {
        if (listHead == 0xFFFFFFFF) {
            return baseColor;
        }
        Color555x finalColor = Uint16ToColor555(baseColor);
        if (frags[listHead].next == 0xFFFFFFFF) {
            return Color555ToUint16(HalfBlendOne(finalColor, frags[listHead].data));
        }
        uint32 lastSeq = 0, lastNode = 0;
        for (uint32 round = 0; round < 32; ++round) {
            uint32 bestNode = 0xFFFFFFFF, bestSeq = 0, bestData = 0;
            uint32 curr = listHead;
            for (uint32 visited = 0; curr != 0xFFFFFFFF && visited < 32; ++visited) {
                const Frag &frag = frags[curr];
                const uint32 seq = BitExtract(frag.data, 16, 16);
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
            finalColor = HalfBlendOne(finalColor, bestData);
            lastSeq = bestSeq;
            lastNode = bestNode;
        }
        return Color555ToUint16(finalColor);
    }

    void MergeThread(uint32 px, uint32 py, uint32 mode) {
        const bool pixel8Bits = BitTest(dpBits, 2);
        const uint32 fbW = (512u << BitExtract(dpBits, 0, 1)) * hrScale;
        const uint32 fbOffset = BitExtract(dpBits, 7, 1) * 256 * 1024 * hrScale * hrScale;
        if (pixel8Bits) {
            if (mode != 0) {
                return;
            }
            const uint32 inx = px * 4, iny = py;
            const uint32 inOffset = inx + iny * fbW;
            uint32 out[4];
            bool has[4];
            bool any = false;
            for (int i = 0; i < 4; ++i) {
                out[i] = spriteOut[inOffset + i];
                has[i] = BitExtract(out[i], 16, 16) != 0;
                any |= has[i];
            }
            if (!any) {
                return;
            }
            for (int i = 0; i < 4; ++i) {
                spriteOut[inOffset + i] = 0;
            }
            const uint32 outOffset = inx + iny * fbW;
            uint32 v = FbLoad(outOffset + fbOffset);
            for (int i = 0; i < 4; ++i) {
                if (has[i]) {
                    v &= ~(0xFFu << (i * 8));
                    v |= BitExtract(out[i], 0, 8) << (i * 8);
                }
            }
            FbStore(outOffset + fbOffset, v);
            return;
        }
        const uint32 inx = px * 2, iny = py;
        const uint32 inOffset = inx + iny * fbW;
        const uint32 outOffset = (inx + iny * fbW) * 2;
        if (mode == 0) {
            uint32 o0 = spriteOut[inOffset], o1 = spriteOut[inOffset + 1];
            const bool h0 = BitExtract(o0, 16, 16) != 0, h1 = BitExtract(o1, 16, 16) != 0;
            if (!h0 && !h1) {
                return;
            }
            spriteOut[inOffset] = 0;
            spriteOut[inOffset + 1] = 0;
            uint32 v = FbLoad(outOffset + fbOffset);
            if (h0) {
                v &= ~0xFFFFu;
                v |= ByteSwap16(o0);
            }
            if (h1) {
                v &= ~0xFFFF0000u;
                v |= ByteSwap16(o1) << 16u;
            }
            FbStore(outOffset + fbOffset, v);
        } else if (mode == 1) {
            const uint32 s0 = std::min(spriteOut[inOffset], 5u), s1 = std::min(spriteOut[inOffset + 1], 5u);
            if (s0 == 0 && s1 == 0) {
                return;
            }
            spriteOut[inOffset] = 0;
            spriteOut[inOffset + 1] = 0;
            uint32 v = FbLoad(outOffset + fbOffset);
            if (s0 != 0) {
                Color555x c = Uint16ToColor555(ByteSwap16(v));
                if (c.a != 0) {
                    c.r >>= s0;
                    c.g >>= s0;
                    c.b >>= s0;
                    v &= ~0xFFFFu;
                    v |= ByteSwap16(Color555ToUint16(c));
                }
            }
            if (s1 != 0) {
                Color555x c = Uint16ToColor555(ByteSwap16(v >> 16u));
                if (c.a != 0) {
                    c.r >>= s1;
                    c.g >>= s1;
                    c.b >>= s1;
                    v &= ~0xFFFF0000u;
                    v |= ByteSwap16(Color555ToUint16(c)) << 16u;
                }
            }
            FbStore(outOffset + fbOffset, v);
        } else {
            const uint32 h0 = listHeads[inOffset], h1 = listHeads[inOffset + 1];
            if (h0 == 0xFFFFFFFF && h1 == 0xFFFFFFFF) {
                return;
            }
            listHeads[inOffset] = 0xFFFFFFFF;
            listHeads[inOffset + 1] = 0xFFFFFFFF;
            uint32 v = FbLoad(outOffset + fbOffset);
            v = (ByteSwap16(HalfTransparentBlend(ByteSwap16(v >> 16u), h1)) << 16u) |
                ByteSwap16(HalfTransparentBlend(ByteSwap16(v), h0));
            FbStore(outOffset + fbOffset, v);
        }
    }

    void Vdp1Draw(const metal::Vdp1DrawParams &p) override {
        VDP1CommonRenderParams common;
        std::memcpy(&common, p.constants, sizeof(common));
        std::memcpy(&dpBits, &common.displayParams, 4);
        Select(p.hr, p.constants);
        VDP1PolyDrawParams pd;
        std::memcpy(&pd, (const uint8 *)p.constants + sizeof(common), sizeof(pd));
        numSpans = pd.numSpans;
        spans = (const VDP1SpanParams *)p.spans;
        prefix = (const uint32 *)p.prefixSums;
        blockStarts = (const uint32 *)p.spanBlockStarts;
        if (p.spanBlockStartsSize < ((p.numPixels + 63) / 64) * 4) {
            std::printf("span table too small: %u bytes for %u pixels\n", p.spanBlockStartsSize, p.numPixels);
            ++tableMismatches;
            return;
        }
        cmds = (const VDP1CommandParams *)p.commands;
        shading = p.shadingMode;
        nextNode = 0;

        std::vector<uint32> order(p.numPixels);
        for (uint32 i = 0; i < p.numPixels; ++i) {
            order[i] = i;
        }
        std::shuffle(order.begin(), order.end(), rng);
        for (uint32 id : order) {
            PolyThread(id);
        }
        if (shading != 3) {
            for (uint32 y = 0; y < p.mergeHeight; ++y) {
                for (uint32 x = 0; x < p.mergeWidth; ++x) {
                    MergeThread(x + common.mergeOriginX, y + common.mergeOriginY, p.mergeMode);
                }
            }
        }
    }
};

void W16(auto &mem, uint32 addr, uint16 v) {
    util::WriteBE<uint16>(&mem[addr], v);
}

} // namespace

int main(int argc, char **argv) {
    const uint32 seed = argc > 1 ? std::atoi(argv[1]) : 1;
    const uint32 scale = argc > 2 ? std::atoi(argv[2]) : 1;
    auto stateSW = std::make_unique<VDPState>();
    auto stateGPU = std::make_unique<VDPState>();
    config::VDP2DebugRender dbg;
    config::VDP2AccessPatternsConfig ap;
    SoftwareVDPRenderer sw(*stateSW, dbg, ap);
    sw.EnableThreadedVDP1(false);
    sw.EnableThreadedVDP2(false);
    EmuGpu gpu;
    gpu.rng.seed(seed);
    MetalVDP1 vdp1(*stateGPU, gpu);
    {
        config::Enhancements enh{};
        enh.internalResolution = scale;
        sw.ConfigureEnhancements(enh);
        vdp1.SetInternalResolution(scale);
    }

    auto w2 = [&](uint32 addr, uint16 v) {
        stateSW->regs2.Write(addr, v);
        stateGPU->regs2.Write(addr, v);
    };
    auto w1 = [&](uint32 addr, uint16 v) {
        stateSW->regs1.Write<false>(addr, v);
        sw.VDP1WriteReg(addr, v);
        stateGPU->regs1.Write<false>(addr, v);
        vdp1.WriteReg();
    };
    w2(0x000, 0x8000);
    stateSW->regs2.LatchTVMD();
    stateGPU->regs2.LatchTVMD();
    sw.VDP2SetResolution(320, 224, false);
    vdp1.SetVRes(224);
    stateSW->state2.UpdateEnabledBGs(stateSW->regs2, dbg);
    stateGPU->state2.UpdateEnabledBGs(stateGPU->regs2, dbg);

    w1(0x000, 0x0000);
    w1(0x006, 0x0000);
    w1(0x008, 0x0000);
    w1(0x00A, (319 / 8) << 9 | 223);
    stateSW->regs1.LatchEraseParameters();
    stateGPU->regs1.LatchEraseParameters();

    auto vw16 = [&](uint32 addr, uint16 v) {
        W16(stateSW->mem1.VRAM, addr, v);
        W16(stateGPU->mem1.VRAM, addr, v);
    };
    // textures: 16bpp RGB gradient 16x16 at 0x20000, 4bpp-lut 16x16 at 0x21000 with table at 0x22000
    for (int y = 0; y < 16; y++) {
        for (int x = 0; x < 16; x++) {
            vw16(0x20000 + (y * 16 + x) * 2, 0x8000 | ((x * 2) << 10) | ((y * 2) << 5) | ((x ^ y) & 31));
        }
    }
    for (int i = 0; i < 16; i++) {
        vw16(0x22000 + i * 2, i == 0 ? 0 : (0x8000 | (i << 10) | ((15 - i) << 5) | (i * 2)));
    }
    for (int i = 0; i < 128; i++) {
        stateSW->mem1.VRAM[0x21000 + i] = stateGPU->mem1.VRAM[0x21000 + i] = uint8(((i * 3) & 0xF) << 4 | ((i * 5) & 0xF));
    }

    uint32 cmd = 0;
    auto put = [&](uint16 ctrl, uint16 link, uint16 pmod, uint16 colr, uint16 srca, uint16 size, sint16 xa, sint16 ya,
                   sint16 xb, sint16 yb, sint16 xc, sint16 yc, sint16 xd, sint16 yd, uint16 gouraud = 0) {
        vw16(cmd + 0x00, ctrl);
        vw16(cmd + 0x02, link);
        vw16(cmd + 0x04, pmod);
        vw16(cmd + 0x06, colr);
        vw16(cmd + 0x08, srca);
        vw16(cmd + 0x0A, size);
        vw16(cmd + 0x0C, xa);
        vw16(cmd + 0x0E, ya);
        vw16(cmd + 0x10, xb);
        vw16(cmd + 0x12, yb);
        vw16(cmd + 0x14, xc);
        vw16(cmd + 0x16, yc);
        vw16(cmd + 0x18, xd);
        vw16(cmd + 0x1A, yd);
        vw16(cmd + 0x1C, gouraud);
        cmd += 0x20;
    };
    // gouraud table at 0x23000
    for (int i = 0; i < 4; i++) {
        vw16(0x23000 + i * 2, (i * 7 + 3) | ((i * 5 + 9) << 5) | ((31 - i * 6) << 10));
    }
    // wide texture 144x32 16bpp at 0x24000 with some transparent texels
    for (int y = 0; y < 32; y++) {
        for (int x = 0; x < 144; x++) {
            const bool hole = ((x * 7 + y * 3) % 23) == 0;
            vw16(0x24000 + (y * 144 + x) * 2, hole ? 0 : (0x8000 | ((x % 32) << 10) | ((y * 3 % 32) << 5) | ((x + y) % 32)));
        }
    }
    put(0x0009, 0, 0, 0, 0, 0, 0, 0, 0, 0, 319, 223, 0, 0);
    put(0x000A, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    // huge quads reaching far beyond the screen (a ground plane close to the camera), textured and flat, drawn
    // under everything else
    // (edges about 2000 pixels long with coordinates within the 11 bits the VDP1 draws at, so beyond 4095 pixels at
    // 3x and 4x)
    put(0x0002, 0, 0x0028, 0, 0x20000 >> 3, (2 << 8) | 16, -1000, 150, 1000, 150, 1020, 1000, -1020, 1000);
    put(0x0004, 0, 0x0000, 0x8210, 0, 0, -1000, -1000, 1000, -1010, 1000, 60, -1000, 40);
    // opaque base: textured rotated sprites and polygons
    put(0x0002, 0, 0x0028, 0, 0x20000 >> 3, (2 << 8) | 16, 20, 20, 150, 40, 130, 150, 30, 120);
    put(0x0004, 0, 0x0000, 0x83E0, 0, 0, 100, 60, 220, 60, 220, 140, 100, 140);
    put(0x0004, 0, 0x0004, 0x801F, 0, 0, 40, 100, 120, 100, 120, 190, 40, 190, 0x23000 >> 3);
    // half-transparent textured sprites, overlapping each other and the base, rotated and axis aligned
    put(0x0002, 0, 0x002B, 0, 0x20000 >> 3, (2 << 8) | 16, 10, 10, 160, 30, 150, 170, 20, 150);
    put(0x0002, 0, 0x002B, 0, 0x20000 >> 3, (2 << 8) | 16, 90, 50, 250, 80, 240, 200, 100, 170);
    put(0x0001, 0, 0x002B, 0, 0x20000 >> 3, (2 << 8) | 16, 150, 20, 0, 0, 40, 60, 0, 0);
    // half-transparent flat polygon + gouraud half-transparent
    put(0x0004, 0, 0x0003, 0x8000 | (31 << 5), 0, 0, 60, 80, 200, 90, 190, 180, 70, 170);
    put(0x0004, 0, 0x0007, 0x801F, 0, 0, 120, 30, 280, 60, 260, 120, 130, 100, 0x23000 >> 3);
    // shadow polygon over everything
    put(0x0004, 0, 0x0001, 0x8000, 0, 0, 50, 120, 280, 130, 270, 210, 60, 200);
    // half-luminance sprite, lookup table sprite, msb polygon
    put(0x0002, 0, 0x002A, 0, 0x20000 >> 3, (2 << 8) | 16, 180, 140, 300, 150, 290, 215, 190, 210);
    put(0x0002, 0, 0x0009, 0x2200 >> 0, 0x21000 >> 3, (2 << 8) | 16, 10, 150, 90, 160, 80, 215, 20, 205);
    put(0x0004, 0, 0x8000, 0, 0, 0, 200, 20, 260, 20, 260, 60, 200, 60);
    // wide sprites partly off screen (clipped at the left and the top), opaque and half-transparent, scaled and not
    put(0x0002, 0, 0x0028, 0, 0x24000 >> 3, (18 << 8) | 32, -72, 10, 71, 10, 71, 41, -72, 41);
    put(0x0002, 0, 0x002B, 0, 0x24000 >> 3, (18 << 8) | 32, -72, 20, 71, 20, 71, 51, -72, 51);
    put(0x0002, 0, 0x002B, 0, 0x24000 >> 3, (18 << 8) | 32, -40, -10, 100, 5, 90, 60, -50, 40);
    put(0x0002, 0, 0x002B, 0, 0x24000 >> 3, (18 << 8) | 32, 30, 120, 170, 100, 170, 200, 30, 190);
    put(0x0002, 0x0010, 0x002B, 0, 0x24000 >> 3, (18 << 8) | 32, 200, 100, 300, 100, 300, 130, 200, 130);
    // shapes one pixel thin in the free strip at the bottom (y 216..223): polygons one pixel tall, one pixel wide and
    // a single pixel, a sprite one texel tall, a distorted sprite with all vertices on one line, lines and a polyline
    put(0x0004, 0, 0x0000, 0x83FF, 0, 0, 5, 218, 60, 218, 60, 218, 5, 218);
    put(0x0004, 0, 0x0000, 0xFC1F, 0, 0, 64, 216, 64, 216, 64, 222, 64, 222);
    put(0x0004, 0, 0x0000, 0xFFE0, 0, 0, 70, 219, 70, 219, 70, 219, 70, 219);
    put(0x0000, 0, 0x0028, 0, 0x20000 >> 3, (2 << 8) | 1, 80, 218, 0, 0, 0, 0, 0, 0);
    put(0x0002, 0, 0x0028, 0, 0x20000 >> 3, (2 << 8) | 16, 100, 220, 160, 220, 160, 220, 100, 220);
    put(0x0006, 0, 0x0000, 0x801F, 0, 0, 170, 217, 230, 217, 0, 0, 0, 0);
    put(0x0006, 0, 0x0000, 0x83E0, 0, 0, 233, 216, 233, 223, 0, 0, 0, 0);
    put(0x0005, 0, 0x0000, 0xFFFF, 0, 0, 240, 217, 260, 217, 260, 222, 240, 222);
    put(0x0006, 0, 0x0000, 0xFC00, 0, 0, 270, 222, 315, 217, 0, 0, 0, 0);
    // lines
    put(0x0006, 0, 0x0003, 0xFFFF, 0, 0, 10, 210, 310, 100, 0, 0, 0, 0);
    put(0x0005, 0, 0x0000, 0x7FFF, 0, 0, 220, 170, 300, 170, 300, 215, 220, 215);
    put(0x8000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    vdp1.ResyncFromCPU();
    if (std::getenv("WARMUP")) {
        // Two erase + swap cycles with nothing drawn so that the scaled up framebuffers are in line when drawing starts
        for (int i = 0; i < 2; i++) {
            sw.VDP1EraseFramebuffer(0);
            vdp1.EraseFramebuffer(0);
            sw.VDP1SwapFramebuffer();
            vdp1.SwapFramebuffer();
            stateSW->fbIndex.Flip();
            stateGPU->fbIndex.Flip();
        }
    }
    int badFrames = 0;
    for (int frame = 0; frame < 4; frame++) {
        // (the erase of the third frame runs out of time)
        const uint64 eraseCycles = frame == 2 ? 20000 : 0;
        sw.VDP1EraseFramebuffer(eraseCycles);
        vdp1.EraseFramebuffer(eraseCycles);
        sw.VDP1BeginFrame();
        vdp1.BeginFrame();
        int cmdNumber = 0;
        for (uint32 a = 0;; a += 0x20) {
            VDP1Command::Control ctrl{.u16 = util::ReadBE<uint16>(&stateSW->mem1.VRAM[a])};
            if (ctrl.end) {
                break;
            }
            if (cmdNumber++ == 9 && frame >= 1) {
                // The CPU writes to the framebuffer while drawing: a word, then a byte, then a word at an odd pixel
                const uint8 fb = stateGPU->fbIndex.draw;
                auto cpuWrite16 = [&](uint32 addr, uint16 v) {
                    util::WriteBE<uint16>(&stateSW->mem1.FBRAM[fb][addr], v);
                    util::WriteBE<uint16>(&stateGPU->mem1.FBRAM[fb][addr], v);
                    sw.VDP1WriteFB(addr, v);
                    vdp1.WriteFB(addr, 2);
                };
                auto cpuWrite8 = [&](uint32 addr, uint8 v) {
                    stateSW->mem1.FBRAM[fb][addr] = v;
                    stateGPU->mem1.FBRAM[fb][addr] = v;
                    sw.VDP1WriteFB(addr, v);
                    vdp1.WriteFB(addr, 1);
                };
                for (uint32 i = 0; i < 40; i++) {
                    cpuWrite16(((100 + i / 8) * 512 + 60 + i) * 2, uint16(0xFC00 + i * 37));
                }
                cpuWrite8(((120) * 512 + 80) * 2 + 1, 0x5A);
                cpuWrite8(((121) * 512 + 81) * 2, 0xC3);
                cpuWrite16(((130) * 512 + 99) * 2, 0x8421);
            }
            if (!ctrl.skip) {
                sw.VDP1ExecuteCommand(a, ctrl);
                vdp1.ExecuteCommand(a, ctrl);
            }
        }
        sw.VDP1EndFrame();
        vdp1.FlushWork();
        const uint8 fbi = stateGPU->fbIndex.draw;
        uint32 diffs = 0, minx = ~0u, miny = ~0u, maxx = 0, maxy = 0;
        for (uint32 y = 0; y < 224; ++y) {
            for (uint32 x = 0; x < 320; ++x) {
                const size_t o = (size_t(y) * 512 + x) * 2;
                const uint8 *g = &gpu.fbram[fbi * 256 * 1024 + o];
                const uint8 *s = &stateSW->mem1.FBRAM[fbi][o];
                if (std::memcmp(g, s, 2) != 0) {
                    if (diffs < 8) {
                        std::printf("  (%u,%u): GPU %02X%02X sw %02X%02X\n", x, y, g[0], g[1], s[0], s[1]);
                    }
                    ++diffs;
                    minx = std::min(minx, x);
                    maxx = std::max(maxx, x);
                    miny = std::min(miny, y);
                    maxy = std::max(maxy, y);
                }
            }
        }
        uint32 filled = 0;
        for (uint32 y = 0; y < 224; ++y) {
            for (uint32 x = 0; x < 320; ++x) {
                const uint8 *s = &stateSW->mem1.FBRAM[fbi][(size_t(y) * 512 + x) * 2];
                filled += (s[0] | s[1]) != 0;
            }
        }
        std::printf("frame %d (FB %u, %u pixels drawn): %u pixels differ", frame, fbi, filled, diffs);
        if (diffs) {
            std::printf(" in x %u..%u y %u..%u", minx, maxx, miny, maxy);
            ++badFrames;
        }
        std::printf("\n");
        if (scale > 1) {
            // The scaled up framebuffer of the draw framebuffer
            const uint16 *swHr = sw.m_hrDraw[fbi].data();
            const size_t stride = size_t(512) * scale;
            uint32 hdiffs = 0, hfilled = 0, hminx = ~0u, hminy = ~0u, hmaxx = 0, hmaxy = 0;
            for (uint32 y = 0; y < 224 * scale; ++y) {
                for (uint32 x = 0; x < 320 * scale; ++x) {
                    const uint8 *g = &gpu.fbramHR[fbi * 256 * 1024 * scale * scale + (size_t(y) * stride + x) * 2];
                    const uint16 gv = (uint16(g[0]) << 8) | g[1];
                    const uint16 sv = swHr[size_t(y) * stride + x];
                    hfilled += sv != 0;
                    if (gv != sv) {
                        if (hdiffs < 8) {
                            std::printf("  hr (%u,%u): GPU %04X sw %04X\n", x, y, gv, sv);
                        }
                        ++hdiffs;
                        hminx = std::min(hminx, x);
                        hmaxx = std::max(hmaxx, x);
                        hminy = std::min(hminy, y);
                        hmaxy = std::max(hmaxy, y);
                    }
                }
            }
            // Every line where the scaled up framebuffer is not the native one scaled up has to be one the renderer
            // reports as changed (VDP2 only composes those again)
            {
                uint32 first = 0, last = 0;
                const bool any = vdp1.HRDirtyLines(fbi, first, last);
                uint32 outside = 0;
                for (uint32 y = 0; y < 224 * scale; ++y) {
                    for (uint32 x = 0; x < 320 * scale; ++x) {
                        const uint8 *g = &gpu.fbramHR[fbi * 256 * 1024 * scale * scale + (size_t(y) * stride + x) * 2];
                        const uint8 *n = &gpu.fbram[fbi * 256 * 1024 + (size_t(y / scale) * 512 + x / scale) * 2];
                        if (std::memcmp(g, n, 2) != 0 && !(any && y / scale >= first && y / scale <= last)) {
                            ++outside;
                        }
                    }
                }
                if (outside) {
                    std::printf("  %u pixels differ from the native ones scaled up outside the reported lines\n", outside);
                    ++badFrames;
                }
            }
            // Shapes one native pixel thin cover all of their native pixels' scaled up pixels (checked on the axis
            // aligned ones in the bottom strip)
            {
                uint32 holes = 0;
                for (uint32 y = 216; y < 224; ++y) {
                    for (uint32 x = 0; x < 265; ++x) {
                        const uint8 *n = &gpu.fbram[fbi * 256 * 1024 + (size_t(y) * 512 + x) * 2];
                        if ((n[0] | n[1]) == 0) {
                            continue;
                        }
                        for (uint32 sy = 0; sy < scale; ++sy) {
                            for (uint32 sx = 0; sx < scale; ++sx) {
                                const uint8 *g = &gpu.fbramHR[fbi * 256 * 1024 * scale * scale +
                                                              (size_t(y * scale + sy) * stride + x * scale + sx) * 2];
                                holes += (g[0] | g[1]) == 0;
                            }
                        }
                    }
                }
                if (holes) {
                    std::printf("  %u scaled up pixels of thin shapes are not drawn\n", holes);
                    ++badFrames;
                }
            }
            if (std::getenv("DUMP") && frame == 0) {
                // Writes the native (scaled up by repetition), GPU and software scaled up framebuffers as PPM images
                auto dump = [&](const char *name, auto pixelAt) {
                    FILE *f = std::fopen(name, "wb");
                    std::fprintf(f, "P6 %u %u 255\n", 320 * scale, 224 * scale);
                    for (uint32 y = 0; y < 224 * scale; ++y) {
                        for (uint32 x = 0; x < 320 * scale; ++x) {
                            const uint16 v = pixelAt(x, y);
                            const uint8 rgb[3] = {uint8((v & 31) << 3), uint8(((v >> 5) & 31) << 3),
                                                  uint8(((v >> 10) & 31) << 3)};
                            std::fwrite(rgb, 1, 3, f);
                        }
                    }
                    std::fclose(f);
                };
                dump("dump_native.ppm", [&](uint32 x, uint32 y) {
                    const uint8 *n = &stateSW->mem1.FBRAM[fbi][(size_t(y / scale) * 512 + x / scale) * 2];
                    return uint16((uint16(n[0]) << 8) | n[1]);
                });
                dump("dump_gpu_native.ppm", [&](uint32 x, uint32 y) {
                    const uint8 *n = &gpu.fbram[fbi * 256 * 1024 + (size_t(y / scale) * 512 + x / scale) * 2];
                    return uint16((uint16(n[0]) << 8) | n[1]);
                });
                dump("dump_gpu_hr.ppm", [&](uint32 x, uint32 y) {
                    const uint8 *g = &gpu.fbramHR[fbi * 256 * 1024 * scale * scale + (size_t(y) * stride + x) * 2];
                    return uint16((uint16(g[0]) << 8) | g[1]);
                });
                dump("dump_sw_hr.ppm", [&](uint32 x, uint32 y) { return swHr[size_t(y) * stride + x]; });
            }
            std::printf("  scaled up %ux (%u pixels drawn): %u pixels differ", scale, hfilled, hdiffs);
            if (hdiffs) {
                std::printf(" in x %u..%u y %u..%u", hminx, hmaxx, hminy, hmaxy);
                ++badFrames;
            }
            std::printf("\n");
        }
        sw.VDP1SwapFramebuffer();
        vdp1.SwapFramebuffer();
        stateSW->fbIndex.Flip();
        stateGPU->fbIndex.Flip();
    }
    if (gpu.tableMismatches) {
        std::printf("%u span table mismatches\n", gpu.tableMismatches);
        ++badFrames;
    }
    return badFrames;
}
