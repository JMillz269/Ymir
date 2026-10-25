// Headless test for the software renderer's internal resolution upscaling.
//
// Renders a synthetic VDP1 scene (flat polygon, textured distorted sprite, lines) through the software VDP renderer at
// the requested scale and writes the frame as a binary PPM. With scale 1 the result is the regular native image.
//
// Usage: hr_render_test <scale> <output.ppm> [--time]
#include <memory>

#include <ymir/hw/vdp/renderer/vdp_renderer_sw.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace ymir;
using namespace ymir::vdp;

static std::vector<uint32> g_frame;
static uint32 g_w, g_h, g_scale;

static void W16(auto &mem, uint32 addr, uint16 v) {
    util::WriteBE<uint16>(&mem[addr], v);
}

int main(int argc, char **argv) {
    const uint32 scale = argc > 1 ? std::atoi(argv[1]) : 2;
    const char *outPath = argc > 2 ? argv[2] : "out.ppm";
    const bool timing = argc > 3 && !std::strcmp(argv[3], "--time");

    VDPState state;
    config::VDP2DebugRender dbg;
    config::VDP2AccessPatternsConfig ap;
    SoftwareVDPRenderer r(state, dbg, ap);

    config::Enhancements enh;
    enh.internalResolution = scale;
    enh.deinterlace = std::getenv("HR_DEINTERLACE") != nullptr;
    enh.transparentMeshes = std::getenv("HR_MESH") != nullptr;
    r.ConfigureEnhancements(enh);

    r.SwCallbacks.FrameComplete = {nullptr, [](uint32 *fb, uint32 w, uint32 h, uint32 s, void *) {
                                       g_w = w;
                                       g_h = h;
                                       g_scale = s;
                                       g_frame.assign(fb, fb + size_t(w) * s * h * s);
                                   }};

    auto w2 = [&](uint32 addr, uint16 v) {
        state.regs2.Write(addr, v);
        r.VDP2WriteReg(addr, v);
    };
    auto w1 = [&](uint32 addr, uint16 v) {
        state.regs1.Write<false>(addr, v);
        r.VDP1WriteReg(addr, v);
    };

    // VDP2: 320x224, display on, sprite layer only (RGB sprites with MSB set), back screen in VRAM at 0x1000
    w2(0x000, 0x8000);
    w2(0x00E, 0x0000);
    w2(0x020, 0x0000);
    w2(0x0AC, 0x0000);
    w2(0x0AE, 0x0800); // BKTAL: address 0x1000 (in words * 2)
    W16(state.mem2.VRAM, 0x1000, 0x5000);
    state.mem2.VRAM[0x1000] = 0x50;
    w2(0x0E0, 0x0020);
    w2(0x0F0, 0x0707);
    state.regs2.LatchTVMD();
    r.VDP2LatchTVMD();
    r.VDP2SetResolution(320, 224, false);

    // VDP1: erase to transparent, then draw
    w1(0x000, 0x0000);
    w1(0x006, 0x0000);                     // EWDR
    w1(0x008, 0x0000);                     // EWLR
    w1(0x00A, (319 / 8) << 9 | 223);       // EWRR
    state.regs1.LatchEraseParameters();

    // texture: 8x8 RGB checker at VRAM 0x20000
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            W16(state.mem1.VRAM, 0x20000 + (y * 8 + x) * 2, ((x ^ y) & 1) ? 0xFFE0 : 0x83E0);
        }
    }

    uint32 cmd = 0;
    auto put = [&](uint16 ctrl, uint16 link, uint16 pmod, uint16 colr, uint16 srca, uint16 size, sint16 xa, sint16 ya,
                   sint16 xb, sint16 yb, sint16 xc, sint16 yc, sint16 xd, sint16 yd) {
        auto &m = state.mem1.VRAM;
        W16(m, cmd + 0x00, ctrl);
        W16(m, cmd + 0x02, link);
        W16(m, cmd + 0x04, pmod);
        W16(m, cmd + 0x06, colr);
        W16(m, cmd + 0x08, srca);
        W16(m, cmd + 0x0A, size);
        W16(m, cmd + 0x0C, xa);
        W16(m, cmd + 0x0E, ya);
        W16(m, cmd + 0x10, xb);
        W16(m, cmd + 0x12, yb);
        W16(m, cmd + 0x14, xc);
        W16(m, cmd + 0x16, yc);
        W16(m, cmd + 0x18, xd);
        W16(m, cmd + 0x1A, yd);
        W16(m, cmd + 0x1C, 0);
        cmd += 0x20;
    };
    // system clip, local coords
    put(0x0009, 0, 0, 0, 0, 0, 0, 0, 0, 0, 319, 223, 0, 0);
    put(0x000A, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    // flat triangle
    put(0x0004, 0, 0x0000, 0x801F, 0, 0, 20, 20, 150, 40, 60, 120, 60, 120);
    // distorted textured sprite (rotated)
    put(0x0002, 0, 0x0028, 0, 0x20000 >> 3, (1 << 8) | 8, 200, 40, 290, 70, 270, 150, 190, 120);
    // axis aligned scaled sprite
    put(0x0001, 0, 0x0028, 0, 0x20000 >> 3, (1 << 8) | 8, 40, 140, 0, 0, 71, 191, 0, 0);
    // polygon green rectangle next to it, tiling test
    put(0x0004, 0, 0x0000, 0x83E0, 0, 0, 120, 150, 160, 150, 160, 190, 120, 190);
    put(0x0004, 0, 0x0000, 0x8000 | (31 << 10), 0, 0, 161, 150, 200, 150, 200, 190, 161, 190);
    // mesh polygon overlapping the textured sprite and the background
    put(0x0004, 0, 0x0100, 0x8000 | (31 << 5) | 31, 0, 0, 150, 100, 260, 100, 260, 180, 150, 180);
    // lines
    put(0x0006, 0, 0x0000, 0xFFFF, 0, 0, 10, 210, 310, 200, 0, 0, 0, 0);
    put(0x0005, 0, 0x0000, 0xFFFF, 0, 0, 220, 170, 300, 170, 300, 215, 220, 215);
    // end
    put(0x8000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    auto renderFrame = [&] {
        r.VDP1EraseFramebuffer(0);
        r.VDP1BeginFrame();
        uint32 a = 0;
        for (;;) {
            VDP1Command::Control ctrl{.u16 = util::ReadBE<uint16>(&state.mem1.VRAM[a])};
            if (ctrl.end) {
                break;
            }
            if (!ctrl.skip) {
                r.VDP1ExecuteCommand(a, ctrl);
            }
            a += 0x20;
        }
        r.VDP1EndFrame();
        r.VDP1SwapFramebuffer();
        state.fbIndex.Flip();

        r.VDP2BeginFrame();
        for (uint32 y = 0; y < 224; y++) {
            r.VDP2RenderLine(y);
        }
        r.VDP2EndFrame();
    };

    renderFrame();
    renderFrame(); // second frame exercises the latched scale / resync paths

    if (timing) {
        const auto t0 = std::chrono::steady_clock::now();
        const int n = 50;
        for (int i = 0; i < n; i++) {
            renderFrame();
        }
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("scale %u: %.3f ms/frame\n", scale, std::chrono::duration<double, std::milli>(t1 - t0).count() / n);
    }

    FILE *f = std::fopen(outPath, "wb");
    const uint32 W = g_w * g_scale, H = g_h * g_scale;
    std::fprintf(f, "P6\n%u %u\n255\n", W, H);
    for (size_t i = 0; i < size_t(W) * H; i++) {
        const uint32 c = g_frame[i];
        const uint8 rgb[3] = {uint8(c >> 0), uint8(c >> 8), uint8(c >> 16)};
        std::fwrite(rgb, 1, 3, f);
    }
    std::fclose(f);
    std::printf("frame %ux%u scale %u\n", g_w, g_h, g_scale);
    return 0;
}
