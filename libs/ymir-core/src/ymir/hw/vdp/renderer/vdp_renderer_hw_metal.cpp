#include <ymir/hw/vdp/renderer/vdp_renderer_hw_metal.hpp>

#include "common_hw/common_hw_defs.hpp"
#include "metal/metal_gpu.hpp"

#include <ymir/util/bit_ops.hpp>
#include <ymir/util/perf_counters.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace ymir::vdp {

namespace {

    /// @brief Packs up boolean values into the least significant bits of an unsigned integer.
    template <std::unsigned_integral T>
    uint32 PackBools(std::span<const bool> bools) {
        T value = 0;
        size_t count = std::min(bools.size(), sizeof(T) * 8);
        for (size_t i = 0; i < count; ++i) {
            if (bools[i]) {
                value |= static_cast<T>(1u) << static_cast<T>(i);
            }
        }
        return value;
    }

} // namespace

// The VDP2 side of this renderer follows the Direct3D 12 renderer closely: the same compute shaders are used with the
// same parameter blocks, and register changes in the middle of a frame split the frame into batches of lines in the
// same way. The difference is how the data gets to the GPU: every pass copies the state it needs when it is submitted
// (see metal::Gpu), so the state a pass was recorded with can't be changed by what the CPU does next.

struct MetalVDPRenderer::Impl {
    Impl(MetalVDPRenderer &owner, VDPState &state, const config::VDP2AccessPatternsConfig &accessPatternsConfig,
         const config::VDP2DebugRender &debugRenderOptions)
        : owner(owner)
        , vdpState(state)
        , accessPatternsConfig(accessPatternsConfig)
        , debugRenderOptions(debugRenderOptions) {
        vramDirty.assign(kVRAMDirtyChunks, 1);
        committedLayerParams = std::make_unique<VDP2LayerRenderParams>();
        cpuLayerRenderParams = std::make_unique<VDP2LayerRenderParams>();
        std::memset(committedLayerParams.get(), 0, sizeof(VDP2LayerRenderParams));
        std::memset(cpuLayerRenderParams.get(), 0, sizeof(VDP2LayerRenderParams));
    }

    MetalVDPRenderer &owner;
    VDPState &vdpState;
    const config::VDP2AccessPatternsConfig &accessPatternsConfig;
    const config::VDP2DebugRender &debugRenderOptions;

    std::unique_ptr<metal::Gpu> gpu;

    // -----------------------------------------------------------------------------------------------------------------
    // State

    static constexpr size_t kVRAMDirtyChunkShift = 8;
    static constexpr size_t kVRAMDirtyChunks = kVDP2VRAMSize >> kVRAMDirtyChunkShift;

    /// @brief VRAM chunks written to since they were last sent to the GPU.
    std::vector<uint8> vramDirty;
    bool vramAnyDirty = true;

    /// @brief CRAM converted to colors, kept up to date as CRAM is written.
    CRAMColorCache cpuCRAMColorCache{};
    /// @brief The CRAM colors the GPU passes use. Updated together with the rest of the state between lines.
    CRAMColorCache committedCRAMColors{};
    /// @brief The raw upper half of CRAM (rotation coefficients) the GPU passes use.
    std::array<uint8, kVDP2CRAMRotCoeffBufferSize> committedCRAMRotCoeffs{};

    uint32 cramGeneration = 0;
    uint32 layerRenderParamsGeneration = 0;
    uint32 composeParamsGeneration = 0;

    // The generations of the state the GPU passes use
    uint32 appliedCramGeneration = ~0u;
    uint32 appliedLayerRenderParamsGeneration = ~0u;
    uint32 appliedComposeParamsGeneration = ~0u;

    /// @brief LNCL (0) and BACK (1) screen colors for every line.
    std::array<std::array<ColorR8G8B8A8, kMaxNormalResV>, 2> cpuLnclBack{};

    /// @brief Rotation parameter base values for every line (parameters A, then B).
    std::array<VDP2RotParamBase, kMaxNormalResV * 2> cpuRotParamBases{};

    VDP2CommonRenderParams cpuCommonRenderParams{};
    std::unique_ptr<VDP2LayerRenderParams> cpuLayerRenderParams;
    /// @brief The layer parameters the GPU passes use (a copy made when the parameters are applied).
    std::unique_ptr<VDP2LayerRenderParams> committedLayerParams;
    VDP2ComposeParams cpuComposeParams{};

    /// @brief Whether to reset the base offset of NBG2-3 scroll Y increments.
    /// Set to true when writing to SCYN2-3.
    std::array<bool, 2> resetNBGScrollY{};

    uint32 nextLayerRenderLine = 0;
    uint32 nextComposeLine = 0;

    bool resChangedToInterlaced = false;
    uint32 HRes = kDefaultResH;
    uint32 VRes = kDefaultResV;
    bool exclusiveMonitor = false;

    // -----------------------------------------------------------------------------------------------------------------
    // Setup and common

    void Reset() {
        vramDirty.assign(kVRAMDirtyChunks, 1);
        vramAnyDirty = true;
        ++cramGeneration;
        ++layerRenderParamsGeneration;
        ++composeParamsGeneration;

        nextLayerRenderLine = 0;
        nextComposeLine = 0;

        VDP2CacheAllCRAMColors();
        VDP2UpdateEnabledLayers();
    }

    void UpdateEnhancements() {
        // Deinterlacing and transparent meshes are not available with this renderer
        cpuCommonRenderParams.enhancements.deinterlace = 0;
        cpuCommonRenderParams.enhancements.transparentMeshes = 0;
    }

    // -----------------------------------------------------------------------------------------------------------------
    // VDP2 rendering

    void VDP2CacheCRAMColor(uint32 address) {
        CRAMColorCache &colorCache = cpuCRAMColorCache;
        switch (vdpState.regs2.vramControl.colorRAMMode) {
        case 0: { // RGB 5:5:5, half CRAM
            const auto value = vdpState.mem2.ReadCRAM<uint16>(address & ~1u);
            const Color555 color5{.u16 = value};
            const Color888 color8 = ConvertRGB555to888(color5);
            ColorR8G8B8A8 &color = colorCache[address >> 1u];
            color.r = color8.r;
            color.g = color8.g;
            color.b = color8.b;
            color.a = color8.msb;
            break;
        }
        case 1: { // RGB 5:5:5, full CRAM
            const auto value = vdpState.mem2.ReadCRAM<uint16>(address & ~1u);
            const Color555 color5{.u16 = value};
            const Color888 color8 = ConvertRGB555to888(color5);
            ColorR8G8B8A8 &color = colorCache[address >> 1u];
            color.r = color8.r;
            color.g = color8.g;
            color.b = color8.b;
            color.a = color8.msb;
            break;
        }
        case 2: [[fallthrough]]; // RGB 8:8:8, full CRAM
        case 3: [[fallthrough]]; // RGB 8:8:8, full CRAM
        default: {
            address = vdpState.mem2.UnmapCRAMAddress<uint8>(address);
            const auto value = vdpState.mem2.ReadCRAM<uint32>(address & ~3u);
            const Color888 color8{.u32 = value};
            ColorR8G8B8A8 &color = colorCache[address >> 2u];
            color.r = color8.r;
            color.g = color8.g;
            color.b = color8.b;
            color.a = color8.msb;
            break;
        }
        }
    }

    void VDP2CacheAllCRAMColors() {
        CRAMColorCache &colorCache = cpuCRAMColorCache;
        switch (vdpState.regs2.vramControl.colorRAMMode) {
        case 0: // RGB 5:5:5, half CRAM
            for (uint32 i = 0; i < 1024; ++i) {
                const auto value = vdpState.mem2.ReadCRAM<uint16>(i * sizeof(uint16));
                const Color555 color5{.u16 = value};
                const Color888 color8 = ConvertRGB555to888(color5);
                colorCache[i].r = color8.r;
                colorCache[i].g = color8.g;
                colorCache[i].b = color8.b;
                colorCache[i].a = color8.msb;
            }
            break;
        case 1: // RGB 5:5:5, full CRAM
            for (uint32 i = 0; i < 2048; ++i) {
                const auto value = vdpState.mem2.ReadCRAM<uint16>(i * sizeof(uint16));
                const Color555 color5{.u16 = value};
                const Color888 color8 = ConvertRGB555to888(color5);
                colorCache[i].r = color8.r;
                colorCache[i].g = color8.g;
                colorCache[i].b = color8.b;
                colorCache[i].a = color8.msb;
            }
            break;
        case 2: [[fallthrough]]; // RGB 8:8:8, full CRAM
        case 3: [[fallthrough]]; // RGB 8:8:8, full CRAM
        default:
            for (uint32 i = 0; i < 1024; ++i) {
                const auto value = vdpState.mem2.ReadCRAM<uint32>(i * sizeof(uint32));
                const Color888 color8{.u32 = value};
                colorCache[i].r = color8.r;
                colorCache[i].g = color8.g;
                colorCache[i].b = color8.b;
                colorCache[i].a = color8.msb;
            }
            break;
        }
    }

    void VDP2WriteVRAM(uint32 address) {
        vramDirty[(address >> kVRAMDirtyChunkShift) & (kVRAMDirtyChunks - 1)] = 1;
        vramAnyDirty = true;
    }

    void VDP2WriteCRAM(uint32 address) {
        ++cramGeneration;
        VDP2CacheCRAMColor(address);
    }

    void VDP2WriteReg(uint32 address, uint16 value) {
        struct DirtyFlags {
            bool render = false;
            bool compose = false;
            bool enabledLayers = false;
            bool cram = false;
        };
        static constexpr auto kDirtyFlags = [] {
            std::array<DirtyFlags, 0x11E / sizeof(uint16) + 1> arr{};

            for (uint32 addr : {
                     0x000 /*TVMD*/,   0x002 /*EXTEN*/,  0x006 /*VRSIZE*/, 0x00E /*RAMCTL*/, 0x010 /*CYCA0L*/,
                     0x012 /*CYCA0U*/, 0x014 /*CYCA1L*/, 0x016 /*CYCA1U*/, 0x018 /*CYCB0L*/, 0x01A /*CYCB0U*/,
                     0x01C /*CYCB1L*/, 0x01E /*CYCB1U*/, 0x020 /*BGON*/,   0x022 /*MZCTL*/,  0x024 /*SFSEL*/,
                     0x026 /*SFCODE*/, 0x028 /*CHCTLA*/, 0x02A /*CHCTLB*/, 0x02C /*BMPNA*/,  0x02E /*BMPNB*/,
                     0x030 /*PNCNA*/,  0x032 /*PNCNB*/,  0x034 /*PNCNC*/,  0x036 /*PNCND*/,  0x038 /*PNCR*/,
                     0x03A /*PLSZ*/,   0x03C /*MPOFN*/,  0x03E /*MPOFR*/,  0x040 /*MPABN0*/, 0x042 /*MPCDN0*/,
                     0x044 /*MPABN1*/, 0x046 /*MPCDN1*/, 0x048 /*MPABN2*/, 0x04A /*MPCDN2*/, 0x04C /*MPABN3*/,
                     0x04E /*MPCDN3*/, 0x050 /*MPABRA*/, 0x052 /*MPCDRA*/, 0x054 /*MPEFRA*/, 0x056 /*MPGHRA*/,
                     0x058 /*MPIJRA*/, 0x05A /*MPKLRA*/, 0x05C /*MPMNRA*/, 0x05E /*MPOPRA*/, 0x060 /*MPABRB*/,
                     0x062 /*MPCDRB*/, 0x064 /*MPEFRB*/, 0x066 /*MPGHRB*/, 0x068 /*MPIJRB*/, 0x06A /*MPKLRB*/,
                     0x06C /*MPMNRB*/, 0x06E /*MPOPRB*/, 0x070 /*SCXIN0*/, 0x072 /*SCXDN0*/, 0x074 /*SCYIN0*/,
                     0x076 /*SCYDN0*/, 0x078 /*ZMXIN0*/, 0x07A /*ZMXDN0*/, 0x07C /*ZMYIN0*/, 0x07E /*ZMYDN0*/,
                     0x080 /*SCXIN1*/, 0x082 /*SCXDN1*/, 0x084 /*SCYIN1*/, 0x086 /*SCYDN1*/, 0x088 /*ZMXIN1*/,
                     0x08A /*ZMXDN1*/, 0x08C /*ZMYIN1*/, 0x08E /*ZMYDN1*/, 0x090 /*SCXN2*/,  0x092 /*SCYN2*/,
                     0x094 /*SCXN3*/,  0x096 /*SCYN3*/,  0x098 /*ZMCTL*/,  0x09A /*SCRCTL*/, 0x09C /*VCSTAU*/,
                     0x09E /*VCSTAL*/, 0x0A0 /*LSTA0U*/, 0x0A2 /*LSTA0L*/, 0x0A4 /*LSTA1U*/, 0x0A6 /*LSTA1L*/,
                     0x0A8 /*LCTAU*/,  0x0AA /*LCTAL*/,  0x0AC /*BKTAU*/,  0x0AE /*BKTAL*/,  0x0B0 /*RPMD*/,
                     0x0B2 /*RPRCTL*/, 0x0B4 /*KTCTL*/,  0x0B6 /*KTAOF*/,  0x0B8 /*OVPNRA*/, 0x0BA /*OVPNRB*/,
                     0x0BC /*RPTAU*/,  0x0BE /*RPTAL*/,  0x0C0 /*WPSX0*/,  0x0C2 /*WPSY0*/,  0x0C4 /*WPEX0*/,
                     0x0C6 /*WPEY0*/,  0x0C8 /*WPSX1*/,  0x0CA /*WPSY1*/,  0x0CC /*WPEX1*/,  0x0CE /*WPEY1*/,
                     0x0D0 /*WCTLA*/,  0x0D2 /*WCTLB*/,  0x0D4 /*WCTLC*/,  0x0D6 /*WCTLD*/,  0x0D8 /*LWTA0U*/,
                     0x0DA /*LWTA0L*/, 0x0DC /*LWTA1U*/, 0x0DE /*LWTA1L*/, 0x0E0 /*SPCTL*/,  0x0E2 /*SDCTL*/,
                     0x0E4 /*CRAOFA*/, 0x0E6 /*CRAOFB*/, 0x0E8 /*LNCLEN*/, 0x0EA /*SFPRMD*/, 0x0EC /*CCCTL*/,
                     0x0EE /*SFCCMD*/, 0x0F0 /*PRISA*/,  0x0F2 /*PRISB*/,  0x0F4 /*PRISC*/,  0x0F6 /*PRISD*/,
                     0x0F8 /*PRINA*/,  0x0FA /*PRINB*/,  0x0FC /*PRIR*/,
                 }) {
                arr[addr / sizeof(uint16)].render = true;
            }

            for (uint32 addr : {
                     0x000 /*TVMD*/,   0x006 /*VRSIZE*/, 0x020 /*BGON*/,  0x0E0 /*SPCTL*/, 0x0E2 /*SDCTL*/,
                     0x0E8 /*LNCLEN*/, 0x0EC /*CCCTL*/,  0x100 /*CCRSA*/, 0x102 /*CCRSB*/, 0x104 /*CCRSC*/,
                     0x106 /*CCRSD*/,  0x108 /*CCRNA*/,  0x10A /*CCRNB*/, 0x10C /*CCRR*/,  0x10E /*CCRLB*/,
                     0x110 /*CLOFEN*/, 0x112 /*CLOFSL*/, 0x114 /*COAR*/,  0x116 /*COAG*/,  0x118 /*COAB*/,
                     0x11A /*COBR*/,   0x11C /*COBG*/,   0x11E /*COBB*/,
                 }) {
                arr[addr / sizeof(uint16)].compose = true;
            }

            for (uint32 addr : {0x020 /*BGON*/, 0x028 /*CHCTLA*/, 0x02A /*CHCTLB*/}) {
                arr[addr / sizeof(uint16)].enabledLayers = true;
            }

            arr[0x00E / sizeof(uint16) /*RAMCTL*/].cram = true;

            return arr;
        }();

        if (address <= 0x11E) {
            const auto &dirtyFlags = kDirtyFlags[address / sizeof(uint16)];
            if (dirtyFlags.render) {
                ++layerRenderParamsGeneration;
            }
            if (dirtyFlags.compose) {
                ++composeParamsGeneration;
            }

            if (dirtyFlags.enabledLayers) {
                VDP2UpdateEnabledLayers();
            }

            if (dirtyFlags.cram) {
                ++cramGeneration;
                VDP2CacheAllCRAMColors();
            }

            switch (address) {
            case 0x092: // SCYN2
                resetNBGScrollY[0] = true;
                break;
            case 0x096: // SCYN3
                resetNBGScrollY[1] = true;
                break;
            }
        }
    }

    /// @brief Sends the VRAM chunks written to since the last call to the GPU.
    void VDP2FlushVRAM() {
        if (!vramAnyDirty) {
            return;
        }
        vramAnyDirty = false;

        constexpr size_t kChunkSize = size_t{1} << kVRAMDirtyChunkShift;
        for (size_t chunk = 0; chunk < kVRAMDirtyChunks;) {
            if (vramDirty[chunk] == 0) {
                ++chunk;
                continue;
            }
            size_t end = chunk;
            while (end < kVRAMDirtyChunks && vramDirty[end] != 0) {
                vramDirty[end] = 0;
                ++end;
            }
            gpu->UploadVRAM(static_cast<uint32>(chunk * kChunkSize), &vdpState.mem2.VRAM[chunk * kChunkSize],
                            static_cast<uint32>((end - chunk) * kChunkSize));
            chunk = end;
        }
    }

    /// @brief Makes the CRAM state the passes use match the current one.
    void VDP2FlushCRAM() {
        if (appliedCramGeneration == cramGeneration) {
            return;
        }
        appliedCramGeneration = cramGeneration;

        committedCRAMColors = cpuCRAMColorCache;
        std::memcpy(committedCRAMRotCoeffs.data(), &vdpState.mem2.CRAM[kVDP2CRAMSize / 2],
                    committedCRAMRotCoeffs.size());
    }

    void VDP2UpdateCommonRenderParams() {
        // cpuCommonRenderParams.startY is updated by the line rendering functions.

        const VDP1Regs &regs1 = vdpState.regs1;
        const VDP2Regs &regs2 = vdpState.regs2;
        VDP2CommonRenderParams &params = cpuCommonRenderParams;

        params.displayParams.displayEnable = regs2.TVMD.DISP;
        params.displayParams.borderColorMode = regs2.TVMD.BDCLMD;
        params.displayParams.interlaceMode = static_cast<HLSLuint>(regs2.TVMD.LSMDn);
        params.displayParams.oddField = regs2.TVSTAT.ODD;
        params.displayParams.exclusiveMonitor = exclusiveMonitor;
        params.displayParams.colorRAMMode = regs2.vramControl.colorRAMMode;
        params.displayParams.hiResH = bit::test<1>(regs2.TVMD.HRESOn);
        params.displayParams.palMode = regs2.TVSTAT.PAL;
        params.displayParams.hresMode = regs2.TVMD.HRESOn;
        params.displayParams.vresMode = regs2.TVMD.VRESOn;
        params.displayParams.dblInterlaceEnable = regs1.dblInterlaceEnable;
        params.displayParams.dblInterlaceDrawLine = regs1.dblInterlaceDrawLine;

        params.layerParams.layerEnabled = PackBools<HLSLuint>(vdpState.state2.layerEnabled);
        params.layerParams.bgEnabled = PackBools<HLSLuint>(regs2.bgEnabled);
        params.layerParams.lineColorEnableRBG0 = regs2.bgParams[0].lineColorScreenEnable;
        params.layerParams.lineColorEnableRBG1 = regs2.bgParams[1].lineColorScreenEnable;
        params.layerParams.mosaicH = regs2.mosaicH - 1;
        params.layerParams.mosaicV = regs2.mosaicV - 1;
        params.layerParams.rotParamMode = static_cast<HLSLuint>(regs2.commonRotParams.rotParamMode);
        params.layerParams.restrictedColorCalc = regs2.restrictedColorCalc;
        params.layerParams.extendedColorCalc = regs2.colorCalcParams.extendedColorCalcEnable && regs2.TVMD.HRESOn < 2;
        params.layerParams.useAdditiveBlend = regs2.colorCalcParams.useAdditiveBlend;
        params.layerParams.useSecondScreenRatio = regs2.colorCalcParams.useSecondScreenRatio;
        params.layerParams.colorGradEnable = regs2.colorCalcParams.colorGradEnable;
        params.layerParams.colorGradScreen = static_cast<HLSLuint>(regs2.colorCalcParams.colorGradScreen);

        auto isCoeff = [](RotDataBankSel sel) { return sel == RotDataBankSel::Coefficients; };

        const VRAMControl &vramCtl = regs2.vramControl;
        params.rotParams.coeffTableCRAM = vramCtl.colorRAMCoeffTableEnable;
        params.rotParams.coeffDataAccess = 0;
        if (isCoeff(vramCtl.rotDataBankSelA0)) {
            params.rotParams.coeffDataAccess |= 1u << 0u;
        }
        if (isCoeff(vramCtl.partitionVRAMA ? vramCtl.rotDataBankSelA1 : vramCtl.rotDataBankSelA0)) {
            params.rotParams.coeffDataAccess |= 1u << 1u;
        }
        if (isCoeff(vramCtl.rotDataBankSelB0)) {
            params.rotParams.coeffDataAccess |= 1u << 2u;
        }
        if (isCoeff(vramCtl.partitionVRAMB ? vramCtl.rotDataBankSelB1 : vramCtl.rotDataBankSelB0)) {
            params.rotParams.coeffDataAccess |= 1u << 3u;
        }
        params.rotParams.coeffDataPerDot = vramCtl.perDotRotationCoeffs;

        const RotationParams &rotParamsA = regs2.rotParams[0];
        params.rotParams.coeffATableEnable = rotParamsA.coeffTableEnable;
        params.rotParams.coeffAUseLineColorData = rotParamsA.coeffUseLineColorData;
        params.rotParams.coeffADataSize = rotParamsA.coeffDataSize;
        params.rotParams.coeffADataMode = static_cast<HLSLuint>(rotParamsA.coeffDataMode);

        const RotationParams &rotParamsB = regs2.rotParams[1];
        params.rotParams.coeffBTableEnable = rotParamsB.coeffTableEnable;
        params.rotParams.coeffBUseLineColorData = rotParamsB.coeffUseLineColorData;
        params.rotParams.coeffBDataSize = rotParamsB.coeffDataSize;
        params.rotParams.coeffBDataMode = static_cast<HLSLuint>(rotParamsB.coeffDataMode);

        params.spriteParams.rotate = regs1.tvmr.fbRotEnable;
        params.spriteParams.pixel8Bits = regs1.tvmr.pixel8Bits;
        params.spriteParams.type = regs2.spriteParams.type;
        params.spriteParams.fbSizeH = std::countr_zero(regs1.tvmr.fbSizeH) - 9;
        params.spriteParams.fbSizeV = std::countr_zero(regs1.tvmr.fbSizeV) - 8;
        params.spriteParams.inHalfResH = false;
        params.spriteParams.outHalfResH = false;
        if (!regs1.tvmr.hdtvEnable && !regs1.tvmr.fbRotEnable) {
            if (regs1.tvmr.pixel8Bits) {
                params.spriteParams.inHalfResH = (regs2.TVMD.HRESOn & 0b110) == 0b000;
            } else {
                params.spriteParams.outHalfResH = (regs2.TVMD.HRESOn & 0b110) == 0b010;
            }
        }
        params.spriteParams.mixedFormat = regs2.spriteParams.mixedFormat;
        params.spriteParams.colorCalcEnable = regs2.spriteParams.colorCalcEnable;
        params.spriteParams.colorCalcValue = regs2.spriteParams.colorCalcValue;
        params.spriteParams.colorCalcCond = static_cast<HLSLuint>(regs2.spriteParams.colorCalcCond);
        params.spriteParams.colorDataOffset = regs2.spriteParams.colorDataOffset >> 8u;
        params.spriteParams.useSpriteWindow = regs2.spriteParams.useSpriteWindow;
        params.spriteParams.windowEnabled = regs2.spriteParams.spriteWindowEnabled;
        params.spriteParams.windowInverted = regs2.spriteParams.spriteWindowInverted;
        params.spriteParams.displayFB = vdpState.fbIndex.display;

        params.spritePriosRatios.x = 0;
        params.spritePriosRatios.y = 0;
        for (uint32 i = 0; i < 4; i++) {
            params.spritePriosRatios.x |= regs2.spriteParams.priorities[i] << (8 * i);
            params.spritePriosRatios.x |= regs2.spriteParams.colorCalcRatios[i] << (8 * i + 3);

            params.spritePriosRatios.y |= regs2.spriteParams.priorities[i + 4] << (8 * i);
            params.spritePriosRatios.y |= regs2.spriteParams.colorCalcRatios[i + 4] << (8 * i + 3);
        }

        params.vcellScroll.tableAddress = regs2.vcellScrollTableAddress;
        params.vcellScroll.inc = regs2.vcellScrollInc >> 2u;

        params.windows.spriteWindowLogic = regs2.spriteParams.windowSet.logic == WindowLogic::And;
        params.windows.spriteW0Enable = regs2.spriteParams.windowSet.enabled[0];
        params.windows.spriteW0Invert = regs2.spriteParams.windowSet.inverted[0];
        params.windows.spriteW1Enable = regs2.spriteParams.windowSet.enabled[1];
        params.windows.spriteW1Invert = regs2.spriteParams.windowSet.inverted[1];

        params.windows.colorCalcWindowLogic = regs2.colorCalcParams.windowSet.logic == WindowLogic::And;
        params.windows.colorCalcW0Enable = regs2.colorCalcParams.windowSet.enabled[0];
        params.windows.colorCalcW0Invert = regs2.colorCalcParams.windowSet.inverted[0];
        params.windows.colorCalcW1Enable = regs2.colorCalcParams.windowSet.enabled[1];
        params.windows.colorCalcW1Invert = regs2.colorCalcParams.windowSet.inverted[1];
        params.windows.colorCalcSWEnable = regs2.colorCalcParams.windowSet.enabled[2];
        params.windows.colorCalcSWInvert = regs2.colorCalcParams.windowSet.inverted[2];
    }

    void VDP2UpdateLayerRenderParams(uint32 y) {
        if (appliedLayerRenderParamsGeneration == layerRenderParamsGeneration) {
            return;
        }
        appliedLayerRenderParamsGeneration = layerRenderParamsGeneration;

        const VDP2Regs &regs2 = vdpState.regs2;

        // These can be either 0 or 8, but we'll condense them to single bits
        auto packVRAMDataOffsets = [](const std::array<uint32, 4> values) {
            uint32 value = 0;
            for (size_t i = 0; i < values.size(); ++i) {
                if (values[i] != 0u) {
                    value |= 1u << i;
                }
            }
            return value;
        };

        // NBG0-3
        for (int i = 0; i < 4; ++i) {
            const BGParams &bgParams = regs2.bgParams[i + 1];
            const NBGLayerState &bgState = vdpState.state2.nbgLayerStates[i];

            const bool bitmap = bgParams.bitmap;

            NBGParams &renderParams = cpuLayerRenderParams->nbg[i];
            renderParams.base.enabled = regs2.bgEnabled[i];
            renderParams.base.enableTransparency = bgParams.enableTransparency;
            renderParams.base.bitmap = bgParams.bitmap;
            renderParams.base.priorityNumber = bgParams.priorityNumber;
            renderParams.base.priorityMode = static_cast<HLSLuint>(bgParams.priorityMode);
            renderParams.base.specialFunctionSelect = bgParams.specialFunctionSelect;
            renderParams.base.cellSizeShift = bgParams.cellSizeShift;
            renderParams.base.colorFormat = static_cast<HLSLuint>(bgParams.colorFormat);
            renderParams.base.cramOffset = bgParams.cramOffset;
            renderParams.base.supplScrollCharNum = bgParams.supplScrollCharNum;
            renderParams.base.supplPalNum = bitmap ? bgParams.supplBitmapPalNum : bgParams.supplScrollPalNum;
            renderParams.base.supplSpecialColorCalc =
                bitmap ? bgParams.supplBitmapSpecialColorCalc : bgParams.supplScrollSpecialColorCalc;
            renderParams.base.supplSpecialPriority =
                bitmap ? bgParams.supplBitmapSpecialPriority : bgParams.supplScrollSpecialPriority;
            renderParams.base.mosaicEnable = bgParams.mosaicEnable;
            renderParams.base.colorCalcEnable = bgParams.colorCalcEnable;
            renderParams.base.extChar = bgParams.extChar;
            renderParams.base.twoWordChar = bgParams.twoWordChar;
            renderParams.base.patNameAccess = PackBools<HLSLuint>(bgParams.patNameAccess);
            renderParams.base.charPatAccess = PackBools<HLSLuint>(bgParams.charPatAccess);
            renderParams.base.charPatDelay = PackBools<HLSLuint>(bgParams.charPatDelay);
            renderParams.base.vramDataOffset = packVRAMDataOffsets(bgParams.vramDataOffset);
            renderParams.base.specialColorCalcMode = static_cast<HLSLuint>(bgParams.specialColorCalcMode);
            renderParams.base.pageShift = {bgParams.pageShiftH, bgParams.pageShiftV};
            renderParams.base.bitmapSize = {bgParams.bitmapSizeH, bgParams.bitmapSizeV};
            renderParams.base.bitmapBaseAddress = bgParams.bitmapBaseAddress;
            renderParams.base.windowParams.base.windowLogicAnd = bgParams.windowSet.logic == WindowLogic::And;
            renderParams.base.windowParams.base.window0Enable = bgParams.windowSet.enabled[0];
            renderParams.base.windowParams.base.window0Invert = bgParams.windowSet.inverted[0];
            renderParams.base.windowParams.base.window1Enable = bgParams.windowSet.enabled[1];
            renderParams.base.windowParams.base.window1Invert = bgParams.windowSet.inverted[1];
            renderParams.base.windowParams.spriteWindowEnable = bgParams.windowSet.enabled[2];
            renderParams.base.windowParams.spriteWindowInvert = bgParams.windowSet.inverted[2];

            renderParams.scrollAmount = {bgParams.scrollAmountH, bgParams.scrollAmountV};
            renderParams.scrollInc = {bgParams.scrollIncH, bgParams.scrollIncV};
            renderParams.pageBaseAddresses = bgParams.pageBaseAddresses;
            renderParams.vcellScrollEnable = bgParams.vcellScrollEnable;
            renderParams.lineScrollXEnable = bgParams.lineScrollXEnable;
            renderParams.lineScrollYEnable = bgParams.lineScrollYEnable;
            renderParams.lineZoomEnable = bgParams.lineZoomEnable;
            renderParams.lineScrollInterval = bgParams.lineScrollInterval;
            renderParams.lineScrollTableAddress = bgState.lineScrollTableAddress;
            renderParams.vcellScrollOffset = bgState.vcellScrollOffset;
            renderParams.vcellScrollDelay = bgState.vcellScrollDelay;
            renderParams.vcellScrollRepeat = bgState.vcellScrollRepeat;
            if (i >= 2 && resetNBGScrollY[i - 2]) {
                renderParams.scrollYIncBase = y;
                resetNBGScrollY[i - 2] = false;
            }
        }

        // RBG0-1 / RotParam A-B
        for (int i = 0; i < 2; ++i) {
            const BGParams &bgParams = regs2.bgParams[i];
            const RotationParams &rotParams = regs2.rotParams[i];

            const bool bitmap = bgParams.bitmap;

            RBGParams &renderParams = cpuLayerRenderParams->rbg[i];
            renderParams.base.enabled = regs2.bgEnabled[i + 4];
            renderParams.base.enableTransparency = bgParams.enableTransparency;
            renderParams.base.bitmap = bgParams.bitmap;
            renderParams.base.priorityNumber = bgParams.priorityNumber;
            renderParams.base.priorityMode = static_cast<HLSLuint>(bgParams.priorityMode);
            renderParams.base.specialFunctionSelect = bgParams.specialFunctionSelect;
            renderParams.base.cellSizeShift = bgParams.cellSizeShift;
            renderParams.base.colorFormat = static_cast<HLSLuint>(bgParams.colorFormat);
            renderParams.base.cramOffset = bgParams.cramOffset;
            renderParams.base.supplScrollCharNum = bgParams.supplScrollCharNum;
            renderParams.base.supplPalNum = bitmap ? bgParams.supplBitmapPalNum : bgParams.supplScrollPalNum;
            renderParams.base.supplSpecialColorCalc =
                bitmap ? bgParams.supplBitmapSpecialColorCalc : bgParams.supplScrollSpecialColorCalc;
            renderParams.base.supplSpecialPriority =
                bitmap ? bgParams.supplBitmapSpecialPriority : bgParams.supplScrollSpecialPriority;
            renderParams.base.mosaicEnable = bgParams.mosaicEnable;
            renderParams.base.colorCalcEnable = bgParams.colorCalcEnable;
            renderParams.base.extChar = bgParams.extChar;
            renderParams.base.twoWordChar = bgParams.twoWordChar;
            renderParams.base.patNameAccess = PackBools<HLSLuint>(bgParams.patNameAccess);
            renderParams.base.charPatAccess = PackBools<HLSLuint>(bgParams.charPatAccess);
            renderParams.base.charPatDelay = PackBools<HLSLuint>(bgParams.charPatDelay);
            renderParams.base.vramDataOffset = packVRAMDataOffsets(bgParams.vramDataOffset);
            renderParams.base.specialColorCalcMode = static_cast<HLSLuint>(bgParams.specialColorCalcMode);
            renderParams.base.pageShift = {rotParams.pageShiftH, rotParams.pageShiftV};
            renderParams.base.bitmapSize = {bgParams.bitmapSizeH, bgParams.bitmapSizeV};
            renderParams.base.bitmapBaseAddress = rotParams.bitmapBaseAddress;
            renderParams.base.windowParams.base.windowLogicAnd = bgParams.windowSet.logic == WindowLogic::And;
            renderParams.base.windowParams.base.window0Enable = bgParams.windowSet.enabled[0];
            renderParams.base.windowParams.base.window0Invert = bgParams.windowSet.inverted[0];
            renderParams.base.windowParams.base.window1Enable = bgParams.windowSet.enabled[1];
            renderParams.base.windowParams.base.window1Invert = bgParams.windowSet.inverted[1];
            renderParams.base.windowParams.spriteWindowEnable = bgParams.windowSet.enabled[2];
            renderParams.base.windowParams.spriteWindowInvert = bgParams.windowSet.inverted[2];

            renderParams.screenOverProcess = static_cast<HLSLuint>(rotParams.screenOverProcess);
            renderParams.screenOverPatternName = rotParams.screenOverPatternName;
            renderParams.pageBaseAddresses[0] = vdpState.state2.rbgPageBaseAddresses[0][i];
            renderParams.pageBaseAddresses[1] = vdpState.state2.rbgPageBaseAddresses[1][i];
        }

        // Windows 0 and 1
        for (int i = 0; i < 2; ++i) {
            const WindowParams &windowParams = regs2.windowParams[i];
            VDP2GlobalWindowParams &renderParams = cpuLayerRenderParams->windows[i];
            renderParams.start = {windowParams.startX, windowParams.startY};
            renderParams.end = {windowParams.endX, windowParams.endY};
            renderParams.lineWindowTableAddress = windowParams.lineWindowTableAddress;
            renderParams.lineWindowTableEnable = windowParams.lineWindowTableEnable;
        }

        VDP2LayerWindowParams &rotWindows = cpuLayerRenderParams->rotWindows;
        rotWindows.windowLogicAnd = regs2.commonRotParams.windowSet.logic == WindowLogic::And;
        rotWindows.window0Enable = regs2.commonRotParams.windowSet.enabled[0];
        rotWindows.window0Invert = regs2.commonRotParams.windowSet.inverted[0];
        rotWindows.window1Enable = regs2.commonRotParams.windowSet.enabled[1];
        rotWindows.window1Invert = regs2.commonRotParams.windowSet.inverted[1];

        VDP2LineBackScreenParams &lnclParams = cpuLayerRenderParams->lineScreenParams;
        lnclParams.baseAddress = regs2.lineScreenParams.baseAddress;
        lnclParams.perLine = regs2.lineScreenParams.perLine;

        VDP2LineBackScreenParams &backParams = cpuLayerRenderParams->backScreenParams;
        backParams.baseAddress = regs2.backScreenParams.baseAddress;
        backParams.perLine = regs2.backScreenParams.perLine;

        // Special function codes
        cpuLayerRenderParams->specialFunctionCodes =
            PackBools<uint32>(regs2.specialFunctionCodes[0].colorMatches) |
            (PackBools<uint32>(regs2.specialFunctionCodes[1].colorMatches) << 8u);

        std::memcpy(committedLayerParams.get(), cpuLayerRenderParams.get(), sizeof(VDP2LayerRenderParams));
    }

    void VDP2UpdateComposeParams() {
        if (appliedComposeParamsGeneration == composeParamsGeneration) {
            return;
        }
        appliedComposeParamsGeneration = composeParamsGeneration;

        const VDP2Regs &regs2 = vdpState.regs2;

        auto &params = cpuComposeParams;
        params.colorCalcEnable = 0                                               //
                                 | (regs2.spriteParams.colorCalcEnable << 0)     //
                                 | (regs2.bgParams[0].colorCalcEnable << 1)      //
                                 | (regs2.bgParams[1].colorCalcEnable << 2)      //
                                 | (regs2.bgParams[2].colorCalcEnable << 3)      //
                                 | (regs2.bgParams[3].colorCalcEnable << 4)      //
                                 | (regs2.bgParams[4].colorCalcEnable << 5)      //
                                 | (regs2.backScreenParams.colorCalcEnable << 6) //
                                 | (regs2.lineScreenParams.colorCalcEnable << 7) //
            ;
        params.colorOffsetEnable = PackBools<HLSLuint>(regs2.colorOffsetEnable);
        params.colorOffsetSelect = PackBools<HLSLuint>(regs2.colorOffsetSelect);
        params.lineColorEnable = 0                                                 //
                                 | (regs2.spriteParams.lineColorScreenEnable << 0) //
                                 | (regs2.bgParams[0].lineColorScreenEnable << 1)  //
                                 | (regs2.bgParams[1].lineColorScreenEnable << 2)  //
                                 | (regs2.bgParams[2].lineColorScreenEnable << 3)  //
                                 | (regs2.bgParams[3].lineColorScreenEnable << 4)  //
                                 | (regs2.bgParams[4].lineColorScreenEnable << 5)  //
            ;

        params.colorOffsetA.r = bit::sign_extend<9>(regs2.colorOffset[0].r);
        params.colorOffsetA.g = bit::sign_extend<9>(regs2.colorOffset[0].g);
        params.colorOffsetA.b = bit::sign_extend<9>(regs2.colorOffset[0].b);

        params.colorOffsetB.r = bit::sign_extend<9>(regs2.colorOffset[1].r);
        params.colorOffsetB.g = bit::sign_extend<9>(regs2.colorOffset[1].g);
        params.colorOffsetB.b = bit::sign_extend<9>(regs2.colorOffset[1].b);

        for (int i = 0; i < 5; ++i) {
            params.bgColorCalcRatios[i] = regs2.bgParams[i].colorCalcRatio;
        }
        params.backLineColorCalcRatios[0] = regs2.backScreenParams.colorCalcRatio;
        params.backLineColorCalcRatios[1] = regs2.lineScreenParams.colorCalcRatio;
    }

    void VDP2CalcAccessPatterns() {
        if (vdpState.regs2.accessPatternsDirty) {
            ++layerRenderParamsGeneration;
        }
        vdpState.state2.CalcAccessPatterns(vdpState.regs2, accessPatternsConfig);
    }

    void VDP2InitNBGs() {
        const VDP2Regs &regs2 = vdpState.regs2;

        for (uint32 i = 0; i < 4; ++i) {
            const BGParams &bgParams = regs2.bgParams[i + 1];
            NBGLayerState &nbgState = vdpState.state2.nbgLayerStates[i];

            // NOTE: fracScrollX/Y are computed from scratch in the shader
            nbgState.scrollIncH = bgParams.scrollIncH;

            if (i < 2) {
                nbgState.lineScrollTableAddress = bgParams.lineScrollTableAddress;
            }
        }

        ++layerRenderParamsGeneration;
    }

    void VDP2UpdateEnabledLayers() {
        vdpState.state2.UpdateEnabledBGs(vdpState.regs2, debugRenderOptions);
    }

    void VDP2CalcVCellScrollDelay() {
        if (vdpState.regs2.accessPatternsDirty) {
            ++layerRenderParamsGeneration;
        }
        vdpState.state2.CalcVCellScrollDelay(vdpState.regs2);
    }

    void VDP2DrawLineColorBackScreens(uint32 y) {
        const VDP2Regs &regs = vdpState.regs2;

        if (regs.displayEnabledLatch) {
            // Read line color screen color
            const LineBackScreenParams &lineParams = regs.lineScreenParams;
            const uint32 lnclY = lineParams.perLine ? y : 0;
            const uint32 lineAddress = lineParams.baseAddress + lnclY * sizeof(uint16);
            const uint32 cramAddress = vdpState.mem2.ReadVRAM<uint16>(lineAddress);
            cpuLnclBack[0][y] = cpuCRAMColorCache[cramAddress & 0x7FF];

            // Read back screen color
            const LineBackScreenParams &backParams = regs.backScreenParams;
            const uint32 backY = backParams.perLine ? y : 0;
            const uint32 backAddress = backParams.baseAddress + backY * sizeof(Color555);
            const Color555 color5{.u16 = vdpState.mem2.ReadVRAM<uint16>(backAddress)};
            const Color888 color8 = ConvertRGB555to888(color5);
            cpuLnclBack[1][y].r = color8.r;
            cpuLnclBack[1][y].g = color8.g;
            cpuLnclBack[1][y].b = color8.b;
            cpuLnclBack[1][y].a = color8.msb;
        } else if (y > 0) {
            cpuLnclBack[0][y] = cpuLnclBack[0][y - 1];
            cpuLnclBack[1][y] = cpuLnclBack[1][y - 1];
        }
    }

    void VDP2UpdateRotationParameterBases(uint16 y) {
        VDP2Regs &regs2 = vdpState.regs2;
        if (!regs2.bgEnabled[4] && !regs2.bgEnabled[5]) {
            // Skip if no RBGs are enabled
            return;
        }

        const bool readAll = y == 0;

        const uint32 baseAddress = regs2.commonRotParams.baseAddress & 0xFFF7C; // mask bit 6 (shifted left by 1)
        for (uint32 i = 0; i < 2; ++i) {
            VDP2RotParamBase &base = cpuRotParamBases[i * kMaxNormalResV + y];
            RotationParams &src = regs2.rotParams[i];

            const uint32 address = baseAddress + i * 0x80;

            base.tableAddress = address;

            if (readAll || src.readXst) {
                base.Xst = bit::extract_signed<6, 28, sint32>(vdpState.mem2.ReadVRAM<uint32>(address + 0x00));
                src.readXst = false;
            } else {
                const VDP2RotParamBase &prevBase = cpuRotParamBases[i * kMaxNormalResV + y - 1];
                base.Xst =
                    prevBase.Xst + bit::extract_signed<6, 18, sint32>(vdpState.mem2.ReadVRAM<uint32>(address + 0x0C));
            }

            if (readAll || src.readYst) {
                base.Yst = bit::extract_signed<6, 28, sint32>(vdpState.mem2.ReadVRAM<uint32>(address + 0x04));
                src.readYst = false;
            } else {
                const VDP2RotParamBase &prevBase = cpuRotParamBases[i * kMaxNormalResV + y - 1];
                base.Yst =
                    prevBase.Yst + bit::extract_signed<6, 18, sint32>(vdpState.mem2.ReadVRAM<uint32>(address + 0x10));
            }

            if (readAll || src.readKAst) {
                const uint32 KAst = bit::extract<6, 31>(vdpState.mem2.ReadVRAM<uint32>(address + 0x54));
                base.KA = src.coeffTableAddressOffset + KAst;
                src.readKAst = false;
            } else {
                const VDP2RotParamBase &prevBase = cpuRotParamBases[i * kMaxNormalResV + y - 1];
                base.KA = prevBase.KA + bit::extract_signed<6, 25>(vdpState.mem2.ReadVRAM<uint32>(address + 0x58));
            }
        }
    }

    /// @brief Makes the state the passes use match the current VDP2 state. The lines drawn before this call have
    /// already been submitted with the previous state.
    void VDP2UpdateState(uint32 y) {
        VDP2FlushVRAM();
        VDP2FlushCRAM();
        VDP2UpdateCommonRenderParams();
        VDP2UpdateLayerRenderParams(y);
        VDP2UpdateComposeParams();
    }

    void VDP2SetResolution(uint32 h, uint32 v, bool exclusive) {
        resChangedToInterlaced = !exclusive && v > vdp::kMaxNormalResV && v != VRes;
        HRes = h;
        VRes = v;
        exclusiveMonitor = exclusive;
    }

    void VDP2BeginFrame() {
        nextLayerRenderLine = 0;
        nextComposeLine = 0;

        VDP2CalcAccessPatterns();
        VDP2InitNBGs();

        cpuCommonRenderParams.displayParams.firstInterlacedFrame = resChangedToInterlaced ? 1u : 0u;
        if (resChangedToInterlaced) {
            resChangedToInterlaced = false;
        }

        VDP2UpdateState(0);
        for (int i = 0; i < 4; ++i) {
            cpuLayerRenderParams->nbg[i].scrollYIncBase = 0;
        }

    }

    /// @brief Sends the VDP1 framebuffer that is being displayed to the GPU.
    void VDP2UploadSpriteFB() {
        // VDP2 reads the VDP1 framebuffer that was just finished. Make sure the render thread is done with it.
        owner.VDP1SyncFB();
        const uint8 displayFB = vdpState.fbIndex.display;
        gpu->UploadSpriteFB(static_cast<uint32>(displayFB * kVDP1FBRAMSize), vdpState.mem1.FBRAM[displayFB].data(),
                            static_cast<uint32>(kVDP1FBRAMSize));
    }

    void VDP2RenderLayerLines(uint32 y) {
        // Bail out if there's nothing to render
        if (y < nextLayerRenderLine) {
            return;
        }

        const uint32 startY = nextLayerRenderLine;
        const uint32 numLines = y - startY + 1;
        nextLayerRenderLine = y + 1;

        cpuCommonRenderParams.startY = startY;

        metal::LayerPassParams pass{};
        pass.common = &cpuCommonRenderParams;
        pass.commonSize = sizeof(cpuCommonRenderParams);
        pass.layerParams = committedLayerParams.get();
        pass.layerParamsSize = sizeof(VDP2LayerRenderParams);
        pass.cramColors = committedCRAMColors.data();
        pass.cramColorsSize = sizeof(committedCRAMColors);
        pass.cramRotCoeffs = committedCRAMRotCoeffs.data();
        pass.cramRotCoeffsSize = static_cast<uint32>(committedCRAMRotCoeffs.size());
        pass.rotParamBases = cpuRotParamBases.data();
        pass.rotParamBasesSize = sizeof(cpuRotParamBases);
        pass.hres = HRes;
        pass.numLines = numLines;
        gpu->RenderLayers(pass);
    }

    void VDP2ComposeLines(uint32 y) {
        // Bail out if there's nothing to render
        if (y < nextComposeLine) {
            return;
        }

        cpuCommonRenderParams.startY = nextComposeLine;

        const uint32 numLines = y - nextComposeLine + 1;
        nextComposeLine = y + 1;

        metal::ComposePassParams pass{};
        pass.common = &cpuCommonRenderParams;
        pass.commonSize = sizeof(cpuCommonRenderParams);
        pass.composeParams = &cpuComposeParams;
        pass.composeParamsSize = sizeof(cpuComposeParams);
        pass.lnclBack = cpuLnclBack.data();
        pass.lnclBackSize = sizeof(cpuLnclBack);
        pass.hres = HRes;
        pass.numLines = numLines;
        gpu->Compose(pass);
    }

    void VDP2RenderLine(uint32 y) {
        if (y == 0) {
            // The frame starts before VDP1 swaps its framebuffers (at the end of the last line of the vertical blank),
            // so the framebuffer to show is only known now
            VDP2UploadSpriteFB();
        }

        VDP2CalcAccessPatterns();
        VDP2CalcVCellScrollDelay();
        VDP2DrawLineColorBackScreens(y);
        VDP2UpdateRotationParameterBases(y);
        vdpState.state2.UpdateRotationPageBaseAddresses(vdpState.regs2);

        // When Y=0, the changes happened during vblank (or, more precisely, between the last Y of the previous frame
        // and the first line of this frame). Otherwise, the changes happened between Y-1 and Y. Therefore, we need to
        // render lines up to Y-1 then sync the state, unless Y=0, in which case we just sync the state.

        if (y > 0) {
            const bool cramDirty = cramGeneration != appliedCramGeneration;
            const bool layerRenderParamsDirty = layerRenderParamsGeneration != appliedLayerRenderParamsGeneration;
            const bool composeParamsDirty = composeParamsGeneration != appliedComposeParamsGeneration;
            const bool renderLayers = vramAnyDirty || cramDirty || layerRenderParamsDirty || composeParamsDirty;
            const bool compose = composeParamsDirty;
            if (renderLayers) {
                VDP2RenderLayerLines(y - 1);
            }
            if (compose) {
                VDP2ComposeLines(y - 1);
            }
        }

        VDP2UpdateState(y);
    }

    void VDP2EndFrame() {
        const uint32 vShift = vdpState.regs2.TVMD.IsInterlaced() ? 1u : 0u;
        const uint32 vres = VRes >> vShift;
        VDP2RenderLayerLines(vres - 1);
        VDP2ComposeLines(VRes - 1);

        // The image of the previous frame is ready by now; this one follows
        const metal::FinishedFrame finished = gpu->FinishFrame(HRes, VRes);
        if (finished.pixels != nullptr) {
            owner.SwCallbacks.FrameComplete(const_cast<uint32 *>(finished.pixels), finished.width, finished.height, 1);
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------------

MetalVDPRenderer::MetalVDPRenderer(VDPState &state, const config::VDP2DebugRender &vdp2DebugRenderOptions,
                                   const config::VDP2AccessPatternsConfig &vdp2AccessPatternsConfig)
    : SoftwareVDPRenderer(state, vdp2DebugRenderOptions, vdp2AccessPatternsConfig)
    , m_impl(std::make_unique<Impl>(*this, state, vdp2AccessPatternsConfig, vdp2DebugRenderOptions)) {}

MetalVDPRenderer::~MetalVDPRenderer() = default;

util::VoidResult<> MetalVDPRenderer::Initialize(void *device) {
    std::string error;
    m_impl->gpu = metal::Gpu::Create(device, error);
    if (!m_impl->gpu) {
        return util::ErrorMessage{error};
    }
    // The software renderer's own VDP2 pass is never used, so it doesn't need a thread
    EnableThreadedVDP2(false);
    m_impl->Reset();
    return {};
}

util::ObjectResult<MetalVDPRenderer>
MetalVDPRenderer::Create(VDPState &state, const config::VDP2DebugRender &vdp2DebugRenderOptions,
                         const config::VDP2AccessPatternsConfig &vdp2AccessPatternsConfig, void *device) {
    if (device == nullptr) {
        return util::ErrorMessage{"No Metal device provided"};
    }
    std::unique_ptr<MetalVDPRenderer> renderer{new MetalVDPRenderer(state, vdp2DebugRenderOptions, vdp2AccessPatternsConfig)};
    util::VoidResult<> result = renderer->Initialize(device);
    if (!result) {
        return result.Error();
    }
    return renderer;
}

// -----------------------------------------------------------------------------
// Configuration

void MetalVDPRenderer::UpdateEnhancements() {
    // These aren't available with the Metal renderer yet; VDP1 is drawn at the native resolution only
    m_enhancements.deinterlace = false;
    m_enhancements.transparentMeshes = false;
    m_enhancements.internalResolution = 1;
    m_enhancements.skipNativeVDP1 = false;
    SoftwareVDPRenderer::UpdateEnhancements();
    m_impl->UpdateEnhancements();
}

// -----------------------------------------------------------------------------
// Basics and save states

void MetalVDPRenderer::Reset(bool hard) {
    SoftwareVDPRenderer::Reset(hard);
    m_impl->Reset();
}

void MetalVDPRenderer::PostLoadStateSync() {
    SoftwareVDPRenderer::PostLoadStateSync();
    m_impl->Reset();
}

// -----------------------------------------------------------------------------
// VDP2 memory and register writes

void MetalVDPRenderer::VDP2WriteVRAM(uint32 address, uint8 value) {
    m_impl->VDP2WriteVRAM(address);
}

void MetalVDPRenderer::VDP2WriteVRAM(uint32 address, uint16 value) {
    // The address is always word-aligned
    m_impl->VDP2WriteVRAM(address);
}

void MetalVDPRenderer::VDP2WriteCRAM(uint32 address, uint8 value) {
    m_impl->VDP2WriteCRAM(address);
}

void MetalVDPRenderer::VDP2WriteCRAM(uint32 address, uint16 value) {
    // The address is always word-aligned
    m_impl->VDP2WriteCRAM(address);
}

void MetalVDPRenderer::VDP2WriteReg(uint32 address, uint16 value) {
    m_impl->VDP2WriteReg(address, value);
}

// -----------------------------------------------------------------------------
// Rendering process

void MetalVDPRenderer::VDP2SetResolution(uint32 h, uint32 v, bool exclusive) {
    // (the software renderer needs the resolution for VDP1 too)
    SoftwareVDPRenderer::VDP2SetResolution(h, v, exclusive);
    m_impl->VDP2SetResolution(h, v, exclusive);
}

void MetalVDPRenderer::VDP2BeginFrame() {
    m_impl->VDP2BeginFrame();
}

void MetalVDPRenderer::VDP2RenderLine(uint32 y) {
    m_impl->VDP2RenderLine(y);
}

void MetalVDPRenderer::VDP2EndFrame() {
    m_impl->VDP2EndFrame();
    Callbacks.VDP2DrawFinished();
    perf::frames.fetch_add(1, std::memory_order_relaxed);
}

// -----------------------------------------------------------------------------
// Debugger

void MetalVDPRenderer::UpdateEnabledLayers() {
    m_impl->VDP2UpdateEnabledLayers();
}

} // namespace ymir::vdp
