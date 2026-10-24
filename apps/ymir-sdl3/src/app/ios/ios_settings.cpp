#include "ios_settings.hpp"

#include <app/events/emu_event_factory.hpp>
#include <app/events/gui_event_factory.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>

namespace app::ios {

namespace {

    template <typename T>
    struct Accessor {
        std::function<T()> get;
        std::function<void(T)> set;
    };

    struct Tables {
        std::unordered_map<std::string, Accessor<bool>> bools;
        std::unordered_map<std::string, Accessor<int>> ints;
        std::unordered_map<std::string, Accessor<float>> floats;
    };

} // namespace

void InstallSettingsAccessors(HostInterface &host, SharedContext &context, Settings &settings, SettingsHooks hooks) {
    using namespace ymir;

    auto tables = std::make_shared<Tables>();
    SharedContext *ctx = &context;
    Settings *cfg = &settings;

    // All setters mark the settings as dirty so that they get saved
    auto addBool = [&](const char *key, std::function<bool()> get, std::function<void(bool)> set) {
        tables->bools[key] = {std::move(get), [cfg, set = std::move(set)](bool value) {
                                  set(value);
                                  cfg->MakeDirty();
                              }};
    };
    auto addInt = [&](const char *key, std::function<int()> get, std::function<void(int)> set) {
        tables->ints[key] = {std::move(get), [cfg, set = std::move(set)](int value) {
                                 set(value);
                                 cfg->MakeDirty();
                             }};
    };
    auto addFloat = [&](const char *key, std::function<float()> get, std::function<void(float)> set) {
        tables->floats[key] = {std::move(get), [cfg, set = std::move(set)](float value) {
                                   set(value);
                                   cfg->MakeDirty();
                               }};
    };

    // -- Video --
    addBool("video.forceIntegerScaling", [cfg] { return cfg->video.forceIntegerScaling; },
            [cfg](bool v) { cfg->video.forceIntegerScaling = v; });
    addBool("video.forceAspectRatio", [cfg] { return cfg->video.forceAspectRatio; },
            [cfg](bool v) { cfg->video.forceAspectRatio = v; });
    addInt(
        "video.aspect",
        [cfg] {
            const Ratio r = cfg->video.forcedAspect;
            if (r == Ratio{3, 2}) {
                return 1;
            } else if (r == Ratio{16, 10}) {
                return 2;
            } else if (r == Ratio{16, 9}) {
                return 3;
            }
            return 0; // 4:3
        },
        [cfg](int v) {
            switch (v) {
            case 1: cfg->video.forcedAspect = {3, 2}; break;
            case 2: cfg->video.forcedAspect = {16, 10}; break;
            case 3: cfg->video.forcedAspect = {16, 9}; break;
            default: cfg->video.forcedAspect = {4, 3}; break;
            }
        });
    addInt("video.rotation", [cfg] { return static_cast<int>(cfg->video.rotation); },
           [cfg](int v) { cfg->video.rotation = static_cast<Settings::Video::DisplayRotation>(std::clamp(v, 0, 3)); });
    addBool("video.hardwareAcceleration", [cfg] { return cfg->video.useHardwareAcceleration.Get(); },
            [cfg](bool v) { cfg->video.useHardwareAcceleration = v; });
    addBool("video.deinterlace", [cfg] { return cfg->video.enhancements.deinterlace.Get(); },
            [cfg](bool v) { cfg->video.enhancements.deinterlace = v; });
    addBool("video.transparentMeshes", [cfg] { return cfg->video.enhancements.transparentMeshes.Get(); },
            [cfg](bool v) { cfg->video.enhancements.transparentMeshes = v; });
    addBool("video.threadedVDP1", [cfg] { return cfg->video.swRenderer.threadedVDP1.Get(); },
            [cfg](bool v) { cfg->video.swRenderer.threadedVDP1 = v; });
    addBool("video.threadedVDP2", [cfg] { return cfg->video.swRenderer.threadedVDP2.Get(); },
            [cfg](bool v) { cfg->video.swRenderer.threadedVDP2 = v; });
    addBool("video.threadedDeinterlacer", [cfg] { return cfg->video.swRenderer.threadedDeinterlacer.Get(); },
            [cfg](bool v) { cfg->video.swRenderer.threadedDeinterlacer = v; });
    addBool("video.sync", [cfg] { return cfg->video.syncInFullscreenMode; },
            [cfg](bool v) { cfg->video.syncInFullscreenMode = v; });
    addBool("video.fullRefreshRate", [cfg] { return cfg->video.useFullRefreshRateWithVideoSync; },
            [cfg](bool v) { cfg->video.useFullRefreshRateWithVideoSync = v; });
    addBool("video.reduceLatency", [cfg] { return cfg->video.reduceLatency; },
            [cfg](bool v) { cfg->video.reduceLatency = v; });

    // -- Audio --
    addFloat("audio.volume", [cfg] { return cfg->audio.volume.Get(); },
             [cfg](float v) { cfg->audio.volume = std::clamp(v, 0.0f, 1.0f); });
    addBool("audio.mute", [cfg] { return cfg->audio.mute.Get(); }, [cfg](bool v) { cfg->audio.mute = v; });
    addInt("audio.interpolation", [cfg] { return static_cast<int>(cfg->audio.interpolation.Get()); },
           [cfg](int v) {
               cfg->audio.interpolation = v == 0 ? core::config::audio::SampleInterpolationMode::NearestNeighbor
                                                 : core::config::audio::SampleInterpolationMode::Linear;
           });
    addBool("audio.threadedSCSP", [cfg] { return cfg->audio.threadedSCSP.Get(); },
            [cfg](bool v) { cfg->audio.threadedSCSP = v; });
    addInt("audio.stepGranularity", [cfg] { return static_cast<int>(cfg->audio.stepGranularity.Get()); },
           [cfg](int v) { cfg->audio.stepGranularity = static_cast<uint32>(std::clamp(v, 0, 5)); });

    // -- System --
    addBool("system.autodetectRegion", [cfg] { return cfg->system.autodetectRegion.Get(); },
            [cfg](bool v) { cfg->system.autodetectRegion = v; });
    addInt("system.videoStandard", [cfg] { return static_cast<int>(cfg->system.videoStandard.Get()); },
           [cfg, ctx](int v) {
               const auto standard = v == 0 ? core::config::sys::VideoStandard::NTSC : core::config::sys::VideoStandard::PAL;
               ctx->EnqueueEvent(events::emu::SetVideoStandard(standard));
               cfg->system.videoStandard = standard;
           });
    addBool("system.emulateSH2Cache", [cfg] { return cfg->system.emulateSH2Cache; },
            [cfg, ctx](bool v) {
                ctx->EnqueueEvent(events::emu::SetEmulateSH2Cache(v));
                cfg->system.emulateSH2Cache = v;
            });
    addInt("system.sh2ClockFactor", [cfg] { return static_cast<int>(cfg->system.sh2ClockFactor.Get()); },
           [cfg](int v) {
               cfg->system.sh2ClockFactor =
                   static_cast<uint32>(std::clamp<int>(v, config_defaults::system::kMinSH2ClockFactor,
                                                       config_defaults::system::kMaxSH2ClockFactor));
           });
    addBool("system.backupRAMPerGame", [cfg] { return cfg->system.internalBackupRAMPerGame; },
            [cfg, ctx](bool v) {
                cfg->system.internalBackupRAMPerGame = v;
                ctx->EnqueueEvent(events::emu::LoadInternalBackupMemory());
            });
    addInt("system.rtcMode", [cfg] { return static_cast<int>(cfg->system.rtc.mode.Get()); },
           [cfg](int v) {
               cfg->system.rtc.mode = v == 0 ? core::config::rtc::Mode::Host : core::config::rtc::Mode::Virtual;
           });
    addInt("system.rtcResetStrategy", [cfg] { return static_cast<int>(cfg->system.rtc.virtHardResetStrategy.Get()); },
           [cfg](int v) {
               using Strategy = core::config::rtc::HardResetStrategy;
               cfg->system.rtc.virtHardResetStrategy =
                   v == 1 ? Strategy::ResetToFixedTime : (v == 2 ? Strategy::Preserve : Strategy::SyncToHost);
           });
    addInt("system.iplVariant", [cfg] { return static_cast<int>(cfg->system.ipl.variant); },
           [cfg](int v) { cfg->system.ipl.variant = static_cast<db::SystemVariant>(std::clamp(v, 0, 4)); });

    // -- CD block --
    addInt("cdblock.readSpeed", [cfg] { return static_cast<int>(cfg->cdblock.readSpeedFactor.Get()); },
           [cfg](int v) { cfg->cdblock.readSpeedFactor = static_cast<uint8>(std::clamp(v, 2, 200)); });
    addBool("cdblock.useLLE", [cfg] { return cfg->cdblock.useLLE; },
            [cfg, ctx](bool v) {
                cfg->cdblock.useLLE = v;
                ctx->EnqueueEvent(events::emu::SetCDBlockLLE(v));
            });

    // -- Cartridge --
    addInt("cartridge.type", [cfg] { return static_cast<int>(cfg->cartridge.type); },
           [cfg, ctx](int v) {
               cfg->cartridge.type = static_cast<Settings::Cartridge::Type>(std::clamp(v, 0, 3));
               ctx->EnqueueEvent(events::emu::InsertCartridgeFromSettings());
           });
    addInt("cartridge.dramCapacity", [cfg] { return static_cast<int>(cfg->cartridge.dram.capacity); },
           [cfg, ctx](int v) {
               cfg->cartridge.dram.capacity = static_cast<Settings::Cartridge::DRAM::Capacity>(std::clamp(v, 0, 2));
               ctx->EnqueueEvent(events::emu::InsertCartridgeFromSettings());
           });
    addInt("cartridge.backupCapacity", [cfg] { return static_cast<int>(cfg->cartridge.backupRAM.capacity); },
           [cfg, ctx](int v) {
               cfg->cartridge.backupRAM.capacity =
                   static_cast<Settings::Cartridge::BackupRAM::Capacity>(std::clamp(v, 0, 3));
               ctx->EnqueueEvent(events::emu::InsertCartridgeFromSettings());
           });
    addBool("cartridge.autoLoad", [cfg] { return cfg->cartridge.autoLoadGameCarts; },
            [cfg](bool v) { cfg->cartridge.autoLoadGameCarts = v; });

    // -- General --
    addFloat("general.speedMain", [cfg] { return static_cast<float>(cfg->general.mainSpeedFactor.Get()); },
             [cfg](float v) { cfg->general.mainSpeedFactor = std::clamp(std::round(v * 10.0) / 10.0, 0.1, 5.0); });
    addFloat("general.speedAlt", [cfg] { return static_cast<float>(cfg->general.altSpeedFactor.Get()); },
             [cfg](float v) { cfg->general.altSpeedFactor = std::clamp(std::round(v * 10.0) / 10.0, 0.1, 5.0); });
    addBool("general.useAltSpeed", [cfg] { return cfg->general.useAltSpeed.Get(); },
            [cfg](bool v) { cfg->general.useAltSpeed = v; });
    addBool("general.unpauseOnDiscLoad", [cfg] { return cfg->general.unpauseOnDiscLoad; },
            [cfg](bool v) { cfg->general.unpauseOnDiscLoad = v; });
    addBool("general.preloadDiscImages", [cfg] { return cfg->general.preloadDiscImagesToRAM; },
            [cfg](bool v) { cfg->general.preloadDiscImagesToRAM = v; });
    addBool("general.boostEmuThread", [cfg] { return cfg->general.boostEmuThreadPriority; },
            [cfg, ctx](bool v) {
                cfg->general.boostEmuThreadPriority = v;
                ctx->EnqueueEvent(events::emu::SetThreadPriority(v));
            });
    addBool("general.rewind", [cfg] { return cfg->general.enableRewindBuffer; },
            [cfg, ctx](bool v) {
                cfg->general.enableRewindBuffer = v;
                ctx->EnqueueEvent(events::gui::EnableRewindBuffer(v));
            });
    addInt("general.rewindCompression", [cfg] { return cfg->general.rewindCompressionLevel; },
           [cfg, ctx](int v) {
               cfg->general.rewindCompressionLevel = std::clamp(v, 0, 16);
               ctx->rewindBuffer.LZ4Accel = 1 << (16 - cfg->general.rewindCompressionLevel);
           });
    addInt("general.screenshotScale", [cfg] { return cfg->general.screenshotScale; },
           [cfg](int v) { cfg->general.screenshotScale = std::clamp(v, 1, 4); });

    // -- GUI --
    addBool("gui.showFrameRate", [cfg] { return cfg->gui.showFrameRateOSD; },
            [cfg](bool v) { cfg->gui.showFrameRateOSD = v; });
    addInt("gui.frameRatePosition", [cfg] { return static_cast<int>(cfg->gui.frameRateOSDPosition); },
           [cfg](int v) {
               cfg->gui.frameRateOSDPosition = static_cast<Settings::GUI::FrameRateOSDPosition>(std::clamp(v, 0, 3));
           });
    addBool("gui.showMessages", [cfg] { return cfg->gui.showMessages; },
            [cfg](bool v) { cfg->gui.showMessages = v; });
    addBool("gui.showSpeedIndicator", [cfg] { return cfg->gui.showSpeedIndicatorForAllSpeeds; },
            [cfg](bool v) { cfg->gui.showSpeedIndicatorForAllSpeeds = v; });

    // -- Input --
    for (int port = 0; port < 2; ++port) {
        const std::string key = "input.port" + std::to_string(port + 1) + "Type";
        tables->ints[key] = {
            [cfg, port] { return static_cast<int>(cfg->input.ports[port].type.Get()); },
            [cfg, port](int v) {
                cfg->input.ports[port].type = static_cast<ymir::peripheral::PeripheralType>(std::clamp(v, 0, 6));
                cfg->MakeDirty();
            }};
    }
    addFloat("input.lsDeadzone", [cfg] { return cfg->input.gamepad.lsDeadzone.Get(); },
             [cfg](float v) { cfg->input.gamepad.lsDeadzone = std::clamp(v, 0.05f, 0.9f); });
    addFloat("input.rsDeadzone", [cfg] { return cfg->input.gamepad.rsDeadzone.Get(); },
             [cfg](float v) { cfg->input.gamepad.rsDeadzone = std::clamp(v, 0.05f, 0.9f); });
    addFloat("input.analogToDigital", [cfg] { return cfg->input.gamepad.analogToDigitalSensitivity.Get(); },
             [cfg](float v) { cfg->input.gamepad.analogToDigitalSensitivity = std::clamp(v, 0.05f, 0.9f); });

    // -- On-screen controls --
    tables->bools["touch.enabled"] = {hooks.getTouchControls, hooks.setTouchControls};

    host.getBool = [tables](const std::string &key, bool defaultValue) {
        auto it = tables->bools.find(key);
        return it != tables->bools.end() && it->second.get ? it->second.get() : defaultValue;
    };
    host.setBool = [tables](const std::string &key, bool value) {
        if (auto it = tables->bools.find(key); it != tables->bools.end() && it->second.set) {
            it->second.set(value);
        }
    };
    host.getInt = [tables](const std::string &key, int defaultValue) {
        auto it = tables->ints.find(key);
        return it != tables->ints.end() && it->second.get ? it->second.get() : defaultValue;
    };
    host.setInt = [tables](const std::string &key, int value) {
        if (auto it = tables->ints.find(key); it != tables->ints.end() && it->second.set) {
            it->second.set(value);
        }
    };
    host.getFloat = [tables](const std::string &key, float defaultValue) {
        auto it = tables->floats.find(key);
        return it != tables->floats.end() && it->second.get ? it->second.get() : defaultValue;
    };
    host.setFloat = [tables](const std::string &key, float value) {
        if (auto it = tables->floats.find(key); it != tables->floats.end() && it->second.set) {
            it->second.set(value);
        }
    };
}

} // namespace app::ios
