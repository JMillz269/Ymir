#pragma once

#include "host_interface.hpp"

#include <app/settings.hpp>
#include <app/services/gfx/gfx_context.hpp>
#include <app/shared_context.hpp>

#include <functional>

namespace app::ios {

/// @brief Things the settings table needs that live outside of Settings.
struct SettingsHooks {
    std::function<bool()> getTouchControls;
    std::function<void(bool)> setTouchControls;
};

/// @brief Applies the upscaler and picture filter settings to the graphics context.
void ApplyVideoFilters(gfx::IGraphicsContext &graphics, const Settings &settings);

/// @brief Fills in the getBool/setBool/getInt/setInt/getFloat/setFloat callbacks of the host interface.
///
/// Every key maps to one option of the emulator's settings, mirroring what the desktop settings window changes (including
/// the events that apply a change to the running emulator). Unknown keys are ignored and return their default value.
///
/// @param[in] host the host interface to fill in
/// @param[in] context the shared context
/// @param[in] settings the settings; must outlive the callbacks
/// @param[in] hooks accessors for state outside of the settings
void InstallSettingsAccessors(HostInterface &host, SharedContext &context, Settings &settings, SettingsHooks hooks);

} // namespace app::ios
