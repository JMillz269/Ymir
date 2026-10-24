#pragma once

#include "input_context.hpp"

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_video.h>

#include <imgui.h>

#include <array>
#include <unordered_map>

namespace app::input {

/// @brief On-screen Saturn control pad for touch-only devices.
///
/// Touches are tracked per finger and translated into gamepad primitives for player 0, so the default Control Pad
/// bindings for port 1 work without any configuration. Fingers may slide between buttons.
class TouchControls {
public:
    explicit TouchControls(InputContext &inputContext);

    /// @brief Enables or disables the overlay. Disabling releases all virtual inputs.
    void SetEnabled(bool enabled);
    bool IsEnabled() const {
        return m_enabled;
    }

    /// @brief Set by the main menu while a hardware gamepad is connected to avoid a ghost player 0.
    void SetHardwareGamepadPresent(bool present);

    /// @brief Whether the on-screen "Menu" button asked for the main menu bar to be shown.
    bool IsMenuRequested() const {
        return m_menuRequested;
    }
    void ClearMenuRequest() {
        m_menuRequested = false;
    }

    /// @brief Feeds an SDL finger event. Ignored if ImGui owns the pointer.
    /// @param[in] evt the SDL event (SDL_EVENT_FINGER_*)
    /// @param[in] displaySize size of the ImGui display, which finger coordinates are scaled to
    /// @param[in] imguiWantsPointer true if ImGui wants to capture the pointer
    void ProcessFingerEvent(const SDL_Event &evt, ImVec2 displaySize, bool imguiWantsPointer);

    /// @brief Recomputes the pressed controls from active fingers and emits input primitives.
    void Update(SDL_Window *window, ImVec2 displaySize);

    /// @brief Draws the overlay on the foreground draw list.
    void Draw(ImDrawList *drawList) const;

private:
    enum Control {
        Up,
        Down,
        Left,
        Right,
        A,
        B,
        C,
        X,
        Y,
        Z,
        L,
        R,
        Start,
        Menu,
        _Count,
    };

    struct Layout {
        ImVec2 dpadCenter;
        float dpadRadius;
        std::array<ImVec2, _Count> centers;
        std::array<ImVec2, _Count> halfSizes; // for pill-shaped buttons; x==y means circle
    };

    void ComputeLayout(SDL_Window *window, ImVec2 displaySize);
    void ApplyState(const std::array<bool, _Count> &state);
    void ReleaseAll();

    InputContext &m_inputContext;
    bool m_enabled = true;
    bool m_hardwareGamepad = false;
    bool m_menuRequested = false;
    bool m_gamepadConnected = false;

    Layout m_layout{};
    std::unordered_map<SDL_FingerID, ImVec2> m_fingers;
    std::array<bool, _Count> m_state{};
};

} // namespace app::input
