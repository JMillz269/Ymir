#pragma once

#include "input_context.hpp"

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_video.h>

#include <imgui.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <optional>
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

    // -- Layout customization --

    /// @brief Sets the file where the layout is stored and loads it if it exists.
    void SetStoragePath(std::filesystem::path path);

    /// @brief Enters or leaves layout editing. While editing, controls can be dragged and don't send inputs.
    /// The layout is saved when leaving the editing mode.
    void SetEditing(bool editing);
    bool IsEditing() const {
        return m_editing;
    }

    /// @brief Index of the control being edited, or -1 if none is selected.
    int SelectedElement() const {
        return m_selected;
    }

    /// @brief Size multiplier of the selected control (1.0 if none is selected).
    float GetSelectedScale() const;
    void SetSelectedScale(float scale);

    /// @brief Size multiplier applied to all controls.
    float GetGlobalScale() const {
        return m_globalScale;
    }
    void SetGlobalScale(float scale);

    /// @brief Opacity of the controls, from 0.1 to 1.0.
    float GetOpacity() const {
        return m_opacity;
    }
    void SetOpacity(float opacity);

    /// @brief Whether moved controls snap to a grid.
    bool GetSnapToGrid() const {
        return m_snapToGrid;
    }
    void SetSnapToGrid(bool snap);

    /// @brief Restores the default placement and sizes of the controls and of the game screen in the current
    /// orientation.
    void ResetLayout();

    /// @brief Where the player moved the game screen to, if they did, in the current orientation.
    struct ScreenOverride {
        float centerX; ///< center of the screen, from 0 to 1 across the window
        float centerY;
        float scale; ///< size multiplier relative to the normal size
    };
    std::optional<ScreenOverride> GetScreenOverride() const;

    /// @brief Tells the editor where the game screen is drawn, so that it can be selected and moved.
    void SetScreenRect(ImVec2 pos, ImVec2 size);

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

    /// @brief Forgets the fingers on the screen and releases every control, for when something else takes the touches
    /// (a native screen over the emulator, the app going to the background).
    void CancelTouches();

    /// @brief Draws the overlay on the foreground draw list.
    /// @param[in] atlas texture with the controller artwork; if not provided, plain shapes are drawn instead
    void Draw(ImDrawList *drawList, std::optional<ImTextureID> atlas = std::nullopt) const;

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

    // Movable parts of the overlay: the D-pad as a whole, and every other button
    enum Element {
        ElDPad,
        ElA,
        ElB,
        ElC,
        ElX,
        ElY,
        ElZ,
        ElL,
        ElR,
        ElStart,
        ElMenu,
        ElCount,
    };

    // Where an element is, relative to the safe area, and how large it is
    struct Placement {
        float x = 0.0f;
        float y = 0.0f;
        float scale = 1.0f;
    };

    using Placements = std::array<Placement, ElCount>;

    // The game screen is selected and edited like an element, with this index
    static constexpr int kScreenElement = ElCount;

    struct ScreenPlacement {
        bool custom = false;
        float x = 0.5f; // center, relative to the window
        float y = 0.5f;
        float scale = 1.0f;
    };

    struct Layout {
        ImVec2 dpadCenter;
        float dpadRadius;
        std::array<ImVec2, _Count> centers;
        std::array<ImVec2, _Count> halfSizes; // for pill-shaped buttons; x==y means circle
    };

    void ComputeLayout(SDL_Window *window, ImVec2 displaySize);
    int HitTestElement(ImVec2 pos) const;
    void ProcessEditingFingerEvent(const SDL_Event &evt, ImVec2 pos);
    void Load();
    void Save();
    void MarkDirty();
    ImVec2 ElementCenter(int element) const;
    void ApplyState(const std::array<bool, _Count> &state);
    void ReleaseAll();
    void DropEndedFingers();

    InputContext &m_inputContext;
    bool m_enabled = true;
    bool m_hardwareGamepad = false;
    bool m_menuRequested = false;
    bool m_gamepadConnected = false;

    Layout m_layout{};

    // Customization
    std::filesystem::path m_storagePath;
    std::filesystem::path m_logPath; ///< where fingers SDL no longer reports are noted (next to the layout)
    bool m_editing = false;
    bool m_snapToGrid = false;
    float m_globalScale = 1.0f;
    float m_opacity = 0.9f;
    std::array<Placements, 2> m_placements{}; // [0]=portrait, [1]=landscape
    std::array<bool, 2> m_placementsReady{};
    std::array<uint32, 2> m_loadedElements{};
    int m_orientation = 0;
    int m_selected = -1;
    SDL_FingerID m_dragFinger = 0;
    bool m_dragging = false;
    ImVec2 m_dragOffset{};
    ImVec4 m_safeRect{}; // left, top, right, bottom
    ImVec2 m_displaySize{};
    std::array<ScreenPlacement, 2> m_screenPlacements{};
    ImVec4 m_screenRect{}; // x, y, width, height of the game screen as drawn
    float m_gridSize = 0.0f;
    bool m_dirty = false;
    std::chrono::steady_clock::time_point m_dirtyTime;
    std::unordered_map<SDL_FingerID, ImVec2> m_fingers;
    std::array<bool, _Count> m_state{};
};

} // namespace app::input
