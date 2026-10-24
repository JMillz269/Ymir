#include "touch_controls.hpp"

#include <algorithm>
#include <cmath>

namespace app::input {

namespace {

    constexpr uint32 kPlayerIndex = 0;

    constexpr const char *kLabels[] = {"", "", "", "", "A", "B", "C", "X", "Y", "Z", "L", "R", "START", "MENU"};

    bool InsideEllipse(ImVec2 p, ImVec2 center, ImVec2 halfSize, float slop) {
        const float dx = (p.x - center.x) / (halfSize.x + slop);
        const float dy = (p.y - center.y) / (halfSize.y + slop);
        return dx * dx + dy * dy <= 1.0f;
    }

} // namespace

TouchControls::TouchControls(InputContext &inputContext)
    : m_inputContext(inputContext) {}

void TouchControls::SetEnabled(bool enabled) {
    if (m_enabled == enabled) {
        return;
    }
    m_enabled = enabled;
    if (!m_enabled) {
        m_fingers.clear();
        ReleaseAll();
    }
}

void TouchControls::SetHardwareGamepadPresent(bool present) {
    if (m_hardwareGamepad == present) {
        return;
    }
    m_hardwareGamepad = present;
    if (present) {
        m_fingers.clear();
        ReleaseAll();
    }
}

void TouchControls::ProcessFingerEvent(const SDL_Event &evt, ImVec2 displaySize, bool imguiWantsPointer) {
    if (!m_enabled || m_hardwareGamepad) {
        return;
    }

    const SDL_TouchFingerEvent &finger = evt.tfinger;
    const ImVec2 pos{finger.x * displaySize.x, finger.y * displaySize.y};
    switch (evt.type) {
    case SDL_EVENT_FINGER_DOWN:
        // Don't steal touches from menus and windows
        if (!imguiWantsPointer) {
            m_fingers[finger.fingerID] = pos;
        }
        break;
    case SDL_EVENT_FINGER_MOTION:
        if (auto it = m_fingers.find(finger.fingerID); it != m_fingers.end()) {
            it->second = pos;
        }
        break;
    case SDL_EVENT_FINGER_UP: [[fallthrough]];
    case SDL_EVENT_FINGER_CANCELED: m_fingers.erase(finger.fingerID); break;
    default: break;
    }
}

void TouchControls::ComputeLayout(SDL_Window *window, ImVec2 displaySize) {
    // Keep controls clear of notches, the Dynamic Island and the home indicator
    SDL_Rect safe{0, 0, (int)displaySize.x, (int)displaySize.y};
    if (window != nullptr) {
        SDL_GetWindowSafeArea(window, &safe);
    }
    const float left = (float)safe.x;
    const float top = (float)safe.y;
    const float right = (float)(safe.x + safe.w);
    const float bottom = (float)(safe.y + safe.h);
    const float unit = std::min(right - left, bottom - top);

    Layout &l = m_layout;

    // D-pad on the left, vertically in the lower half
    l.dpadRadius = unit * 0.24f;
    l.dpadCenter = {left + l.dpadRadius * 1.25f, bottom - l.dpadRadius * 1.35f};
    const float arm = l.dpadRadius * 0.62f;
    l.centers[Up] = {l.dpadCenter.x, l.dpadCenter.y - arm};
    l.centers[Down] = {l.dpadCenter.x, l.dpadCenter.y + arm};
    l.centers[Left] = {l.dpadCenter.x - arm, l.dpadCenter.y};
    l.centers[Right] = {l.dpadCenter.x + arm, l.dpadCenter.y};
    for (int c : {Up, Down, Left, Right}) {
        l.halfSizes[c] = {l.dpadRadius * 0.42f, l.dpadRadius * 0.42f};
    }

    // Face buttons on the right in the Saturn's two-row arrangement (X Y Z over A B C)
    const float rb = unit * 0.095f;
    const float pitch = rb * 2.25f;
    const float rowBottomY = bottom - rb * 1.7f;
    const float rowTopY = rowBottomY - pitch * 0.95f;
    const float colRight = right - rb * 1.6f;
    const int bottomRow[] = {A, B, C};
    const int topRow[] = {X, Y, Z};
    for (int i = 0; i < 3; ++i) {
        const float x = colRight - pitch * (2 - i);
        l.centers[bottomRow[i]] = {x, rowBottomY};
        l.centers[topRow[i]] = {x - pitch * 0.35f, rowTopY};
        l.halfSizes[bottomRow[i]] = {rb, rb};
        l.halfSizes[topRow[i]] = {rb, rb};
    }

    // Shoulder buttons at the top corners
    const float shoulderW = unit * 0.20f;
    const float shoulderH = unit * 0.075f;
    l.centers[L] = {left + shoulderW * 1.1f, top + shoulderH * 1.6f};
    l.centers[R] = {right - shoulderW * 1.1f, top + shoulderH * 1.6f};
    l.halfSizes[L] = l.halfSizes[R] = {shoulderW, shoulderH};

    // Start and Menu along the bottom edge
    const float midX = (left + right) * 0.5f;
    const float pillW = unit * 0.13f;
    const float pillH = unit * 0.05f;
    l.centers[Menu] = {midX - pillW * 1.3f, bottom - pillH * 1.8f};
    l.centers[Start] = {midX + pillW * 1.3f, bottom - pillH * 1.8f};
    l.halfSizes[Menu] = l.halfSizes[Start] = {pillW, pillH};
}

void TouchControls::Update(SDL_Window *window, ImVec2 displaySize) {
    if (!m_enabled || m_hardwareGamepad) {
        return;
    }

    ComputeLayout(window, displaySize);

    std::array<bool, _Count> state{};
    for (const auto &[id, pos] : m_fingers) {
        // D-pad: one finger drives up to two adjacent directions, so diagonals work and sliding feels natural
        const float dx = pos.x - m_layout.dpadCenter.x;
        const float dy = pos.y - m_layout.dpadCenter.y;
        const float dist = std::sqrt(dx * dx + dy * dy);
        if (dist <= m_layout.dpadRadius * 1.5f && dist >= m_layout.dpadRadius * 0.18f) {
            // Inclusion threshold: sin(22.5deg) of the distance on the minor axis counts as a diagonal
            const float t = dist * 0.38f;
            state[Left] |= dx < -t;
            state[Right] |= dx > t;
            state[Up] |= dy < -t;
            state[Down] |= dy > t;
            continue;
        }

        for (int c = A; c < _Count; ++c) {
            if (InsideEllipse(pos, m_layout.centers[c], m_layout.halfSizes[c], m_layout.halfSizes[c].y * 0.3f)) {
                state[c] = true;
            }
        }
    }

    ApplyState(state);
}

void TouchControls::ApplyState(const std::array<bool, _Count> &state) {
    if (!m_gamepadConnected) {
        m_inputContext.ConnectGamepad(kPlayerIndex);
        m_gamepadConnected = true;
    }

    auto button = [&](Control c, GamepadButton btn) {
        if (state[c] != m_state[c]) {
            m_inputContext.ProcessPrimitive(kPlayerIndex, btn, state[c]);
        }
    };
    auto trigger = [&](Control c, GamepadAxis1D axis) {
        if (state[c] != m_state[c]) {
            m_inputContext.ProcessPrimitive(kPlayerIndex, axis, state[c] ? 1.0f : 0.0f);
        }
    };

    button(Up, GamepadButton::DpadUp);
    button(Down, GamepadButton::DpadDown);
    button(Left, GamepadButton::DpadLeft);
    button(Right, GamepadButton::DpadRight);

    // Same physical layout the SDL gamepad defaults use: A=X, B=A, C=B, X=LB, Y=Y, Z=RB
    button(A, GamepadButton::X);
    button(B, GamepadButton::A);
    button(C, GamepadButton::B);
    button(X, GamepadButton::LeftBumper);
    button(Y, GamepadButton::Y);
    button(Z, GamepadButton::RightBumper);
    button(Start, GamepadButton::Start);
    trigger(L, GamepadAxis1D::LeftTrigger);
    trigger(R, GamepadAxis1D::RightTrigger);

    if (state[Menu] && !m_state[Menu]) {
        m_menuRequested = !m_menuRequested;
    }

    m_state = state;
}

void TouchControls::ReleaseAll() {
    if (m_gamepadConnected) {
        ApplyState({});
        m_inputContext.ProcessAxes();
        m_inputContext.ResetGamepadInputs(kPlayerIndex);
    }
}

void TouchControls::Draw(ImDrawList *drawList) const {
    if (!m_enabled || m_hardwareGamepad || drawList == nullptr) {
        return;
    }

    const ImU32 idle = IM_COL32(255, 255, 255, 70);
    const ImU32 active = IM_COL32(255, 255, 255, 150);
    const ImU32 outline = IM_COL32(255, 255, 255, 140);
    const ImU32 text = IM_COL32(255, 255, 255, 200);

    // D-pad cross
    for (int c : {Up, Down, Left, Right}) {
        const ImVec2 ctr = m_layout.centers[c];
        const ImVec2 hs = m_layout.halfSizes[c];
        drawList->AddRectFilled({ctr.x - hs.x, ctr.y - hs.y}, {ctr.x + hs.x, ctr.y + hs.y},
                                m_state[c] ? active : idle, hs.x * 0.3f);
        drawList->AddRect({ctr.x - hs.x, ctr.y - hs.y}, {ctr.x + hs.x, ctr.y + hs.y}, outline, hs.x * 0.3f);
    }

    for (int c = A; c < _Count; ++c) {
        const ImVec2 ctr = m_layout.centers[c];
        const ImVec2 hs = m_layout.halfSizes[c];
        const ImU32 fill = m_state[c] ? active : idle;
        if (hs.x == hs.y) {
            drawList->AddCircleFilled(ctr, hs.x, fill, 32);
            drawList->AddCircle(ctr, hs.x, outline, 32, 2.0f);
        } else {
            drawList->AddRectFilled({ctr.x - hs.x, ctr.y - hs.y}, {ctr.x + hs.x, ctr.y + hs.y}, fill, hs.y);
            drawList->AddRect({ctr.x - hs.x, ctr.y - hs.y}, {ctr.x + hs.x, ctr.y + hs.y}, outline, hs.y, 0, 2.0f);
        }
        const ImVec2 textSize = ImGui::CalcTextSize(kLabels[c]);
        drawList->AddText({ctr.x - textSize.x * 0.5f, ctr.y - textSize.y * 0.5f}, text, kLabels[c]);
    }
}

} // namespace app::input
