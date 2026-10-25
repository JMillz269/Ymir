#include "touch_controls.hpp"
#include "touch_controls_atlas.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

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
    const SDL_TouchFingerEvent &finger = evt.tfinger;
    const ImVec2 pos{finger.x * displaySize.x, finger.y * displaySize.y};

    if (m_editing) {
        ProcessEditingFingerEvent(evt, pos);
        return;
    }

    if (!m_enabled || m_hardwareGamepad) {
        return;
    }

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
    const float safeW = std::max(1.0f, right - left);
    const float safeH = std::max(1.0f, bottom - top);
    const float unit = std::min(safeW, safeH);
    m_safeRect = {left, top, right, bottom};
    m_displaySize = displaySize;
    m_gridSize = unit / 16.0f;

    // Default layout, which also gives the base size of every control
    Layout base{};

    // D-pad on the left, vertically in the lower half
    base.dpadRadius = unit * 0.24f;
    base.dpadCenter = {left + base.dpadRadius * 1.25f, bottom - base.dpadRadius * 1.35f};

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
        base.centers[bottomRow[i]] = {x, rowBottomY};
        base.centers[topRow[i]] = {x - pitch * 0.35f, rowTopY};
        base.halfSizes[bottomRow[i]] = {rb, rb};
        base.halfSizes[topRow[i]] = {rb, rb};
    }

    // Shoulder buttons at the top corners
    const float shoulderW = unit * 0.20f;
    const float shoulderH = unit * 0.075f;
    base.centers[L] = {left + shoulderW * 1.1f, top + shoulderH * 1.6f};
    base.centers[R] = {right - shoulderW * 1.1f, top + shoulderH * 1.6f};
    base.halfSizes[L] = base.halfSizes[R] = {shoulderW, shoulderH};

    // Start and Menu along the bottom edge
    const float midX = (left + right) * 0.5f;
    const float pillW = unit * 0.13f;
    const float pillH = unit * 0.05f;
    base.centers[Menu] = {midX - pillW * 1.3f, bottom - pillH * 1.8f};
    base.centers[Start] = {midX + pillW * 1.3f, bottom - pillH * 1.8f};
    base.halfSizes[Menu] = base.halfSizes[Start] = {pillW, pillH};

    // Each orientation has its own layout, initialized from the defaults
    m_orientation = safeW > safeH ? 1 : 0;
    Placements &placements = m_placements[m_orientation];
    if (!m_placementsReady[m_orientation]) {
        placements[ElDPad] = {(base.dpadCenter.x - left) / safeW, (base.dpadCenter.y - top) / safeH, 1.0f};
        for (int c = A; c < _Count; ++c) {
            placements[c - A + 1] = {(base.centers[c].x - left) / safeW, (base.centers[c].y - top) / safeH, 1.0f};
        }
        m_placementsReady[m_orientation] = true;
    }

    // Apply the placements and sizes
    Layout &l = m_layout;
    auto place = [&](int element) {
        const Placement &pl = placements[element];
        return ImVec2{left + std::clamp(pl.x, 0.0f, 1.0f) * safeW, top + std::clamp(pl.y, 0.0f, 1.0f) * safeH};
    };

    const float dpadScale = placements[ElDPad].scale * m_globalScale;
    l.dpadRadius = base.dpadRadius * dpadScale;
    l.dpadCenter = place(ElDPad);
    const float arm = l.dpadRadius * 0.62f;
    l.centers[Up] = {l.dpadCenter.x, l.dpadCenter.y - arm};
    l.centers[Down] = {l.dpadCenter.x, l.dpadCenter.y + arm};
    l.centers[Left] = {l.dpadCenter.x - arm, l.dpadCenter.y};
    l.centers[Right] = {l.dpadCenter.x + arm, l.dpadCenter.y};
    for (int c : {Up, Down, Left, Right}) {
        l.halfSizes[c] = {l.dpadRadius * 0.42f, l.dpadRadius * 0.42f};
    }

    for (int c = A; c < _Count; ++c) {
        const float scale = placements[c - A + 1].scale * m_globalScale;
        l.centers[c] = place(c - A + 1);
        l.halfSizes[c] = {base.halfSizes[c].x * scale, base.halfSizes[c].y * scale};
    }
}

ImVec2 TouchControls::ElementCenter(int element) const {
    if (element == kScreenElement) {
        return {m_screenRect.x + m_screenRect.z * 0.5f, m_screenRect.y + m_screenRect.w * 0.5f};
    }
    return element == ElDPad ? m_layout.dpadCenter : m_layout.centers[A + element - 1];
}

int TouchControls::HitTestElement(ImVec2 pos) const {
    // Later elements are drawn on top
    for (int e = ElCount - 1; e >= 0; --e) {
        if (e == ElDPad) {
            const float dx = pos.x - m_layout.dpadCenter.x;
            const float dy = pos.y - m_layout.dpadCenter.y;
            if (std::sqrt(dx * dx + dy * dy) <= m_layout.dpadRadius * 1.1f) {
                return e;
            }
        } else {
            const int c = A + e - 1;
            if (InsideEllipse(pos, m_layout.centers[c], m_layout.halfSizes[c], m_layout.halfSizes[c].y * 0.2f)) {
                return e;
            }
        }
    }
    return -1;
}

void TouchControls::ProcessEditingFingerEvent(const SDL_Event &evt, ImVec2 pos) {
    const SDL_TouchFingerEvent &finger = evt.tfinger;
    switch (evt.type) {
    case SDL_EVENT_FINGER_DOWN:
        if (m_dragging) {
            break; // one finger moves a control at a time
        }
        m_selected = HitTestElement(pos);
        if (m_selected < 0 && pos.x >= m_screenRect.x && pos.x <= m_screenRect.x + m_screenRect.z &&
            pos.y >= m_screenRect.y && pos.y <= m_screenRect.y + m_screenRect.w) {
            m_selected = kScreenElement; // controls are on top of the game screen
        }
        if (m_selected >= 0) {
            const ImVec2 center = ElementCenter(m_selected);
            m_dragOffset = {center.x - pos.x, center.y - pos.y};
            m_dragFinger = finger.fingerID;
            m_dragging = true;
        }
        break;
    case SDL_EVENT_FINGER_MOTION:
        if (m_dragging && finger.fingerID == m_dragFinger && m_selected >= 0) {
            const float left = m_safeRect.x, top = m_safeRect.y, right = m_safeRect.z, bottom = m_safeRect.w;
            float cx = std::clamp(pos.x + m_dragOffset.x, left, right);
            float cy = std::clamp(pos.y + m_dragOffset.y, top, bottom);
            if (m_snapToGrid && m_gridSize > 1.0f) {
                cx = left + std::round((cx - left) / m_gridSize) * m_gridSize;
                cy = top + std::round((cy - top) / m_gridSize) * m_gridSize;
                cx = std::clamp(cx, left, right);
                cy = std::clamp(cy, top, bottom);
            }
            if (m_selected == kScreenElement) {
                // The game screen is positioned relative to the whole window rather than the safe area
                ScreenPlacement &sp = m_screenPlacements[m_orientation];
                sp.custom = true;
                sp.x = std::clamp(cx / std::max(1.0f, m_displaySize.x), 0.0f, 1.0f);
                sp.y = std::clamp(cy / std::max(1.0f, m_displaySize.y), 0.0f, 1.0f);
            } else {
                Placement &pl = m_placements[m_orientation][m_selected];
                pl.x = (cx - left) / std::max(1.0f, right - left);
                pl.y = (cy - top) / std::max(1.0f, bottom - top);
            }
            MarkDirty();
        }
        break;
    case SDL_EVENT_FINGER_UP: [[fallthrough]];
    case SDL_EVENT_FINGER_CANCELED:
        if (m_dragging && finger.fingerID == m_dragFinger) {
            m_dragging = false;
        }
        break;
    default: break;
    }
}

// -------------------------------------------------------------------------------------------------------------------
// Customization

void TouchControls::SetEditing(bool editing) {
    if (m_editing == editing) {
        return;
    }
    m_editing = editing;
    m_dragging = false;
    m_selected = -1;
    m_fingers.clear();
    if (editing) {
        // Make sure nothing is held down and that the controls are visible
        ReleaseAll();
        m_enabled = true;
    } else {
        Save();
    }
}

float TouchControls::GetSelectedScale() const {
    if (m_selected == kScreenElement) {
        return m_screenPlacements[m_orientation].scale;
    }
    return m_selected >= 0 ? m_placements[m_orientation][m_selected].scale : 1.0f;
}

void TouchControls::SetSelectedScale(float scale) {
    if (m_selected == kScreenElement) {
        ScreenPlacement &sp = m_screenPlacements[m_orientation];
        sp.scale = std::clamp(scale, 0.3f, 3.0f);
        // Resizing keeps the current position, whether the screen was moved or not
        if (!sp.custom) {
            sp.custom = true;
            const ImVec2 center = ElementCenter(kScreenElement);
            sp.x = std::clamp(center.x / std::max(1.0f, m_displaySize.x), 0.0f, 1.0f);
            sp.y = std::clamp(center.y / std::max(1.0f, m_displaySize.y), 0.0f, 1.0f);
        }
        MarkDirty();
    } else if (m_selected >= 0) {
        m_placements[m_orientation][m_selected].scale = std::clamp(scale, 0.4f, 2.5f);
        MarkDirty();
    }
}

void TouchControls::SetGlobalScale(float scale) {
    m_globalScale = std::clamp(scale, 0.5f, 1.6f);
    MarkDirty();
}

void TouchControls::SetOpacity(float opacity) {
    m_opacity = std::clamp(opacity, 0.1f, 1.0f);
    MarkDirty();
}

void TouchControls::SetSnapToGrid(bool snap) {
    m_snapToGrid = snap;
    MarkDirty();
}

std::optional<TouchControls::ScreenOverride> TouchControls::GetScreenOverride() const {
    const ScreenPlacement &sp = m_screenPlacements[m_orientation];
    if (!sp.custom) {
        return std::nullopt;
    }
    return ScreenOverride{sp.x, sp.y, sp.scale};
}

void TouchControls::SetScreenRect(ImVec2 pos, ImVec2 size) {
    m_screenRect = {pos.x, pos.y, size.x, size.y};
}

void TouchControls::ResetLayout() {
    m_screenPlacements[m_orientation] = {};
    // Placements are rebuilt from the defaults on the next layout computation
    m_placementsReady[m_orientation] = false;
    m_globalScale = 1.0f;
    m_selected = -1;
    m_dragging = false;
    MarkDirty();
}

void TouchControls::MarkDirty() {
    m_dirty = true;
    m_dirtyTime = std::chrono::steady_clock::now();
}

void TouchControls::SetStoragePath(std::filesystem::path path) {
    m_storagePath = std::move(path);
    Load();
}

void TouchControls::Load() {
    std::ifstream in{m_storagePath};
    if (!in) {
        return;
    }
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss{line};
        std::string key;
        ss >> key;
        if (key == "scale") {
            float v;
            if (ss >> v) {
                m_globalScale = std::clamp(v, 0.5f, 1.6f);
            }
        } else if (key == "alpha") {
            float v;
            if (ss >> v) {
                m_opacity = std::clamp(v, 0.1f, 1.0f);
            }
        } else if (key == "snap") {
            int v;
            if (ss >> v) {
                m_snapToGrid = v != 0;
            }
        } else if (key == "s") {
            int orientation, custom;
            ScreenPlacement sp;
            if ((ss >> orientation >> custom >> sp.x >> sp.y >> sp.scale) && orientation >= 0 && orientation < 2) {
                sp.custom = custom != 0;
                sp.x = std::clamp(sp.x, 0.0f, 1.0f);
                sp.y = std::clamp(sp.y, 0.0f, 1.0f);
                sp.scale = std::clamp(sp.scale, 0.3f, 3.0f);
                m_screenPlacements[orientation] = sp;
            }
        } else if (key == "p") {
            int orientation, element;
            Placement pl;
            if ((ss >> orientation >> element >> pl.x >> pl.y >> pl.scale) && orientation >= 0 && orientation < 2 &&
                element >= 0 && element < ElCount) {
                pl.scale = std::clamp(pl.scale, 0.4f, 2.5f);
                m_placements[orientation][element] = pl;
                m_loadedElements[orientation] |= 1u << element;
            }
        }
    }
    // Orientations whose elements were all loaded don't need the defaults
    for (int o = 0; o < 2; ++o) {
        m_placementsReady[o] = m_loadedElements[o] == (1u << ElCount) - 1u;
    }
}

void TouchControls::Save() {
    m_dirty = false;
    if (m_storagePath.empty()) {
        return;
    }
    std::ofstream out{m_storagePath, std::ios::trunc};
    if (!out) {
        return;
    }
    out << "scale " << m_globalScale << "\n";
    out << "alpha " << m_opacity << "\n";
    out << "snap " << (m_snapToGrid ? 1 : 0) << "\n";
    for (int o = 0; o < 2; ++o) {
        const ScreenPlacement &sp = m_screenPlacements[o];
        out << "s " << o << " " << (sp.custom ? 1 : 0) << " " << sp.x << " " << sp.y << " " << sp.scale << "\n";
    }
    for (int o = 0; o < 2; ++o) {
        if (!m_placementsReady[o]) {
            continue;
        }
        for (int e = 0; e < ElCount; ++e) {
            const Placement &pl = m_placements[o][e];
            out << "p " << o << " " << e << " " << pl.x << " " << pl.y << " " << pl.scale << "\n";
        }
    }
}

void TouchControls::Update(SDL_Window *window, ImVec2 displaySize) {
    if (m_dirty && std::chrono::steady_clock::now() - m_dirtyTime > std::chrono::milliseconds(800)) {
        Save();
    }

    if (m_editing) {
        // The controls are only being arranged: keep the layout up to date and send no inputs
        ComputeLayout(window, displaySize);
        return;
    }

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

void TouchControls::Draw(ImDrawList *drawList, std::optional<ImTextureID> atlas) const {
    if (drawList == nullptr || (!m_editing && (!m_enabled || m_hardwareGamepad))) {
        return;
    }

    const ImU32 selectedFill = IM_COL32(60, 150, 255, 60);
    const ImU32 selectedOutline = IM_COL32(120, 190, 255, 255);

    if (m_editing) {
        // Dim the game so that the controls stand out, and show the grid if snapping is on
        drawList->AddRectFilled({-1.0f, -1.0f}, {16384.0f, 16384.0f}, IM_COL32(0, 0, 0, 120));
        if (m_snapToGrid && m_gridSize > 1.0f) {
            const ImU32 gridColor = IM_COL32(255, 255, 255, 40);
            for (float x = m_safeRect.x; x <= m_safeRect.z + 0.5f; x += m_gridSize) {
                drawList->AddLine({x, m_safeRect.y}, {x, m_safeRect.w}, gridColor);
            }
            for (float y = m_safeRect.y; y <= m_safeRect.w + 0.5f; y += m_gridSize) {
                drawList->AddLine({m_safeRect.x, y}, {m_safeRect.z, y}, gridColor);
            }
        }
        drawList->AddRect({m_safeRect.x, m_safeRect.y}, {m_safeRect.z, m_safeRect.w}, IM_COL32(255, 255, 255, 70), 0.0f,
                          0, 1.5f);

        // The game screen can be moved and resized too
        if (m_screenRect.z > 1.0f && m_screenRect.w > 1.0f) {
            const ImVec2 p0{m_screenRect.x, m_screenRect.y};
            const ImVec2 p1{m_screenRect.x + m_screenRect.z, m_screenRect.y + m_screenRect.w};
            if (m_selected == kScreenElement) {
                drawList->AddRectFilled(p0, p1, selectedFill);
                drawList->AddRect(p0, p1, selectedOutline, 0.0f, 0, 3.0f);
            } else {
                drawList->AddRect(p0, p1, IM_COL32(255, 255, 255, 110), 0.0f, 0, 1.5f);
            }
            drawList->AddText({p0.x + 8.0f, p0.y + 6.0f}, IM_COL32(255, 255, 255, 170), "GAME SCREEN");
        }
    }

    const int alpha = (int)std::clamp(m_opacity * 255.0f, 0.0f, 255.0f);
    const ImU32 idleTint = IM_COL32(225, 225, 225, alpha);
    const ImU32 pressedTint = IM_COL32(255, 255, 255, alpha);

    // Draws a sprite so that the control itself (without the soft glow around it) is bodyWidth wide
    const auto drawSprite = [&](const touch_atlas::Rect &r, ImVec2 center, float bodyWidth, bool pressed) {
        const float factor = bodyWidth / std::max(1.0f, (float)r.bodyW) * (pressed ? 0.94f : 1.0f);
        const float w = r.w * factor;
        const float h = r.h * factor;
        const ImVec2 p0{center.x - w * 0.5f, center.y - h * 0.5f};
        const ImVec2 p1{p0.x + w, p0.y + h};
        const ImVec2 uv0{(float)r.x / touch_atlas::kWidth, (float)r.y / touch_atlas::kHeight};
        const ImVec2 uv1{(float)(r.x + r.w) / touch_atlas::kWidth, (float)(r.y + r.h) / touch_atlas::kHeight};
        drawList->AddImage(*atlas, p0, p1, uv0, uv1, pressed ? pressedTint : idleTint);
    };

    if (atlas) {
        const bool dpadSelected = m_editing && m_selected == ElDPad;
        if (dpadSelected) {
            drawList->AddCircleFilled(m_layout.dpadCenter, m_layout.dpadRadius * 1.08f, selectedFill, 48);
            drawList->AddCircle(m_layout.dpadCenter, m_layout.dpadRadius * 1.08f, selectedOutline, 48, 3.0f);
        }
        drawSprite(touch_atlas::kDpad, m_layout.dpadCenter, m_layout.dpadRadius * 2.08f, false);
        // Light up the arms that are held down
        for (int c : {Up, Down, Left, Right}) {
            if (m_state[c]) {
                const ImVec2 ctr = m_layout.centers[c];
                const ImVec2 hs = m_layout.halfSizes[c];
                drawList->AddRectFilled({ctr.x - hs.x, ctr.y - hs.y}, {ctr.x + hs.x, ctr.y + hs.y},
                                        IM_COL32(255, 255, 255, 90), hs.x * 0.3f);
            }
        }

        const touch_atlas::Rect *sprites[_Count] = {};
        sprites[A] = &touch_atlas::kA;
        sprites[B] = &touch_atlas::kB;
        sprites[C] = &touch_atlas::kC;
        sprites[X] = &touch_atlas::kX;
        sprites[Y] = &touch_atlas::kY;
        sprites[Z] = &touch_atlas::kZ;
        sprites[L] = &touch_atlas::kL;
        sprites[R] = &touch_atlas::kR;
        sprites[Start] = &touch_atlas::kStart;
        sprites[Menu] = &touch_atlas::kMode;

        for (int c = A; c < _Count; ++c) {
            const ImVec2 ctr = m_layout.centers[c];
            const ImVec2 hs = m_layout.halfSizes[c];
            if (m_editing && m_selected == c - A + 1) {
                if (hs.x == hs.y) {
                    drawList->AddCircleFilled(ctr, hs.x * 1.1f, selectedFill, 32);
                    drawList->AddCircle(ctr, hs.x * 1.1f, selectedOutline, 32, 3.0f);
                } else {
                    const ImVec2 pad{hs.x * 0.04f, hs.y * 0.15f};
                    drawList->AddRectFilled({ctr.x - hs.x - pad.x, ctr.y - hs.y - pad.y},
                                            {ctr.x + hs.x + pad.x, ctr.y + hs.y + pad.y}, selectedFill, hs.y);
                    drawList->AddRect({ctr.x - hs.x - pad.x, ctr.y - hs.y - pad.y},
                                      {ctr.x + hs.x + pad.x, ctr.y + hs.y + pad.y}, selectedOutline, hs.y, 0, 3.0f);
                }
            }
            drawSprite(*sprites[c], ctr, hs.x * 2.0f, m_state[c]);
        }
        return;
    }

    // No artwork available: plain shapes
    const float scaleAlpha = m_opacity / 0.9f;
    const auto a = [&](int v) { return (int)std::clamp(v * scaleAlpha, 0.0f, 255.0f); };
    const ImU32 idle = IM_COL32(255, 255, 255, a(70));
    const ImU32 active = IM_COL32(255, 255, 255, a(150));
    const ImU32 outline = IM_COL32(255, 255, 255, a(140));
    const ImU32 text = IM_COL32(255, 255, 255, a(200));

    const bool dpadSelected = m_editing && m_selected == ElDPad;
    for (int c : {Up, Down, Left, Right}) {
        const ImVec2 ctr = m_layout.centers[c];
        const ImVec2 hs = m_layout.halfSizes[c];
        drawList->AddRectFilled({ctr.x - hs.x, ctr.y - hs.y}, {ctr.x + hs.x, ctr.y + hs.y},
                                dpadSelected ? selectedOutline : (m_state[c] ? active : idle), hs.x * 0.3f);
        drawList->AddRect({ctr.x - hs.x, ctr.y - hs.y}, {ctr.x + hs.x, ctr.y + hs.y},
                          dpadSelected ? selectedOutline : outline, hs.x * 0.3f);
    }

    for (int c = A; c < _Count; ++c) {
        const ImVec2 ctr = m_layout.centers[c];
        const ImVec2 hs = m_layout.halfSizes[c];
        const bool selected = m_editing && m_selected == c - A + 1;
        const ImU32 fill = selected ? selectedOutline : (m_state[c] ? active : idle);
        const ImU32 line = selected ? selectedOutline : outline;
        if (hs.x == hs.y) {
            drawList->AddCircleFilled(ctr, hs.x, fill, 32);
            drawList->AddCircle(ctr, hs.x, line, 32, 2.0f);
        } else {
            drawList->AddRectFilled({ctr.x - hs.x, ctr.y - hs.y}, {ctr.x + hs.x, ctr.y + hs.y}, fill, hs.y);
            drawList->AddRect({ctr.x - hs.x, ctr.y - hs.y}, {ctr.x + hs.x, ctr.y + hs.y}, line, hs.y, 0, 2.0f);
        }
        const ImVec2 textSize = ImGui::CalcTextSize(kLabels[c]);
        drawList->AddText({ctr.x - textSize.x * 0.5f, ctr.y - textSize.y * 0.5f}, text, kLabels[c]);
    }
}

} // namespace app::input
