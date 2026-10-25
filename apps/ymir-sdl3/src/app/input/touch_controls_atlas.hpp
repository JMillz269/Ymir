#pragma once

// Texture atlas of the on-screen controller artwork (res/embed/images/touch_controls.png).
// Rectangles are in pixels. bodyW is the width of the control itself, without the soft glow around it.

namespace app::input::touch_atlas {

inline constexpr int kWidth = 2048;
inline constexpr int kHeight = 1024;

struct Rect {
    int x, y, w, h;
    int bodyW;
};

inline constexpr Rect kDpad = {4, 4, 539, 552, 507};
inline constexpr Rect kL = {547, 4, 438, 185, 406};
inline constexpr Rect kR = {989, 4, 437, 185, 405};
inline constexpr Rect kStart = {1430, 4, 295, 148, 263};
inline constexpr Rect kMode = {1729, 4, 290, 149, 258};
inline constexpr Rect kA = {4, 560, 232, 237, 200};
inline constexpr Rect kB = {240, 560, 232, 237, 200};
inline constexpr Rect kC = {476, 560, 231, 234, 199};
inline constexpr Rect kX = {711, 560, 233, 237, 201};
inline constexpr Rect kY = {948, 560, 233, 235, 201};
inline constexpr Rect kZ = {1185, 560, 231, 234, 199};

} // namespace app::input::touch_atlas
