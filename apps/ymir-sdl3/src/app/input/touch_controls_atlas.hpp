#pragma once

// Texture atlas of the on-screen controller artwork (res/embed/images/touch_controls.png).
// Rectangles are in pixels: x, y, width, height.

namespace app::input::touch_atlas {

inline constexpr int kWidth = 2048;
inline constexpr int kHeight = 2048;

struct Rect {
    int x, y, w, h;
};

inline constexpr Rect kDpad = {4, 4, 726, 744};
inline constexpr Rect kL = {734, 4, 598, 232};
inline constexpr Rect kR = {1336, 4, 558, 232};
inline constexpr Rect kStart = {4, 752, 364, 164};
inline constexpr Rect kMode = {372, 752, 354, 164};
inline constexpr Rect kA = {730, 752, 288, 300};
inline constexpr Rect kB = {1022, 752, 290, 302};
inline constexpr Rect kC = {1316, 752, 288, 296};
inline constexpr Rect kX = {1608, 752, 290, 300};
inline constexpr Rect kY = {4, 1058, 286, 294};
inline constexpr Rect kZ = {294, 1058, 284, 294};

} // namespace app::input::touch_atlas
