# Ymir on iOS

This is the iOS port of Ymir. It runs the same emulator core as the desktop builds, rendered with the Metal graphics backend and a native SwiftUI interface. It targets arm64 iPhones and iPads on iOS 16 or later.

## Installing

The [iOS workflow](../.github/workflows/ios.yaml) builds an **unsigned** `ymir-ios-arm64-*.ipa` and uploads it as a build artifact. Download it from the workflow run and install it with a sideloading tool (AltStore, SideStore, Sideloadly, TrollStore, etc.), which signs it with your own certificate.

## Adding BIOS images and games

Ymir stores everything in its Documents folder, which shows up in the Files app under **On My iPhone > Ymir**:

| Folder             | Contents                          |
| ------------------ | --------------------------------- |
| `roms/ipl`         | Saturn BIOS (IPL ROM) images      |
| `roms/cdb`         | CD block ROM images (optional)    |
| `roms/cart`        | Cartridge ROM images              |
| `games`            | Discs imported from the Games tab |
| `backup`, `state`, `savestates`, `screenshots` | Saves and state    |
| `exports`          | Targets of save/export dialogs    |

## Native interface

The iOS app has a native SwiftUI interface (sources in `apps/ymir-sdl3/src/ios`) instead of the desktop menu bar:

- **Games** lists the disc images found in the Ymir folder (`.cue`, `.chd`, `.iso`, `.ccd`, `.mds`). The emulator stays idle until you tap a game. Swipe or long-press a game to delete it. The **+** button imports games or a BIOS with the system document picker and opens the Ymir folder in the Files app. Long-press a game to set its artwork (from Photos, Files or a Google Images search where you touch and hold the image you want) or to rename it; the name only changes in the app. Artwork is stored in `covers`, and a `.png` or `.jpg` named like the disc image next to it is used too.
- **Settings** replaces the menu bar and the desktop settings window with native pages: Display, Audio, Controls (peripherals, deadzones), Emulation (speed, rewind), System & Region, Cartridge, CD Drive, Performance & Accuracy and Interface, plus BIOS import. The only thing not available is editing individual button and hotkey bindings; controllers use the default bindings.
- On iOS 26 and later the tab bar and the buttons use Liquid Glass; older systems get translucent materials.

Import a BIOS first (Games tab banner or Settings > System). Files can also be copied by hand into `roms/ipl` and `games` with the Files app, or shared to Ymir with **Open in...**. Pick every file of a multi-file image (e.g. the `.cue` and all `.bin` files) at once so they end up side by side; `.chd` and `.iso` images are single files and avoid this.

## Controls

An on-screen Saturn control pad is drawn over the game: D-pad, A/B/C, X/Y/Z, L/R, Start and a Menu button. It is mapped to the default port 1 Control Pad binds, so it works out of the box. Fingers can slide between buttons and multi-touch works.

- The **Menu** button on screen, or **Select/Back** on a controller, opens the in-game menu: resume, pause, fast forward, reset, disc tray, save states, screenshots, settings, a way back to the game library and **Close Game**. The game is paused while the menu is open.
- **Settings > Controls > On-Screen Controls** turns the overlay off.
- The overlay hides itself while a hardware controller is connected. MFi, Xbox, PlayStation and other controllers supported by iOS work through SDL's gamepad support and use the same default binds as on desktop.

## JIT

The emulator core is an interpreter. It does not generate machine code at runtime, so it does not need JIT, and no JIT/TXM-specific code exists in this port. Any future dynamic recompiler would need to be added to the core first.

## Known limitations

- Save and export dialogs don't exist on iOS; they write to `Documents/exports`.
- Update checks are disabled. Rebuild from the workflow to update.
- The Metal backend only handles presentation; VDP rendering runs on the CPU, as on macOS.
- This port has not been run on a device by its author at the time of writing; expect rough edges and report them.
