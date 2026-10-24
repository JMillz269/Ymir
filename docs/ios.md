# Ymir on iOS

This is the iOS port of Ymir. It runs the same emulator core as the desktop builds, rendered with the Metal graphics backend. It targets arm64 iPhones and iPads on iOS 16 or later.

## Installing

The [iOS workflow](../.github/workflows/ios.yaml) builds an **unsigned** `ymir-ios-arm64-*.ipa` and uploads it as a build artifact. Download it from the workflow run and install it with a sideloading tool (AltStore, SideStore, Sideloadly, TrollStore, etc.), which signs it with your own certificate.

## Adding BIOS images and games

Ymir stores everything in its Documents folder, which shows up in the Files app under **On My iPhone > Ymir**:

| Folder             | Contents                          |
| ------------------ | --------------------------------- |
| `roms/ipl`         | Saturn BIOS (IPL ROM) images      |
| `roms/cdb`         | CD block ROM images (optional)    |
| `roms/cart`        | Cartridge ROM images              |
| `games`            | Discs imported with **File > Load disc image** |
| `backup`, `state`, `savestates`, `screenshots` | Saves and state    |
| `exports`          | Targets of save/export dialogs    |

Copy your BIOS into `roms/ipl` using the Files app, then open **File > Load disc image**. The system document picker copies the selected files into `games`. Pick every file of a multi-file image (e.g. the `.cue` and all `.bin` files) at once so they end up side by side. `.chd` and `.iso` images are single files and avoid this.

You can also share a disc image to Ymir from the Files app with **Open in...**.

## Controls

An on-screen Saturn control pad is drawn over the game: D-pad, A/B/C, X/Y/Z, L/R, Start and a Menu button. It is mapped to the default port 1 Control Pad binds, so it works out of the box. Fingers can slide between buttons and multi-touch works.

- The **Menu** button toggles the main menu bar. Tapping near the top of the screen also reveals it.
- **View > Touch controls** turns the overlay off.
- The overlay hides itself while a hardware controller is connected. MFi, Xbox, PlayStation and other controllers supported by iOS work through SDL's gamepad support and use the same default binds as on desktop.

## JIT

The emulator core is an interpreter. It does not generate machine code at runtime, so it does not need JIT, and no JIT/TXM-specific code exists in this port. Any future dynamic recompiler would need to be added to the core first.

## Known limitations

- Save and export dialogs don't exist on iOS; they write to `Documents/exports`.
- Update checks are disabled. Rebuild from the workflow to update.
- The Metal backend only handles presentation; VDP rendering runs on the CPU, as on macOS.
- This port has not been run on a device by its author at the time of writing; expect rough edges and report them.
