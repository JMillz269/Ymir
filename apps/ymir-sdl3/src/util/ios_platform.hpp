#pragma once

#include <SDL3/SDL_dialog.h>
#include <SDL3/SDL_video.h>

#include <filesystem>

namespace util::ios {

/// @brief Retrieves the app's Documents directory.
///
/// This is the directory exposed in the Files app (under "On My iPhone" > "Ymir"), and is where the profile lives.
std::filesystem::path GetDocumentsPath();

/// @brief Shows a file dialog backed by the system document picker, mimicking SDL_ShowFileDialogWithProperties.
///
/// Opened files are copied into <Documents>/games so that they stay accessible after the app restarts. SDL's own file
/// dialogs are not implemented on iOS.
///
/// Save file dialogs don't exist on iOS; they resolve immediately to <Documents>/exports/<default file name> and folder
/// dialogs resolve to <Documents>/exports, both reachable from the Files app.
///
/// @param[in] window the window to present from
/// @param[in] type the type of dialog
/// @param[in] allowMany whether multiple files can be selected (open file dialogs only)
/// @param[in] location default location; only its file name is used for save dialogs
/// @param[in] userdata user pointer passed to the callback
/// @param[in] callback the callback invoked with the selected paths, or an empty list if cancelled
void ShowFileDialog(SDL_Window *window, SDL_FileDialogType type, bool allowMany, const char *location, void *userdata,
                    SDL_DialogFileCallback callback);

/// @brief Registers the window that hosts the emulator view so that the native UI can be presented on top of it.
void AttachGameWindow(SDL_Window *window);

/// @brief Presents the native game library and settings (full screen) over the emulator view.
void ShowLauncher();

/// @brief Presents the native in-game menu (sheet) over the emulator view.
void ShowGameMenu();

/// @brief Whether any native UI is currently covering the emulator view.
bool IsNativeUIVisible();

} // namespace util::ios
