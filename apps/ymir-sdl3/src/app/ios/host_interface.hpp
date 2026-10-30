#pragma once

// Interface between the emulator frontend and the native iOS (Swift) user interface.
//
// The emulator fills in the callbacks once it is up and running and clears them when it shuts down. All callbacks are
// invoked from the main thread, the same one that runs the SDL event loop, so they can safely enqueue events and touch
// frontend state. Only standard library types are used so that this header can be included from Objective-C++.

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace app::ios {

struct SaveSlotInfo {
    int index = 0;
    bool present = false;
    std::string description; // e.g. date and time; empty if the slot is empty
};

struct HostInterface {
    // -- State queries --
    std::function<bool()> isPaused;
    std::function<bool()> isFastForward;
    std::function<bool()> isDiscLoaded;
    std::function<std::filesystem::path()> loadedDiscPath;
    std::function<bool()> hasIPLROM;      // true if an IPL ROM is loaded into the emulator
    std::function<std::string()> iplName; // human readable name of the loaded IPL ROM
    std::function<std::vector<SaveSlotInfo>()> saveSlots;
    std::function<std::filesystem::path(int)> saveStatePath; // file of a slot's save state; empty if there's none

    // -- Actions --
    std::function<void(const std::filesystem::path &)> launchGame;
    std::function<void(bool)> setPaused;
    std::function<void(bool)> setFastForward;
    std::function<void()> softReset;
    std::function<void()> hardReset;
    std::function<void()> openCloseTray;
    std::function<void()> ejectDisc;
    std::function<void()> quitGame; // unloads the game and leaves the emulator idle
    std::function<void(int)> saveState;
    std::function<void(int)> loadState;
    std::function<void(int)> deleteState;
    std::function<void()> takeScreenshot;
    std::function<std::string()> reloadIPLROM; // rescans the IPL ROM folder; returns an error message or "" on success

    // -- Settings --
    std::function<bool(const std::string &, bool)> getBool;    // key, default
    std::function<void(const std::string &, bool)> setBool;    // key, value
    std::function<float(const std::string &, float)> getFloat; // key, default
    std::function<void(const std::string &, float)> setFloat;  // key, value
    std::function<std::string(const std::string &)> getPath;   // key
    std::function<void(const std::string &, const std::string &)> setPath; // key, absolute path
    std::function<int(const std::string &, int)> getInt;       // key, default (also used for enumerations)
    std::function<void(const std::string &, int)> setInt;      // key, value
};

/// @brief Gets the global host interface. Callbacks are empty while the emulator isn't running.
inline HostInterface &GetHost() {
    static HostInterface host{};
    return host;
}

} // namespace app::ios
