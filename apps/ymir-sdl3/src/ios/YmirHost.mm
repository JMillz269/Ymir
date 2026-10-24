#import "YmirHost.h"

#include <app/ios/host_interface.hpp>

#include <string>

static __weak UIWindow *s_gameWindow = nil;

static NSString *ToNSString(const std::string &str) {
    NSString *result = [NSString stringWithUTF8String:str.c_str()];
    return result != nil ? result : @"";
}

static NSString *ToNSString(const std::filesystem::path &path) {
    return ToNSString(path.string());
}

@implementation YmirHost

+ (UIWindow *)gameWindow {
    return s_gameWindow;
}

+ (void)setGameWindow:(UIWindow *)window {
    s_gameWindow = window;
}

+ (BOOL)isRunning {
    return static_cast<bool>(app::ios::GetHost().launchGame);
}

+ (BOOL)isDiscLoaded {
    auto &host = app::ios::GetHost();
    return host.isDiscLoaded ? host.isDiscLoaded() : NO;
}

+ (BOOL)hasBIOS {
    auto &host = app::ios::GetHost();
    return host.hasIPLROM ? host.hasIPLROM() : NO;
}

+ (NSString *)biosName {
    auto &host = app::ios::GetHost();
    return host.iplName ? ToNSString(host.iplName()) : @"";
}

+ (NSString *)loadedGamePath {
    auto &host = app::ios::GetHost();
    if (!host.loadedDiscPath) {
        return nil;
    }
    auto path = host.loadedDiscPath();
    return path.empty() ? nil : ToNSString(path);
}

+ (NSString *)documentsPath {
    NSArray<NSString *> *dirs = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    return dirs.count > 0 ? dirs[0] : NSHomeDirectory();
}

+ (BOOL)paused {
    auto &host = app::ios::GetHost();
    return host.isPaused ? host.isPaused() : NO;
}

+ (void)setPaused:(BOOL)paused {
    auto &host = app::ios::GetHost();
    if (host.setPaused) {
        host.setPaused(paused);
    }
}

+ (BOOL)fastForward {
    auto &host = app::ios::GetHost();
    return host.isFastForward ? host.isFastForward() : NO;
}

+ (void)setFastForward:(BOOL)fastForward {
    auto &host = app::ios::GetHost();
    if (host.setFastForward) {
        host.setFastForward(fastForward);
    }
}

+ (void)launchGameAtPath:(NSString *)path {
    auto &host = app::ios::GetHost();
    if (host.launchGame) {
        host.launchGame(std::filesystem::path{std::string{path.UTF8String}});
    }
}

+ (void)softReset {
    auto &host = app::ios::GetHost();
    if (host.softReset) {
        host.softReset();
    }
}

+ (void)hardReset {
    auto &host = app::ios::GetHost();
    if (host.hardReset) {
        host.hardReset();
    }
}

+ (void)openCloseTray {
    auto &host = app::ios::GetHost();
    if (host.openCloseTray) {
        host.openCloseTray();
    }
}

+ (void)ejectDisc {
    auto &host = app::ios::GetHost();
    if (host.ejectDisc) {
        host.ejectDisc();
    }
}

+ (void)saveStateToSlot:(NSInteger)slot {
    auto &host = app::ios::GetHost();
    if (host.saveState) {
        host.saveState(static_cast<int>(slot));
    }
}

+ (void)loadStateFromSlot:(NSInteger)slot {
    auto &host = app::ios::GetHost();
    if (host.loadState) {
        host.loadState(static_cast<int>(slot));
    }
}

+ (void)quitGame {
    auto &host = app::ios::GetHost();
    if (host.quitGame) {
        host.quitGame();
    }
}

+ (void)takeScreenshot {
    auto &host = app::ios::GetHost();
    if (host.takeScreenshot) {
        host.takeScreenshot();
    }
}

+ (NSString *)reloadBIOS {
    auto &host = app::ios::GetHost();
    if (!host.reloadIPLROM) {
        return @"The emulator is not running";
    }
    std::string error = host.reloadIPLROM();
    return error.empty() ? nil : ToNSString(error);
}

+ (NSArray<NSDictionary<NSString *, id> *> *)saveSlots {
    NSMutableArray *result = [NSMutableArray array];
    auto &host = app::ios::GetHost();
    if (host.saveSlots) {
        for (const auto &slot : host.saveSlots()) {
            [result addObject:@{
                @"index" : @(slot.index),
                @"present" : @(slot.present),
                @"text" : ToNSString(slot.description),
            }];
        }
    }
    return result;
}

+ (BOOL)boolForKey:(NSString *)key defaultValue:(BOOL)defaultValue {
    auto &host = app::ios::GetHost();
    return host.getBool ? host.getBool(key.UTF8String, defaultValue) : defaultValue;
}

+ (void)setBool:(BOOL)value forKey:(NSString *)key {
    auto &host = app::ios::GetHost();
    if (host.setBool) {
        host.setBool(key.UTF8String, value);
    }
}

+ (NSString *)pathForKey:(NSString *)key {
    auto &host = app::ios::GetHost();
    return host.getPath ? ToNSString(host.getPath(key.UTF8String)) : @"";
}

+ (void)setPath:(NSString *)path forKey:(NSString *)key {
    auto &host = app::ios::GetHost();
    if (host.setPath) {
        host.setPath(key.UTF8String, path.UTF8String);
    }
}

+ (NSInteger)intForKey:(NSString *)key defaultValue:(NSInteger)defaultValue {
    auto &host = app::ios::GetHost();
    return host.getInt ? host.getInt(key.UTF8String, static_cast<int>(defaultValue)) : defaultValue;
}

+ (void)setInt:(NSInteger)value forKey:(NSString *)key {
    auto &host = app::ios::GetHost();
    if (host.setInt) {
        host.setInt(key.UTF8String, static_cast<int>(value));
    }
}

+ (float)floatForKey:(NSString *)key defaultValue:(float)defaultValue {
    auto &host = app::ios::GetHost();
    return host.getFloat ? host.getFloat(key.UTF8String, defaultValue) : defaultValue;
}

+ (void)setFloat:(float)value forKey:(NSString *)key {
    auto &host = app::ios::GetHost();
    if (host.setFloat) {
        host.setFloat(key.UTF8String, value);
    }
}

@end
