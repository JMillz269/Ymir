#import "YmirHost.h"

#include <app/ios/host_interface.hpp>

#include <ymir/hw/sh2/sh2_recompiler.hpp>
#include <ymir/util/perf_counters.hpp>
#include <ymir/jit/jit_memory.hpp>

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

+ (BOOL)jitAvailable {
    return ymir::jit::ProbeJit().Available();
}

+ (NSString *)jitStatus {
    return ToNSString(ymir::jit::ProbeJit().Message());
}

+ (NSString *)jitReport {
    const auto g = ymir::sh2::SH2Recompiler::GetGlobalStats();
    const double avgMicros = g.blocksCompiled > 0 ? static_cast<double>(g.compileNanoseconds) / g.blocksCompiled / 1000.0 : 0.0;
    const std::string helpers = ymir::sh2::SH2Recompiler::HelperReport(8);
    NSString *main = [NSString stringWithFormat:@"%llu blocks compiled (%.1f MB), %llu cache flushes, compile time %.0f µs average / %.1f ms slowest, %llu idle slices, %llu poll skips, %.1f%% of time in long slices, %llu conflict evictions, %llu invalidations, %.0f%% of instructions go through the interpreter",
                                      g.blocksCompiled, g.bytesGenerated / 1048576.0, g.cacheFlushes, avgMicros,
                                      g.slowestCompileNanoseconds / 1.0e6, ymir::sh2::SH2Recompiler::IdleSlices(), ymir::sh2::SH2Recompiler::PollSkips(), ymir::sh2::SH2Recompiler::IdleTimeShare() * 100.0, g.conflictEvictions, g.invalidations,
                                      (g.inlineInstructions + g.helperInstructions) > 0 ? 100.0 * g.helperInstructions / (g.inlineInstructions + g.helperInstructions) : 0.0];
    // Time spent per emulated frame over the last interval
    static uint64_t lastSH2 = 0, lastOther = 0, lastVDP1 = 0, lastVDP2 = 0, lastFrames = 0, lastEmuWait = 0, lastGuiWait = 0;
    const uint64_t sh2 = ymir::perf::sh2Ns.load(), other = ymir::perf::otherNs.load();
    const uint64_t vdp1 = ymir::perf::vdp1Ns.load(), vdp2 = ymir::perf::vdp2Ns.load(), frames = ymir::perf::frames.load();
    const uint64_t emuWait = ymir::perf::emuWaitNs.load(), guiWait = ymir::perf::guiWaitNs.load();
    // Keep showing the last measurement while no frames are being emulated (the page refreshes every second)
    static NSString *lastPerf = @"";
    NSString *perf = lastPerf;
    if (frames > lastFrames) {
        const double n = static_cast<double>(frames - lastFrames) * 1.0e6;
        perf = [NSString stringWithFormat:@"\nPer frame: SH-2 and SCU %.2f ms, VDP timing/sound/CD %.2f ms, VDP1 %.2f ms, VDP2 %.2f ms (the VDP times run on other threads when threaded)\nWaiting per frame: emulator for the display %.2f ms, display for the emulator %.2f ms",
                                          (sh2 - lastSH2) / n, (other - lastOther) / n, (vdp1 - lastVDP1) / n, (vdp2 - lastVDP2) / n,
                                          (emuWait - lastEmuWait) / n, (guiWait - lastGuiWait) / n];
        lastPerf = perf;
    }
    lastSH2 = sh2; lastOther = other; lastVDP1 = vdp1; lastVDP2 = vdp2; lastFrames = frames; lastEmuWait = emuWait; lastGuiWait = guiWait;
    return [[main stringByAppendingString:perf] stringByAppendingFormat:@"\nMost used interpreter calls: %s", helpers.empty() ? "none yet" : helpers.c_str()];
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

+ (void)deleteStateInSlot:(NSInteger)slot {
    auto &host = app::ios::GetHost();
    if (host.deleteState) {
        host.deleteState(static_cast<int>(slot));
    }
}

+ (NSString *)pathForSaveStateSlot:(NSInteger)slot {
    auto &host = app::ios::GetHost();
    if (!host.saveStatePath) {
        return nil;
    }
    auto path = host.saveStatePath(static_cast<int>(slot));
    return path.empty() ? nil : ToNSString(path);
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
