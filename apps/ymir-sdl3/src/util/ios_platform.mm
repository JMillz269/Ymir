#include "ios_platform.hpp"

#import <UIKit/UIKit.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#import "ios/YmirHost.h"

#include <SDL3/SDL_properties.h>

#import <objc/message.h>

#include <string>
#include <vector>

namespace {

// Invoked once the picker finishes. Paths are UTF-8, the list passed to the SDL callback is NULL-terminated.
using PickerCompletion = void (^)(NSArray<NSURL *> *urls);

} // namespace

@interface YmirDocumentPickerDelegate : NSObject <UIDocumentPickerDelegate>
@property(nonatomic, copy) PickerCompletion completion;
@property(nonatomic, strong) YmirDocumentPickerDelegate *selfRetain; // keeps the delegate alive until we're done
@end

@implementation YmirDocumentPickerDelegate
- (void)finishWithURLs:(NSArray<NSURL *> *)urls {
    PickerCompletion completion = self.completion;
    self.completion = nil;
    self.selfRetain = nil;
    if (completion != nil) {
        completion(urls);
    }
}
- (void)documentPicker:(UIDocumentPickerViewController *)controller didPickDocumentsAtURLs:(NSArray<NSURL *> *)urls {
    [self finishWithURLs:urls];
}
- (void)documentPickerWasCancelled:(UIDocumentPickerViewController *)controller {
    [self finishWithURLs:@[]];
}
@end

namespace util::ios {

std::filesystem::path GetDocumentsPath() {
    @autoreleasepool {
        NSArray<NSString *> *dirs = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
        NSString *path = dirs.count > 0 ? dirs[0] : NSHomeDirectory();
        return std::filesystem::path{std::string{path.UTF8String}};
    }
}

static NSURL *EnsureSubdirectory(NSString *name) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSURL *docs = [NSURL fileURLWithPath:[NSString stringWithUTF8String:GetDocumentsPath().c_str()] isDirectory:YES];
    NSURL *dir = [docs URLByAppendingPathComponent:name isDirectory:YES];
    [fm createDirectoryAtURL:dir withIntermediateDirectories:YES attributes:nil error:nil];
    return dir;
}

static void InvokeCallback(SDL_DialogFileCallback callback, void *userdata, const std::vector<std::string> &paths) {
    std::vector<const char *> list;
    list.reserve(paths.size() + 1);
    for (const std::string &path : paths) {
        list.push_back(path.c_str());
    }
    list.push_back(nullptr);
    callback(userdata, list.data(), -1);
}

static UIViewController *TopViewController(SDL_Window *window) {
    UIWindow *uiWindow = nil;
    if (window != nullptr) {
        SDL_PropertiesID props = SDL_GetWindowProperties(window);
        uiWindow = (__bridge UIWindow *)SDL_GetPointerProperty(props, SDL_PROP_WINDOW_UIKIT_WINDOW_POINTER, nullptr);
    }
    UIViewController *vc = uiWindow.rootViewController;
    while (vc.presentedViewController != nil) {
        vc = vc.presentedViewController;
    }
    return vc;
}

void ShowFileDialog(SDL_Window *window, SDL_FileDialogType type, bool allowMany, const char *location, void *userdata,
                    SDL_DialogFileCallback callback) {
    if (callback == nullptr) {
        return;
    }

    @autoreleasepool {
        if (type == SDL_FILEDIALOG_SAVEFILE || type == SDL_FILEDIALOG_OPENFOLDER) {
            NSURL *exports = EnsureSubdirectory(@"exports");
            std::string path = exports.path.UTF8String;
            if (type == SDL_FILEDIALOG_SAVEFILE) {
                std::filesystem::path name = location != nullptr ? std::filesystem::path{location}.filename()
                                                                 : std::filesystem::path{"export"};
                path = (std::filesystem::path{path} / name).string();
            }
            InvokeCallback(callback, userdata, {path});
            return;
        }

        UIViewController *presenter = TopViewController(window);
        if (presenter == nil) {
            InvokeCallback(callback, userdata, {});
            return;
        }

        // File extensions can't be reliably mapped to UTIs (e.g. .chd/.ccd/.mds have no registered type), so allow
        // everything and let the emulator reject unsupported files
        UIDocumentPickerViewController *picker =
            [[UIDocumentPickerViewController alloc] initForOpeningContentTypes:@[ UTTypeItem ] asCopy:YES];
        picker.allowsMultipleSelection = allowMany;

        YmirDocumentPickerDelegate *delegate = [[YmirDocumentPickerDelegate alloc] init];
        delegate.selfRetain = delegate;
        delegate.completion = ^(NSArray<NSURL *> *urls) {
            // Picked files are copies in the temporary inbox; move them under Documents/games so they persist and so
            // that multi-file images (CUE/BIN) end up side by side
            NSFileManager *fm = [NSFileManager defaultManager];
            NSURL *games = EnsureSubdirectory(@"games");
            std::vector<std::string> paths;
            for (NSURL *url in urls) {
                NSURL *dest = [games URLByAppendingPathComponent:url.lastPathComponent];
                [fm removeItemAtURL:dest error:nil];
                NSError *error = nil;
                if (![fm moveItemAtURL:url toURL:dest error:&error] && ![fm copyItemAtURL:url toURL:dest error:&error]) {
                    continue;
                }
                paths.push_back(dest.path.UTF8String);
            }
            InvokeCallback(callback, userdata, paths);
        };
        picker.delegate = delegate;
        [presenter presentViewController:picker animated:YES completion:nil];
    }
}

static void CallShell(const char *selectorName) {
    Class shell = NSClassFromString(@"YmirShell");
    SEL selector = sel_registerName(selectorName);
    if (shell != nil && [shell respondsToSelector:selector]) {
        ((void (*)(id, SEL))objc_msgSend)(shell, selector);
    }
}

void AttachGameWindow(SDL_Window *window) {
    if (window == nullptr) {
        return;
    }
    SDL_PropertiesID props = SDL_GetWindowProperties(window);
    YmirHost.gameWindow = (__bridge UIWindow *)SDL_GetPointerProperty(props, SDL_PROP_WINDOW_UIKIT_WINDOW_POINTER, nullptr);
}

void ShowLauncher() {
    CallShell("showLauncher");
}

void ShowGameMenu() {
    CallShell("showGameMenu");
}

void ToggleGameMenu() {
    CallShell("toggleGameMenu");
}

void PumpNativeUI(double seconds) {
    CFRunLoopRunInMode(kCFRunLoopDefaultMode, seconds, false);
}

bool IsNativeUIVisible() {
    Class shell = NSClassFromString(@"YmirShell");
    SEL selector = sel_registerName("isNativeUIVisible");
    if (shell != nil && [shell respondsToSelector:selector]) {
        return ((BOOL (*)(id, SEL))objc_msgSend)(shell, selector);
    }
    return false;
}

} // namespace util::ios
