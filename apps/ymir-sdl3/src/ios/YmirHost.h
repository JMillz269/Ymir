#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

/// Objective-C facade over the emulator frontend, used by the Swift user interface.
/// All methods must be called from the main thread.
@interface YmirHost : NSObject

/// The UIWindow that hosts the emulator's SDL view, or nil before the emulator window exists.
@property(class, nonatomic, weak, nullable) UIWindow *gameWindow;

/// Whether the emulator frontend is up and accepting commands.
@property(class, nonatomic, readonly) BOOL isRunning;
@property(class, nonatomic, readonly) BOOL isDiscLoaded;
@property(class, nonatomic, readonly) BOOL hasBIOS;
@property(class, nonatomic, readonly) NSString *biosName;
@property(class, nonatomic, readonly, nullable) NSString *loadedGamePath;
@property(class, nonatomic, readonly) NSString *documentsPath;

@property(class, nonatomic) BOOL paused;
@property(class, nonatomic) BOOL fastForward;

+ (void)launchGameAtPath:(NSString *)path NS_SWIFT_NAME(launchGame(path:));
+ (void)softReset NS_SWIFT_NAME(softReset());
+ (void)hardReset NS_SWIFT_NAME(hardReset());
+ (void)openCloseTray NS_SWIFT_NAME(openCloseTray());
+ (void)ejectDisc NS_SWIFT_NAME(ejectDisc());
+ (void)saveStateToSlot:(NSInteger)slot NS_SWIFT_NAME(saveState(slot:));
+ (void)loadStateFromSlot:(NSInteger)slot NS_SWIFT_NAME(loadState(slot:));
+ (void)quitGame NS_SWIFT_NAME(quitGame());
+ (void)takeScreenshot NS_SWIFT_NAME(takeScreenshot());

/// Rescans the BIOS folder and loads an IPL ROM. Returns an error message, or nil on success.
+ (nullable NSString *)reloadBIOS NS_SWIFT_NAME(reloadBIOS());

/// Returns an array of dictionaries with keys "index" (NSNumber), "present" (NSNumber) and "text" (NSString).
+ (NSArray<NSDictionary<NSString *, id> *> *)saveSlots NS_SWIFT_NAME(saveSlots());

+ (BOOL)boolForKey:(NSString *)key defaultValue:(BOOL)defaultValue NS_SWIFT_NAME(boolValue(key:default:));
+ (void)setBool:(BOOL)value forKey:(NSString *)key NS_SWIFT_NAME(setBoolValue(_:key:));
+ (NSInteger)intForKey:(NSString *)key defaultValue:(NSInteger)defaultValue NS_SWIFT_NAME(intValue(key:default:));
+ (void)setInt:(NSInteger)value forKey:(NSString *)key NS_SWIFT_NAME(setIntValue(_:key:));
+ (float)floatForKey:(NSString *)key defaultValue:(float)defaultValue NS_SWIFT_NAME(floatValue(key:default:));
+ (void)setFloat:(float)value forKey:(NSString *)key NS_SWIFT_NAME(setFloatValue(_:key:));

@end

NS_ASSUME_NONNULL_END
