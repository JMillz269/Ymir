import SwiftUI
import UIKit

/// Hosting controller used for all native screens. Reports when it was dismissed, including by swiping a sheet down.
final class ShellHostingController<Content: View>: UIHostingController<Content> {
    var onDismissed: (() -> Void)?

    override var prefersStatusBarHidden: Bool { false }

    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        if isBeingDismissed {
            onDismissed?()
        }
    }
}

/// State shared between the in-game menu and the code that presents it.
@MainActor
final class MenuState: ObservableObject {
    /// Whether the game should stay paused after the menu closes. The game is always paused while the menu is open.
    @Published var keepPaused = false
}

/// Presents the native user interface (game library, settings and in-game menu) over the emulator's view.
///
/// The emulator itself is driven by SDL, which owns the application and its window. This class is invoked from the
/// frontend through the Objective-C runtime (see `util::ios::ShowLauncher` and `util::ios::ShowGameMenu`).
@MainActor
@objc(YmirShell)
final class YmirShell: NSObject {
    private static var launcherController: UIViewController?
    private static var menuController: UIViewController?
    private static var pausedByLauncher = false
    private static var afterMenuClosed: (() -> Void)?
    private static let menuState = MenuState()

    @objc static func isNativeUIVisible() -> Bool {
        launcherController != nil || menuController != nil
    }

    // MARK: Presentation

    private static func topViewController() -> UIViewController? {
        var window: UIWindow? = YmirHost.gameWindow
        if window == nil {
            window = UIApplication.shared.connectedScenes
                .compactMap { $0 as? UIWindowScene }
                .flatMap { $0.windows }
                .first { $0.isKeyWindow }
        }
        var controller = window?.rootViewController
        while let presented = controller?.presentedViewController {
            controller = presented
        }
        return controller
    }

    /// Shows the game library and settings.
    @objc static func showLauncher() {
        guard launcherController == nil else { return }

        // Never stack the library on top of the in-game menu
        if menuController != nil {
            closeMenu { YmirShell.showLauncher() }
            return
        }
        guard let presenter = topViewController() else { return }

        // Don't let the game run behind the library
        if YmirHost.isDiscLoaded && !YmirHost.paused {
            YmirHost.paused = true
            pausedByLauncher = true
        }

        let root = LauncherRoot(
            onLaunch: { game in YmirShell.launch(game) },
            onResume: { YmirShell.resumeFromLauncher() }
        )
        let controller = ShellHostingController(rootView: root)
        // Keep the emulator view in the hierarchy underneath, so that its Metal layer keeps presenting frames
        controller.modalPresentationStyle = .overFullScreen
        controller.modalPresentationCapturesStatusBarAppearance = true
        launcherController = controller
        presenter.present(controller, animated: true)
    }

    /// Shows the in-game menu as a sheet.
    @objc static func showGameMenu() {
        guard menuController == nil, launcherController == nil, let presenter = topViewController() else { return }

        menuState.keepPaused = YmirHost.paused
        if !YmirHost.paused {
            YmirHost.paused = true
        }

        let root = GameMenuView(state: menuState)
        let controller = ShellHostingController(rootView: root)
        controller.modalPresentationStyle = .pageSheet
        controller.modalPresentationCapturesStatusBarAppearance = true
        if let sheet = controller.sheetPresentationController {
            sheet.detents = [.medium(), .large()]
            sheet.prefersGrabberVisible = true
        }
        controller.onDismissed = { YmirShell.menuDidClose() }
        menuController = controller
        presenter.present(controller, animated: true)
    }

    // MARK: Actions

    private static func launch(_ game: GameEntry) {
        GameLibrary.markPlayed(game)
        YmirHost.launchGame(path: game.url.path)
        pausedByLauncher = false
        dismissLauncher()
    }

    private static func resumeFromLauncher() {
        if pausedByLauncher {
            YmirHost.paused = false
            pausedByLauncher = false
        }
        dismissLauncher()
    }

    private static func dismissLauncher(completion: (() -> Void)? = nil) {
        guard let controller = launcherController else {
            completion?()
            return
        }
        launcherController = nil
        controller.dismiss(animated: true, completion: completion)
    }

    /// Closes the in-game menu and optionally runs something once it is gone.
    static func closeMenu(then action: (() -> Void)? = nil) {
        guard let controller = menuController else {
            action?()
            return
        }
        afterMenuClosed = action
        controller.dismiss(animated: true)
    }

    private static func menuDidClose() {
        guard menuController != nil else { return }
        menuController = nil
        YmirHost.paused = menuState.keepPaused
        let action = afterMenuClosed
        afterMenuClosed = nil
        action?()
    }

    /// Opens the in-game menu, or closes it when it is already open (the Select button of a controller does this).
    @objc static func toggleGameMenu() {
        if menuController != nil {
            closeMenu()
        } else {
            showGameMenu()
        }
    }

    /// Returns from the in-game menu to the game library, leaving the game paused where it is.
    static func showLibraryFromMenu() {
        closeMenu { YmirShell.showLauncher() }
    }

    /// Unloads the game and returns to the game library.
    static func closeGame() {
        closeMenu {
            YmirHost.quitGame()
            YmirShell.showLauncher()
        }
    }
}
