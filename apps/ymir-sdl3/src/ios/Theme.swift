import SwiftUI
import UIKit

// MARK: - Liquid Glass helpers

extension View {
    /// Fills the view with Liquid Glass on iOS 26 and later, and with a translucent material on older systems.
    @ViewBuilder
    func ymirGlass<S: Shape>(in shape: S, interactive: Bool = true) -> some View {
        #if compiler(>=6.2)
        if #available(iOS 26.0, *) {
            if interactive {
                self.glassEffect(.regular.interactive(), in: shape)
            } else {
                self.glassEffect(.regular, in: shape)
            }
        } else {
            self.background(.ultraThinMaterial, in: shape)
        }
        #else
        self.background(.ultraThinMaterial, in: shape)
        #endif
    }

    /// Prominent call-to-action button style: a tinted glass button on iOS 26 and later, bordered prominent before.
    @ViewBuilder
    func ymirProminentButton() -> some View {
        #if compiler(>=6.2)
        if #available(iOS 26.0, *) {
            self.buttonStyle(.glassProminent)
        } else {
            self.buttonStyle(.borderedProminent)
        }
        #else
        self.buttonStyle(.borderedProminent)
        #endif
    }

    /// Secondary button style: a glass button on iOS 26 and later, bordered before.
    @ViewBuilder
    func ymirSecondaryButton() -> some View {
        #if compiler(>=6.2)
        if #available(iOS 26.0, *) {
            self.buttonStyle(.glass)
        } else {
            self.buttonStyle(.bordered)
        }
        #else
        self.buttonStyle(.bordered)
        #endif
    }
}

// MARK: - Paths

enum AppPaths {
    static var documents: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    static var games: URL {
        documents.appendingPathComponent("games", isDirectory: true)
    }

    static var bios: URL {
        documents.appendingPathComponent("roms", isDirectory: true).appendingPathComponent("ipl", isDirectory: true)
    }

    static var covers: URL {
        documents.appendingPathComponent("covers", isDirectory: true)
    }
}
