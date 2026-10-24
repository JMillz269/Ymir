import SwiftUI
import UniformTypeIdentifiers

/// Values backed by the emulator's settings. Changes are written through to the emulator immediately.
@MainActor
final class SettingsModel: ObservableObject {
    @Published var integerScaling = false { didSet { write(integerScaling, "video.forceIntegerScaling") } }
    @Published var keepAspectRatio = true { didSet { write(keepAspectRatio, "video.forceAspectRatio") } }
    @Published var threadedVDP = false { didSet { write(threadedVDP, "video.threadedVDP") } }
    @Published var showFPS = false { didSet { write(showFPS, "gui.showFrameRateOSD") } }
    @Published var mute = false { didSet { write(mute, "audio.mute") } }
    @Published var volume: Double = 1 {
        didSet {
            guard !loading else { return }
            YmirHost.setFloatValue(Float(volume), key: "audio.volume")
        }
    }
    @Published var touchControls = true { didSet { write(touchControls, "touch.enabled") } }
    @Published var rewind = false { didSet { write(rewind, "general.enableRewindBuffer") } }
    @Published var biosName = ""
    @Published var hasBIOS = false

    private var loading = false

    init() {
        load()
    }

    func load() {
        loading = true
        integerScaling = YmirHost.boolValue(key: "video.forceIntegerScaling", default: false)
        keepAspectRatio = YmirHost.boolValue(key: "video.forceAspectRatio", default: true)
        threadedVDP = YmirHost.boolValue(key: "video.threadedVDP", default: false)
        showFPS = YmirHost.boolValue(key: "gui.showFrameRateOSD", default: false)
        mute = YmirHost.boolValue(key: "audio.mute", default: false)
        volume = Double(YmirHost.floatValue(key: "audio.volume", default: 1))
        touchControls = YmirHost.boolValue(key: "touch.enabled", default: true)
        rewind = YmirHost.boolValue(key: "general.enableRewindBuffer", default: false)
        refreshBIOS()
        loading = false
    }

    func refreshBIOS() {
        hasBIOS = YmirHost.hasBIOS
        biosName = YmirHost.biosName
    }

    private func write(_ value: Bool, _ key: String) {
        guard !loading else { return }
        YmirHost.setBoolValue(value, key: key)
    }
}

/// The app's settings. Everything that used to live in the menu bar of the desktop frontend is reachable from here.
struct SettingsView: View {
    @StateObject private var model = SettingsModel()
    @State private var showImporter = false
    @State private var isImporting = false
    @State private var biosAlert: String?

    private var version: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?"
    }

    var body: some View {
        Form {
            Section {
                HStack {
                    Label("BIOS", systemImage: "cpu")
                    Spacer()
                    if model.hasBIOS {
                        Text(model.biosName)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    } else {
                        Text("Not loaded")
                            .foregroundStyle(.orange)
                    }
                }
                Button {
                    showImporter = true
                } label: {
                    Label(model.hasBIOS ? "Replace BIOS…" : "Import BIOS…", systemImage: "square.and.arrow.down")
                }
            } header: {
                Text("System")
            } footer: {
                Text("A Sega Saturn BIOS (IPL ROM) image is required to run games.")
            }

            Section("Display") {
                Toggle("Integer Scaling", isOn: $model.integerScaling)
                Toggle("Keep Aspect Ratio", isOn: $model.keepAspectRatio)
                Toggle("Multithreaded Renderer", isOn: $model.threadedVDP)
                Toggle("Show Frame Rate", isOn: $model.showFPS)
            }

            Section("Audio") {
                Toggle("Mute", isOn: $model.mute)
                HStack(spacing: 12) {
                    Image(systemName: "speaker.fill").foregroundStyle(.secondary)
                    Slider(value: $model.volume, in: 0...1)
                    Image(systemName: "speaker.wave.3.fill").foregroundStyle(.secondary)
                }
                .disabled(model.mute)
            }

            Section {
                Toggle("On-Screen Controls", isOn: $model.touchControls)
                AdvancedRow(title: "Controller Bindings", systemImage: "gamecontroller", tab: "input")
                AdvancedRow(title: "Hotkeys", systemImage: "keyboard", tab: "hotkeys")
            } header: {
                Text("Controls")
            } footer: {
                Text("MFi, Xbox, PlayStation and other Bluetooth controllers work automatically. On-screen controls hide while a controller is connected.")
            }

            Section {
                Toggle("Rewind Buffer", isOn: $model.rewind)
            } header: {
                Text("Emulation")
            } footer: {
                Text("Lets you step back in time. Uses more memory.")
            }

            Section {
                AdvancedRow(title: "General", systemImage: "slider.horizontal.3", tab: "general")
                AdvancedRow(title: "System & Region", systemImage: "globe", tab: "system")
                AdvancedRow(title: "Video", systemImage: "tv", tab: "video")
                AdvancedRow(title: "Audio", systemImage: "speaker.wave.2", tab: "audio")
                AdvancedRow(title: "Cartridge", systemImage: "memorychip", tab: "cartridge")
                AdvancedRow(title: "CD Block", systemImage: "opticaldisc", tab: "cdblock")
                AdvancedRow(title: "Tweaks", systemImage: "wrench.and.screwdriver", tab: "tweaks")
            } header: {
                Text("Advanced")
            } footer: {
                Text("Opens the emulator's built-in settings window over the game.")
            }

            Section {
                Button {
                    FilesApp.openDocuments()
                } label: {
                    Label("Open Ymir Folder in Files", systemImage: "folder")
                }
            } header: {
                Text("Files")
            } footer: {
                Text("Games, BIOS images, save data and screenshots live in the Ymir folder, which is visible in the Files app.")
            }

            Section("About") {
                HStack {
                    Text("Ymir")
                    Spacer()
                    Text(version).foregroundStyle(.secondary)
                }
                Text("Sega Saturn emulator")
                    .foregroundStyle(.secondary)
            }
        }
        .onAppear { model.load() }
        .overlay {
            if isImporting {
                ProgressView("Loading BIOS…")
                    .padding(24)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            }
        }
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.item], allowsMultipleSelection: false) { result in
            guard case .success(let urls) = result, !urls.isEmpty else { return }
            importBIOS(urls)
        }
        .alert("BIOS Not Recognized", isPresented: Binding(get: { biosAlert != nil }, set: { if !$0 { biosAlert = nil } })) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(biosAlert ?? "")
        }
    }

    private func importBIOS(_ urls: [URL]) {
        isImporting = true
        Task {
            _ = await Task.detached(priority: .userInitiated) { () -> ImportService.Outcome in
                ImportService.copy(urls, to: AppPaths.bios)
            }.value
            _ = YmirHost.reloadBIOS()
            // The emulator loads the BIOS on its next frame
            try? await Task.sleep(nanoseconds: 1_200_000_000)
            model.refreshBIOS()
            isImporting = false
            if !model.hasBIOS {
                biosAlert = "That file doesn't look like a valid Sega Saturn BIOS. Try a different image."
            }
        }
    }
}

/// A row that opens one of the emulator's own settings pages.
private struct AdvancedRow: View {
    let title: String
    let systemImage: String
    let tab: String

    var body: some View {
        Button {
            YmirShell.openAdvancedSettings(tab)
        } label: {
            HStack {
                Label(title, systemImage: systemImage)
                    .foregroundStyle(.primary)
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(.tertiary)
            }
            .contentShape(Rectangle())
        }
    }
}
