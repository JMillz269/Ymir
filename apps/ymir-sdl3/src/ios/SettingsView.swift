import SwiftUI
import UniformTypeIdentifiers

// MARK: - Setting bindings

/// Bindings to the emulator's settings. Every value is read from, and written straight to, the emulator.
final class SettingStore: ObservableObject {
    static let shared = SettingStore()

    func bool(_ key: String, default defaultValue: Bool = false) -> Binding<Bool> {
        Binding(
            get: { YmirHost.boolValue(key: key, default: defaultValue) },
            set: { [weak self] value in
                YmirHost.setBoolValue(value, key: key)
                self?.objectWillChange.send()
            }
        )
    }

    func int(_ key: String, default defaultValue: Int = 0) -> Binding<Int> {
        Binding(
            get: { YmirHost.intValue(key: key, default: defaultValue) },
            set: { [weak self] value in
                YmirHost.setIntValue(value, key: key)
                self?.objectWillChange.send()
            }
        )
    }

    /// An integer setting edited with a slider.
    func intSlider(_ key: String, default defaultValue: Int = 0) -> Binding<Double> {
        Binding(
            get: { Double(YmirHost.intValue(key: key, default: defaultValue)) },
            set: { [weak self] value in
                YmirHost.setIntValue(Int(value.rounded()), key: key)
                self?.objectWillChange.send()
            }
        )
    }

    func float(_ key: String, default defaultValue: Double = 0) -> Binding<Double> {
        Binding(
            get: { Double(YmirHost.floatValue(key: key, default: Float(defaultValue))) },
            set: { [weak self] value in
                YmirHost.setFloatValue(Float(value), key: key)
                self?.objectWillChange.send()
            }
        )
    }
}

// MARK: - Reusable rows

/// A slider with a title and the current value on top.
private struct SettingSlider: View {
    let title: String
    let value: Binding<Double>
    let range: ClosedRange<Double>
    var step: Double = 0.01
    var format: (Double) -> String = { String(format: "%.0f%%", $0 * 100) }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title)
                Spacer()
                Text(format(value.wrappedValue))
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            Slider(value: value, in: range, step: step)
        }
        .padding(.vertical, 2)
    }
}

private func percent(_ value: Double) -> String {
    String(format: "%.0f%%", value * 100)
}

// MARK: - Settings home

/// The app's settings. Everything that was in the desktop frontend's menus and settings window lives here.
struct SettingsView: View {
    @ObservedObject private var store = SettingStore.shared
    @State private var hasBIOS = YmirHost.hasBIOS
    @State private var biosName = YmirHost.biosName
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
                    if hasBIOS {
                        Text(biosName)
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
                    Label(hasBIOS ? "Replace BIOS…" : "Import BIOS…", systemImage: "square.and.arrow.down")
                }
            } footer: {
                Text("A Sega Saturn BIOS (IPL ROM) image is required to run games.")
            }

            Section {
                NavigationLink {
                    DisplaySettingsView()
                } label: {
                    Label("Display", systemImage: "tv")
                }
                NavigationLink {
                    AudioSettingsView()
                } label: {
                    Label("Audio", systemImage: "speaker.wave.2")
                }
                NavigationLink {
                    ControlsSettingsView()
                } label: {
                    Label("Controls", systemImage: "gamecontroller")
                }
                NavigationLink {
                    EmulationSettingsView()
                } label: {
                    Label("Emulation", systemImage: "forward.frame")
                }
            }

            Section {
                NavigationLink {
                    SystemSettingsView()
                } label: {
                    Label("System & Region", systemImage: "globe")
                }
                NavigationLink {
                    CartridgeSettingsView()
                } label: {
                    Label("Cartridge", systemImage: "memorychip")
                }
                NavigationLink {
                    CDBlockSettingsView()
                } label: {
                    Label("CD Drive", systemImage: "opticaldisc")
                }
                NavigationLink {
                    PerformanceSettingsView()
                } label: {
                    Label("Performance & Accuracy", systemImage: "speedometer")
                }
                NavigationLink {
                    InterfaceSettingsView()
                } label: {
                    Label("Interface", systemImage: "rectangle.on.rectangle")
                }
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
        .onAppear {
            hasBIOS = YmirHost.hasBIOS
            biosName = YmirHost.biosName
        }
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
            let error = YmirHost.reloadBIOS()
            hasBIOS = YmirHost.hasBIOS
            biosName = YmirHost.biosName
            isImporting = false
            if !hasBIOS {
                biosAlert = error ?? "That file doesn't look like a valid Sega Saturn BIOS. Try a different image."
            }
        }
    }
}

// MARK: - Display

struct DisplaySettingsView: View {
    @ObservedObject private var store = SettingStore.shared

    var body: some View {
        Form {
            Section("Scaling") {
                Toggle("Stretch to Fill Screen", isOn: store.bool("video.forceStretch"))
                SettingSlider(title: "Extra Portrait Height", value: store.float("video.portraitStretch", default: 0),
                              range: 0...1, format: percent)
                    .disabled(!YmirHost.boolValue(key: "video.forceStretch", default: false))
                Toggle("Integer Scaling", isOn: store.bool("video.forceIntegerScaling"))
                    .disabled(YmirHost.boolValue(key: "video.forceStretch", default: false))
                Toggle("Force Aspect Ratio", isOn: store.bool("video.forceAspectRatio", default: true))
                Picker("Aspect Ratio", selection: store.int("video.aspect")) {
                    Text("4:3").tag(0)
                    Text("3:2").tag(1)
                    Text("16:10").tag(2)
                    Text("16:9").tag(3)
                }
                .disabled(!YmirHost.boolValue(key: "video.forceAspectRatio", default: true)
                    || YmirHost.boolValue(key: "video.forceStretch", default: false))
                Picker("Rotation", selection: store.int("video.rotation")) {
                    Text("None").tag(0)
                    Text("90° Clockwise").tag(1)
                    Text("180°").tag(2)
                    Text("90° Counterclockwise").tag(3)
                }
            }

            Section {
                Picker("MetalFX", selection: store.int("video.upscaling")) {
                    Text("Off").tag(0)
                    Text("Spatial").tag(1)
                    Text("Temporal").tag(2)
                }
                .disabled(!YmirHost.boolValue(key: "video.upscalingSupported", default: false))
            } header: {
                Text("Upscaling")
            } footer: {
                Text("MetalFX upscales the game's image with the GPU. Spatial works on each frame on its own and looks cleaner and sharper than plain scaling. Temporal also blends in previous frames for extra stability, but the Saturn's 2D video has no motion data, so fast movement can show slight ghosting.")
            }

            Section {
                Toggle("Sharp Pixels", isOn: store.bool("video.sharpPixels"))
                SettingSlider(title: "Sharpening", value: store.float("video.sharpness"), range: 0...1, format: percent)
                SettingSlider(title: "Scanlines", value: store.float("video.scanlines"), range: 0...1, format: percent)
                SettingSlider(title: "Aperture Grille", value: store.float("video.mask"), range: 0...1, format: percent)
            } header: {
                Text("Picture Filters")
            } footer: {
                Text("Sharp Pixels keeps every game pixel crisp and only smooths its edges, which looks better than plain smooth scaling. Sharpening adds crispness after upscaling. Scanlines and the aperture grille imitate a CRT TV; they look best on a high resolution screen like yours.")
            }

            Section {
                Picker("Internal Resolution", selection: store.int("video.internalResolution", default: 1)) {
                    Text("Native").tag(1)
                    Text("2×").tag(2)
                    Text("3×").tag(3)
                    Text("4×").tag(4)
                }
            } header: {
                Text("Resolution")
            } footer: {
                Text("Draws the game's 3D models, sprites and lines at several times the Saturn's own resolution, so their edges are smooth and detailed instead of blocky. Backgrounds drawn by the Saturn's second video chip stay at their original sharpness. Higher values need much more processing power. Modes that can't be enlarged (high resolution screens, interlaced video, 8-bit or rotated sprite buffers) are drawn at native resolution, and so is everything when Transparent Meshes is on.")
            }

            Section {
                Toggle("Deinterlacing", isOn: store.bool("video.deinterlace"))
                Toggle("Transparent Meshes", isOn: store.bool("video.transparentMeshes"))
            } header: {
                Text("Enhancements")
            } footer: {
                Text("Deinterlacing smooths high resolution interlaced video. Transparent meshes replaces the checkerboard patterns games use for transparency with real transparency.")
            }

            Section {
                Toggle("Video Sync", isOn: store.bool("video.sync", default: true))
                Toggle("Use Full Refresh Rate", isOn: store.bool("video.fullRefreshRate"))
                Toggle("Reduce Latency", isOn: store.bool("video.reduceLatency"))
            } header: {
                Text("Synchronization")
            } footer: {
                Text("Video sync paces emulation to the display. Reduce Latency helps on displays with a low refresh rate.")
            }
        }
        .navigationTitle("Display")
        .navigationBarTitleDisplayMode(.inline)
    }
}

// MARK: - Audio

struct AudioSettingsView: View {
    @ObservedObject private var store = SettingStore.shared

    var body: some View {
        Form {
            Section {
                Toggle("Mute", isOn: store.bool("audio.mute"))
                SettingSlider(title: "Volume", value: store.float("audio.volume", default: 1), range: 0...1, format: percent)
            }

            Section {
                Picker("Interpolation", selection: store.int("audio.interpolation", default: 1)) {
                    Text("Nearest Neighbor").tag(0)
                    Text("Linear").tag(1)
                }
                Toggle("Multithreaded SCSP", isOn: store.bool("audio.threadedSCSP"))
                Stepper(
                    "Step Granularity: \(store.int("audio.stepGranularity").wrappedValue)",
                    value: store.int("audio.stepGranularity"),
                    in: 0...5
                )
            } header: {
                Text("Sound Chip")
            } footer: {
                Text("Linear interpolation matches the real SCSP. Higher step granularity is more accurate and slower.")
            }
        }
        .navigationTitle("Audio")
        .navigationBarTitleDisplayMode(.inline)
    }
}

// MARK: - Controls

struct ControlsSettingsView: View {
    @ObservedObject private var store = SettingStore.shared

    private static let peripherals = [
        (0, "None"), (1, "Saturn Control Pad"), (2, "Saturn 3D Control Pad"), (3, "Arcade Racer"),
        (4, "Mission Stick"), (5, "Virtua Gun"), (6, "Shuttle Mouse"),
    ]

    var body: some View {
        Form {
            Section {
                Toggle("On-Screen Controls", isOn: store.bool("touch.enabled", default: true))
                Button {
                    YmirShell.beginTouchEditing()
                } label: {
                    Label("Customize Layout…", systemImage: "arrow.up.and.down.and.arrow.left.and.right")
                }
                SettingSlider(title: "Size", value: store.float("touch.globalScale", default: 1), range: 0.5...1.6,
                              format: percent)
                SettingSlider(title: "Opacity", value: store.float("touch.opacity", default: 0.9), range: 0.1...1,
                              format: percent)
            } footer: {
                Text("MFi, Xbox, PlayStation and other Bluetooth controllers work automatically, using the default bindings. On-screen controls hide while a controller is connected. Press Select on a controller to open the menu.")
            }

            Section {
                ForEach(1...2, id: \.self) { port in
                    Picker("Port \(port)", selection: store.int("input.port\(port)Type", default: port == 1 ? 1 : 0)) {
                        ForEach(Self.peripherals, id: \.0) { entry in
                            Text(entry.1).tag(entry.0)
                        }
                    }
                }
            } header: {
                Text("Connected Peripherals")
            } footer: {
                Text("Touch controls and the default controller bindings drive the Saturn Control Pad on port 1.")
            }

            Section {
                SettingSlider(title: "Left Stick Deadzone", value: store.float("input.lsDeadzone", default: 0.15),
                              range: 0.05...0.9, format: percent)
                SettingSlider(title: "Right Stick Deadzone", value: store.float("input.rsDeadzone", default: 0.15),
                              range: 0.05...0.9, format: percent)
                SettingSlider(title: "Analog to Digital Sensitivity",
                              value: store.float("input.analogToDigital", default: 0.5), range: 0.05...0.9, format: percent)
            } header: {
                Text("Controller")
            } footer: {
                Text("The deadzone is the stick travel that is ignored. Analog to digital sensitivity is how far a stick must move to press a digital button.")
            }
        }
        .navigationTitle("Controls")
        .navigationBarTitleDisplayMode(.inline)
    }
}

// MARK: - Emulation

struct EmulationSettingsView: View {
    @ObservedObject private var store = SettingStore.shared

    var body: some View {
        Form {
            Section {
                Toggle("Use Alternate Speed", isOn: store.bool("general.useAltSpeed"))
                SettingSlider(title: "Primary Speed", value: store.float("general.speedMain", default: 1),
                              range: 0.1...5, step: 0.1, format: percent)
                SettingSlider(title: "Alternate Speed", value: store.float("general.speedAlt", default: 0.5),
                              range: 0.1...5, step: 0.1, format: percent)
            } header: {
                Text("Speed")
            } footer: {
                Text("Switch between the two speeds to slow the game down or run it faster. Fast forward in the in-game menu ignores them.")
            }

            Section("Behavior") {
                Toggle("Unpause After Loading a Game", isOn: store.bool("general.unpauseOnDiscLoad", default: true))
                Toggle("Preload Disc Images to RAM", isOn: store.bool("general.preloadDiscImages"))
            }

            Section {
                Toggle("Rewind Buffer", isOn: store.bool("general.rewind"))
                SettingSlider(title: "Compression", value: store.intSlider("general.rewindCompression"),
                              range: 0...16, step: 1, format: { String(format: "%.0f", $0) })
            } header: {
                Text("Rewind")
            } footer: {
                Text("Lets you step back in time, at the cost of memory and some performance. Higher compression uses less memory but is slower.")
            }

            Section {
                Stepper(
                    "Screenshot Scale: \(store.int("general.screenshotScale", default: 1).wrappedValue)×",
                    value: store.int("general.screenshotScale", default: 1),
                    in: 1...4
                )
            } header: {
                Text("Screenshots")
            } footer: {
                Text("Screenshots are saved in the Ymir folder under screenshots.")
            }
        }
        .navigationTitle("Emulation")
        .navigationBarTitleDisplayMode(.inline)
    }
}

// MARK: - System

struct SystemSettingsView: View {
    @ObservedObject private var store = SettingStore.shared

    var body: some View {
        Form {
            Section {
                Toggle("Autodetect Region from Games", isOn: store.bool("system.autodetectRegion", default: true))
                Picker("Video Standard", selection: store.int("system.videoStandard")) {
                    Text("NTSC").tag(0)
                    Text("PAL").tag(1)
                }
                Picker("Preferred BIOS", selection: store.int("system.iplVariant")) {
                    Text("Any").tag(0)
                    Text("Saturn").tag(1)
                    Text("Hi-Saturn").tag(2)
                    Text("V-Saturn").tag(3)
                    Text("Dev Kit").tag(4)
                }
            } header: {
                Text("Region")
            } footer: {
                Text("With autodetection on, the region and video standard follow the loaded game. Reset the console after changing them.")
            }

            Section {
                Picker("Mode", selection: store.int("system.rtcMode")) {
                    Text("Host Clock").tag(0)
                    Text("Virtual").tag(1)
                }
                if store.int("system.rtcMode").wrappedValue == 1 {
                    Picker("On Hard Reset", selection: store.int("system.rtcResetStrategy")) {
                        Text("Sync to Host").tag(0)
                        Text("Reset to Fixed Time").tag(1)
                        Text("Preserve").tag(2)
                    }
                }
            } header: {
                Text("Clock")
            } footer: {
                Text("The virtual clock is deterministic and only moves while the emulator runs.")
            }

            Section {
                Toggle("Separate Backup Memory per Game", isOn: store.bool("system.backupRAMPerGame"))
            } header: {
                Text("Saves")
            } footer: {
                Text("Creates an internal backup memory image for each game. Backup files are in the Ymir folder under backup.")
            }
        }
        .navigationTitle("System & Region")
        .navigationBarTitleDisplayMode(.inline)
    }
}

// MARK: - Cartridge

struct CartridgeSettingsView: View {
    @ObservedObject private var store = SettingStore.shared

    var body: some View {
        Form {
            Section {
                Toggle("Switch to Recommended Cartridges", isOn: store.bool("cartridge.autoLoad", default: true))
            } footer: {
                Text("Automatically inserts the cartridge a game needs, such as a RAM expansion.")
            }

            Section {
                Picker("Cartridge", selection: store.int("cartridge.type")) {
                    Text("None").tag(0)
                    Text("Backup RAM").tag(1)
                    Text("DRAM Expansion").tag(2)
                    Text("ROM").tag(3)
                }

                switch store.int("cartridge.type").wrappedValue {
                case 1:
                    Picker("Capacity", selection: store.int("cartridge.backupCapacity")) {
                        Text("4 Mbit").tag(0)
                        Text("8 Mbit").tag(1)
                        Text("16 Mbit").tag(2)
                        Text("32 Mbit").tag(3)
                    }
                    FileSettingRow(title: "Image", key: "cartridge.backupPath",
                                   folder: AppPaths.documents.appendingPathComponent("backup", isDirectory: true))
                case 2:
                    Picker("Capacity", selection: store.int("cartridge.dramCapacity")) {
                        Text("48 Mbit").tag(0)
                        Text("32 Mbit").tag(1)
                        Text("8 Mbit").tag(2)
                    }
                case 3:
                    FileSettingRow(title: "ROM Image", key: "cartridge.romPath",
                                   folder: AppPaths.documents.appendingPathComponent("roms/cart", isDirectory: true))
                default:
                    EmptyView()
                }
            } header: {
                Text("Inserted Cartridge")
            } footer: {
                Text("Chosen files are copied into the Ymir folder, under roms/cart and backup.")
            }
        }
        .navigationTitle("Cartridge")
        .navigationBarTitleDisplayMode(.inline)
    }
}

// MARK: - CD drive

struct CDBlockSettingsView: View {
    @ObservedObject private var store = SettingStore.shared

    var body: some View {
        Form {
            Section {
                SettingSlider(title: "Read Speed", value: store.intSlider("cdblock.readSpeed", default: 2),
                              range: 2...200, step: 1, format: { String(format: "%.0f×", $0) })
            } footer: {
                Text("The maximum read speed of the emulated drive. 2× is how fast a real Saturn reads discs.")
            }

            Section {
                Toggle("Low Level Emulation", isOn: store.bool("cdblock.useLLE"))
                FileSettingRow(title: "CD Block ROM", key: "cdblock.romPath",
                               folder: AppPaths.documents.appendingPathComponent("roms/cdb", isDirectory: true))
            } header: {
                Text("Accuracy")
            } footer: {
                Text("Emulates the CD block's own CPU. Needs a CD block ROM image, and is much slower.")
            }
        }
        .navigationTitle("CD Drive")
        .navigationBarTitleDisplayMode(.inline)
    }
}

// MARK: - Performance

struct PerformanceSettingsView: View {
    @ObservedObject private var store = SettingStore.shared
    @State private var jitAvailable = YmirHost.jitAvailable
    @State private var jitStatus = YmirHost.jitStatus
    @State private var jitReport = YmirHost.jitReport

    var body: some View {
        Form {
            Section {
                Button("Recommended") {
                    apply(sh2Cache: false, interpolation: 1, granularity: 0, lle: false)
                }
                Button("Best Accuracy") {
                    apply(sh2Cache: true, interpolation: 1, granularity: 5, lle: false)
                }
                Button("Best Performance") {
                    apply(sh2Cache: false, interpolation: 0, granularity: 0, lle: false)
                }
            } header: {
                Text("Presets")
            } footer: {
                Text("Presets change the options below and the sound chip settings in Audio.")
            }

            Section {
                Picker("CPU Execution", selection: store.bool("system.useRecompiler")) {
                    Text("Interpreter").tag(false)
                    Text("Recompiler (JIT)").tag(true)
                }
                .disabled(!jitAvailable || store.bool("system.emulateSH2Cache").wrappedValue)

                HStack(spacing: 10) {
                    Image(systemName: jitAvailable ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                        .foregroundStyle(jitAvailable ? Color.green : Color.orange)
                    Text(jitStatus)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                if jitAvailable && store.bool("system.useRecompiler").wrappedValue {
                    Text(jitReport)
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .onReceive(Timer.publish(every: 1, on: .main, in: .common).autoconnect()) { _ in
                            jitReport = YmirHost.jitReport
                        }
                }
                if jitAvailable && store.bool("system.useRecompiler").wrappedValue {
                    Toggle("Keep Registers in CPU Registers", isOn: store.bool("system.jitRegisterCache", default: true))
                    Toggle("Link Blocks Directly", isOn: store.bool("system.jitBlockLinking", default: true))
                }
                if !jitAvailable {
                    Button {
                        requestJIT()
                    } label: {
                        Label("Enable JIT…", systemImage: "bolt.fill")
                    }
                }
            } header: {
                Text("CPU")
            } footer: {
                Text(recompilerFooter)
            }

            Section {
                Toggle("Emulate SH-2 Cache", isOn: store.bool("system.emulateSH2Cache"))
                SettingSlider(title: "SH-2 Clock", value: store.intSlider("system.sh2ClockFactor", default: 100),
                              range: 25...500, step: 5, format: { String(format: "%.0f%%", $0) })
                Button("Reset Clock to 100%") {
                    store.intSlider("system.sh2ClockFactor").wrappedValue = 100
                }
            } header: {
                Text("Accuracy")
            } footer: {
                Text("A few games need the SH-2 cache. Changing the clock may break games, desync audio or cause crashes: above 100% reduces slowdowns, below 100% helps slower devices.")
            }

            Section {
                Toggle("Threaded VDP1", isOn: store.bool("video.threadedVDP1", default: true))
                Toggle("Threaded VDP2", isOn: store.bool("video.threadedVDP2", default: true))
                Toggle("Threaded Deinterlacer", isOn: store.bool("video.threadedDeinterlacer", default: true))
                Toggle("Threaded Sound Chip", isOn: store.bool("audio.threadedSCSP"))
            } header: {
                Text("Multithreading")
            } footer: {
                Text("Spreads emulation over several CPU cores for speed.")
            }

            Section {
                Toggle("Boost Emulator Thread Priority", isOn: store.bool("general.boostEmuThread"))
            } footer: {
                Text("May reduce stuttering, at the cost of more battery use.")
            }
        }
        .navigationTitle("Performance")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear { refreshJIT() }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)) { _ in
            refreshJIT()
        }
    }

    private var recompilerFooter: String {
        if store.bool("system.emulateSH2Cache").wrappedValue {
            return "The recompiler is not used while SH-2 cache emulation is on."
        }
        return "Translates the Saturn's SH-2 processors to native code instead of interpreting them. Experimental: games "
            + "may run faster, or may misbehave; turn it off if something looks wrong. It needs a JIT enabler "
            + "(StikDebug, StosDebug, ...) attached to Ymir, and the change applies to the running game immediately."
    }

    private func refreshJIT() {
        jitAvailable = YmirHost.jitAvailable
        jitStatus = YmirHost.jitStatus
    }

    /// Asks a JIT enabler app to attach to Ymir. Tries the URL schemes of the common enablers in turn.
    private func requestJIT() {
        let bundleID = Bundle.main.bundleIdentifier ?? ""
        let schemes = ["stikdebug", "stikjit", "stosdebug"]
        func attempt(_ index: Int) {
            guard index < schemes.count, let url = URL(string: "\(schemes[index])://enable-jit?bundle-id=\(bundleID)") else {
                jitStatus = "Couldn't reach a JIT enabler. Open StikDebug (or a similar app) and enable JIT for Ymir there."
                return
            }
            UIApplication.shared.open(url, options: [:]) { opened in
                if !opened {
                    attempt(index + 1)
                }
            }
        }
        attempt(0)
    }

    private func apply(sh2Cache: Bool, interpolation: Int, granularity: Int, lle: Bool) {
        store.bool("system.emulateSH2Cache").wrappedValue = sh2Cache
        store.intSlider("system.sh2ClockFactor").wrappedValue = 100
        store.int("audio.interpolation").wrappedValue = interpolation
        store.int("audio.stepGranularity").wrappedValue = granularity
        store.intSlider("cdblock.readSpeed").wrappedValue = 2
        store.bool("cdblock.useLLE").wrappedValue = lle
    }
}

// MARK: - Interface

struct InterfaceSettingsView: View {
    @ObservedObject private var store = SettingStore.shared

    var body: some View {
        Form {
            Section {
                Toggle("Show Frame Rate", isOn: store.bool("gui.showFrameRate"))
                Picker("Position", selection: store.int("gui.frameRatePosition")) {
                    Text("Top Left").tag(0)
                    Text("Top Right").tag(1)
                    Text("Bottom Left").tag(2)
                    Text("Bottom Right").tag(3)
                }
                .disabled(!YmirHost.boolValue(key: "gui.showFrameRate", default: false))
            } header: {
                Text("Frame Rate")
            }

            Section {
                Toggle("Show Messages", isOn: store.bool("gui.showMessages", default: true))
                Toggle("Always Show Speed Indicator", isOn: store.bool("gui.showSpeedIndicator"))
            } header: {
                Text("On-Screen Display")
            }
        }
        .navigationTitle("Interface")
        .navigationBarTitleDisplayMode(.inline)
    }
}

// MARK: - File settings

/// A setting that points to a file (a cartridge image, a ROM...). Picked files are copied into the Ymir folder.
private struct FileSettingRow: View {
    let title: String
    let key: String
    let folder: URL

    @State private var showImporter = false
    @State private var name = URL(fileURLWithPath: "").lastPathComponent

    init(title: String, key: String, folder: URL) {
        self.title = title
        self.key = key
        self.folder = folder
        _name = State(initialValue: URL(fileURLWithPath: YmirHost.pathValue(key: key)).lastPathComponent)
    }

    var body: some View {
        HStack {
            Text(title)
            Spacer()
            Text(name.isEmpty ? "None" : name)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        Button {
            showImporter = true
        } label: {
            Label("Choose File…", systemImage: "doc.badge.plus")
        }
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.item], allowsMultipleSelection: false) { result in
            guard case .success(let urls) = result, let url = urls.first else { return }
            let folder = self.folder
            Task {
                _ = await Task.detached(priority: .userInitiated) { () -> ImportService.Outcome in
                    ImportService.copy([url], to: folder)
                }.value
                let destination = folder.appendingPathComponent(url.lastPathComponent)
                YmirHost.setPathValue(destination.path, key: key)
                name = destination.lastPathComponent
                SettingStore.shared.objectWillChange.send()
            }
        }
    }
}
