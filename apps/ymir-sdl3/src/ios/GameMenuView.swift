import SwiftUI
import UIKit

/// The in-game menu, presented as a sheet over the running game. It replaces the desktop frontend's menu bar.
struct GameMenuView: View {
    @ObservedObject var state: MenuState

    @State private var fastForward = YmirHost.fastForward
    @State private var touchControls = YmirHost.boolValue(key: "touch.enabled", default: true)
    @State private var confirmHardReset = false
    @State private var confirmClose = false
    @State private var gameTitle = GameMenuView.currentGameTitle()

    private static func currentGameTitle() -> String {
        guard let path = YmirHost.loadedGamePath else { return "No Game Loaded" }
        return URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
    }

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Button {
                        state.keepPaused = false
                        YmirShell.closeMenu()
                    } label: {
                        Label("Resume", systemImage: "play.fill")
                            .font(.headline)
                    }

                    Toggle(isOn: $state.keepPaused) {
                        Label("Pause Emulation", systemImage: "pause.fill")
                    }

                    Toggle(isOn: $fastForward) {
                        Label("Fast Forward", systemImage: "forward.fill")
                    }
                    .onChange(of: fastForward) { value in
                        YmirHost.fastForward = value
                    }
                } header: {
                    Text(gameTitle)
                        .textCase(nil)
                        .font(.headline)
                        .foregroundStyle(.primary)
                }

                Section("Game") {
                    Button {
                        YmirHost.softReset()
                        state.keepPaused = false
                        YmirShell.closeMenu()
                    } label: {
                        Label("Reset", systemImage: "arrow.counterclockwise")
                    }

                    Button(role: .destructive) {
                        confirmHardReset = true
                    } label: {
                        Label("Hard Reset", systemImage: "power")
                    }

                    Button {
                        YmirHost.openCloseTray()
                        state.keepPaused = false
                        YmirShell.closeMenu()
                    } label: {
                        Label("Open/Close Disc Tray", systemImage: "eject")
                    }

                    NavigationLink {
                        SaveStatesView(state: state)
                    } label: {
                        Label("Save States", systemImage: "externaldrive")
                    }

                    Button {
                        YmirHost.takeScreenshot()
                        state.keepPaused = false
                        YmirShell.closeMenu()
                    } label: {
                        Label("Take Screenshot", systemImage: "camera")
                    }
                }

                Section("Controls") {
                    Toggle(isOn: $touchControls) {
                        Label("On-Screen Controls", systemImage: "gamecontroller")
                    }
                    .onChange(of: touchControls) { value in
                        YmirHost.setBoolValue(value, key: "touch.enabled")
                    }

                    Button {
                        YmirShell.beginTouchEditing()
                    } label: {
                        Label("Customize Layout…", systemImage: "arrow.up.and.down.and.arrow.left.and.right")
                    }
                }

                Section {
                    NavigationLink {
                        SettingsView()
                            .navigationTitle("Settings")
                            .navigationBarTitleDisplayMode(.inline)
                    } label: {
                        Label("Settings", systemImage: "gearshape")
                    }

                    Button {
                        YmirShell.showLibraryFromMenu()
                    } label: {
                        Label("Game Library", systemImage: "square.grid.2x2")
                    }

                    Button(role: .destructive) {
                        confirmClose = true
                    } label: {
                        Label("Close Game", systemImage: "xmark.circle")
                    }
                    .confirmationDialog("Close this game?", isPresented: $confirmClose, titleVisibility: .visible) {
                        Button("Close Game", role: .destructive) {
                            YmirShell.closeGame()
                        }
                        Button("Cancel", role: .cancel) {}
                    } message: {
                        Text("Unsaved progress is lost. Use a save state first to keep it.")
                    }
                }
            }
            .navigationTitle("Menu")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        YmirShell.closeMenu()
                    }
                    .fontWeight(.semibold)
                }
            }
            .confirmationDialog("Hard reset the console?", isPresented: $confirmHardReset, titleVisibility: .visible) {
                Button("Hard Reset", role: .destructive) {
                    YmirHost.hardReset()
                    state.keepPaused = false
                    YmirShell.closeMenu()
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Unsaved progress is lost, like pressing the power button on a real Saturn.")
            }
        }
    }
}

/// Lists the save state slots and lets the player save to, load from, export or delete any of them.
struct SaveStatesView: View {
    @ObservedObject var state: MenuState

    @State private var slots: [Slot] = SaveStatesView.readSlots()
    @State private var slotToDelete: Slot?
    @State private var exportItem: ExportItem?
    @State private var exportError: String?

    struct Slot: Identifiable {
        let index: Int
        let present: Bool
        let text: String
        var id: Int { index }
    }

    /// A save state file copied under a readable name, for the share sheet
    struct ExportItem: Identifiable {
        let url: URL
        var id: URL { url }
    }

    private static func readSlots() -> [Slot] {
        YmirHost.saveSlots().map { entry in
            Slot(
                index: (entry["index"] as? NSNumber)?.intValue ?? 0,
                present: (entry["present"] as? NSNumber)?.boolValue ?? false,
                text: (entry["text"] as? String) ?? ""
            )
        }
    }

    /// Copies the slot's file to a temporary folder as "<game> - Slot N.savestate" and offers it to share or save.
    private func export(_ slot: Slot) {
        guard let path = YmirHost.saveStatePath(slot: slot.index) else {
            exportError = "Slot \(slot.index + 1) has no save state file yet."
            return
        }
        let game = YmirHost.loadedGamePath.map { URL(fileURLWithPath: $0).deletingPathExtension().lastPathComponent }
            ?? "Ymir"
        let name = "\(game) - Slot \(slot.index + 1).savestate".replacingOccurrences(of: "/", with: "-")
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("SaveStateExport", isDirectory: true)
        let target = folder.appendingPathComponent(name)
        do {
            try? FileManager.default.removeItem(at: folder)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: URL(fileURLWithPath: path), to: target)
            exportItem = ExportItem(url: target)
        } catch {
            exportError = error.localizedDescription
        }
    }

    var body: some View {
        List {
            Section {
                ForEach(slots) { slot in
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Slot \(slot.index + 1)")
                            Text(slot.present ? slot.text : "Empty")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button("Save") {
                            YmirHost.saveState(slot: slot.index)
                            slots = Self.readSlots()
                            // Saving happens on the emulator thread; read again once it had time to finish
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { slots = Self.readSlots() }
                        }
                        .ymirSecondaryButton()
                        Button("Load") {
                            YmirHost.loadState(slot: slot.index)
                            state.keepPaused = false
                            YmirShell.closeMenu()
                        }
                        .ymirSecondaryButton()
                        .disabled(!slot.present)
                        Menu {
                            Button {
                                export(slot)
                            } label: {
                                Label("Export…", systemImage: "square.and.arrow.up")
                            }
                            Button(role: .destructive) {
                                slotToDelete = slot
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        } label: {
                            Image(systemName: "ellipsis.circle")
                                .font(.title3)
                                .frame(minWidth: 32, minHeight: 32)
                        }
                        .disabled(!slot.present)
                        .accessibilityLabel("More for slot \(slot.index + 1)")
                    }
                    .buttonStyle(.borderless)
                    .swipeActions(edge: .trailing) {
                        if slot.present {
                            Button(role: .destructive) {
                                slotToDelete = slot
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                    }
                }
            } footer: {
                Text("Save states are stored per game and appear in the Files app under Ymir › savestates. Export shares a copy, for example to Files or AirDrop.")
            }
        }
        .navigationTitle("Save States")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear { slots = Self.readSlots() }
        .confirmationDialog(
            "Delete the save state in slot \((slotToDelete?.index ?? 0) + 1)?",
            isPresented: Binding(get: { slotToDelete != nil }, set: { if !$0 { slotToDelete = nil } }),
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                if let slot = slotToDelete {
                    YmirHost.deleteState(slot: slot.index)
                    slots = Self.readSlots()
                }
                slotToDelete = nil
            }
            Button("Cancel", role: .cancel) { slotToDelete = nil }
        } message: {
            Text("This can't be undone.")
        }
        .sheet(item: $exportItem) { item in
            ShareSheet(items: [item.url])
        }
        .alert(
            "Couldn't export the save state",
            isPresented: Binding(get: { exportError != nil }, set: { if !$0 { exportError = nil } })
        ) {
            Button("OK", role: .cancel) { exportError = nil }
        } message: {
            Text(exportError ?? "")
        }
    }
}

/// The system share sheet, for exporting files
struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
