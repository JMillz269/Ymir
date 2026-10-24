import SwiftUI
import UniformTypeIdentifiers

/// Root of the native app: a tab bar (Liquid Glass on iOS 26 and later) with the game library and the settings.
struct LauncherRoot: View {
    let onLaunch: (GameEntry) -> Void
    let onResume: () -> Void

    @StateObject private var library = GameLibrary()
    @State private var selection = 0

    var body: some View {
        TabView(selection: $selection) {
            GamesView(library: library, onLaunch: onLaunch, onResume: onResume)
                .tabItem { Label("Games", systemImage: "gamecontroller.fill") }
                .tag(0)

            NavigationStack {
                SettingsView()
                    .navigationTitle("Settings")
            }
            .tabItem { Label("Settings", systemImage: "gearshape.fill") }
            .tag(1)
        }
    }
}

// MARK: - Games tab

private enum ImportKind {
    case games
    case bios
}

private struct AlertInfo: Identifiable {
    let id = UUID()
    let title: String
    let message: String
    var actionTitle: String?
    var action: (() -> Void)?
}

struct GamesView: View {
    @ObservedObject var library: GameLibrary
    let onLaunch: (GameEntry) -> Void
    let onResume: () -> Void

    @State private var search = ""
    @State private var showImporter = false
    @State private var importKind = ImportKind.games
    @State private var isImporting = false
    @State private var alert: AlertInfo?
    @State private var pendingDelete: GameEntry?
    @State private var hasBIOS = YmirHost.hasBIOS
    @State private var activePath: String? = YmirHost.loadedGamePath

    private var filteredGames: [GameEntry] {
        let query = search.trimmingCharacters(in: .whitespaces)
        if query.isEmpty { return library.games }
        return library.games.filter { $0.title.localizedCaseInsensitiveContains(query) }
    }

    private var activeGame: GameEntry? {
        guard let activePath else { return nil }
        return library.games.first { $0.url.path == activePath }
    }

    var body: some View {
        NavigationStack {
            List {
                if !hasBIOS {
                    Section {
                        BIOSBanner { startImport(.bios) }
                    }
                }

                if library.games.isEmpty {
                    Section {
                        EmptyLibraryView(
                            importGames: { startImport(.games) },
                            openFiles: { FilesApp.openDocuments() }
                        )
                    }
                } else {
                    Section {
                        ForEach(filteredGames) { game in
                            Button {
                                play(game)
                            } label: {
                                GameRow(game: game, isActive: game.url.path == activePath)
                            }
                            .buttonStyle(.plain)
                            .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                                Button(role: .destructive) {
                                    pendingDelete = game
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                            }
                            .contextMenu {
                                Button {
                                    play(game)
                                } label: {
                                    Label("Play", systemImage: "play.fill")
                                }
                                Button(role: .destructive) {
                                    pendingDelete = game
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                            }
                        }
                    }
                }
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Games")
            .searchable(text: $search, prompt: "Search games")
            .refreshable { refresh() }
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Menu {
                        Button {
                            startImport(.games)
                        } label: {
                            Label("Import Games…", systemImage: "square.and.arrow.down")
                        }
                        Button {
                            startImport(.bios)
                        } label: {
                            Label("Import BIOS…", systemImage: "cpu")
                        }
                        Button {
                            FilesApp.openDocuments()
                        } label: {
                            Label("Open Ymir Folder in Files", systemImage: "folder")
                        }
                    } label: {
                        Image(systemName: "plus")
                    }
                }
            }
            .safeAreaInset(edge: .bottom) {
                if let game = activeGame {
                    ResumeBar(game: game, onResume: onResume)
                        .padding(.horizontal, 16)
                        .padding(.bottom, 8)
                }
            }
            .overlay {
                if isImporting {
                    ProgressView("Importing…")
                        .padding(24)
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                }
            }
        }
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.item], allowsMultipleSelection: true) { result in
            guard case .success(let urls) = result, !urls.isEmpty else { return }
            runImport(urls)
        }
        .alert(item: $alert) { info in
            if let title = info.actionTitle, let action = info.action {
                return Alert(
                    title: Text(info.title),
                    message: Text(info.message),
                    primaryButton: .default(Text(title), action: action),
                    secondaryButton: .cancel()
                )
            }
            return Alert(title: Text(info.title), message: Text(info.message), dismissButton: .default(Text("OK")))
        }
        .confirmationDialog(
            "Delete this game?",
            isPresented: Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }),
            titleVisibility: .visible,
            presenting: pendingDelete
        ) { game in
            Button("Delete \(game.title)", role: .destructive) {
                library.delete(game)
                pendingDelete = nil
            }
            Button("Cancel", role: .cancel) {
                pendingDelete = nil
            }
        } message: { _ in
            Text("The disc image is removed from this device.")
        }
        .onAppear { refresh() }
    }

    private func refresh() {
        library.reload()
        hasBIOS = YmirHost.hasBIOS
        activePath = YmirHost.isDiscLoaded ? YmirHost.loadedGamePath : nil
    }

    private func play(_ game: GameEntry) {
        guard hasBIOS else {
            alert = AlertInfo(
                title: "BIOS Required",
                message: "Ymir needs a Sega Saturn BIOS (IPL ROM) to run games. Import one first.",
                actionTitle: "Import BIOS…",
                action: { startImport(.bios) }
            )
            return
        }
        onLaunch(game)
    }

    private func startImport(_ kind: ImportKind) {
        importKind = kind
        showImporter = true
    }

    private func runImport(_ urls: [URL]) {
        let kind = importKind
        isImporting = true
        Task {
            let folder = kind == .games ? AppPaths.games : AppPaths.bios
            let outcome = await Task.detached(priority: .userInitiated) { () -> ImportService.Outcome in
                ImportService.copy(urls, to: folder)
            }.value

            var biosError: String?
            if kind == .bios {
                biosError = YmirHost.reloadBIOS()
            }
            refresh()
            isImporting = false

            if outcome.failed > 0 {
                alert = AlertInfo(
                    title: "Import Incomplete",
                    message: "\(outcome.failed) file(s) could not be copied."
                )
            } else if kind == .bios && !hasBIOS {
                alert = AlertInfo(
                    title: "BIOS Not Recognized",
                    message: biosError ?? "That file doesn't look like a valid Sega Saturn BIOS. Try a different image."
                )
            }
        }
    }
}

// MARK: - Rows and banners

private struct GameRow: View {
    let game: GameEntry
    let isActive: Bool

    var body: some View {
        HStack(spacing: 16) {
            CoverArtView(game: game, size: 60)
            VStack(alignment: .leading, spacing: 4) {
                Text(game.title)
                    .font(.body)
                    .foregroundStyle(.primary)
                    .multilineTextAlignment(.leading)
                    .lineLimit(2)
                Text(game.subtitle)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            if isActive {
                Image(systemName: "play.circle.fill")
                    .font(.title3)
                    .foregroundStyle(.tint)
            }
        }
        .padding(.vertical, 6)
        .contentShape(Rectangle())
    }
}

private struct ResumeBar: View {
    let game: GameEntry
    let onResume: () -> Void

    var body: some View {
        Button(action: onResume) {
            HStack(spacing: 12) {
                CoverArtView(game: game, size: 40)
                VStack(alignment: .leading, spacing: 2) {
                    Text(game.title)
                        .font(.subheadline.weight(.semibold))
                        .lineLimit(1)
                    Text("Tap to resume")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 8)
                Image(systemName: "play.fill")
                    .font(.title3)
            }
            .foregroundStyle(.primary)
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .ymirGlass(in: Capsule())
    }
}

private struct BIOSBanner: View {
    let onImport: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("BIOS required", systemImage: "exclamationmark.triangle.fill")
                .font(.headline)
                .foregroundStyle(.orange)
            Text("Ymir needs a Sega Saturn BIOS (IPL ROM) image to run games. Import one to get started.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Button("Import BIOS…", action: onImport)
                .ymirProminentButton()
        }
        .padding(.vertical, 6)
    }
}

private struct EmptyLibraryView: View {
    let importGames: () -> Void
    let openFiles: () -> Void

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "opticaldisc")
                .font(.system(size: 44, weight: .light))
                .foregroundStyle(.secondary)
            Text("No Games Yet")
                .font(.headline)
            Text("Import disc images (CUE/BIN, CHD, ISO, CCD, MDS) or drop them into the Ymir folder in the Files app.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            HStack(spacing: 12) {
                Button("Import Games…", action: importGames)
                    .ymirProminentButton()
                Button("Open Files", action: openFiles)
                    .ymirSecondaryButton()
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 18)
    }
}

enum FilesApp {
    /// Opens the Files app at this app's Documents folder.
    static func openDocuments() {
        let path = AppPaths.documents.path
        if let url = URL(string: "shareddocuments://" + path) {
            UIApplication.shared.open(url)
        }
    }
}
