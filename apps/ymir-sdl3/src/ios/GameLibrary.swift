import SwiftUI
import UIKit

// MARK: - Model

struct GameEntry: Identifiable, Hashable {
    let url: URL
    let sizeBytes: Int64
    /// Name chosen by the user, if any.
    var customName: String?
    /// Modification time of the cover art file; changes whenever the artwork changes.
    var coverStamp: TimeInterval = 0

    var id: String { url.path }
    /// The file name without extension.
    var stem: String { url.deletingPathExtension().lastPathComponent }
    /// The name shown in the app.
    var title: String { customName ?? stem }
    var format: String { url.pathExtension.uppercased() }

    var subtitle: String {
        let size = ByteCountFormatter.string(fromByteCount: sizeBytes, countStyle: .file)
        return "\(format) · \(size)"
    }

    /// Path relative to the Documents folder; stable across app reinstalls, unlike the absolute path.
    var relativePath: String {
        let root = AppPaths.documents.path
        let path = url.path
        if path.hasPrefix(root) {
            return String(path.dropFirst(root.count)).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        }
        return path
    }

    /// All files that make up this disc image (the descriptor file plus its data tracks).
    func relatedFiles() -> [URL] {
        let fm = FileManager.default
        let dir = url.deletingLastPathComponent()
        var files = [url]

        switch url.pathExtension.lowercased() {
        case "cue":
            let text = (try? String(contentsOf: url, encoding: .utf8))
                ?? (try? String(contentsOf: url, encoding: .isoLatin1))
                ?? ""
            for line in text.split(whereSeparator: \.isNewline) {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard trimmed.uppercased().hasPrefix("FILE") else { continue }
                let parts = trimmed.split(separator: "\"", omittingEmptySubsequences: false)
                guard parts.count >= 3 else { continue }
                let file = dir.appendingPathComponent(String(parts[1]))
                if fm.fileExists(atPath: file.path) && !files.contains(file) {
                    files.append(file)
                }
            }
        case "ccd":
            for ext in ["img", "sub"] {
                let file = dir.appendingPathComponent(stem + "." + ext)
                if fm.fileExists(atPath: file.path) { files.append(file) }
            }
        case "mds":
            let file = dir.appendingPathComponent(stem + ".mdf")
            if fm.fileExists(atPath: file.path) { files.append(file) }
        default:
            break
        }
        return files
    }

    /// Cover art placed next to the game or in Documents/covers, named like the game. Optional.
    func coverURL() -> URL? {
        let fm = FileManager.default
        // Artwork chosen in the app lives in covers; images next to the game are used as a fallback
        let folders = [AppPaths.covers, url.deletingLastPathComponent()]
        for folder in folders {
            for ext in ["png", "jpg", "jpeg"] {
                let candidate = folder.appendingPathComponent(stem + "." + ext)
                if fm.fileExists(atPath: candidate.path) { return candidate }
            }
        }
        return nil
    }
}

// MARK: - Library

/// How the games are ordered in the library.
enum GameSortMode: String, CaseIterable, Identifiable {
    case recent
    case titleAscending
    case titleDescending
    case manual

    var id: String { rawValue }

    var label: String {
        switch self {
        case .recent: return "Recently Played"
        case .titleAscending: return "Title (A to Z)"
        case .titleDescending: return "Title (Z to A)"
        case .manual: return "Manual"
        }
    }

    var systemImage: String {
        switch self {
        case .recent: return "clock"
        case .titleAscending: return "arrow.down"
        case .titleDescending: return "arrow.up"
        case .manual: return "hand.draw"
        }
    }
}

@MainActor
final class GameLibrary: ObservableObject {
    @Published private(set) var games: [GameEntry] = []

    @Published var sortMode: GameSortMode = {
        UserDefaults.standard.string(forKey: "ymir.sortMode").flatMap(GameSortMode.init(rawValue:)) ?? .recent
    }() {
        didSet {
            UserDefaults.standard.set(sortMode.rawValue, forKey: Self.sortModeKey)
            if sortMode == .manual && oldValue != .manual && Self.manualOrder().isEmpty {
                // Manual ordering starts from what the player was looking at
                Self.saveManualOrder(games.map(\.relativePath))
            }
            applySort()
        }
    }

    private var scanned: [GameEntry] = []

    private static let discExtensions: Set<String> = ["cue", "chd", "iso", "ccd", "mds"]
    // Folders of the emulator profile that never contain games
    private static let ignoredFolders: Set<String> = [
        "roms", "backup", "savestates", "state", "screenshots", "dumps", "covers", "exports",
    ]
    private static let lastPlayedKey = "ymir.lastPlayed"
    private static let customNamesKey = "ymir.customNames"
    private static let sortModeKey = "ymir.sortMode"
    private static let manualOrderKey = "ymir.manualOrder"

    init() {
        reload()
    }

    func reload() {
        let fm = FileManager.default
        try? fm.createDirectory(at: AppPaths.games, withIntermediateDirectories: true)

        let names = Self.customNames()
        var found: [GameEntry] = []

        let keys: [URLResourceKey] = [.isDirectoryKey, .fileSizeKey]
        if let enumerator = fm.enumerator(at: AppPaths.documents, includingPropertiesForKeys: keys,
                                          options: [.skipsHiddenFiles]) {
            for case let url as URL in enumerator {
                let values = try? url.resourceValues(forKeys: Set(keys))
                if values?.isDirectory == true {
                    if Self.ignoredFolders.contains(url.lastPathComponent.lowercased()) {
                        enumerator.skipDescendants()
                    }
                    continue
                }
                guard Self.discExtensions.contains(url.pathExtension.lowercased()) else { continue }

                var entry = GameEntry(url: url, sizeBytes: 0)
                let total = entry.relatedFiles().reduce(Int64(0)) { sum, file in
                    let size = (try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
                    return sum + Int64(size)
                }
                let stamp = entry.coverURL()
                    .flatMap { try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate }?
                    .timeIntervalSince1970 ?? 0
                entry = GameEntry(url: url, sizeBytes: total, customName: names[entry.relativePath], coverStamp: stamp)
                found.append(entry)
            }
        }

        scanned = found
        applySort()
    }

    /// Orders the scanned games according to the sort mode.
    private func applySort() {
        let alphabetical: (GameEntry, GameEntry) -> Bool = {
            $0.title.localizedStandardCompare($1.title) == .orderedAscending
        }
        switch sortMode {
        case .recent:
            let lastPlayed = Self.lastPlayed()
            games = scanned.sorted { a, b in
                let da = lastPlayed[a.relativePath] ?? 0
                let db = lastPlayed[b.relativePath] ?? 0
                if da != db { return da > db }
                return alphabetical(a, b)
            }
        case .titleAscending:
            games = scanned.sorted(by: alphabetical)
        case .titleDescending:
            games = scanned.sorted { alphabetical($1, $0) }
        case .manual:
            let order = Self.manualOrder()
            let position = Dictionary(order.enumerated().map { ($1, $0) }, uniquingKeysWith: { first, _ in first })
            // Games that were never placed (new ones) go at the end, alphabetically
            games = scanned.sorted { a, b in
                switch (position[a.relativePath], position[b.relativePath]) {
                case let (pa?, pb?): return pa < pb
                case (_?, nil): return true
                case (nil, _?): return false
                case (nil, nil): return alphabetical(a, b)
                }
            }
        }
    }

    /// Moves games in the manual order, like List's onMove. Only meaningful in manual mode.
    func moveGames(from source: IndexSet, to destination: Int) {
        var reordered = games
        reordered.move(fromOffsets: source, toOffset: destination)
        games = reordered
        Self.saveManualOrder(reordered.map(\.relativePath))
    }

    private static func manualOrder() -> [String] {
        UserDefaults.standard.stringArray(forKey: manualOrderKey) ?? []
    }

    private static func saveManualOrder(_ order: [String]) {
        UserDefaults.standard.set(order, forKey: manualOrderKey)
    }

    func delete(_ game: GameEntry) {
        for file in game.relatedFiles() {
            try? FileManager.default.removeItem(at: file)
        }
        var played = Self.lastPlayed()
        played.removeValue(forKey: game.relativePath)
        UserDefaults.standard.set(played, forKey: Self.lastPlayedKey)
        removeArtworkFiles(for: game)
        rename(game, to: nil, reloadAfter: false)
        reload()
    }

    // MARK: Names

    /// Changes the name the game has in the app. An empty name restores the file name.
    func rename(_ game: GameEntry, to name: String?, reloadAfter: Bool = true) {
        var names = Self.customNames()
        let trimmed = name?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if trimmed.isEmpty || trimmed == game.stem {
            names.removeValue(forKey: game.relativePath)
        } else {
            names[game.relativePath] = trimmed
        }
        UserDefaults.standard.set(names, forKey: Self.customNamesKey)
        if reloadAfter { reload() }
    }

    // MARK: Artwork

    func hasCustomArtwork(_ game: GameEntry) -> Bool {
        ["png", "jpg", "jpeg"].contains {
            FileManager.default.fileExists(atPath: AppPaths.covers.appendingPathComponent(game.stem + "." + $0).path)
        }
    }

    /// Saves an image as the game's artwork. Returns false if the data isn't an image.
    @discardableResult
    func setArtwork(_ data: Data, for game: GameEntry) -> Bool {
        guard let image = UIImage(data: data) else { return false }

        // Covers are shown small; keep the files light
        let maxSide: CGFloat = 800
        let scale = min(1, maxSide / max(image.size.width, image.size.height))
        let size = CGSize(width: image.size.width * scale, height: image.size.height * scale)
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let resized = UIGraphicsImageRenderer(size: size, format: format).image { _ in
            image.draw(in: CGRect(origin: .zero, size: size))
        }
        guard let jpeg = resized.jpegData(compressionQuality: 0.88) else { return false }

        removeArtworkFiles(for: game)
        do {
            try FileManager.default.createDirectory(at: AppPaths.covers, withIntermediateDirectories: true)
            try jpeg.write(to: AppPaths.covers.appendingPathComponent(game.stem + ".jpg"), options: .atomic)
        } catch {
            return false
        }
        reload()
        return true
    }

    func removeArtwork(for game: GameEntry) {
        removeArtworkFiles(for: game)
        reload()
    }

    private func removeArtworkFiles(for game: GameEntry) {
        for ext in ["png", "jpg", "jpeg"] {
            try? FileManager.default.removeItem(at: AppPaths.covers.appendingPathComponent(game.stem + "." + ext))
        }
    }

    private static func customNames() -> [String: String] {
        (UserDefaults.standard.dictionary(forKey: customNamesKey) as? [String: String]) ?? [:]
    }

    static func markPlayed(_ game: GameEntry) {
        var played = lastPlayed()
        played[game.relativePath] = Date().timeIntervalSince1970
        UserDefaults.standard.set(played, forKey: lastPlayedKey)
    }

    private static func lastPlayed() -> [String: Double] {
        (UserDefaults.standard.dictionary(forKey: lastPlayedKey) as? [String: Double]) ?? [:]
    }
}

// MARK: - Importing files

enum ImportService {
    struct Outcome {
        var copied = 0
        var failed = 0
    }

    /// Copies files picked with the document picker into a folder inside Documents.
    /// Meant to be called off the main thread, since disc images can be large.
    static func copy(_ urls: [URL], to folder: URL) -> Outcome {
        let fm = FileManager.default
        try? fm.createDirectory(at: folder, withIntermediateDirectories: true)

        var outcome = Outcome()
        for url in urls {
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }

            let destination = folder.appendingPathComponent(url.lastPathComponent)
            do {
                if fm.fileExists(atPath: destination.path) {
                    try fm.removeItem(at: destination)
                }
                try fm.copyItem(at: url, to: destination)
                outcome.copied += 1
            } catch {
                outcome.failed += 1
            }
        }
        return outcome
    }
}

// MARK: - Cover art

private final class CoverCache: @unchecked Sendable {
    static let shared = CoverCache()
    private let cache = NSCache<NSString, UIImage>()

    func image(for game: GameEntry, pixelSize: CGFloat) async -> UIImage? {
        let key = "\(game.id)|\(game.coverStamp)@\(Int(pixelSize))" as NSString
        if let cached = cache.object(forKey: key) {
            return cached
        }
        let url = game.coverURL()
        let image: UIImage? = await Task.detached(priority: .utility) { () -> UIImage? in
            guard let url, let full = UIImage(contentsOfFile: url.path) else { return nil }
            return full.preparingThumbnail(of: CGSize(width: pixelSize, height: pixelSize)) ?? full
        }.value
        if let image {
            cache.setObject(image, forKey: key)
        }
        return image
    }
}

struct CoverArtView: View {
    let game: GameEntry
    var size: CGFloat = 60

    @State private var image: UIImage?
    @Environment(\.displayScale) private var displayScale

    var body: some View {
        ZStack {
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
            } else {
                placeholder
            }
        }
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: size * 0.22, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: size * 0.22, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.1), lineWidth: 0.5)
        )
        .task(id: "\(game.id)|\(game.coverStamp)") {
            image = await CoverCache.shared.image(for: game, pixelSize: size * displayScale)
        }
    }

    /// A stable colour per title, so that games are easy to tell apart without cover art.
    private var hue: Double {
        var hash: UInt32 = 5381
        for byte in game.title.utf8 {
            hash = (hash &* 33) &+ UInt32(byte)
        }
        return Double(hash % 360) / 360.0
    }

    private var placeholder: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(hue: hue, saturation: 0.55, brightness: 0.88),
                    Color(hue: (hue + 0.08).truncatingRemainder(dividingBy: 1), saturation: 0.75, brightness: 0.55),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            Image(systemName: "opticaldisc.fill")
                .font(.system(size: size * 0.46, weight: .semibold))
                .foregroundStyle(.white.opacity(0.92))
        }
    }
}
