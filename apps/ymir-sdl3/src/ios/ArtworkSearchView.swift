import SwiftUI
import WebKit

/// A web browser showing Google Images results for a game. Long-pressing an image picks it as the game's artwork.
struct ArtworkSearchView: View {
    let game: GameEntry
    let onPick: (Data) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var candidate: Candidate?
    @State private var isLoading = false
    @State private var failed = false

    struct Candidate: Identifiable {
        let id = UUID()
        let source: String
    }

    private var searchURL: URL {
        var components = URLComponents(string: "https://www.google.com/search")!
        components.queryItems = [
            URLQueryItem(name: "tbm", value: "isch"),
            URLQueryItem(name: "q", value: "\(game.title) Sega Saturn cover"),
        ]
        return components.url!
    }

    var body: some View {
        NavigationStack {
            ZStack(alignment: .bottom) {
                ImageSearchWebView(url: searchURL) { source in
                    candidate = Candidate(source: source)
                }
                .ignoresSafeArea(edges: .bottom)

                Text("Open a result, then touch and hold the image to use it")
                    .font(.footnote)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .ymirGlass(in: Capsule(), interactive: false)
                    .padding(.bottom, 12)
                    .allowsHitTesting(false)

                if isLoading {
                    ProgressView("Downloading…")
                        .padding(24)
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                        .frame(maxHeight: .infinity)
                }
            }
            .alert("Couldn't Download the Image", isPresented: $failed) {
                Button("OK", role: .cancel) {}
            } message: {
                Text("Try a different image.")
            }
            .navigationTitle("Find Artwork")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") { dismiss() }
                }
            }
            .alert(item: $candidate) { item in
                Alert(
                    title: Text("Use This Image?"),
                    message: Text("It will become the artwork for \(game.title)."),
                    primaryButton: .default(Text("Use Image")) { download(item.source) },
                    secondaryButton: .cancel()
                )
            }
        }
    }

    private func download(_ source: String) {
        isLoading = true
        Task {
            let data = await ImageDownloader.data(from: source)
            isLoading = false
            if let data, UIImage(data: data) != nil {
                onPick(data)
                dismiss()
            } else {
                failed = true
            }
        }
    }
}

private enum ImageDownloader {
    static func data(from source: String) async -> Data? {
        if source.hasPrefix("data:") {
            guard let comma = source.firstIndex(of: ",") else { return nil }
            let payload = String(source[source.index(after: comma)...])
            return Data(base64Encoded: payload.removingPercentEncoding ?? payload)
        }
        guard let url = URL(string: source), url.scheme == "https" || url.scheme == "http" else { return nil }
        var request = URLRequest(url: url, timeoutInterval: 30)
        request.setValue("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
                         forHTTPHeaderField: "User-Agent")
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard (response as? HTTPURLResponse)?.statusCode ?? 200 < 400 else { return nil }
            return data
        } catch {
            return nil
        }
    }
}

/// Web view that reports the source of an image the user touches and holds.
private struct ImageSearchWebView: UIViewRepresentable {
    let url: URL
    let onImage: (String) -> Void

    private static let script = """
    (function () {
      var timer = null;
      var startX = 0, startY = 0;
      function imageAt(target) {
        var el = target;
        while (el && el.tagName !== 'IMG') { el = el.parentElement; }
        return el;
      }
      var style = document.createElement('style');
      style.textContent = 'img { -webkit-touch-callout: none; }';
      document.documentElement.appendChild(style);
      document.addEventListener('touchstart', function (e) {
        var img = imageAt(e.target);
        if (!img) { return; }
        startX = e.touches[0].clientX; startY = e.touches[0].clientY;
        timer = setTimeout(function () {
          var src = img.currentSrc || img.src;
          if (src) { window.webkit.messageHandlers.pickImage.postMessage(src); }
        }, 600);
      }, { passive: true });
      function cancel() { if (timer) { clearTimeout(timer); timer = null; } }
      document.addEventListener('touchend', cancel, { passive: true });
      document.addEventListener('touchcancel', cancel, { passive: true });
      document.addEventListener('touchmove', function (e) {
        if (Math.abs(e.touches[0].clientX - startX) > 10 || Math.abs(e.touches[0].clientY - startY) > 10) { cancel(); }
      }, { passive: true });
    })();
    """

    func makeCoordinator() -> Coordinator {
        Coordinator(onImage: onImage)
    }

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.userContentController.add(context.coordinator, name: "pickImage")
        configuration.userContentController.addUserScript(
            WKUserScript(source: Self.script, injectionTime: .atDocumentEnd, forMainFrameOnly: false)
        )
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.allowsBackForwardNavigationGestures = true
        webView.load(URLRequest(url: url))
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {}

    final class Coordinator: NSObject, WKScriptMessageHandler {
        let onImage: (String) -> Void

        init(onImage: @escaping (String) -> Void) {
            self.onImage = onImage
        }

        func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
            if let source = message.body as? String {
                DispatchQueue.main.async { self.onImage(source) }
            }
        }
    }
}
