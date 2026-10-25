import SwiftUI

/// Toolbar shown over the game while the on-screen controls are being arranged. Controls are dragged directly in the
/// game view; a tap selects one, and the sliders here resize it.
struct TouchEditorBar: View {
    @ObservedObject private var store = SettingStore.shared
    @State private var selected = -1

    private let timer = Timer.publish(every: 0.12, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(spacing: 10) {
            HStack(spacing: 10) {
                Button("Done") {
                    YmirShell.endTouchEditing()
                }
                .ymirProminentButton()

                Button("Reset") {
                    store.bool("touch.reset").wrappedValue = true
                }
                .ymirSecondaryButton()

                Toggle(isOn: store.bool("touch.snap")) {
                    Label("Grid", systemImage: "square.grid.3x3")
                }
                .toggleStyle(.button)

                Spacer(minLength: 0)

                Button {
                    YmirShell.moveTouchEditorBar()
                } label: {
                    Image(systemName: "arrow.up.arrow.down")
                }
                .ymirSecondaryButton()
            }

            sizeRow(
                title: selected >= 0 ? "Selected" : "Tap a control",
                value: store.float("touch.selectedScale", default: 1),
                range: 0.4...2.5
            )
            .disabled(selected < 0)
            .opacity(selected < 0 ? 0.5 : 1)

            sizeRow(title: "All sizes", value: store.float("touch.globalScale", default: 1), range: 0.5...1.6)
            sizeRow(title: "Opacity", value: store.float("touch.opacity", default: 0.9), range: 0.1...1)
        }
        .padding(14)
        .ymirGlass(in: RoundedRectangle(cornerRadius: 26, style: .continuous), interactive: false)
        .onReceive(timer) { _ in
            // The selection changes inside the game view, which SwiftUI doesn't observe
            let current = YmirHost.intValue(key: "touch.selected", default: -1)
            if current != selected {
                selected = current
                store.objectWillChange.send()
            }
        }
    }

    private func sizeRow(title: String, value: Binding<Double>, range: ClosedRange<Double>) -> some View {
        HStack(spacing: 12) {
            Text(title)
                .font(.footnote)
                .frame(width: 92, alignment: .leading)
            Slider(value: value, in: range)
            Text(String(format: "%.0f%%", value.wrappedValue * 100))
                .font(.footnote.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 44, alignment: .trailing)
        }
    }
}
