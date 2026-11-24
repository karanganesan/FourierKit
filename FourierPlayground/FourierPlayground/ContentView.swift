import FourierKit
import FourierKitUI
import SwiftUI

struct ContentView: View {
    @Environment(AppModel.self) private var app
    @SceneStorage("selection") private var storedSelection = SidebarItem.mode(.listen).rawValue
    @SceneStorage("showStats") private var storedStats = false
    @State private var restored = false
    @Environment(\.openSettings) private var openSettings

    var body: some View {
        @Bindable var app = app
        // three panes in one row on one surface. a split view would paint its own sidebar and
        // inspector materials and draw a divider between them, and the window is meant to read as
        // a single sheet of glass.
        HStack(spacing: 0) {
            if app.showSidebar {
                Sidebar()
                    .frame(width: 224)
                    .transition(.move(edge: .leading).combined(with: .opacity))
            }
            Detail()
                // with the sidebar (224) and the stats (320) this sets the window's minimum size
                .frame(minWidth: 440, maxWidth: .infinity, minHeight: 480, maxHeight: .infinity)
            if app.showStats {
                FourierStatsView(app.state)
                    .frame(width: 320)
                    .transition(.move(edge: .trailing).combined(with: .opacity))
            }
            if app.showCode {
                CodePane()
                    .frame(width: 400)
                    .transition(.move(edge: .trailing).combined(with: .opacity))
            }
        }
        .animation(.smooth(duration: 0.24), value: app.showSidebar)
        .animation(.smooth(duration: 0.24), value: app.showStats)
        .animation(.smooth(duration: 0.24), value: app.showCode)
        .toolbar {
            ToolbarItem(placement: .navigation) {
                Button("sidebar", systemImage: "sidebar.leading") { app.showSidebar.toggle() }
                    .help(app.showSidebar ? "hide the sidebar" : "show the sidebar")
            }
            PlaygroundToolbar()
        }
        // one glass surface for the whole window, in light and dark, with no fills on top of it
        .fourierGlassWindow()
        // no bar behind the toolbar: its buttons float on the window like everything else
        .toolbarBackgroundVisibility(.hidden, for: .windowToolbar)
        .navigationTitle(navigationTitle)
        .task { await app.state.observe() }
        .onChange(of: app.state.isReady, initial: true) { _, ready in
            if ready { app.markReady() }
        }
        .onAppear(perform: restore)
        .onChange(of: app.selection) { _, selection in storedSelection = selection.rawValue }
        .onChange(of: app.showStats) { _, show in storedStats = show }
        .onChange(of: app.isBusy, initial: true) { _, busy in
            #if DEBUG
            MainThreadWatchdog.shared.setMeasuring(busy, label: app.selection.rawValue)
            #endif
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            Task { await app.state.refreshMicrophone() }
        }
        .background(WindowSizer(size: app.launchOptions.windowSize))
    }

    private var navigationTitle: String {
        switch app.selection {
        case .mode(let mode): mode.title
        case .recent(let id): app.recents.item(id)?.title ?? "recent"
        }
    }

    /// scene restoration, unless a launch argument chose the mode
    private func restore() {
        guard !restored else { return }
        restored = true
        if app.launchOptions.settingsTab != nil { openSettings() }
        guard !app.launchOptions.isAutomated else { return }
        if let item = SidebarItem(rawValue: storedSelection) { app.selection = item }
        app.showStats = storedStats
    }
}

// MARK: sidebar

private struct Sidebar: View {
    @Environment(AppModel.self) private var app
    @Environment(\.undoManager) private var undoManager
    @FocusState private var focused: Bool

    /// every row in order, for the arrow keys
    private var order: [SidebarItem] {
        Mode.allCases.map(SidebarItem.mode) + app.recents.items.map { SidebarItem.recent($0.id) }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(Mode.allCases) { mode in
                    SidebarRow(item: .mode(mode)) {
                        Label {
                            Text(mode.title)
                        } icon: {
                            Image(systemName: mode.symbol).frame(width: 24)
                        }
                    }
                }

                Text("recent")
                    .fourierTextStyle(.caption)
                    .fourierInk(.tertiary)
                    .padding(.horizontal, 8)
                    .padding(.top, 24)
                    .padding(.bottom, 8)

                if app.recents.items.isEmpty {
                    Text("transcripts you make show up here")
                        .fourierTextStyle(.caption)
                        .fourierInk(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.horizontal, 8)
                } else {
                    ForEach(app.recents.items) { item in
                        SidebarRow(item: .recent(item.id)) {
                            VStack(alignment: .leading, spacing: 0) {
                                Text(item.title).lineLimit(1)
                                Text("\(item.mode.title) · \(item.date.formatted(.relative(presentation: .named)).lowercased())")
                                    .fourierTextStyle(.caption)
                                    .fourierInk(.tertiary)
                                    .lineLimit(1)
                            }
                            .padding(.vertical, 8)
                            .accessibilityElement(children: .combine)
                        }
                        .contextMenu {
                            Button("copy") { Pasteboard.copy(item.text) }
                            ShareLink("share…", item: item.text)
                            Divider()
                            Button("delete", role: .destructive) { delete(item.id) }
                        }
                    }
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .scrollEdgeEffectStyle(.soft, for: .top)
        .focusable()
        .focusEffectDisabled()
        .focused($focused)
        .onMoveCommand { direction in
            guard let index = order.firstIndex(of: app.selection) else { return }
            switch direction {
            case .up where index > 0: app.selection = order[index - 1]
            case .down where index < order.count - 1: app.selection = order[index + 1]
            default: break
            }
        }
        .onDeleteCommand {
            if case .recent(let id) = app.selection { delete(id) }
        }
        .accessibilityLabel("sidebar")
    }

    private func delete(_ id: UUID) {
        RecentActions.delete(id, app: app, undoManager: undoManager)
    }
}

/// one sidebar row. the selection is drawn here, as the same quiet tint of the text colour in every
/// state, because the system's own highlight turns into a solid accent block when the list has focus.
private struct SidebarRow<Content: View>: View {
    @Environment(AppModel.self) private var app
    let item: SidebarItem
    @ViewBuilder var content: Content
    @State private var hovering = false

    var body: some View {
        let selected = app.selection == item
        Button {
            app.selection = item
        } label: {
            content
                .fourierTextStyle(.body)
                .fourierInk(.primary)
                .padding(.horizontal, 8)
                .frame(maxWidth: .infinity, minHeight: 40, alignment: .leading)
                .contentShape(.rect)
        }
        .buttonStyle(.plain)
        .background {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color.primary.opacity(selected ? 0.12 : hovering ? 0.06 : 0))
        }
        .onHover { hovering = $0 }
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

enum RecentActions {
    @MainActor
    static func delete(_ id: UUID, app: AppModel, undoManager: UndoManager?) {
        guard let (item, index) = app.recents.delete(id) else { return }
        if app.selection == .recent(id) {
            let items = app.recents.items
            app.selection = items.isEmpty ? .mode(item.mode) : .recent(items[min(index, items.count - 1)].id)
        }
        let store = app.recents
        undoManager?.registerUndo(withTarget: store) { store in
            MainActor.assumeIsolated { store.restore(item, at: index) }
        }
        undoManager?.setActionName("delete transcript")
    }
}

// MARK: detail

private struct Detail: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        Group {
            if !app.hasBeenReady {
                // first run: one clear prepare screen until the model is ready
                FourierPrepareView(app.state, startsPrepare: false)
                    .padding(32)
            } else {
                switch app.selection {
                case .mode(.transcribe): TranscribeScreen()
                case .mode(.record): RecordScreen()
                case .mode(.listen): ListenScreen()
                case .recent(let id): RecentScreen(id: id)
                }
            }
        }
        .animation(.smooth, value: app.hasBeenReady)
    }
}

// MARK: toolbar

private struct PlaygroundToolbar: ToolbarContent {
    @Environment(AppModel.self) private var app
    @State private var exporting = false

    var body: some ToolbarContent {
        @Bindable var app = app
        let text = app.currentText
        ToolbarItemGroup(placement: .primaryAction) {
            if app.canToggleCurrent {
                Button(app.toggleTitle, systemImage: toggleSymbol) { app.toggleCurrent() }
                    .help(app.toggleTitle)
                    .contentTransition(.symbolEffect(.replace))
            }
            Button("copy", systemImage: "doc.on.doc") {
                if let text { Pasteboard.copy(text) }
            }
            .help("copy transcript")
            .disabled(text == nil)
            ShareLink(item: text ?? "") {
                Label("share", systemImage: "square.and.arrow.up")
            }
            .help("share transcript")
            .disabled(text == nil)
            Button("export", systemImage: "arrow.down.document") { exporting = true }
                .help("export as text")
                .disabled(text == nil)
                .fileExporter(isPresented: $exporting, document: TextFile(text: text ?? ""), contentType: .plainText,
                              defaultFilename: "transcript") { _ in }
            Button("code", systemImage: "chevron.left.forwardslash.chevron.right") { app.showCode.toggle() }
                .help(app.showCode ? "hide the code" : "show the code for this screen")
        }
        ToolbarSpacer(.fixed, placement: .primaryAction)
        ToolbarItem(placement: .primaryAction) {
            Button("stats", systemImage: "chart.xyaxis.line") { app.showStats.toggle() }
                .help(app.showStats ? "hide live stats" : "show live stats")
        }
    }

    private var toggleSymbol: String {
        switch app.selection {
        case .mode(.listen): app.listen.isRunning ? "stop.fill" : "mic.fill"
        case .mode(.record): app.record.isRunning ? "stop.fill" : "record.circle"
        case .mode(.transcribe): app.transcribe.isRunning ? "xmark" : "folder"
        case .recent: "play.fill"
        }
    }
}

// MARK: window size, for automation

/// sets the window's content size once, when `-fourierWindowSize` asks for it
private struct WindowSizer: NSViewRepresentable {
    let size: CGSize?

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        guard let size else { return view }
        DispatchQueue.main.async {
            guard let window = view.window else { return }
            window.setContentSize(size)
            window.center()
        }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}
