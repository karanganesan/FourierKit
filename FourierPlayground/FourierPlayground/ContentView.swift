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
        }
        .animation(.smooth(duration: 0.24), value: app.showSidebar)
        .animation(.smooth(duration: 0.24), value: app.showStats)
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
    @Environment(\.controlActiveState) private var activeState
    @FocusState private var focused: Bool

    @Environment(\.colorScheme) private var scheme
    /// the colour that reads on top of the accent: the accent is white in dark mode and black in light mode
    private var onAccent: Color { scheme == .dark ? .black : .white }

    /// the focused selection is an accent pill, so its label takes the opposite colour, like a selected tab
    private func inverted(_ item: SidebarItem) -> Bool {
        // the selection is a quiet tint of the text colour in every state, so labels never invert
        false
    }

    var body: some View {
        @Bindable var app = app
        List(selection: $app.selection) {
            Section {
                ForEach(Mode.allCases) { mode in
                    let ink = inverted(.mode(mode)) ? onAccent : Color.primary
                    Label {
                        Text(mode.title).fourierTextStyle(.body).foregroundStyle(ink)
                    } icon: {
                        Image(systemName: mode.symbol).foregroundStyle(ink)
                    }
                    .tag(SidebarItem.mode(mode))
                }
            }

            Section {
                if app.recents.items.isEmpty {
                    Text("transcripts you make show up here")
                        .fourierTextStyle(.caption)
                        .fourierInk(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                        .selectionDisabled()
                } else {
                    ForEach(app.recents.items) { item in
                        RecentRow(item: item, inverted: inverted(.recent(item.id)), onAccent: onAccent)
                            .tag(SidebarItem.recent(item.id))
                            .contextMenu {
                                Button("copy") { Pasteboard.copy(item.text) }
                                ShareLink("share…", item: item.text)
                                Divider()
                                Button("delete", role: .destructive) { delete(item.id) }
                            }
                    }
                }
            } header: {
                Text("recent").fourierTextStyle(.caption)
            }
        }
        .listStyle(.sidebar)
        .tint(Color.primary.opacity(0.16))
        .scrollContentBackground(.hidden)
        .scrollEdgeEffectStyle(.soft, for: .top)
        .focused($focused)
        .onDeleteCommand {
            if case .recent(let id) = app.selection { delete(id) }
        }
    }

    private func delete(_ id: UUID) {
        RecentActions.delete(id, app: app, undoManager: undoManager)
    }
}

private struct RecentRow: View {
    let item: RecentTranscript
    let inverted: Bool
    let onAccent: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(item.title)
                .fourierTextStyle(.body)
                .foregroundStyle(inverted ? onAccent : Color.primary)
                .lineLimit(1)
            Text("\(item.mode.title) · \(item.date.formatted(.relative(presentation: .named)).lowercased())")
                .fourierTextStyle(.caption)
                .foregroundStyle(inverted ? onAccent.opacity(0.64) : Color.primary.opacity(0.40))
                .lineLimit(1)
        }
        .padding(.vertical, 4)
        .accessibilityElement(children: .combine)
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
