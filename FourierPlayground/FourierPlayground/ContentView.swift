import FourierKit
import FourierKitUI
import SwiftUI

struct ContentView: View {
    @Environment(AppModel.self) private var app
    @SceneStorage("selection") private var storedSelection = SidebarItem.mode(.listen).rawValue
    @SceneStorage("showStats") private var storedStats = false
    @State private var restored = false
    #if os(macOS)
    @Environment(\.openSettings) private var openSettings
    #else
    @Environment(\.horizontalSizeClass) private var sizeClass
    @Environment(\.scenePhase) private var scenePhase
    #endif

    #if os(macOS)
    var body: some View {
        Panes()
            .background { TextPresence() }
            .toolbar {
                ToolbarItem(placement: .navigation) {
                    ToolIcon("sidebar", symbol: "sidebar.leading", help: "show or hide the sidebar", width: 40) { app.showSidebar.toggle() }
                }
                PlaygroundToolbar()
            }
        // one glass surface for the whole window, in light and dark, with no fills on top of it
        .fourierGlassWindow()
        // no bar behind the toolbar: its buttons float on the window like everything else
        .toolbarBackgroundVisibility(.hidden, for: .windowToolbar)
        // in full screen macos moves the toolbar into a strip of its own and paints a bar behind
        // it that no setting removes. so the toolbar steps aside there and comes down on hover,
        // and the content runs to the top of the screen.
        .windowToolbarFullScreenVisibility(.onHover)
        .navigationTitle(navigationTitle)
        .task { await app.state.observe() }
        .onChange(of: app.state.isReady, initial: true) { _, ready in
            if ready { app.markReady() }
        }
        .onAppear(perform: restore)
        .onChange(of: app.selection) { _, selection in storedSelection = selection.rawValue }
        .onChange(of: app.isBusy, initial: true) { _, busy in
            #if DEBUG
            MainThreadWatchdog.shared.setMeasuring(busy, label: app.selection.rawValue)
            #endif
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            Task { await app.state.refreshMicrophone() }
        }
        .background(WindowSizer(size: app.launchOptions.windowSize) { app.stopCapture() })
    }
    #else
    /// iphone, and ipad in slide over or a narrow split, get one screen at a time under a tab bar.
    /// a full or wide ipad window gets the mac's row of panes. the size class decides, not the device.
    var body: some View {
        @Bindable var app = app
        Group {
            if sizeClass == .compact {
                CompactRoot()
            } else {
                RegularRoot(title: navigationTitle)
            }
        }
        .background { TextPresence() }
        .task { await app.state.observe() }
        .onChange(of: app.state.isReady, initial: true) { _, ready in
            if ready { app.markReady() }
        }
        .onAppear(perform: restore)
        .onChange(of: app.selection) { _, selection in storedSelection = selection.rawValue }
        .onChange(of: app.isBusy, initial: true) { _, busy in
            #if DEBUG
            MainThreadWatchdog.shared.setMeasuring(busy, label: app.selection.rawValue)
            #endif
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active { Task { await app.state.refreshMicrophone() } }
            // there is no background audio, so capture ends here and what was heard is transcribed
            if phase == .background { app.stopCapture() }
        }
        .sheet(isPresented: $app.showSettings) {
            SettingsView()
                .environment(app)
        }
    }
    #endif

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
        #if os(macOS)
        if app.launchOptions.settingsTab != nil { openSettings() }
        #else
        if app.launchOptions.settingsTab != nil { app.showSettings = true }
        #endif
        guard !app.launchOptions.isAutomated else { return }
        if let item = SidebarItem(rawValue: storedSelection) { app.selection = item }
        #if os(macOS)
        app.showStats = storedStats
        #else
        // on a phone stats is a sheet, and a sheet that opens by itself at launch is in the way
        if sizeClass != .compact { app.showStats = storedStats }
        #endif
    }
}

// MARK: panes

/// the sidebar, the content and the trailing pane in one row on one surface. a split view would
/// paint its own sidebar and inspector materials and draw a divider between them, and the window
/// is meant to read as a single sheet of glass.
///
/// this is a view of its own so that opening or closing a pane re-evaluates only this row. the
/// toolbar lives on the parent and is never rebuilt by a pane.
private struct Panes: View {
    @Environment(AppModel.self) private var app
    @SceneStorage("showStats") private var storedStats = false
    #if os(iOS)
    @State private var width: CGFloat = .infinity
    #endif

    /// the mac window cannot shrink below the three panes. an ipad screen can be narrower than
    /// them, so there the sidebar steps aside while a trailing pane needs the room.
    private var showsSidebar: Bool {
        #if os(macOS)
        app.showSidebar
        #else
        let trailing: CGFloat = app.showCode ? 400 : app.showStats ? 320 : 0
        return app.showSidebar && (trailing == 0 || width >= 224 + 440 + trailing)
        #endif
    }

    #if os(macOS)
    private let detailMinWidth: CGFloat = 440
    private let detailMinHeight: CGFloat = 480
    #else
    private let detailMinWidth: CGFloat = 0
    private let detailMinHeight: CGFloat = 0
    #endif

    var body: some View {
        HStack(spacing: 0) {
            if showsSidebar {
                Sidebar()
                    .frame(width: 224)
                    .transition(.opacity)
            }
            Detail()
                // with the sidebar (224) and the widest trailing pane (400) this sets the window's minimum size
                .frame(minWidth: detailMinWidth, maxWidth: .infinity, minHeight: detailMinHeight, maxHeight: .infinity)
            if app.showStats {
                FourierStatsView(app.state)
                    .frame(width: 320)
                    .transition(.opacity)
            }
            if app.showCode {
                CodePane()
                    .frame(width: 400)
                    .transition(.opacity)
            }
        }
        .animation(.smooth(duration: 0.16), value: app.showSidebar)
        .animation(.smooth(duration: 0.16), value: app.showStats)
        .animation(.smooth(duration: 0.16), value: app.showCode)
        .onChange(of: app.showStats) { _, show in storedStats = show }
        #if os(iOS)
        .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { width = $0 }
        #endif
    }
}

/// one toolbar button, drawn here so its states are ours: a faint tint of the text colour on
/// hover and on press, and nothing darker. the system's own toolbar button goes black when pressed.
struct ToolIcon: View {
    let title: String
    let symbol: String
    let help: String
    let action: () -> Void
    /// 32 inside a group. a button alone in its own glass needs 40, or the glass squeezes into an oval.
    var width: CGFloat = 32
    @Environment(\.isEnabled) private var enabled
    @State private var hovering = false

    init(_ title: String, symbol: String, help: String, width: CGFloat = 32, action: @escaping () -> Void) {
        self.title = title
        self.symbol = symbol
        self.help = help
        self.width = width
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: 16, weight: .medium))
                .frame(width: width, height: 32)
                .contentShape(.rect)
        }
        .buttonStyle(ToolIconStyle(hovering: hovering && enabled))
        .onHover { hovering = $0 }
        .help(help)
        .accessibilityLabel(title)
    }
}

private struct ToolIconStyle: ButtonStyle {
    let hovering: Bool
    @Environment(\.isEnabled) private var enabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(Color.primary.opacity(enabled ? 1 : 0.4))
            .background {
                Capsule().fill(Color.primary.opacity(configuration.isPressed ? 0.16 : hovering ? 0.08 : 0))
            }
    }
}

// MARK: sidebar

private struct Sidebar: View {
    @Environment(AppModel.self) private var app
    @Environment(\.undoManager) private var undoManager
    @FocusState private var focused: Bool
    @State private var pendingDelete: [UUID] = []
    @State private var showingDeleted = false

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

                if app.recents.saveFailed {
                    Text("transcripts could not be saved. they stay here until the next try.")
                        .fourierTextStyle(.caption)
                        .foregroundStyle(Color.red)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.horizontal, 8)
                        .padding(.bottom, 8)
                }

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

                if !app.recents.deleted.isEmpty {
                    Button { showingDeleted = true } label: {
                        Text("recently deleted · \(app.recents.deleted.count)")
                            .fourierTextStyle(.caption)
                            .fourierInk(.tertiary)
                            .padding(.horizontal, 8)
                            .frame(maxWidth: .infinity, minHeight: 40, alignment: .leading)
                            .contentShape(.rect)
                    }
                    .buttonStyle(.plain)
                    .padding(.top, 16)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .fourierScrollEdges()
        .sheet(isPresented: $showingDeleted) { RecentlyDeletedSheet() }
        #if os(macOS)
        .focusable()
        .focusEffectDisabled()
        .focused($focused)
        .onMoveCommand { direction in
            guard let index = order.firstIndex(of: app.selection) else { return }
            switch direction {
            case .up where index > 0: app.pick(order[index - 1], command: false, shift: false)
            case .down where index < order.count - 1: app.pick(order[index + 1], command: false, shift: false)
            default: break
            }
        }
        .onDeleteCommand(perform: deleteSelection)
        #endif
        .alert("delete \(pendingDelete.count) transcripts?", isPresented: Binding(get: { !pendingDelete.isEmpty }, set: { if !$0 { pendingDelete = [] } })) {
            Button("delete", role: .destructive, action: deletePending)
            Button("cancel", role: .cancel) { pendingDelete = [] }
        } message: {
            Text("they move to recently deleted on \(DeviceText.this) and stay there for 30 days.")
        }
        #if os(macOS)
        // control a selects every recent, as command a does from the edit menu
        .onKeyPress(keys: ["a"]) { press in
            guard press.modifiers.contains(.control) || press.modifiers.contains(.command) else { return .ignored }
            app.selectAllRecents()
            return .handled
        }
        #endif
        .accessibilityLabel("sidebar")
    }

    private func delete(_ id: UUID) {
        RecentActions.delete(id, app: app, undoManager: undoManager)
    }

    /// the delete key: the one selected transcript goes at once and can be undone, or found again in
    /// recently deleted. several picked together ask first. a held key deletes once, never a run.
    private func deleteSelection() {
        #if os(macOS)
        if let event = NSApp.currentEvent, event.type == .keyDown, event.isARepeat { return }
        #endif
        let picked = app.recents.items.filter { app.selectedRecents.contains($0.id) }.map(\.id)
        if picked.count > 1 {
            pendingDelete = picked
        } else if case .recent(let id) = app.selection {
            delete(id)
        }
    }

    private func deletePending() {
        let picked = pendingDelete
        pendingDelete = []
        guard !picked.isEmpty else { return }
        undoManager?.beginUndoGrouping()
        for id in picked.reversed() { delete(id) }
        undoManager?.endUndoGrouping()
        undoManager?.setActionName("delete \(picked.count) transcripts")
        app.selectedRecents = []
    }
}

/// transcripts deleted in the last 30 days, each with a restore
struct RecentlyDeletedSheet: View {
    @Environment(AppModel.self) private var app
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Group {
                if app.recents.deleted.isEmpty {
                    ContentUnavailableView {
                        Label {
                            Text("nothing deleted").fourierTextStyle(.title)
                        } icon: {
                            Image(systemName: "trash")
                        }
                    } description: {
                        Text("deleted transcripts are kept here for 30 days.")
                            .fourierTextStyle(.body)
                            .fourierInk(.secondary)
                    }
                } else {
                    List(app.recents.deleted) { entry in
                        HStack(spacing: 16) {
                            VStack(alignment: .leading, spacing: 0) {
                                Text(entry.item.title)
                                    .fourierTextStyle(.body)
                                    .fourierInk(.primary)
                                    .lineLimit(1)
                                Text("\(entry.item.mode.title) · deleted \(entry.deletedAt.formatted(.relative(presentation: .named)).lowercased())")
                                    .fourierTextStyle(.caption)
                                    .fourierInk(.tertiary)
                                    .lineLimit(1)
                            }
                            Spacer(minLength: 0)
                            Button("restore") { app.recents.restoreDeleted(entry.id) }
                                .fourierGlassButton()
                                .fourierTextStyle(.body)
                        }
                        .padding(.vertical, 8)
                        .listRowBackground(Color.clear)
                    }
                    .listStyle(.plain)
                    .scrollContentBackground(.hidden)
                }
            }
            .navigationTitle("recently deleted")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("done") { dismiss() }
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 440, minHeight: 360)
        #endif
    }
}

/// one sidebar row. the selection is drawn here, as the same quiet tint of the text colour in every
/// state, because the system's own highlight turns into a solid accent block when the list has focus.
private struct SidebarRow<Content: View>: View {
    @Environment(AppModel.self) private var app
    let item: SidebarItem
    @ViewBuilder var content: Content
    @State private var hovering = false

    private var isPicked: Bool {
        if case .recent(let id) = item { return app.selectedRecents.contains(id) }
        return false
    }

    var body: some View {
        let selected = app.selection == item || isPicked
        Button {
            #if os(macOS)
            let flags = NSEvent.modifierFlags
            app.pick(item, command: flags.contains(.command), shift: flags.contains(.shift))
            #else
            // a tap picks one. several at once is the recent list's select mode on a phone.
            app.pick(item, command: false, shift: false)
            #endif
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

#if os(iOS)
/// a wide ipad window: the mac's row of panes under a navigation bar, settings as a sheet
private struct RegularRoot: View {
    @Environment(AppModel.self) private var app
    let title: String

    var body: some View {
        NavigationStack {
            Panes()
                .toolbar {
                    ToolbarItem(placement: .topBarLeading) {
                        ToolIcon("sidebar", symbol: "sidebar.leading", help: "show or hide the sidebar", width: 40) { app.showSidebar.toggle() }
                    }
                    ToolbarItem(placement: .topBarLeading) {
                        ToolIcon("settings", symbol: "gearshape", help: "settings", width: 40) { app.showSettings = true }
                    }
                    PlaygroundToolbar()
                }
                .navigationTitle(title)
                .navigationBarTitleDisplayMode(.inline)
                .fourierGlassWindow()
        }
    }
}
#endif

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
        .safeAreaInset(edge: .bottom, spacing: 0) { RunningBar() }
        .overlay(alignment: .top) { DisplayOnlyNote() }
    }
}

/// stays on screen for a display-only launch (`-fourierSkipPrepare`), so a screenshot of it cannot pass for a working app
struct DisplayOnlyNote: View {
    var body: some View {
        if LaunchOptions.current.skipPrepare {
            Text("display only: no model is loaded")
                .fourierTextStyle(.caption)
                .fourierInk(.tertiary)
                .padding(.top, 8)
                .allowsHitTesting(false)
        }
    }
}

/// while something is capturing, a stop control on every screen except its own, which has one
/// already. so listening or recording never runs on out of sight.
struct RunningBar: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        let shown = [Mode.listen, .record].filter { running($0) && app.selection != .mode($0) }
        if !shown.isEmpty {
            VStack(spacing: 8) {
                ForEach(shown) { mode in bar(mode) }
            }
            .frame(maxWidth: .infinity)
            .padding(.bottom, 16)
            .transition(.opacity)
        }
    }

    private func running(_ mode: Mode) -> Bool {
        switch mode {
        case .listen: app.listen.isRunning
        case .record: app.record.isRunning
        case .transcribe: false
        }
    }

    private func bar(_ mode: Mode) -> some View {
        let listening = mode == .listen
        let status = listening ? FourierMicStatus(app.listen.phase) : app.record.micStatus
        let canStop = status == .starting || status == .running
        let word = status == .starting ? "starting" : status == .finishing ? (listening ? "finishing" : "transcribing") : (listening ? "listening" : "recording")
        let stop = { if listening { app.listen.stop() } else { app.record.toggle() } }
        return FourierControlBar(action: canStop ? stop : nil) {
            if status == .running {
                FourierLevelMeter(level: listening ? app.listen.level : app.record.level, isActive: true, barCount: 16)
                    .frame(width: 120, height: 24)
                    .padding(.leading, 8)
            }
        } center: {
            FourierMicButton(status: status, diameter: 48, action: stop) { _ in
                Image(systemName: "stop.fill")
                    .font(.system(size: 24, weight: .semibold))
            }
            .plain()
            .disabled(!canStop)
            .accessibilityLabel(listening ? "stop listening" : "stop recording")
        } trailing: {
            Text(word)
                .fourierTextStyle(.caption)
                .fourierInk(.tertiary)
        }
        .animation(.smooth, value: status)
    }
}

// MARK: toolbar

/// watches for text appearing or going away, in a view of its own, so that is the only thing
/// re-evaluated while live text changes many times a second.
private struct TextPresence: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        Color.clear
            .onChange(of: app.currentText != nil, initial: true) { _, present in
                if app.hasText != present { app.hasText = present }
            }
    }
}

private struct PlaygroundToolbar: ToolbarContent {
    @Environment(AppModel.self) private var app
    @State private var exporting = false
    @State private var exportText = ""

    var body: some ToolbarContent {
        // the toolbar reads only values that change when the user does something. the text itself
        // is fetched when a button is pressed, never while drawing.
        let hasText = app.hasText
        ToolbarItem(placement: .primaryAction) {
            HStack(spacing: 0) {
                if app.canToggleCurrent {
                    ToolIcon(app.toggleTitle, symbol: toggleSymbol, help: app.toggleTitle) { app.toggleCurrent() }
                }
                ToolIcon("copy", symbol: "doc.on.doc", help: "copy transcript") {
                    if let text = app.currentText { Pasteboard.copy(text) }
                }
                .disabled(!hasText)
                ToolIcon("share", symbol: "square.and.arrow.up", help: "share transcript") {
                    if let text = app.currentText { SharePicker.show(text) }
                }
                .disabled(!hasText)
                ToolIcon("export", symbol: "arrow.down.document", help: "export as text") {
                    exportText = app.currentText ?? ""
                    exporting = true
                }
                .disabled(!hasText)
                ToolIcon("code", symbol: "chevron.left.forwardslash.chevron.right", help: "show or hide the code for this screen") {
                    app.showCode.toggle()
                }
            }
            .padding(.horizontal, 4)
            .fileExporter(isPresented: $exporting, document: TextFile(text: exportText), contentType: .plainText,
                          defaultFilename: "transcript") { _ in }
        }
        ToolbarSpacer(.fixed, placement: .primaryAction)
        ToolbarItem(placement: .primaryAction) {
            ToolIcon("stats", symbol: "chart.xyaxis.line", help: "show or hide live stats", width: 40) { app.showStats.toggle() }
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

// MARK: the window itself

#if os(macOS)

/// reaches the window for what swiftui has no modifier for: a title bar with no background of its
/// own, in a window and in full screen, and the size and full screen that automation asks for.
private struct WindowSizer: NSViewRepresentable {
    let size: CGSize?
    /// called when the window closes: the microphone must not stay on behind a window nobody can see
    let onClose: () -> Void

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async {
            guard let window = view.window else { return }
            // the toolbar floats on the window's own surface. without this, full screen paints a bar behind it.
            window.titlebarAppearsTransparent = true
            window.titlebarSeparatorStyle = .none
            if let size {
                window.setContentSize(size)
                window.center()
            }
            if LaunchOptions.current.fullScreen { window.toggleFullScreen(nil) }
            NotificationCenter.default.addObserver(forName: NSWindow.willCloseNotification, object: window, queue: .main) { _ in
                MainActor.assumeIsolated { onClose() }
            }
        }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}
#endif
