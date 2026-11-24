#if os(iOS)
import FourierKit
import FourierKitUI
import SwiftUI

/// the phone layout: one screen at a time. the three modes and the recent list are tabs, stats and
/// code open as sheets from the navigation bar, settings is a sheet from its gear.
struct CompactRoot: View {
    enum Page: Hashable {
        case mode(Mode)
        case recent
    }

    @Environment(AppModel.self) private var app
    @State private var page = Page.mode(.listen)
    @State private var recentPath: [UUID] = []

    var body: some View {
        @Bindable var app = app
        TabView(selection: $page) {
            ForEach(Mode.allCases) { mode in
                Tab(mode.title, systemImage: mode.symbol, value: Page.mode(mode)) {
                    NavigationStack {
                        CompactScreen(item: .mode(mode))
                    }
                }
            }
            Tab("recent", systemImage: "clock", value: Page.recent) {
                NavigationStack(path: $recentPath) {
                    RecentList()
                        .navigationDestination(for: UUID.self) { id in
                            CompactScreen(item: .recent(id))
                        }
                }
            }
        }
        .sheet(isPresented: $app.showStats) {
            CompactSheet(title: "live stats") { app.showStats = false } content: {
                FourierStatsView(app.state)
            }
        }
        .sheet(isPresented: $app.showCode) {
            CompactSheet(title: "code") { app.showCode = false } content: {
                CodePane()
            }
        }
        .onAppear { if app.choseFirstScreen { follow() } }
        // launch arguments and scene restoration choose a screen once the recent list is read
        .onChange(of: app.choseFirstScreen) { follow() }
        .onChange(of: page) { _, page in
            switch page {
            case .mode(let mode): app.selection = .mode(mode)
            case .recent: if let id = recentPath.last { app.selection = .recent(id) }
            }
        }
        .onChange(of: recentPath) { _, path in
            if let id = path.last { app.selection = .recent(id) }
        }
    }

    /// shows whatever the app has selected. only at launch: after that the tabs lead and the
    /// selection follows them, so deleting a transcript never moves the user to another screen.
    private func follow() {
        if app.launchOptions.recentList {
            page = .recent
            return
        }
        switch app.selection {
        case .mode(let mode):
            page = .mode(mode)
        case .recent(let id):
            guard app.recents.item(id) != nil else { return }
            page = .recent
            recentPath = [id]
        }
    }
}

// MARK: one screen

/// a mode or a recent transcript, full screen, with its actions in the navigation bar
private struct CompactScreen: View {
    @Environment(AppModel.self) private var app
    let item: SidebarItem
    @State private var exporting = false
    @State private var exportText = ""

    var body: some View {
        Group {
            switch item {
            case .mode(let mode):
                if !app.hasBeenReady {
                    // first run: one clear prepare screen until the model is ready
                    FourierPrepareView(app.state, startsPrepare: false)
                        .padding(16)
                } else {
                    switch mode {
                    case .transcribe: TranscribeScreen()
                    case .record: RecordScreen()
                    case .listen: ListenScreen()
                    }
                }
            case .recent(let id):
                RecentScreen(id: id)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .animation(.smooth, value: app.hasBeenReady)
        .fourierGlassWindow()
        .navigationTitle(title)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            if case .mode = item {
                ToolbarItem(placement: .topBarLeading) {
                    Button("settings", systemImage: "gearshape") { app.showSettings = true }
                }
            }
            ToolbarItemGroup(placement: .topBarTrailing) {
                // the empty screen has its own choose button; this one is for the next file, or to cancel
                if item == .mode(.transcribe), app.hasBeenReady, app.transcribe.isRunning || app.transcribe.transcript != nil {
                    Button(app.toggleTitle, systemImage: app.transcribe.isRunning ? "xmark" : "folder") { app.toggleCurrent() }
                }
                Menu("transcript", systemImage: "square.and.arrow.up") {
                    Button("copy", systemImage: "doc.on.doc") {
                        if let text = app.currentText { Pasteboard.copy(text) }
                    }
                    Button("share…", systemImage: "square.and.arrow.up") {
                        if let text = app.currentText { SharePicker.show(text) }
                    }
                    Button("export as text…", systemImage: "arrow.down.document") {
                        exportText = app.currentText ?? ""
                        exporting = true
                    }
                }
                .disabled(!app.hasText)
                Button("code", systemImage: "chevron.left.forwardslash.chevron.right") { app.showCode = true }
                Button("live stats", systemImage: "chart.xyaxis.line") { app.showStats = true }
            }
        }
        .fileExporter(isPresented: $exporting, document: TextFile(text: exportText), contentType: .plainText,
                      defaultFilename: "transcript") { _ in }
    }

    private var title: String {
        switch item {
        case .mode(let mode): mode.title
        case .recent(let id): app.recents.item(id)?.title ?? "recent"
        }
    }
}

// MARK: recent

/// recent transcripts, newest first. swipe to delete, or select several and delete them together.
private struct RecentList: View {
    @Environment(AppModel.self) private var app
    @Environment(\.undoManager) private var undoManager
    @State private var selecting = false
    @State private var picked: Set<UUID> = []
    @State private var confirming = false

    var body: some View {
        let items = app.recents.items
        Group {
            if items.isEmpty {
                ContentUnavailableView {
                    Label {
                        Text("no transcripts yet").fourierTextStyle(.title)
                    } icon: {
                        Image(systemName: "clock")
                    }
                } description: {
                    Text("transcripts you make show up here.")
                        .fourierTextStyle(.body)
                        .fourierInk(.secondary)
                }
            } else {
                List(selection: $picked) {
                    ForEach(items) { item in
                        NavigationLink(value: item.id) {
                            RecentRow(item: item)
                        }
                        .listRowBackground(Color.clear)
                        .swipeActions {
                            Button("delete", systemImage: "trash", role: .destructive) { delete(item.id) }
                        }
                        .contextMenu {
                            Button("copy", systemImage: "doc.on.doc") { Pasteboard.copy(item.text) }
                            ShareLink("share…", item: item.text)
                            Divider()
                            Button("delete", systemImage: "trash", role: .destructive) { delete(item.id) }
                        }
                    }
                }
                .listStyle(.plain)
                .scrollContentBackground(.hidden)
                .environment(\.editMode, .constant(selecting ? .active : .inactive))
                .fourierScrollEdges()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .fourierGlassWindow()
        .navigationTitle("recent")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarLeading) {
                if selecting {
                    Button(picked.count == items.count ? "select none" : "select all") {
                        picked = picked.count == items.count ? [] : Set(items.map(\.id))
                    }
                } else {
                    Button("settings", systemImage: "gearshape") { app.showSettings = true }
                }
            }
            if selecting {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("delete", systemImage: "trash", role: .destructive) { confirming = true }
                        .disabled(picked.isEmpty)
                }
                ToolbarSpacer(.fixed, placement: .topBarTrailing)
            }
            if !items.isEmpty || selecting {
                ToolbarItem(placement: .topBarTrailing) {
                    Button(selecting ? "done" : "select") {
                        selecting.toggle()
                        picked = []
                    }
                }
            }
        }
        .onChange(of: items.isEmpty) { _, empty in
            if empty { selecting = false }
        }
        .alert(picked.count == 1 ? "delete 1 transcript?" : "delete \(picked.count) transcripts?", isPresented: $confirming) {
            Button("delete", role: .destructive, action: deletePicked)
            Button("cancel", role: .cancel) {}
        } message: {
            Text("they are removed from \(DeviceText.this). you can undo until you quit.")
        }
    }

    private func delete(_ id: UUID) {
        RecentActions.delete(id, app: app, undoManager: undoManager)
    }

    private func deletePicked() {
        let ids = app.recents.items.map(\.id).filter(picked.contains)
        guard !ids.isEmpty else { return }
        undoManager?.beginUndoGrouping()
        for id in ids.reversed() { delete(id) }
        undoManager?.endUndoGrouping()
        undoManager?.setActionName("delete \(ids.count) transcripts")
        picked = []
        selecting = false
    }
}

private struct RecentRow: View {
    let item: RecentTranscript

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(item.title)
                .fourierTextStyle(.body)
                .fourierInk(.primary)
                .lineLimit(1)
            Text("\(item.mode.title) · \(item.date.formatted(.relative(presentation: .named)).lowercased())")
                .fourierTextStyle(.caption)
                .fourierInk(.tertiary)
                .lineLimit(1)
        }
        .padding(.vertical, 8)
        .accessibilityElement(children: .combine)
    }
}

// MARK: sheets

/// stats or code in a sheet: half height to glance at, full height to read
private struct CompactSheet<Content: View>: View {
    let title: String
    let close: () -> Void
    @ViewBuilder let content: Content

    var body: some View {
        NavigationStack {
            content
                .navigationTitle(title)
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button(role: .close, action: close)
                    }
                }
        }
        .presentationDetents([.medium, .large])
        .presentationBackgroundInteraction(.enabled(upThrough: .medium))
    }
}
#endif
