import FourierKit
import SwiftUI

/// live text: settled words in primary, the draft after them in a softer step that turns primary as it settles.
///
/// text is lowercased for display only. each phrase is its own paragraph, so closed phrases never redraw and only
/// the phrase being spoken changes; the text grows downward from a fixed top edge, so nothing above it moves.
/// an empty view keeps one placeholder line, so the layout does not jump when words arrive.
public struct FourierLiveText: View {
    /// one paragraph: settled text and the draft after it
    public struct Paragraph: Sendable, Hashable, Identifiable {
        public var id: Int
        public var settled: String
        public var draft: String

        public init(id: Int, settled: String, draft: String = "") {
            self.id = id
            self.settled = settled
            self.draft = draft
        }
    }

    private let paragraphs: [Paragraph]
    private let placeholder: String
    private let style: FourierTextStyle

    /// - parameters:
    ///   - paragraphs: phrases in order; only the last one normally has a draft
    ///   - placeholder: shown while there is no text
    ///   - style: a step of the type scale, `.largeTitle` by default
    public init(paragraphs: [Paragraph], placeholder: String = "", style: FourierTextStyle = .largeTitle) {
        self.paragraphs = paragraphs.filter { !$0.settled.isEmpty || !$0.draft.isEmpty }
        self.placeholder = placeholder
        self.style = style
    }

    /// one paragraph of settled text and a draft
    public init(settled: String, draft: String = "", placeholder: String = "", style: FourierTextStyle = .largeTitle) {
        self.init(paragraphs: [Paragraph(id: 0, settled: settled, draft: draft)], placeholder: placeholder, style: style)
    }

    /// a live update, one paragraph per phrase
    public init(_ update: LiveUpdate?, placeholder: String = "", style: FourierTextStyle = .largeTitle) {
        self.init(paragraphs: Self.paragraphs(update), placeholder: placeholder, style: style)
    }

    /// a listen model's text: live while running, the exact transcript once finished
    public init(_ listen: FourierListenModel, placeholder: String = "", style: FourierTextStyle = .largeTitle) {
        let paragraphs: [Paragraph]
        if let transcript = listen.transcript {
            paragraphs = transcript.pieces.isEmpty
                ? [Paragraph(id: 0, settled: transcript.text)]
                : transcript.pieces.enumerated().map { Paragraph(id: $0.offset, settled: $0.element.text) }
        } else {
            paragraphs = Self.paragraphs(listen.update)
        }
        self.init(paragraphs: paragraphs, placeholder: placeholder, style: style)
    }

    private static func paragraphs(_ update: LiveUpdate?) -> [Paragraph] {
        guard let update else { return [] }
        if update.phrases.isEmpty { return [Paragraph(id: 0, settled: update.settled, draft: update.draft)] }
        return update.phrases.enumerated().map { Paragraph(id: $0.offset, settled: $0.element.settled, draft: $0.element.draft) }
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: style.lineHeight / 2) {
            if paragraphs.isEmpty {
                Text(placeholder.isEmpty ? " " : placeholder.lowercased())
                    .fourierInk(.quaternary)
            } else {
                ForEach(paragraphs) { paragraph in
                    LiveParagraph(paragraph: paragraph).equatable()
                }
            }
        }
        .fourierTextStyle(style)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text(accessibilityText))
        .accessibilityAddTraits(.updatesFrequently)
    }

    private var accessibilityText: String {
        let text = paragraphs.map { ($0.settled + $0.draft).trimmingCharacters(in: .whitespaces) }.joined(separator: " ")
        return (text.isEmpty ? placeholder : text).lowercased()
    }
}

/// one phrase. equatable, so a closed phrase is never re-rendered while the open one changes.
private struct LiveParagraph: View, Equatable {
    let paragraph: FourierLiveText.Paragraph
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.colorSchemeContrast) private var contrast

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool { lhs.paragraph == rhs.paragraph }

    var body: some View {
        let soft = Color.primary.opacity(FourierInk.draft.opacity(increasedContrast: contrast == .increased))
        let settled = Self.trimmedStart(paragraph.settled).lowercased()
        let draft = (settled.isEmpty ? Self.trimmedStart(paragraph.draft) : paragraph.draft).lowercased()
        Text("\(Text(settled).foregroundStyle(Color.primary))\(Text(draft).foregroundStyle(soft))")
            .multilineTextAlignment(.leading)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
            // a crossfade, not a glyph morph: two draws per change instead of one per frame
            .contentTransition(.opacity)
            .animation(reduceMotion ? nil : .smooth(duration: 0.16), value: paragraph)
    }

    private static func trimmedStart(_ text: String) -> String {
        String(text.drop { $0.isWhitespace })
    }
}
