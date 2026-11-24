import CoreText
import Foundation
import SwiftUI
import Synchronization

// MARK: type scale

/// one step of the type scale: size, weight and line height on an 8 pt grid.
public struct FourierTextStyle: Sendable, Hashable {
    /// point size
    public var size: CGFloat
    /// the weight axis, 100 (thin) to 900 (black)
    public var weight: CGFloat
    /// distance between baselines, in points
    public var lineHeight: CGFloat
    /// letter spacing in points; negative tightens display sizes
    public var tracking: CGFloat
    /// the roundness axis, 0 to 100. nil keeps the font's default.
    public var roundness: CGFloat?

    public init(size: CGFloat, weight: CGFloat = 400, lineHeight: CGFloat? = nil, tracking: CGFloat = 0, roundness: CGFloat? = nil) {
        self.size = size
        self.weight = weight
        self.lineHeight = lineHeight ?? (size * 1.5 / 8).rounded(.up) * 8
        self.tracking = tracking
        self.roundness = roundness
    }

    // three sizes, and nothing in between: 12, 16 and 32. a heading is the body size in a
    // heavier weight, not a size of its own. every line height sits on the 8 pt grid.

    /// 12 on 16: section headers, meta lines, hints, status
    public static let caption = FourierTextStyle(size: 12, weight: 500, lineHeight: 16)
    /// 16 on 24: every row, label, value, button and paragraph
    public static let body = FourierTextStyle(size: 16, weight: 400, lineHeight: 24)
    /// 16 on 24, semibold: the one heading of a screen or a panel
    public static let title = FourierTextStyle(size: 16, weight: 600, lineHeight: 24)
    /// 32 on 40: transcript text and the big clock
    public static let largeTitle = FourierTextStyle(size: 32, weight: 450, lineHeight: 40, tracking: -0.4)

    public func weight(_ weight: CGFloat) -> FourierTextStyle {
        var copy = self
        copy.weight = weight
        return copy
    }
}

/// how numbers are set.
public enum FourierNumbers: Sendable, Hashable {
    case proportional
    /// equal-width figures, for numbers that change in place
    case tabular
}

// MARK: font choice

/// the typeface the kit's views draw with. set it with `.fourierFont(_:)`.
public enum FourierFont: Sendable {
    /// allo variable, bundled with the kit and registered on first use. falls back to the system font if it cannot load.
    case allo
    /// the system font at the same sizes and weights
    case system
    /// any font for each step of the scale
    case custom(@Sendable (FourierTextStyle, FourierNumbers) -> Font)

    /// the font for a step of the scale
    public func font(_ style: FourierTextStyle, numbers: FourierNumbers = .proportional) -> Font {
        switch self {
        case .allo:
            if let font = AlloFont.font(style, numbers: numbers) { return font }
            return Self.systemFont(style, numbers: numbers)
        case .system:
            return Self.systemFont(style, numbers: numbers)
        case .custom(let make):
            return make(style, numbers)
        }
    }

    /// true once allo variable is registered and usable in this process
    public static var isAlloAvailable: Bool { AlloFont.isAvailable }

    /// why allo variable could not load, nil when it loaded
    public static var alloLoadError: String? { AlloFont.loadError }

    static func systemFont(_ style: FourierTextStyle, numbers: FourierNumbers) -> Font {
        let font = Font.system(size: style.size, weight: Font.Weight(axis: style.weight))
        return numbers == .tabular ? font.monospacedDigit() : font
    }
}

extension Font.Weight {
    init(axis: CGFloat) {
        switch axis {
        case ..<150: self = .ultraLight
        case ..<250: self = .thin
        case ..<350: self = .light
        case ..<450: self = .regular
        case ..<550: self = .medium
        case ..<650: self = .semibold
        case ..<750: self = .bold
        case ..<850: self = .heavy
        default: self = .black
        }
    }
}

extension EnvironmentValues {
    /// the typeface for every kit view below this point
    @Entry public var fourierFont: FourierFont = .allo
}

extension View {
    /// replaces allo variable for every kit view below this point, e.g. `.fourierFont(.system)`.
    public func fourierFont(_ font: FourierFont) -> some View {
        environment(\.fourierFont, font)
    }

    /// sets text below this point in a step of the type scale, with its line height and tracking.
    public func fourierTextStyle(_ style: FourierTextStyle, numbers: FourierNumbers = .proportional) -> some View {
        modifier(FourierTextStyleModifier(style: style, numbers: numbers))
    }
}

private struct FourierTextStyleModifier: ViewModifier {
    let style: FourierTextStyle
    let numbers: FourierNumbers
    @Environment(\.fourierFont) private var fourierFont
    @Environment(\.legibilityWeight) private var legibilityWeight

    func body(content: Content) -> some View {
        var style = style
        if legibilityWeight == .bold { style.weight = min(900, style.weight + 200) }
        let natural = AlloFont.naturalLineHeight(style)
        return content
            .font(fourierFont.font(style, numbers: numbers))
            .tracking(style.tracking)
            .lineSpacing(max(0, style.lineHeight - natural))
    }
}

// MARK: allo variable

/// registers the bundled allo variable for this process once and builds fonts on its axes.
enum AlloFont {
    static let family = "Allo Variable"
    private static let weightAxis = 0x7767_6874 as NSNumber      // 'wght'
    private static let opticalSizeAxis = 0x6F70_737A as NSNumber // 'opsz'
    private static let roundnessAxis = 0x524F_4E44 as NSNumber   // 'ROND'

    private struct Key: Hashable {
        var style: FourierTextStyle
        var numbers: FourierNumbers
    }

    private struct Cache: @unchecked Sendable {
        var fonts: [Key: Font] = [:]
        var lineHeights: [FourierTextStyle: CGFloat] = [:]
    }

    private static let cache = Mutex(Cache())

    /// registration runs once, on first use, from whichever thread asks first
    private static let registration: String? = {
        guard let url = Bundle.module.url(forResource: "AlloVariable", withExtension: "woff2", subdirectory: "Fonts")
            ?? Bundle.module.url(forResource: "AlloVariable", withExtension: "woff2")
        else { return "AlloVariable.woff2 is missing from the FourierKitUI bundle" }
        var error: Unmanaged<CFError>?
        if !CTFontManagerRegisterFontsForURL(url as CFURL, .process, &error) {
            let error = error?.takeRetainedValue()
            // already registered, e.g. by another copy of the kit, is fine
            if let error, CFErrorGetCode(error) != CTFontManagerError.alreadyRegistered.rawValue {
                return "core text could not register the font: \(CFErrorCopyDescription(error) as String)"
            }
        }
        let probe = make(.body, numbers: .proportional)
        let family = CTFontCopyFamilyName(probe) as String
        guard family == Self.family else { return "core text returned \(family) instead of \(Self.family)" }
        return nil
    }()

    static var isAvailable: Bool { registration == nil }
    static var loadError: String? { registration }

    static func font(_ style: FourierTextStyle, numbers: FourierNumbers) -> Font? {
        guard isAvailable else { return nil }
        let key = Key(style: style, numbers: numbers)
        if let font = cache.withLock({ $0.fonts[key] }) { return font }
        let font = Font(make(style, numbers: numbers))
        cache.withLock { $0.fonts[key] = font }
        return font
    }

    /// ascent, descent and leading of the font at this step, so line spacing can land on the line height
    static func naturalLineHeight(_ style: FourierTextStyle) -> CGFloat {
        if let height = cache.withLock({ $0.lineHeights[style] }) { return height }
        let font: CTFont = isAvailable ? make(style, numbers: .proportional) : CTFontCreateUIFontForLanguage(.system, style.size, nil)!
        let height = CTFontGetAscent(font) + CTFontGetDescent(font) + CTFontGetLeading(font)
        cache.withLock { $0.lineHeights[style] = height }
        return height
    }

    private static func make(_ style: FourierTextStyle, numbers: FourierNumbers) -> CTFont {
        var variation: [NSNumber: NSNumber] = [
            weightAxis: NSNumber(value: Double(min(900, max(100, style.weight)))),
            // optical size follows the point size within the font's 9...32 range
            opticalSizeAxis: NSNumber(value: Double(min(32, max(9, style.size)))),
        ]
        if let roundness = style.roundness {
            variation[roundnessAxis] = NSNumber(value: Double(min(100, max(0, roundness))))
        }
        var attributes: [CFString: Any] = [
            kCTFontFamilyNameAttribute: family,
            kCTFontVariationAttribute: variation,
        ]
        if numbers == .tabular {
            attributes[kCTFontFeatureSettingsAttribute] = [[
                kCTFontOpenTypeFeatureTag: "tnum",
                kCTFontOpenTypeFeatureValue: 1,
            ] as [CFString: Any]]
        }
        let descriptor = CTFontDescriptorCreateWithAttributes(attributes as CFDictionary)
        return CTFontCreateWithFontDescriptor(descriptor, style.size, nil)
    }
}
