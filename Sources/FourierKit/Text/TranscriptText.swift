/// turns the decoded model output into the final transcript, the way the transformers processor
/// does for `return_format="transcription_only"` (`_parse_single_output` with its repetition fix).
///
/// python strings index by code point, so everything here works on unicode scalars, and
/// whitespace, line breaks and case follow python's `str` rules.
enum TranscriptText {
    /// splits `language <name><asr_text>text`, after collapsing runaway repetitions.
    static func parse(_ decoded: String) -> (language: String?, transcription: String) {
        var text = Array(decoded.unicodeScalars)
        text = strip(text)
        if text.isEmpty { return (nil, "") }

        let assistant = Array("assistant\n".unicodeScalars)
        if let at = find(assistant, in: text) {
            text = Array(text[(at + assistant.count)...])
        }

        text = fixRepetitions(text)

        let marker = Array("<asr_text>".unicodeScalars)
        guard let at = find(marker, in: text) else {
            return (nil, string(strip(text)))
        }
        let prefix = strip(Array(text[..<at]))
        let transcription = string(strip(Array(text[(at + marker.count)...])))

        if asciiLowercasedEquals(prefix, "language none") {
            return (nil, transcription)
        }

        var language: [Unicode.Scalar]? = nil
        for rawLine in splitLines(prefix) {
            let line = strip(rawLine)
            if line.isEmpty { continue }
            if line.count >= 9, asciiLowercasedEquals(Array(line[0..<9]), "language ") {
                let value = strip(Array(line[9...]))
                if !value.isEmpty { language = value }
            } else {
                language = line
            }
            break
        }
        let name = language.map(string)
        return (name?.isEmpty == false ? name : nil, transcription)
    }

    // MARK: python string semantics

    /// `str.isspace`.
    static func isPythonSpace(_ s: Unicode.Scalar) -> Bool {
        switch s.value {
        case 0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000:
            return true
        default:
            return false
        }
    }

    /// `str.strip()`.
    static func strip(_ s: [Unicode.Scalar]) -> [Unicode.Scalar] {
        var lo = 0
        var hi = s.count
        while lo < hi && isPythonSpace(s[lo]) { lo += 1 }
        while hi > lo && isPythonSpace(s[hi - 1]) { hi -= 1 }
        return Array(s[lo..<hi])
    }

    /// `str.splitlines()`.
    static func splitLines(_ s: [Unicode.Scalar]) -> [[Unicode.Scalar]] {
        var lines: [[Unicode.Scalar]] = []
        var start = 0
        var i = 0
        while i < s.count {
            switch s[i].value {
            case 0x0A, 0x0B, 0x0C, 0x1C, 0x1D, 0x1E, 0x85, 0x2028, 0x2029:
                lines.append(Array(s[start..<i]))
                i += 1
                start = i
            case 0x0D:
                lines.append(Array(s[start..<i]))
                i += (i + 1 < s.count && s[i + 1] == "\n") ? 2 : 1
                start = i
            default:
                i += 1
            }
        }
        if start < s.count { lines.append(Array(s[start...])) }
        return lines
    }

    /// `s.lower() == target` for an ascii lowercase target. only ascii capitals lower onto ascii
    /// letters that appear in the targets used here, so folding ascii alone is exact.
    private static func asciiLowercasedEquals(_ s: [Unicode.Scalar], _ target: String) -> Bool {
        let t = Array(target.unicodeScalars)
        guard s.count == t.count else { return false }
        for (a, b) in zip(s, t) {
            let v = (a.value >= 0x41 && a.value <= 0x5A) ? a.value + 0x20 : a.value
            if v != b.value { return false }
        }
        return true
    }

    private static func find(_ needle: [Unicode.Scalar], in hay: [Unicode.Scalar]) -> Int? {
        guard !needle.isEmpty, hay.count >= needle.count else { return needle.isEmpty ? 0 : nil }
        outer: for i in 0...(hay.count - needle.count) {
            for k in 0..<needle.count where hay[i + k] != needle[k] { continue outer }
            return i
        }
        return nil
    }

    private static func string(_ s: [Unicode.Scalar]) -> String {
        String(String.UnicodeScalarView(s))
    }

    // MARK: repetition fix

    static let repetitionThreshold = 20
    static let maxPatternLength = 20

    /// `_detect_and_fix_repetitions`: a char repeated more than 20 times becomes one, then the
    /// first pattern (up to 20 chars) that repeats at least 20 times in a row is kept once.
    static func fixRepetitions(_ text: [Unicode.Scalar]) -> [Unicode.Scalar] {
        fixPatternRepeats(fixCharRepeats(text))
    }

    static func fixCharRepeats(_ s: [Unicode.Scalar]) -> [Unicode.Scalar] {
        var out: [Unicode.Scalar] = []
        out.reserveCapacity(s.count)
        var i = 0
        while i < s.count {
            var count = 1
            while i + count < s.count && s[i + count] == s[i] { count += 1 }
            if count > repetitionThreshold {
                out.append(s[i])
            } else {
                out.append(contentsOf: s[i..<(i + count)])
            }
            i += count
        }
        return out
    }

    /// python recurses on the tail after each collapsed pattern; this loops instead.
    static func fixPatternRepeats(_ input: [Unicode.Scalar]) -> [Unicode.Scalar] {
        let thresh = repetitionThreshold
        var out: [Unicode.Scalar] = []
        out.reserveCapacity(input.count)
        var s = input[...]
        while true {
            let n = s.count
            if n < thresh * 2 {
                out.append(contentsOf: s)
                return out
            }
            let base = s.startIndex
            var i = 0
            var collapsedTo: Int? = nil
            while i <= n - thresh * 2 {
                for k in 1...maxPatternLength {
                    if i + k * thresh > n { break }
                    var valid = true
                    reps: for rep in 1..<thresh {
                        let at = i + rep * k
                        for m in 0..<k where s[base + at + m] != s[base + i + m] {
                            valid = false
                            break reps
                        }
                    }
                    if valid {
                        var end = i + thresh * k
                        while end + k <= n {
                            var same = true
                            for m in 0..<k where s[base + end + m] != s[base + i + m] {
                                same = false
                                break
                            }
                            if !same { break }
                            end += k
                        }
                        out.append(contentsOf: s[(base + i)..<(base + i + k)])
                        collapsedTo = end
                        break
                    }
                }
                if collapsedTo != nil { break }
                out.append(s[base + i])
                i += 1
            }
            guard let end = collapsedTo else {
                out.append(contentsOf: s[(base + i)...])
                return out
            }
            s = s[(base + end)...]
        }
    }
}
