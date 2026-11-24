import Foundation

/// the qwen2 byte-level bpe tokenizer, read from the checkpoint's `tokenizer.json`.
///
/// encoding follows the file: added tokens are cut out of the raw text first (leftmost, longest),
/// every other piece is nfc-normalised, split by the pre-tokenizer pattern, mapped to byte-level
/// symbols and merged by rank. decoding maps tokens back to bytes and repairs invalid utf-8 with
/// U+FFFD, like the fast tokenizer's lossy decode.
struct Tokenizer: Sendable {
    enum LoadError: Error, CustomStringConvertible {
        case unsupported(String)
        case malformed(String)

        var description: String {
            switch self {
            case .unsupported(let what): "unsupported tokenizer.json: \(what)"
            case .malformed(let what): "malformed tokenizer.json: \(what)"
            }
        }
    }

    private struct AddedToken: Sendable {
        var id: Int32
        var scalars: [Unicode.Scalar]
        var special: Bool
    }

    /// token string -> id, for the bpe vocabulary and the added tokens.
    private let tokenToID: [String: Int32]
    /// decode bytes of each id, flattened; ids without a token have an empty range.
    private let idBytes: [UInt8]
    private let idByteOffsets: [Int32]
    /// true for ids that skipSpecialTokens drops.
    private let idIsSpecial: [Bool]
    /// true for ids that exist (in the vocabulary or as added tokens).
    private let idExists: [Bool]
    /// the bpe symbol id of each raw byte.
    private let byteSymbol: [Int32]
    /// (left id << 32 | right id) -> (rank << 32 | merged id).
    private let merges: [UInt64: UInt64]
    /// added tokens grouped by their first scalar, longest first.
    private let addedByFirstScalar: [Unicode.Scalar: [AddedToken]]

    /// one past the largest token id, the size python's `len(tokenizer)` reports.
    let vocabularySize: Int

    init(contentsOf tokenizerJSON: URL) throws {
        let root = try JSONValue.parse(contentsOf: tokenizerJSON)
        guard let model = root["model"] else { throw LoadError.malformed("no model") }
        guard model["type"]?.string == "BPE" else { throw LoadError.unsupported("model type is not BPE") }
        if model["ignore_merges"]?.bool == true { throw LoadError.unsupported("ignore_merges") }
        if let normalizer = root["normalizer"], !normalizer.isNull, normalizer["type"]?.string != "NFC" {
            throw LoadError.unsupported("normalizer other than NFC")
        }
        try Self.checkPreTokenizer(root["pre_tokenizer"])
        if let decoder = root["decoder"], !decoder.isNull, decoder["type"]?.string != "ByteLevel" {
            throw LoadError.unsupported("decoder other than ByteLevel")
        }

        let unicodeToByte = ByteLevel.unicodeToByte

        guard let vocabMembers = model["vocab"]?.members else { throw LoadError.malformed("no vocab") }
        var tokenToID: [String: Int32] = [:]
        tokenToID.reserveCapacity(vocabMembers.count + 256)
        var maxID = -1
        for (token, value) in vocabMembers {
            guard let n = value.number, let id = Int32(exactly: n), id >= 0 else {
                throw LoadError.malformed("vocab id for \(token)")
            }
            tokenToID[token] = id
            maxID = max(maxID, Int(id))
        }

        var added: [AddedToken] = []
        for entry in root["added_tokens"]?.array ?? [] {
            guard let content = entry["content"]?.string, let n = entry["id"]?.number,
                  let id = Int32(exactly: n) else { throw LoadError.malformed("added token") }
            if entry["lstrip"]?.bool == true || entry["rstrip"]?.bool == true
                || entry["single_word"]?.bool == true || entry["normalized"]?.bool == true {
                throw LoadError.unsupported("added token options on \(content)")
            }
            added.append(AddedToken(id: id, scalars: Array(content.unicodeScalars),
                                    special: entry["special"]?.bool ?? false))
            maxID = max(maxID, Int(id))
        }

        let size = maxID + 1
        var vocabToken = [String?](repeating: nil, count: size)
        for (token, id) in tokenToID { vocabToken[Int(id)] = token }

        var special = [Bool](repeating: false, count: size)
        var exists = [Bool](repeating: false, count: size)
        var addedContent = [String?](repeating: nil, count: size)
        for token in added {
            special[Int(token.id)] = token.special
            addedContent[Int(token.id)] = String(String.UnicodeScalarView(token.scalars))
        }

        // decode bytes per id. added tokens win over the vocabulary, like the fast tokenizer.
        // the byte-level decoder maps each char back to its byte, or keeps the token's own utf-8
        // when some char has no byte.
        var bytes: [UInt8] = []
        bytes.reserveCapacity(size * 7)
        var offsets = [Int32](repeating: 0, count: size + 1)
        for id in 0..<size {
            offsets[id] = Int32(bytes.count)
            guard let text = addedContent[id] ?? vocabToken[id] else { continue }
            exists[id] = true
            var mapped: [UInt8] = []
            mapped.reserveCapacity(text.unicodeScalars.count)
            var ok = true
            for scalar in text.unicodeScalars {
                if scalar.value < 0x200, case let b = unicodeToByte[Int(scalar.value)], b >= 0 {
                    mapped.append(UInt8(b))
                } else {
                    ok = false
                    break
                }
            }
            if ok { bytes.append(contentsOf: mapped) } else { bytes.append(contentsOf: text.utf8) }
        }
        offsets[size] = Int32(bytes.count)

        var byteSymbol = [Int32](repeating: -1, count: 256)
        for b in 0..<256 {
            let symbol = String(ByteLevel.byteToUnicode[b])
            guard let id = tokenToID[symbol] else { throw LoadError.malformed("no symbol for byte \(b)") }
            byteSymbol[b] = id
        }

        guard let mergeList = model["merges"]?.array else { throw LoadError.malformed("no merges") }
        var merges: [UInt64: UInt64] = [:]
        merges.reserveCapacity(mergeList.count)
        for (rank, entry) in mergeList.enumerated() {
            let left: String
            let right: String
            switch entry {
            case .array(let pair) where pair.count == 2:
                guard let l = pair[0].string, let r = pair[1].string else { throw LoadError.malformed("merge \(rank)") }
                left = l
                right = r
            case .string(let s):
                let parts = s.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: false)
                guard parts.count == 2 else { throw LoadError.malformed("merge \(rank)") }
                left = String(parts[0])
                right = String(parts[1])
            default:
                throw LoadError.malformed("merge \(rank)")
            }
            guard let l = tokenToID[left], let r = tokenToID[right], let m = tokenToID[left + right] else {
                throw LoadError.malformed("merge \(rank) uses unknown tokens")
            }
            let key = UInt64(UInt32(l)) << 32 | UInt64(UInt32(r))
            // a repeated pair keeps its first (best) rank
            if merges[key] == nil {
                merges[key] = UInt64(rank) << 32 | UInt64(UInt32(m))
            }
        }

        for token in added { tokenToID[String(String.UnicodeScalarView(token.scalars))] = token.id }

        var byFirst: [Unicode.Scalar: [AddedToken]] = [:]
        for token in added where !token.scalars.isEmpty {
            byFirst[token.scalars[0], default: []].append(token)
        }
        for key in byFirst.keys { byFirst[key]!.sort { $0.scalars.count > $1.scalars.count } }

        self.tokenToID = tokenToID
        self.idBytes = bytes
        self.idByteOffsets = offsets
        self.idIsSpecial = special
        self.idExists = exists
        self.byteSymbol = byteSymbol
        self.merges = merges
        self.addedByFirstScalar = byFirst
        self.vocabularySize = size
    }

    private static func checkPreTokenizer(_ value: JSONValue?) throws {
        guard let value, let list = value["pretokenizers"]?.array, list.count == 2,
              list[0]["type"]?.string == "Split",
              list[0]["pattern"]?["Regex"]?.string == PreTokenizer.pattern,
              list[0]["behavior"]?.string == "Isolated",
              list[0]["invert"]?.bool == false,
              list[1]["type"]?.string == "ByteLevel",
              list[1]["add_prefix_space"]?.bool == false,
              list[1]["use_regex"]?.bool == false
        else { throw LoadError.unsupported("pre_tokenizer is not the qwen2 split + byte-level") }
    }

    func id(of token: String) -> Int32? {
        tokenToID[token]
    }

    // MARK: encode

    func encode(_ text: String) -> [Int32] {
        let scalars = Array(text.unicodeScalars)
        var output: [Int32] = []
        output.reserveCapacity(scalars.count / 3 + 8)
        var cache: [String: [Int32]] = [:]
        var segmentStart = 0
        var i = 0
        while i < scalars.count {
            if let candidates = addedByFirstScalar[scalars[i]],
               let match = candidates.first(where: { Self.matches($0.scalars, scalars, at: i) }) {
                if segmentStart < i { encodeSegment(scalars[segmentStart..<i], into: &output, cache: &cache) }
                output.append(match.id)
                i += match.scalars.count
                segmentStart = i
            } else {
                i += 1
            }
        }
        if segmentStart < scalars.count {
            encodeSegment(scalars[segmentStart..<scalars.count], into: &output, cache: &cache)
        }
        return output
    }

    private static func matches(_ token: [Unicode.Scalar], _ text: [Unicode.Scalar], at start: Int) -> Bool {
        guard start + token.count <= text.count else { return false }
        for k in 0..<token.count where text[start + k] != token[k] { return false }
        return true
    }

    private func encodeSegment(_ raw: ArraySlice<Unicode.Scalar>, into output: inout [Int32],
                               cache: inout [String: [Int32]]) {
        let rawString = String(String.UnicodeScalarView(raw))
        let normalized = Array(rawString.precomposedStringWithCanonicalMapping.unicodeScalars)
        var utf8: [UInt8] = []
        for piece in PreTokenizer.split(normalized) {
            let word = String(String.UnicodeScalarView(normalized[piece]))
            if let hit = cache[word] {
                output.append(contentsOf: hit)
                continue
            }
            utf8.removeAll(keepingCapacity: true)
            utf8.append(contentsOf: word.utf8)
            let ids = bpe(utf8)
            if cache.count < 4096 { cache[word] = ids }
            output.append(contentsOf: ids)
        }
    }

    /// merges the byte symbols of one pre-token, always taking the lowest rank and, among equal
    /// ranks, the leftmost pair. a lazy min-heap keeps long words from going quadratic.
    private func bpe(_ bytes: [UInt8]) -> [Int32] {
        let n = bytes.count
        var ids = bytes.map { byteSymbol[Int($0)] }
        if n < 2 { return ids }
        if n == 2 {
            if let m = merges[Self.key(ids[0], ids[1])] { return [Int32(truncatingIfNeeded: m)] }
            return ids
        }
        var next = Array(1...n)
        next[n - 1] = -1
        var prev = Array(-1..<(n - 1))
        var alive = [Bool](repeating: true, count: n)

        var heap = MergeHeap()
        for p in 0..<(n - 1) {
            if let m = merges[Self.key(ids[p], ids[p + 1])] {
                heap.push(.init(rank: UInt32(m >> 32), position: Int32(p), left: ids[p], right: ids[p + 1],
                                merged: Int32(truncatingIfNeeded: m)))
            }
        }
        while let top = heap.pop() {
            let p = Int(top.position)
            guard alive[p], ids[p] == top.left else { continue }
            let q = next[p]
            guard q >= 0, ids[q] == top.right else { continue }
            ids[p] = top.merged
            alive[q] = false
            let r = next[q]
            next[p] = r
            if r >= 0 { prev[r] = p }
            let l = prev[p]
            if l >= 0, let m = merges[Self.key(ids[l], ids[p])] {
                heap.push(.init(rank: UInt32(m >> 32), position: Int32(l), left: ids[l], right: ids[p],
                                merged: Int32(truncatingIfNeeded: m)))
            }
            if r >= 0, let m = merges[Self.key(ids[p], ids[r])] {
                heap.push(.init(rank: UInt32(m >> 32), position: Int32(p), left: ids[p], right: ids[r],
                                merged: Int32(truncatingIfNeeded: m)))
            }
        }
        var out: [Int32] = []
        var p = 0
        while p >= 0 {
            out.append(ids[p])
            p = next[p]
        }
        return out
    }

    @inline(__always)
    private static func key(_ a: Int32, _ b: Int32) -> UInt64 {
        UInt64(UInt32(bitPattern: a)) << 32 | UInt64(UInt32(bitPattern: b))
    }

    // MARK: decode

    func decode(_ ids: [Int32], skipSpecialTokens: Bool) -> String {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(ids.count * 4)
        for id in ids {
            let i = Int(id)
            // ids the tokenizer does not know are dropped, as the fast tokenizer does
            guard i >= 0, i < vocabularySize, idExists[i] else { continue }
            if skipSpecialTokens && idIsSpecial[i] { continue }
            let start = Int(idByteOffsets[i])
            let end = Int(idByteOffsets[i + 1])
            bytes.append(contentsOf: idBytes[start..<end])
        }
        return UTF8Repair.decode(bytes)
    }

    /// true when `skipSpecialTokens` drops this id.
    func isSpecial(_ id: Int32) -> Bool {
        let i = Int(id)
        return i >= 0 && i < vocabularySize && idIsSpecial[i]
    }
}

private struct MergeHeap {
    struct Entry {
        var rank: UInt32
        var position: Int32
        var left: Int32
        var right: Int32
        var merged: Int32

        @inline(__always)
        func before(_ other: Entry) -> Bool {
            rank != other.rank ? rank < other.rank : position < other.position
        }
    }

    private var items: [Entry] = []

    mutating func push(_ e: Entry) {
        items.append(e)
        var c = items.count - 1
        while c > 0 {
            let p = (c - 1) / 2
            guard items[c].before(items[p]) else { break }
            items.swapAt(c, p)
            c = p
        }
    }

    mutating func pop() -> Entry? {
        guard let last = items.popLast() else { return nil }
        guard !items.isEmpty else { return last }
        let top = items[0]
        items[0] = last
        var p = 0
        let n = items.count
        while true {
            let l = 2 * p + 1
            if l >= n { break }
            var c = l
            if l + 1 < n, items[l + 1].before(items[l]) { c = l + 1 }
            guard items[c].before(items[p]) else { break }
            items.swapAt(c, p)
            p = c
        }
        return top
    }
}

/// gpt-2's reversible byte <-> printable char table.
enum ByteLevel {
    static let byteToUnicode: [Character] = {
        var table = [Character](repeating: " ", count: 256)
        var extra: UInt32 = 0
        for b in 0..<256 {
            let printable = (0x21...0x7E).contains(b) || (0xA1...0xAC).contains(b) || (0xAE...0xFF).contains(b)
            if printable {
                table[b] = Character(Unicode.Scalar(UInt32(b))!)
            } else {
                table[b] = Character(Unicode.Scalar(256 + extra)!)
                extra += 1
            }
        }
        return table
    }()

    /// indexed by scalar value below 0x200, -1 when the scalar is not a byte symbol.
    static let unicodeToByte: [Int16] = {
        var table = [Int16](repeating: -1, count: 0x200)
        for (b, c) in byteToUnicode.enumerated() {
            table[Int(c.unicodeScalars.first!.value)] = Int16(b)
        }
        return table
    }()
}

/// the qwen2 split pattern, hand-written:
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
/// with the oniguruma semantics the fast tokenizer uses (\s is unicode White_Space, and the
/// case-insensitive contractions also accept U+017F for s).
enum PreTokenizer {
    static let pattern = #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    private static let letter: UInt8 = 1
    private static let number: UInt8 = 2
    private static let space: UInt8 = 4
    private static let newline: UInt8 = 8
    private static let unicodeVersionMajor = 16

    @inline(__always)
    static func classify(_ s: Unicode.Scalar) -> UInt8 {
        let v = s.value
        if v < 0x80 {
            switch v {
            case 0x0A, 0x0D: return space | newline
            case 0x09, 0x0B, 0x0C, 0x20: return space
            case 0x30...0x39: return number
            case 0x41...0x5A, 0x61...0x7A: return letter
            default: return 0
            }
        }
        switch v {
        case 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000: return space
        default: break
        }
        let properties = s.properties
        // the fast tokenizer's regex tables are unicode 16; chars assigned later are still
        // unassigned (neither letter nor number) there.
        guard let age = properties.age, age.major <= unicodeVersionMajor else { return 0 }
        switch properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: return letter
        case .decimalNumber, .letterNumber, .otherNumber: return number
        default: return 0
        }
    }

    /// ranges of the pieces, covering the input in order.
    static func split(_ text: [Unicode.Scalar]) -> [Range<Int>] {
        let n = text.count
        var classes = [UInt8](repeating: 0, count: n)
        for i in 0..<n { classes[i] = classify(text[i]) }
        var pieces: [Range<Int>] = []
        pieces.reserveCapacity(n / 3 + 1)
        var i = 0
        while i < n {
            let end = matchEnd(text, classes, i)
            pieces.append(i..<end)
            i = end
        }
        return pieces
    }

    @inline(__always)
    private static func lower(_ s: Unicode.Scalar) -> UInt32 {
        let v = s.value
        return (v >= 0x41 && v <= 0x5A) ? v + 0x20 : v
    }

    private static func matchEnd(_ t: [Unicode.Scalar], _ c: [UInt8], _ i: Int) -> Int {
        let n = t.count

        // 1. contractions
        if t[i] == "'" && i + 1 < n {
            let a = lower(t[i + 1])
            if a == 0x73 || a == 0x17F || a == 0x74 || a == 0x6D || a == 0x64 { return i + 2 }  // s ſ t m d
            if i + 2 < n {
                let b = lower(t[i + 2])
                if (a == 0x72 || a == 0x76) && b == 0x65 { return i + 3 }  // re ve
                if a == 0x6C && b == 0x6C { return i + 3 }  // ll
            }
        }

        // 2. an optional non-letter/number/newline char, then letters
        if c[i] & letter == 0 && c[i] & number == 0 && c[i] & newline == 0,
           i + 1 < n, c[i + 1] & letter != 0 {
            var j = i + 1
            while j < n && c[j] & letter != 0 { j += 1 }
            return j
        }
        if c[i] & letter != 0 {
            var j = i
            while j < n && c[j] & letter != 0 { j += 1 }
            return j
        }

        // 3. one number
        if c[i] & number != 0 { return i + 1 }

        // 4. optional space, then symbols, then newlines
        let isSymbol = { (k: Int) -> Bool in c[k] & (letter | number | space) == 0 }
        var start4 = -1
        if t[i] == " ", i + 1 < n, isSymbol(i + 1) {
            start4 = i + 1
        } else if isSymbol(i) {
            start4 = i
        }
        if start4 >= 0 {
            var j = start4
            while j < n && isSymbol(j) { j += 1 }
            while j < n && (t[j] == "\r" || t[j] == "\n") { j += 1 }
            return j
        }

        // remaining: t[i] is whitespace
        var runEnd = i
        while runEnd < n && c[runEnd] & space != 0 { runEnd += 1 }

        // 5. whitespace up to and including the last newline of the run
        var lastNewline = -1
        for k in stride(from: runEnd - 1, through: i, by: -1) where c[k] & newline != 0 {
            lastNewline = k
            break
        }
        if lastNewline >= 0 { return lastNewline + 1 }

        // 6. whitespace not followed by a non-space
        if runEnd == n { return runEnd }
        if runEnd - i >= 2 { return runEnd - 1 }

        // 7. whitespace
        return runEnd
    }
}

/// utf-8 decoding that replaces each maximal invalid subsequence with U+FFFD, the unicode
/// recommended practice that rust's `from_utf8_lossy` and python's `errors="replace"` follow.
enum UTF8Repair {
    static func decode(_ bytes: [UInt8]) -> String {
        var scalars = String.UnicodeScalarView()
        let n = bytes.count
        var i = 0
        while i < n {
            let b0 = bytes[i]
            if b0 < 0x80 {
                scalars.append(Unicode.Scalar(b0))
                i += 1
                continue
            }
            var need = 0
            var lo: UInt8 = 0x80
            var hi: UInt8 = 0xBF
            var value: UInt32 = 0
            switch b0 {
            case 0xC2...0xDF: need = 1; value = UInt32(b0 & 0x1F)
            case 0xE0: need = 2; lo = 0xA0; value = UInt32(b0 & 0x0F)
            case 0xE1...0xEC, 0xEE...0xEF: need = 2; value = UInt32(b0 & 0x0F)
            case 0xED: need = 2; hi = 0x9F; value = UInt32(b0 & 0x0F)
            case 0xF0: need = 3; lo = 0x90; value = UInt32(b0 & 0x07)
            case 0xF1...0xF3: need = 3; value = UInt32(b0 & 0x07)
            case 0xF4: need = 3; hi = 0x8F; value = UInt32(b0 & 0x07)
            default:
                scalars.append("\u{FFFD}")
                i += 1
                continue
            }
            var j = i + 1
            var ok = true
            for k in 0..<need {
                guard j < n else { ok = false; break }
                let b = bytes[j]
                let (l, h) = k == 0 ? (lo, hi) : (UInt8(0x80), UInt8(0xBF))
                guard b >= l && b <= h else { ok = false; break }
                value = value << 6 | UInt32(b & 0x3F)
                j += 1
            }
            if ok {
                scalars.append(Unicode.Scalar(value)!)
            } else {
                scalars.append("\u{FFFD}")
            }
            i = j
        }
        return String(scalars)
    }
}
