/// `torch.topk(largest=True, sorted=True)` on the cpu, including the order it gives exact ties.
///
/// torch sorts (value, index) pairs with the c++ standard library: a heap-based `partial_sort`
/// when `k * 64 <= n`, otherwise `nth_element` followed by a sort of the first `k - 1`. those
/// are not stable, so equal values (the -inf bans and the -1e9 masks) come out in the order the
/// algorithms leave them. this file reproduces libc++'s steps so the order matches; it was
/// checked against torch 2.9.1 on tie-heavy inputs.
enum TopK {
    struct Entry: Sendable {
        var value: Float
        var index: Int32
    }

    /// torch's comparator for `largest`: nan ranks above everything, then greater first.
    @inline(__always)
    static func better(_ x: Entry, _ y: Entry) -> Bool {
        (x.value.isNaN && !y.value.isNaN) || x.value > y.value
    }

    static func usesHeap(n: Int, k: Int) -> Bool { k * 64 <= n }

    /// top `k` of a small array, `k < values.count`, through the nth_element path or the heap path.
    static func select(_ values: [Float], k: Int) -> [Entry] {
        var entries = values.enumerated().map { Entry(value: $0.element, index: Int32($0.offset)) }
        return select(&entries, k: k)
    }

    static func select(_ entries: inout [Entry], k: Int) -> [Entry] {
        let n = entries.count
        precondition(k >= 1 && k <= n)
        if usesHeap(n: n, k: k) {
            entries.withUnsafeMutableBufferPointer { a in
                makeHeap(a, k)
                for i in k..<n where better(a[i], a[0]) {
                    a.swapAt(i, 0)
                    siftDown(a, k, 0)
                }
                sortHeap(a, k)
            }
        } else {
            entries.withUnsafeMutableBufferPointer { a in
                nthElement(a, 0, k - 1, n)
                smallSort(a, 0, k - 1)
            }
        }
        return Array(entries[0..<k])
    }

    // MARK: heap (partial_sort)

    /// the heap keeps the worst kept entry on top.
    static func siftDown(_ a: UnsafeMutableBufferPointer<Entry>, _ n: Int, _ startIndex: Int) {
        var start = startIndex
        var child = start
        if n < 2 || (n - 2) / 2 < child { return }
        child = 2 * child + 1
        if child + 1 < n && better(a[child], a[child + 1]) { child += 1 }
        if better(a[child], a[start]) { return }
        let top = a[start]
        while true {
            a[start] = a[child]
            start = child
            if (n - 2) / 2 < child { break }
            child = 2 * child + 1
            if child + 1 < n && better(a[child], a[child + 1]) { child += 1 }
            if better(a[child], top) { break }
        }
        a[start] = top
    }

    static func makeHeap(_ a: UnsafeMutableBufferPointer<Entry>, _ n: Int) {
        guard n > 1 else { return }
        for start in stride(from: (n - 2) / 2, through: 0, by: -1) {
            siftDown(a, n, start)
        }
    }

    private static func floydSiftDown(_ a: UnsafeMutableBufferPointer<Entry>, _ n: Int) -> Int {
        var hole = 0
        var child = 0
        while true {
            var c = 2 * child + 1
            if c + 1 < n && better(a[c], a[c + 1]) { c += 1 }
            child = c
            a[hole] = a[c]
            hole = c
            if child > (n - 2) / 2 { return hole }
        }
    }

    private static func siftUp(_ a: UnsafeMutableBufferPointer<Entry>, _ end: Int) {
        // end is one past the entry to move up
        var len = end
        guard len > 1 else { return }
        len = (len - 2) / 2
        var last = end - 1
        if better(a[len], a[last]) {
            let t = a[last]
            var parent = len
            while true {
                a[last] = a[parent]
                last = parent
                if len == 0 { break }
                len = (len - 1) / 2
                parent = len
                if !better(a[parent], t) { break }
            }
            a[last] = t
        }
    }

    private static func popHeap(_ a: UnsafeMutableBufferPointer<Entry>, _ n: Int) {
        guard n > 1 else { return }
        let top = a[0]
        var hole = floydSiftDown(a, n)
        let last = n - 1
        if hole == last {
            a[hole] = top
        } else {
            a[hole] = a[last]
            hole += 1
            a[last] = top
            siftUp(a, hole)
        }
    }

    static func sortHeap(_ a: UnsafeMutableBufferPointer<Entry>, _ count: Int) {
        var n = count
        while n > 1 {
            popHeap(a, n)
            n -= 1
        }
    }

    // MARK: nth_element + sort

    @discardableResult
    private static func sort3(_ a: UnsafeMutableBufferPointer<Entry>, _ x: Int, _ y: Int, _ z: Int) -> Int {
        if !better(a[y], a[x]) {
            if !better(a[z], a[y]) { return 0 }
            a.swapAt(y, z)
            if better(a[y], a[x]) {
                a.swapAt(x, y)
                return 2
            }
            return 1
        }
        if better(a[z], a[y]) {
            a.swapAt(x, z)
            return 1
        }
        a.swapAt(x, y)
        if better(a[z], a[y]) {
            a.swapAt(y, z)
            return 2
        }
        return 1
    }

    private static func selectionSort(_ a: UnsafeMutableBufferPointer<Entry>, _ lo: Int, _ hi: Int) {
        var first = lo
        while first < hi - 1 {
            var m = first
            for j in (first + 1)..<hi where better(a[j], a[m]) { m = j }
            if m != first { a.swapAt(first, m) }
            first += 1
        }
    }

    private static func nthElement(_ a: UnsafeMutableBufferPointer<Entry>, _ lo: Int, _ nth: Int, _ hi: Int) {
        var first = lo
        var last = hi
        while true {
            if nth == last { return }
            let len = last - first
            if len <= 1 { return }
            if len == 2 {
                if better(a[last - 1], a[first]) { a.swapAt(first, last - 1) }
                return
            }
            if len == 3 {
                sort3(a, first, first + 1, last - 1)
                return
            }
            if len <= 7 {
                selectionSort(a, first, last)
                return
            }
            var m = first + len / 2
            let lm1 = last - 1
            var swaps = sort3(a, first, m, lm1)
            var i = first
            var j = lm1
            if !better(a[i], a[m]) {
                // the pivot equals the first entry: look for a guard below it
                var guarded = false
                while true {
                    j -= 1
                    if i == j { break }
                    if better(a[j], a[m]) {
                        guarded = true
                        break
                    }
                }
                if guarded {
                    a.swapAt(i, j)
                    swaps += 1
                } else {
                    // everything is >= the first entry: split off the entries equal to it
                    i += 1
                    j = last - 1
                    if !better(a[first], a[j]) {
                        while true {
                            if i == j { return }
                            if better(a[first], a[i]) {
                                a.swapAt(i, j)
                                swaps += 1
                                i += 1
                                break
                            }
                            i += 1
                        }
                    }
                    if i == j { return }
                    while true {
                        while !better(a[first], a[i]) { i += 1 }
                        repeat { j -= 1 } while better(a[first], a[j])
                        if i >= j { break }
                        a.swapAt(i, j)
                        swaps += 1
                        i += 1
                    }
                    if nth < i { return }
                    first = i
                    continue
                }
            }
            i += 1
            if i < j {
                while true {
                    while better(a[i], a[m]) { i += 1 }
                    repeat { j -= 1 } while !better(a[j], a[m])
                    if i >= j { break }
                    a.swapAt(i, j)
                    swaps += 1
                    if m == i { m = j }
                    i += 1
                }
            }
            if i != m && better(a[m], a[i]) {
                a.swapAt(i, m)
                swaps += 1
            }
            if nth == i { return }
            if swaps == 0 {
                // already partitioned: stop if the side holding nth is sorted
                if nth < i {
                    var p = first
                    var q = first
                    var sorted = false
                    while true {
                        q += 1
                        if q == i { sorted = true; break }
                        if better(a[q], a[p]) { break }
                        p = q
                    }
                    if sorted { return }
                } else {
                    var p = i
                    var q = i
                    var sorted = false
                    while true {
                        q += 1
                        if q == last { sorted = true; break }
                        if better(a[q], a[p]) { break }
                        p = q
                    }
                    if sorted { return }
                }
            }
            if nth < i {
                last = i
            } else {
                first = i + 1
            }
        }
    }

    private static func smallSort(_ a: UnsafeMutableBufferPointer<Entry>, _ first: Int, _ last: Int) {
        let n = last - first
        if n <= 1 { return }
        if n == 2 {
            if better(a[last - 1], a[first]) { a.swapAt(first, last - 1) }
            return
        }
        if n == 3 {
            sort3(a, first, first + 1, last - 1)
            return
        }
        precondition(n < 30, "topk ties are only reproduced for k <= 30 on this path")
        for i in (first + 1)..<last {
            var j = i - 1
            if better(a[i], a[j]) {
                let t = a[i]
                var k = j
                j = i
                while true {
                    a[j] = a[k]
                    j = k
                    if j == first { break }
                    k -= 1
                    if !better(t, a[k]) { break }
                }
                a[j] = t
            }
        }
    }
}
