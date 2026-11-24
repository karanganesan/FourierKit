import Foundation

/// float32 math rounded the way torch's arm64 cpu kernels round it, so the search sees the
/// same log-probabilities transformers does.
///
/// torch's vectorised `exp` on arm64 is sleef's `expf4_u10` (a degree-6 polynomial after a
/// ln2 range reduction, with fused multiply-adds); this reimplements that recipe. it was checked
/// bit for bit against `torch.exp` (torch 2.9.1) on millions of inputs.
enum TorchFloat {
    private static let log2e: Float = 1.442695040888963407359924681001892137426645954152985934135449406931
    private static let ln2Hi: Float = 0.693145751953125
    private static let ln2Lo: Float = 1.428606765330187045e-06
    private static let magic: Float = 12_582_912
    private static let magicBits = SIMD4<Int32>(repeating: Int32(bitPattern: (12_582_912 as Float).bitPattern))

    @inline(__always)
    static func exp(_ d: SIMD4<Float>) -> SIMD4<Float> {
        // inputs outside [-104, 100] give garbage here and are replaced at the end
        let qf = (d * log2e).rounded(.toNearestOrEven)
        // integer-valued floats read back as ints through the 1.5 * 2^23 bias (the generic
        // simd float-to-int init is very slow)
        let q = unsafeBitCast(qf + magic, to: SIMD4<Int32>.self) &- magicBits
        var s = d.addingProduct(qf, -ln2Hi)
        s = s.addingProduct(qf, -ln2Lo)
        var u = SIMD4<Float>(repeating: 0.000198527617612853646278381)
        u = SIMD4<Float>(repeating: 0.00139304355252534151077271).addingProduct(u, s)
        u = SIMD4<Float>(repeating: 0.00833336077630519866943359).addingProduct(u, s)
        u = SIMD4<Float>(repeating: 0.0416664853692054748535156).addingProduct(u, s)
        u = SIMD4<Float>(repeating: 0.166666671633720397949219).addingProduct(u, s)
        u = SIMD4<Float>(repeating: 0.5).addingProduct(u, s)
        u = 1 + s.addingProduct(s * s, u)
        // scale by 2^q in two halves so large |q| does not overflow the exponent field
        let q1 = q &>> 1
        let q2 = q &- q1
        u = u * pow2(q1) * pow2(q2)
        u.replace(with: 0, where: d .< -104)
        u.replace(with: .infinity, where: 100 .< d)
        u.replace(with: .nan, where: d .!= d)
        return u
    }

    @inline(__always)
    private static func pow2(_ q: SIMD4<Int32>) -> SIMD4<Float> {
        unsafeBitCast((q &+ 0x7F) &<< 23, to: SIMD4<Float>.self)
    }

    static func exp(_ x: Float) -> Float {
        exp(SIMD4<Float>(repeating: x))[0]
    }

    /// sleef's `logf4_u10`: split off the exponent around 0.75, then atanh-style series in
    /// double-float (head + tail) arithmetic. finite positive normal inputs only; anything else
    /// falls back to a correctly rounded log, which only matters for rows torch turns into nan.
    static func log(_ x: Float) -> Float {
        guard x.isFinite, x >= .leastNormalMagnitude else { return Float(Foundation.log(Double(x))) }
        let e = Int32(((x * (1.0 / 0.75)).bitPattern >> 23) & 0xFF) - 0x7F
        let m = Float(bitPattern: UInt32(bitPattern: Int32(bitPattern: x.bitPattern) &- (e &<< 23)))
        var s = mul(DF(0.69314718246459960938, -1.904654323148236017e-09), Float(e))
        let xq = div(add2(-1, m), add2(1, m))
        let x2 = xq.hi * xq.hi
        var t: Float = 0.3027294874
        t = (0.3996108174 as Float).addingProduct(t, x2)
        t = (0.6666694880 as Float).addingProduct(t, x2)
        s = add(s, DF(xq.hi * 2, xq.lo * 2))
        s = add(s, x2 * xq.hi * t)
        return s.hi + s.lo
    }

    private struct DF {
        var hi: Float
        var lo: Float
        init(_ hi: Float, _ lo: Float) { self.hi = hi; self.lo = lo }
    }

    @inline(__always) private static func fma(_ a: Float, _ b: Float, _ c: Float) -> Float {
        c.addingProduct(a, b)
    }

    @inline(__always) private static func mul(_ x: DF, _ y: Float) -> DF {
        let s = x.hi * y
        return DF(s, fma(x.lo, y, fma(x.hi, y, -s)))
    }

    @inline(__always) private static func add2(_ x: Float, _ y: Float) -> DF {
        let s = x + y
        let v = s - x
        return DF(s, (x - (s - v)) + (y - v))
    }

    @inline(__always) private static func add(_ x: DF, _ y: DF) -> DF {
        let s = x.hi + y.hi
        return DF(s, (((x.hi - s) + y.hi) + x.lo) + y.lo)
    }

    @inline(__always) private static func add(_ x: DF, _ y: Float) -> DF {
        let s = x.hi + y
        return DF(s, ((x.hi - s) + y) + x.lo)
    }

    @inline(__always) private static func div(_ n: DF, _ d: DF) -> DF {
        let t = 1 / d.hi
        let s = n.hi * t
        let u = fma(t, n.hi, -s)
        let v = fma(-d.lo, t, fma(-d.hi, t, 1))
        return DF(s, fma(s, v, fma(n.lo, t, u)))
    }
}
