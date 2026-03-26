import Testing
import Foundation
@testable import TurboQuantCore

private func l2norm(_ v: [Float]) -> Float {
    sqrt(v.reduce(0.0) { $0 + $1 * $1 })
}

@Suite("Random Orthogonal Rotation Tests", .serialized)
struct TransformTests {

    let dim: UInt32 = 32     // unique dim to avoid cross-suite global state contention
    let seed: UInt64 = 0xDEAD_BEEF_CAFE_1234

    private func makeVector(dim: UInt32, offset: Float = 0.0) -> [Float] {
        (0..<Int(dim)).map { i in sin(Float(i) * 0.3 + offset) }
    }

    // MARK: - Init / cleanup

    @Test("tq_rotation_init succeeds and returns 0")
    func rotationInitSucceeds() {
        let ret = tq_rotation_init(dim, seed)
        #expect(ret == 0, "tq_rotation_init returned \(ret)")
        tq_rotation_cleanup()
    }

    // MARK: - Invertibility

    @Test("rotate then inverse-rotate recovers original vector")
    func rotationIsInvertible() {
        tq_rotation_init(dim, seed)
        let x = makeVector(dim: dim)
        var y = [Float](repeating: 0, count: Int(dim))
        var xhat = [Float](repeating: 0, count: Int(dim))

        x.withUnsafeBufferPointer { xb in
            y.withUnsafeMutableBufferPointer { yb in
                tq_rotate(xb.baseAddress, yb.baseAddress, dim)
            }
        }
        y.withUnsafeBufferPointer { yb in
            xhat.withUnsafeMutableBufferPointer { xhb in
                tq_rotate_inverse(yb.baseAddress, xhb.baseAddress, dim)
            }
        }

        for i in 0..<Int(dim) {
            #expect(abs(xhat[i] - x[i]) < 1e-4,
                    "Element \(i): got \(xhat[i]), expected \(x[i])")
        }
        tq_rotation_cleanup()
    }

    // MARK: - Norm preservation

    @Test("rotation preserves L2 norm")
    func rotationPreservesNorm() {
        // Use tq_rotate_inplace which atomically inits + rotates under mutex
        let x = makeVector(dim: dim)
        var y = x  // copy
        tq_rotate_inplace(&y, dim, seed)
        let normBefore = l2norm(x)
        let normAfter  = l2norm(y)
        #expect(abs(normAfter - normBefore) < 1e-3,
                "Norm before: \(normBefore), after: \(normAfter)")
    }

    // MARK: - Determinism

    @Test("same seed produces same rotation")
    func sameSeedIsDeterministic() {
        let x = makeVector(dim: dim)
        var y1 = [Float](repeating: 0, count: Int(dim))
        var y2 = [Float](repeating: 0, count: Int(dim))

        tq_rotation_init(dim, seed)
        x.withUnsafeBufferPointer { xb in
            y1.withUnsafeMutableBufferPointer { yb in tq_rotate(xb.baseAddress, yb.baseAddress, dim) }
        }
        tq_rotation_cleanup()

        tq_rotation_init(dim, seed)
        x.withUnsafeBufferPointer { xb in
            y2.withUnsafeMutableBufferPointer { yb in tq_rotate(xb.baseAddress, yb.baseAddress, dim) }
        }
        tq_rotation_cleanup()

        for i in 0..<Int(dim) {
            #expect(y1[i] == y2[i], "Element \(i) differs between identical seeds")
        }
    }

    @Test("different seeds produce different rotations")
    func differentSeedsProduceDifferentResults() {
        let x = makeVector(dim: dim)
        var y1 = [Float](repeating: 0, count: Int(dim))
        var y2 = [Float](repeating: 0, count: Int(dim))

        tq_rotation_init(dim, seed)
        x.withUnsafeBufferPointer { xb in
            y1.withUnsafeMutableBufferPointer { yb in tq_rotate(xb.baseAddress, yb.baseAddress, dim) }
        }
        tq_rotation_cleanup()

        tq_rotation_init(dim, seed &+ 1)
        x.withUnsafeBufferPointer { xb in
            y2.withUnsafeMutableBufferPointer { yb in tq_rotate(xb.baseAddress, yb.baseAddress, dim) }
        }
        tq_rotation_cleanup()

        let anyDiffers = (0..<Int(dim)).contains { y1[$0] != y2[$0] }
        #expect(anyDiffers, "Different seeds should produce different rotations")
    }

    // MARK: - Backward-compatible inplace wrappers

    @Test("tq_rotate_inplace: rotate then inverse recovers original")
    func inplaceWrapperIsInvertible() {
        var x    = makeVector(dim: dim)
        let orig = x

        x.withUnsafeMutableBufferPointer { buf in
            tq_rotate_inplace(buf.baseAddress, dim, seed)
            tq_rotate_inverse_inplace(buf.baseAddress, dim, seed)
        }

        for i in 0..<Int(dim) {
            #expect(abs(x[i] - orig[i]) < 1e-4,
                    "Element \(i): got \(x[i]), expected \(orig[i])")
        }
    }

    // MARK: - Query rotation (dot-product preservation)

    @Test("tq_rotate_query: dot(Π@k, Π@q) == dot(k, q)")
    func queryRotationPreservesDotProduct() {
        tq_rotation_init(dim, seed)
        let k = makeVector(dim: dim, offset: 0.0)
        let q = makeVector(dim: dim, offset: 1.1)

        let dotOriginal = zip(k, q).reduce(0.0 as Float) { $0 + $1.0 * $1.1 }

        var kRotated = [Float](repeating: 0, count: Int(dim))
        var qRotated = [Float](repeating: 0, count: Int(dim))

        k.withUnsafeBufferPointer { kb in
            kRotated.withUnsafeMutableBufferPointer { yb in
                tq_rotate(kb.baseAddress, yb.baseAddress, dim)
            }
        }
        q.withUnsafeBufferPointer { qb in
            qRotated.withUnsafeMutableBufferPointer { yb in
                tq_rotate_query(qb.baseAddress, yb.baseAddress, dim, seed)
            }
        }

        let dotRotated = zip(kRotated, qRotated).reduce(0.0 as Float) { $0 + $1.0 * $1.1 }
        #expect(abs(dotRotated - dotOriginal) < 1e-3,
                "dot after rotation: \(dotRotated), expected: \(dotOriginal)")
        tq_rotation_cleanup()
    }
}
