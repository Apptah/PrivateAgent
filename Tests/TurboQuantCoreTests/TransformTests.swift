import Testing
import Foundation
@testable import TurboQuantCore

// Helper: compute L2 norm of a float array
private func l2norm(_ v: [Float]) -> Float {
    sqrt(v.reduce(0.0) { $0 + $1 * $1 })
}

@Suite("Structured Rotation Transform Tests")
struct TransformTests {

    let dim: UInt32 = 128  // must be power of 2
    let seed: UInt64 = 0xDEAD_BEEF_CAFE_1234

    // Build a repeatable test vector
    private func makeVector(dim: UInt32, offset: Float = 0.0) -> [Float] {
        (0..<Int(dim)).map { i in sin(Float(i) * 0.3 + offset) }
    }

    // MARK: - Invertibility

    @Test("Rotate then inverse rotate recovers original vector")
    func rotationIsInvertible() {
        var x = makeVector(dim: dim)
        let original = x

        x.withUnsafeMutableBufferPointer { buf in
            tq_rotate_inplace(buf.baseAddress, dim, seed)
            tq_rotate_inverse_inplace(buf.baseAddress, dim, seed)
        }

        for i in 0..<Int(dim) {
            #expect(abs(x[i] - original[i]) < 1e-4,
                    "Element \(i): got \(x[i]), expected \(original[i])")
        }
    }

    // MARK: - Norm preservation

    @Test("Rotation preserves L2 norm")
    func rotationPreservesNorm() {
        var x = makeVector(dim: dim)
        let normBefore = l2norm(x)

        x.withUnsafeMutableBufferPointer { buf in
            tq_rotate_inplace(buf.baseAddress, dim, seed)
        }

        let normAfter = l2norm(x)
        #expect(abs(normAfter - normBefore) < 1e-3,
                "Norm before: \(normBefore), after: \(normAfter)")
    }

    // MARK: - Determinism

    @Test("Same seed produces same rotation")
    func sameSeedIsDeterministic() {
        var x1 = makeVector(dim: dim)
        var x2 = makeVector(dim: dim)

        x1.withUnsafeMutableBufferPointer { tq_rotate_inplace($0.baseAddress, dim, seed) }
        x2.withUnsafeMutableBufferPointer { tq_rotate_inplace($0.baseAddress, dim, seed) }

        for i in 0..<Int(dim) {
            #expect(x1[i] == x2[i], "Element \(i) differs between identical rotations")
        }
    }

    @Test("Different seeds produce different rotations")
    func differentSeedsProduceDifferentResults() {
        var x1 = makeVector(dim: dim)
        var x2 = makeVector(dim: dim)

        x1.withUnsafeMutableBufferPointer { tq_rotate_inplace($0.baseAddress, dim, seed) }
        x2.withUnsafeMutableBufferPointer { tq_rotate_inplace($0.baseAddress, dim, seed &+ 1) }

        // At least one element must differ
        let anyDiffers = (0..<Int(dim)).contains { x1[$0] != x2[$0] }
        #expect(anyDiffers, "Different seeds should produce different rotations")
    }

    // MARK: - Query rotation

    @Test("tq_rotate_query: dot(R@k, R@q) == dot(k, q)")
    func queryRotationPreservesDotProduct() {
        let k = makeVector(dim: dim, offset: 0.0)
        let q = makeVector(dim: dim, offset: 1.1)

        // Compute original dot product
        let dotOriginal = zip(k, q).reduce(0.0 as Float) { $0 + $1.0 * $1.1 }

        // Rotate k forward: R @ k
        var kRotated = k
        kRotated.withUnsafeMutableBufferPointer { tq_rotate_inplace($0.baseAddress, dim, seed) }

        // Rotate q via tq_rotate_query (also applies R, same forward rotation)
        var qRotated = [Float](repeating: 0.0, count: Int(dim))
        q.withUnsafeBufferPointer { qBuf in
            qRotated.withUnsafeMutableBufferPointer { qOutBuf in
                tq_rotate_query(qBuf.baseAddress, qOutBuf.baseAddress, dim, seed)
            }
        }

        // dot(R@k, R@q) = k^T R^T R q = k^T q  (R orthogonal: R^T R = I)
        let dotRotated = zip(kRotated, qRotated).reduce(0.0 as Float) { $0 + $1.0 * $1.1 }
        #expect(abs(dotRotated - dotOriginal) < 1e-3,
                "dot after rotation: \(dotRotated), expected: \(dotOriginal)")
    }

    @Test("tq_rotate_query output inverse-rotated recovers input")
    func queryRotationRecoversByInverseRotation() {
        let q = makeVector(dim: dim, offset: 2.7)

        // q_out = R @ q  (tq_rotate_query applies forward rotation)
        var qOut = [Float](repeating: 0.0, count: Int(dim))
        q.withUnsafeBufferPointer { qBuf in
            qOut.withUnsafeMutableBufferPointer { outBuf in
                tq_rotate_query(qBuf.baseAddress, outBuf.baseAddress, dim, seed)
            }
        }

        // Applying inverse rotation to q_out should recover q: R^{-1} @ (R @ q) = q
        qOut.withUnsafeMutableBufferPointer { tq_rotate_inverse_inplace($0.baseAddress, dim, seed) }

        for i in 0..<Int(dim) {
            #expect(abs(qOut[i] - q[i]) < 1e-4,
                    "Element \(i): got \(qOut[i]), expected \(q[i])")
        }
    }
}
