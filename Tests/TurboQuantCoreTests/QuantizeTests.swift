import Testing
import Foundation
@testable import TurboQuantCore

private func maxAbsError(_ a: [Float], _ b: [Float]) -> Float {
    zip(a, b).map { abs($0.0 - $0.1) }.max() ?? 0
}

/// Roundtrip helper using the new Lloyd-Max API.
/// We need a rotation matrix to be initialised first so coordinates are
/// approximately N(0, 1/d) as the codebook expects.
private func lloydmaxRoundtrip(input: [Float], bits: UInt8) -> [Float] {
    let dim = UInt32(input.count)
    var codes  = [UInt8](repeating: 0, count: input.count)
    var output = [Float](repeating: 0, count: input.count)

    tq_quantize_lloydmax(input, dim, &codes, bits)
    tq_dequantize_lloydmax(codes, &output, dim, bits)
    return output
}

@Suite("TurboQuant Lloyd-Max Quantize Tests", .serialized)
struct QuantizeTests {

    // Unit-norm vector whose rotated coordinates are approximately N(0,1/d).
    // We synthesise a rotated unit vector directly so the codebook scale is correct.
    private func makeRotatedUnitVector(dim: Int, seed: UInt64) -> [Float] {
        // Generate a random N(0,1) vector and normalise it.
        var state = seed
        var v = (0..<dim).map { _ -> Float in
            state &+= 0x9e3779b97f4a7c15
            var z = state
            z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
            z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
            z = z ^ (z >> 31)
            return Float(bitPattern: UInt32(z >> 32) & 0x7fffffff) / Float(0x7fffffff) * 2 - 1
        }
        // Normalise
        let norm = sqrt(v.reduce(0.0) { $0 + $1 * $1 })
        if norm > 1e-10 { v = v.map { $0 / norm } }
        return v
    }

    // MARK: - Codebook sanity

    @Test("1-bit codebook has 2 entries with correct signs")
    func codebook1bit() {
        let cb = tq_lloydmax_codebook(1)!
        #expect(cb[0] < 0, "codebook_1bit[0] should be negative")
        #expect(cb[1] > 0, "codebook_1bit[1] should be positive")
        #expect(abs(cb[0] + cb[1]) < 1e-6, "1-bit codebook should be symmetric")
    }

    @Test("4-bit codebook has 16 monotonically increasing entries")
    func codebook4bit() {
        let cb = tq_lloydmax_codebook(4)!
        for k in 0..<15 {
            #expect(cb[k] < cb[k + 1], "codebook_4bit[\(k)] not < [\(k+1)]")
        }
    }

    @Test("codebook is symmetric for all bit widths")
    func codebookSymmetry() {
        for bits: UInt8 in 1...4 {
            let levels = 1 << Int(bits)
            let cb = tq_lloydmax_codebook(bits)!
            for k in 0..<(levels / 2) {
                let mirror = levels - 1 - k
                #expect(abs(cb[k] + cb[mirror]) < 1e-4,
                        "\(bits)-bit codebook not symmetric at k=\(k)")
            }
        }
    }

    // MARK: - Roundtrip accuracy on rotated unit vectors

    @Test("4-bit Lloyd-Max roundtrip max error < 0.3 on rotated unit vector")
    func fourBitRoundtrip() {
        let dim = 128
        let v = makeRotatedUnitVector(dim: dim, seed: 0xCAFE1234)
        let out = lloydmaxRoundtrip(input: v, bits: 4)
        let err = maxAbsError(v, out)
        #expect(err < 0.3, "4-bit max error \(err) exceeds 0.3")
    }

    @Test("3-bit error >= 4-bit error (more bits = lower error)")
    func fewerBitsMoreError() {
        let dim = 128
        let v = makeRotatedUnitVector(dim: dim, seed: 0xBEEF5678)
        let out4 = lloydmaxRoundtrip(input: v, bits: 4)
        let out3 = lloydmaxRoundtrip(input: v, bits: 3)
        let err4 = maxAbsError(v, out4)
        let err3 = maxAbsError(v, out3)
        #expect(err3 >= err4, "3-bit error (\(err3)) should be >= 4-bit (\(err4))")
    }

    // MARK: - QJL encode sanity (via new API)

    @Test("tq_qjl_encode produces non-trivial packed bits")
    func qjlNonTrivial() {
        let dim: UInt32 = 128
        let seed: UInt64 = 0xDEADBEEF

        tq_rotation_init(dim, seed)
        tq_qjl_init(dim, seed ^ 0xDEADBEEFCAFEBABE)

        let v = makeRotatedUnitVector(dim: Int(dim), seed: 0x11223344)

        // Build a residual by quantising and subtracting
        var codes = [UInt8](repeating: 0, count: Int(dim))
        var dequant = [Float](repeating: 0, count: Int(dim))
        tq_quantize_lloydmax(v, dim, &codes, 3)
        tq_dequantize_lloydmax(codes, &dequant, dim, 3)
        let residual = zip(v, dequant).map { $0.0 - $0.1 }

        let numBitBytes = (Int(dim) + 7) / 8
        var bits = [UInt8](repeating: 0, count: numBitBytes)
        var gamma: Float = 0
        tq_qjl_encode(residual, dim, &bits, &gamma)

        #expect(gamma > 0, "Residual norm gamma should be positive")

        let allZero = bits.allSatisfy { $0 == 0x00 }
        let allOnes = bits.allSatisfy { $0 == 0xFF }
        #expect(!allZero, "QJL output was all zeros")
        #expect(!allOnes, "QJL output was all ones")

        let oneCount = bits.reduce(0) { $0 + Int($1.nonzeroBitCount) }
        let ratio = Float(oneCount) / Float(numBitBytes * 8)
        #expect(ratio > 0.1 && ratio < 0.9, "Bit balance \(ratio) outside [0.1, 0.9]")

        tq_qjl_cleanup()
        tq_rotation_cleanup()
    }
}
