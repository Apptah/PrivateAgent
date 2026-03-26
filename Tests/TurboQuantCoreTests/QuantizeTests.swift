import Testing
import Foundation
@testable import TurboQuantCore

// ── Helpers ──

/// Wrap C quantize/dequantize for a given bit width.
private func roundtrip(
    input: [Float],
    blockSize: Int,
    bitsX2: UInt16
) -> [Float] {
    let n = input.count
    let numBlocks = (n + blockSize - 1) / blockSize

    var codes  = [UInt8](repeating: 0, count: n)
    var scales = [Float](repeating: 0, count: numBlocks)
    var zeros  = [Float](repeating: 0, count: numBlocks)
    var output = [Float](repeating: 0, count: n)

    input.withUnsafeBufferPointer { inPtr in
        codes.withUnsafeMutableBufferPointer { codesPtr in
            scales.withUnsafeMutableBufferPointer { scalesPtr in
                zeros.withUnsafeMutableBufferPointer { zerosPtr in
                    tq_quantize_scalar(
                        inPtr.baseAddress,
                        codesPtr.baseAddress,
                        scalesPtr.baseAddress,
                        zerosPtr.baseAddress,
                        n,
                        blockSize,
                        bitsX2
                    )
                }
            }
        }
    }

    codes.withUnsafeBufferPointer { codesPtr in
        scales.withUnsafeBufferPointer { scalesPtr in
            zeros.withUnsafeBufferPointer { zerosPtr in
                output.withUnsafeMutableBufferPointer { outPtr in
                    tq_dequantize_scalar(
                        codesPtr.baseAddress,
                        scalesPtr.baseAddress,
                        zerosPtr.baseAddress,
                        outPtr.baseAddress,
                        n,
                        blockSize,
                        bitsX2
                    )
                }
            }
        }
    }

    return output
}

private func maxAbsError(_ a: [Float], _ b: [Float]) -> Float {
    zip(a, b).map { abs($0.0 - $0.1) }.max() ?? 0
}

// ── Tests ──

@Suite("TurboQuant Quantize Tests")
struct QuantizeTests {

    // Generate a sin-wave input in [-1, 1]
    static let sinWave: [Float] = (0..<256).map { i in
        Float(sin(Double(i) * .pi / 64.0))
    }

    // ── Roundtrip at 4-bit ──

    @Test("4-bit roundtrip max error < 0.15 on sin wave")
    func fourBitRoundtrip() {
        let output = roundtrip(input: Self.sinWave, blockSize: 32, bitsX2: 8)
        let err = maxAbsError(Self.sinWave, output)
        // 4-bit has 16 levels over range 2.0 → step ≈ 0.133; allow some margin
        #expect(err < 0.15, "4-bit max error \(err) exceeds 0.15")
    }

    // ── 3.5-bit has >= error than 4-bit ──

    @Test("3.5-bit error >= 4-bit error (fewer levels = more error)")
    func halfBitMoreError() {
        let out4   = roundtrip(input: Self.sinWave, blockSize: 32, bitsX2: 8)
        let out3_5 = roundtrip(input: Self.sinWave, blockSize: 32, bitsX2: 7)

        let err4   = maxAbsError(Self.sinWave, out4)
        let err3_5 = maxAbsError(Self.sinWave, out3_5)

        // 3.5-bit (≈11 levels) should be less precise than 4-bit (16 levels)
        #expect(err3_5 >= err4,
            "Expected 3.5-bit error (\(err3_5)) >= 4-bit error (\(err4))")
    }

    // ── QJL encode produces non-trivial bits ──

    @Test("QJL encode produces non-trivial packed bits")
    func qjlNonTrivial() {
        let input = Self.sinWave
        let n = input.count

        // Compute dequantized via 4-bit roundtrip
        let numBlocks = (n + 31) / 32
        var codes  = [UInt8](repeating: 0, count: n)
        var scales = [Float](repeating: 0, count: numBlocks)
        var zeros  = [Float](repeating: 0, count: numBlocks)
        var dequant = [Float](repeating: 0, count: n)

        input.withUnsafeBufferPointer { inPtr in
            codes.withUnsafeMutableBufferPointer { codesPtr in
                scales.withUnsafeMutableBufferPointer { scalesPtr in
                    zeros.withUnsafeMutableBufferPointer { zerosPtr in
                        tq_quantize_scalar(inPtr.baseAddress, codesPtr.baseAddress,
                                           scalesPtr.baseAddress, zerosPtr.baseAddress,
                                           n, 32, 8)
                    }
                }
            }
        }
        codes.withUnsafeBufferPointer { codesPtr in
            scales.withUnsafeBufferPointer { scalesPtr in
                zeros.withUnsafeBufferPointer { zerosPtr in
                    dequant.withUnsafeMutableBufferPointer { outPtr in
                        tq_dequantize_scalar(codesPtr.baseAddress, scalesPtr.baseAddress,
                                             zerosPtr.baseAddress, outPtr.baseAddress,
                                             n, 32, 8)
                    }
                }
            }
        }

        // Encode residuals
        let numBitBytes = (n + 7) / 8
        var bits = [UInt8](repeating: 0, count: numBitBytes)

        input.withUnsafeBufferPointer { inPtr in
            dequant.withUnsafeBufferPointer { dqPtr in
                bits.withUnsafeMutableBufferPointer { bitsPtr in
                    tq_qjl_encode(inPtr.baseAddress, dqPtr.baseAddress,
                                  bitsPtr.baseAddress, n, 0xDEADBEEF_CAFEBABE)
                }
            }
        }

        // The bits should not all be zero or all be 0xFF
        let allZero = bits.allSatisfy { $0 == 0x00 }
        let allOnes = bits.allSatisfy { $0 == 0xFF }

        #expect(!allZero, "QJL output was all zeros — PRNG or residual logic broken")
        #expect(!allOnes, "QJL output was all ones — sign logic broken")

        // Bit balance: between 20% and 80% ones (random-ish)
        let totalBits = numBitBytes * 8
        let oneCount  = bits.reduce(0) { $0 + Int($1.nonzeroBitCount) }
        let ratio = Float(oneCount) / Float(totalBits)
        #expect(ratio > 0.2 && ratio < 0.8,
            "QJL bit balance \(ratio) out of expected [0.2, 0.8] range")
    }
}
