import Testing
import Foundation
@testable import TurboQuantCore

private func maxAbsError(_ a: [Float], _ b: [Float]) -> Float {
    zip(a, b).map { abs($0.0 - $0.1) }.max() ?? 0
}

private func roundtrip(input: [Float], blockSize: UInt32, bitsX2: UInt16) -> [Float] {
    let dim = UInt32(input.count)
    let numBlocks = Int((dim + blockSize - 1) / blockSize)

    var codes  = [UInt8](repeating: 0, count: Int(dim))
    var scales = [Float](repeating: 0, count: numBlocks)
    var zeros  = [Float](repeating: 0, count: numBlocks)
    var output = [Float](repeating: 0, count: Int(dim))

    tq_quantize_scalar(input, dim, &codes, &scales, &zeros, blockSize, bitsX2)
    tq_dequantize_scalar(codes, scales, zeros, &output, dim, blockSize, bitsX2)

    return output
}

@Suite("TurboQuant Quantize Tests")
struct QuantizeTests {

    static let sinWave: [Float] = (0..<256).map { Float(sin(Double($0) * .pi / 64.0)) }

    @Test("4-bit roundtrip max error < 0.15 on sin wave")
    func fourBitRoundtrip() {
        let output = roundtrip(input: Self.sinWave, blockSize: 32, bitsX2: 8)
        let err = maxAbsError(Self.sinWave, output)
        #expect(err < 0.15, "4-bit max error \(err) exceeds 0.15")
    }

    @Test("3.5-bit error >= 4-bit error")
    func halfBitMoreError() {
        let out4   = roundtrip(input: Self.sinWave, blockSize: 32, bitsX2: 8)
        let out3_5 = roundtrip(input: Self.sinWave, blockSize: 32, bitsX2: 7)
        let err4   = maxAbsError(Self.sinWave, out4)
        let err3_5 = maxAbsError(Self.sinWave, out3_5)
        #expect(err3_5 >= err4, "3.5-bit error (\(err3_5)) should >= 4-bit (\(err4))")
    }

    @Test("QJL encode produces non-trivial packed bits")
    func qjlNonTrivial() {
        let dim: UInt32 = 256
        let blockSize: UInt32 = 32
        let numBlocks = Int((dim + blockSize - 1) / blockSize)

        var codes  = [UInt8](repeating: 0, count: Int(dim))
        var scales = [Float](repeating: 0, count: numBlocks)
        var zeros  = [Float](repeating: 0, count: numBlocks)
        var dequant = [Float](repeating: 0, count: Int(dim))

        tq_quantize_scalar(Self.sinWave, dim, &codes, &scales, &zeros, blockSize, 8)
        tq_dequantize_scalar(codes, scales, zeros, &dequant, dim, blockSize, 8)

        let numBitBytes = (Int(dim) + 7) / 8
        var bits = [UInt8](repeating: 0, count: numBitBytes)
        tq_qjl_encode(Self.sinWave, dequant, &bits, dim, 0xDEADBEEF)

        let allZero = bits.allSatisfy { $0 == 0x00 }
        let allOnes = bits.allSatisfy { $0 == 0xFF }
        #expect(!allZero, "QJL output was all zeros")
        #expect(!allOnes, "QJL output was all ones")

        let oneCount = bits.reduce(0) { $0 + Int($1.nonzeroBitCount) }
        let ratio = Float(oneCount) / Float(numBitBytes * 8)
        #expect(ratio > 0.2 && ratio < 0.8, "Bit balance \(ratio) outside [0.2, 0.8]")
    }
}
