import Testing
import Foundation
@testable import TurboQuantCore

private func makeDeterministicVector(dim: Int, seed: UInt64) -> [Float] {
    var state = seed
    return (0..<dim).map { _ -> Float in
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        z = z ^ (z >> 31)
        return Float(Int64(bitPattern: z)) / Float(Int64.max)
    }
}

private func bf16DotProduct(_ a: [Float], _ b: [Float]) -> Float {
    func toBF16(_ f: Float) -> Float {
        Float(bitPattern: f.bitPattern & 0xFFFF0000)
    }
    return zip(a, b).reduce(0.0) { $0 + toBF16($1.0) * toBF16($1.1) }
}

/// Build a desc that matches the new API (block_size ignored by new path, kept for struct compat).
private func makeDesc(bitsX2: UInt16 = 8, seed: UInt64 = 42) -> PA_QuantizedKVDesc {
    PA_QuantizedKVDesc(
        key_bits_x2: bitsX2,
        value_bits_x2: bitsX2,
        block_size: 32,                // unused by new path, kept for struct compat
        transform_kind: UInt32(PA_TRANSFORM_STRUCTURED_ROTATION.rawValue),
        transform_seed: seed,
        residual_bits: 1,
        main_codes_offset: 0,
        aux_params_offset: 0,
        qjl_bits_offset: 0,
        aux_params_kind: 0
    )
}

@Suite("Compressed KV Operations")
struct CompressedKVTests {

    let dim: Int = 128   // smaller dim so QR is fast in CI

    // MARK: - Compressed size

    @Test("tq_compressed_size returns 8 + dim + ceil(dim/8)")
    func compressedSizeFormula() {
        for d in [64, 128, 256] as [UInt32] {
            let expected = 8 + d + (d + 7) / 8
            let got = tq_compressed_size(d)
            #expect(got == expected,
                    "dim=\(d): expected \(expected), got \(got)")
        }
    }

    // MARK: - Compress → decompress roundtrip

    @Test("V decompress roundtrip max absolute error < 0.8")
    func vDecompressRoundtrip() {
        let vVec = makeDeterministicVector(dim: dim, seed: 0xDEAD_BEEF_0003)
        var desc = makeDesc()

        let bufSize = Int(tq_compressed_size(UInt32(dim)))
        var vCompressed = [UInt8](repeating: 0, count: bufSize)
        let written = tq_compress_kv(vVec, UInt32(dim), &vCompressed, &desc)
        #expect(written == UInt32(bufSize), "Expected \(bufSize) bytes written, got \(written)")

        var vOut = [Float](repeating: 0, count: dim)
        tq_decompress_v_tile(vCompressed, &vOut, UInt32(dim), &desc)

        let maxErr = zip(vVec, vOut).map { abs($0.0 - $0.1) }.max() ?? 0
        #expect(maxErr < 0.8, "V roundtrip max error \(maxErr) exceeds 0.8")
    }

    // MARK: - Compressed dot accuracy

    @Test("Compressed dot product relative error < 0.8 vs bf16 reference")
    func compressedDotAccuracy() {
        let kVec = makeDeterministicVector(dim: dim, seed: 0xCAFE_BABE_0001)
        let qVec = makeDeterministicVector(dim: dim, seed: 0xCAFE_BABE_0002)
        var desc = makeDesc()

        let ref = bf16DotProduct(kVec, qVec)

        let bufSize = Int(tq_compressed_size(UInt32(dim)))
        var kCompressed = [UInt8](repeating: 0, count: bufSize)
        let written = tq_compress_kv(kVec, UInt32(dim), &kCompressed, &desc)
        #expect(written > 0, "tq_compress_kv returned 0 bytes")

        // tq_compressed_dot takes query in ORIGINAL space (new API)
        let score = tq_compressed_dot(qVec, UInt32(dim), kCompressed, &desc)

        let absRef = abs(ref)
        if absRef > 1e-4 {
            let relErr = abs(score - ref) / absRef
            #expect(relErr < 0.8, "Relative error \(relErr) exceeds 0.8 (ref=\(ref), got=\(score))")
        } else {
            let absErr = abs(score - ref)
            #expect(absErr < 0.5, "Absolute error \(absErr) exceeds 0.5 for near-zero ref")
        }
    }

    // MARK: - Compression ratio

    @Test("Compressed size < bf16 size for dim=128")
    func compressionRatio() {
        let kVec = makeDeterministicVector(dim: dim, seed: 0x1234_5678_9ABC)
        var desc = makeDesc()

        let bufSize = Int(tq_compressed_size(UInt32(dim)))
        var kCompressed = [UInt8](repeating: 0, count: bufSize)
        let written = Int(tq_compress_kv(kVec, UInt32(dim), &kCompressed, &desc))

        let bf16Size = dim * 2  // 2 bytes per element
        #expect(written > 0, "No bytes written")
        #expect(written < bf16Size,
                "Compressed size \(written)B not smaller than bf16 size \(bf16Size)B")
    }

    // MARK: - Norm preservation check

    @Test("Stored norm matches L2 norm of input")
    func storedNormIsCorrect() {
        let v = makeDeterministicVector(dim: dim, seed: 0xABCD_EF01)
        var desc = makeDesc()

        let bufSize = Int(tq_compressed_size(UInt32(dim)))
        var buf = [UInt8](repeating: 0, count: bufSize)
        tq_compress_kv(v, UInt32(dim), &buf, &desc)

        // First 4 bytes are the norm (float32)
        let storedNorm = buf.withUnsafeBytes { $0.load(as: Float.self) }
        let trueNorm = sqrt(v.reduce(0.0) { $0 + $1 * $1 })

        #expect(abs(storedNorm - trueNorm) < 1e-4,
                "Stored norm \(storedNorm) vs true norm \(trueNorm)")
    }
}
