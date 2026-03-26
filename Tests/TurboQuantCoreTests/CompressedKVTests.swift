import Testing
import Foundation
@testable import TurboQuantCore

// Helpers shared by these tests
private func makeDeterministicVector(dim: Int, seed: UInt64) -> [Float] {
    var state = seed
    return (0..<dim).map { _ -> Float in
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        z = z ^ (z >> 31)
        // Map u64 → [-1, 1]
        return Float(Int64(bitPattern: z)) / Float(Int64.max)
    }
}

private func bf16DotProduct(_ a: [Float], _ b: [Float]) -> Float {
    // bf16 truncates the lower 16 bits of a Float's mantissa
    func toBF16(_ f: Float) -> Float {
        let bits = f.bitPattern & 0xFFFF0000
        return Float(bitPattern: bits)
    }
    return zip(a, b).reduce(0.0) { acc, pair in
        acc + toBF16(pair.0) * toBF16(pair.1)
    }
}

// Default descriptor used across tests
private func makeDesc() -> PA_QuantizedKVDesc {
    PA_QuantizedKVDesc(
        key_bits_x2: 8,          // 4-bit
        value_bits_x2: 8,
        block_size: 32,
        transform_kind: UInt32(PA_TRANSFORM_STRUCTURED_ROTATION.rawValue),
        transform_seed: 42,
        residual_bits: 1,
        main_codes_offset: 0,
        aux_params_offset: 0,
        qjl_bits_offset: 0,
        aux_params_kind: 0
    )
}

@Suite("Compressed KV Operations")
struct CompressedKVTests {

    let dim: Int = 256

    // ── Test 1: Compressed dot accuracy ────────────────────────────────────────

    @Test("Compressed dot product relative error < 0.5 vs bf16 reference")
    func compressedDotAccuracy() throws {
        let kVec = makeDeterministicVector(dim: dim, seed: 0xCAFE_BABE_0001)
        let qVec = makeDeterministicVector(dim: dim, seed: 0xCAFE_BABE_0002)

        var desc = makeDesc()

        // Reference: bf16 dot product (uncompressed)
        let ref = bf16DotProduct(kVec, qVec)

        // Compress K
        let bufSize = dim + 2 * (Int((UInt32(dim) + desc.block_size - 1) / desc.block_size) * 4) + (dim + 7) / 8 + 64
        var kCompressed = [UInt8](repeating: 0, count: bufSize)
        let written = tq_compress_kv(kVec, UInt32(dim), &kCompressed, &desc)
        #expect(written > 0, "tq_compress_kv returned 0 bytes")

        // Rotate Q (so it matches the rotated-K space)
        var qTransformed = [Float](repeating: 0, count: dim)
        tq_rotate_query(qVec, &qTransformed, UInt32(dim), desc.transform_seed)

        // Compressed dot
        let score = tq_compressed_dot(qTransformed, UInt32(dim), kCompressed, &desc)

        // Relative error check (guard against near-zero reference)
        let absRef = abs(ref)
        if absRef > 1e-4 {
            let relErr = abs(score - ref) / absRef
            #expect(relErr < 0.5, "Relative error \(relErr) exceeds 0.5 (ref=\(ref), got=\(score))")
        } else {
            let absErr = abs(score - ref)
            #expect(absErr < 0.5, "Absolute error \(absErr) exceeds 0.5 for near-zero ref")
        }
    }

    // ── Test 2: V decompress roundtrip ─────────────────────────────────────────

    @Test("V decompress roundtrip max absolute error < 0.5")
    func vDecompressRoundtrip() throws {
        let vVec = makeDeterministicVector(dim: dim, seed: 0xDEAD_BEEF_0003)

        var desc = makeDesc()

        let nb = (dim + Int(desc.block_size) - 1) / Int(desc.block_size)
        let bufSize = dim + nb * 8 + (dim + 7) / 8 + 64
        var vCompressed = [UInt8](repeating: 0, count: bufSize)
        let written = tq_compress_kv(vVec, UInt32(dim), &vCompressed, &desc)
        #expect(written > 0, "tq_compress_kv returned 0 bytes for V")

        var vOut = [Float](repeating: 0, count: dim)
        tq_decompress_v_tile(vCompressed, &vOut, UInt32(dim), &desc)

        let maxErr = zip(vVec, vOut).map { abs($0.0 - $0.1) }.max() ?? 0
        #expect(maxErr < 0.5, "V roundtrip max error \(maxErr) exceeds 0.5")
    }

    // ── Test 3: Compression ratio ───────────────────────────────────────────────

    @Test("Compressed size smaller than bf16 size")
    func compressionRatio() throws {
        let kVec = makeDeterministicVector(dim: dim, seed: 0x1234_5678_9ABC)

        var desc = makeDesc()

        let nb = (dim + Int(desc.block_size) - 1) / Int(desc.block_size)
        let bufSize = dim + nb * 8 + (dim + 7) / 8 + 64
        var kCompressed = [UInt8](repeating: 0, count: bufSize)
        let written = Int(tq_compress_kv(kVec, UInt32(dim), &kCompressed, &desc))

        let bf16Size = dim * 2  // bf16 = 2 bytes per element

        #expect(written > 0, "No bytes written")
        #expect(written < bf16Size,
                "Compressed size \(written)B not smaller than bf16 size \(bf16Size)B")
    }
}
