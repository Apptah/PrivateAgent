import Testing
@testable import TurboQuantCore

@Suite("TurboQuant Types Tests")
struct TQTypesTests {

    @Test("TurboQuantCore target compiles and links FlashMoECore")
    func targetCompiles() {
        // If this test runs, the target graph is correct
        #expect(true)
    }
}
