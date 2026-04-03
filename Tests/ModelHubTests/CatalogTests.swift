import Testing
@testable import ModelHub

@Suite("ModelCatalog Tests")
struct CatalogTests {

    @Test("Bundled catalog has entries")
    func bundledCatalog() async {
        let entries = await ModelCatalog.shared.entries
        #expect(entries.count >= 2)
        #expect(entries[0].id == "gemma4-26b-a4b-q4")
        #expect(entries[0].files.count > 0)
        #expect(entries[0].totalSizeGB > 10)
    }

    @Test("Each entry has required config files")
    func validFiles() async {
        for entry in await ModelCatalog.shared.entries {
            #expect(!entry.files.isEmpty)
            #expect(entry.files.contains { $0.filename == "config.json" })
            #expect(entry.files.contains { $0.filename == "model_weights.bin" })
        }
    }

    @Test("Lookup entry by id")
    func lookupById() async {
        let entry = await ModelCatalog.shared.entry(id: "gemma4-26b-a4b-q4")
        #expect(entry != nil)
        #expect(entry?.displayName.contains("26B") == true)
    }

    @Test("Missing id returns nil")
    func lookupMissing() async {
        let entry = await ModelCatalog.shared.entry(id: "does-not-exist")
        #expect(entry == nil)
    }
}
