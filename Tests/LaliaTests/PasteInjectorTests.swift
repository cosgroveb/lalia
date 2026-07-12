import AppKit
import Foundation
import Testing
@testable import Lalia

struct PasteInjectorTests {
    @Test @MainActor func snapshotsAndRestoresAllReadableRepresentationsInOrder() {
        _ = NSApplication.shared
        let pasteboard = NSPasteboard.withUniqueName()
        let first = NSPasteboardItem()
        let second = NSPasteboardItem()
        let firstTypes: [(NSPasteboard.PasteboardType, Data)] = [
            (.string, Data("first string".utf8)),
            (.rtf, Data([0x00, 0xFF])),
        ]
        let secondTypes: [(NSPasteboard.PasteboardType, Data)] = [
            (.string, Data("second string".utf8)),
            (.rtf, Data([0x01, 0x02, 0x03])),
        ]
        firstTypes.forEach { first.setData($0.1, forType: $0.0) }
        secondTypes.forEach { second.setData($0.1, forType: $0.0) }
        pasteboard.clearContents()
        #expect(pasteboard.writeObjects([first, second]))
        #expect(pasteboard.pasteboardItems?.count == 2)
        let expected = pasteboard.pasteboardItems?.map { item in
            item.types.compactMap { type in
                item.data(forType: type).map { (type, $0) }
            }
        } ?? []
        #expect(expected.allSatisfy { $0.count >= 2 })

        let snapshot = PasteboardSnapshot(pasteboard)
        pasteboard.clearContents()
        pasteboard.setData(Data("replacement".utf8), forType: .string)
        #expect(throws: Never.self) { try snapshot.restore(to: pasteboard) }

        let restored = pasteboard.pasteboardItems ?? []
        #expect(restored.count == 2)
        for (item, representations) in zip(restored, expected) {
            let restoredTypes = item.types.filter { type in
                representations.contains { $0.0 == type }
            }
            #expect(restoredTypes == representations.map(\.0))
            for (type, data) in representations {
                #expect(item.data(forType: type) == data)
            }
        }
    }
}
