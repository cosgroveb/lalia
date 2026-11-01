import AppKit
import Testing
@testable import Lalia

@Suite(.serialized) @MainActor struct HistoryDeletionConfirmationTests {
    @Test func confirmationUsesExactCopyAndKeys() {
        _ = NSApplication.shared
        for count in [1, 3] {
            let alert = HistoryDeletionConfirmation.makeAlert(count: count)
            #expect(alert.messageText == (count == 1 ? "Delete this transcript?" : "Delete 3 transcripts?"))
            #expect(alert.informativeText == (count == 1
                ? "Its audio and generated notes will also be permanently deleted. This can’t be undone."
                : "Their audio and generated notes will also be permanently deleted. This can’t be undone."))
            #expect(alert.buttons.map(\.title) == ["Cancel", "Delete"])
            #expect(alert.buttons.map(\.keyEquivalent) == ["\u{1b}", ""])
            #expect(alert.buttons.allSatisfy { $0.keyEquivalent != "\r" })
        }
    }
}
