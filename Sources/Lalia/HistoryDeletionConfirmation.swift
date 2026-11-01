import AppKit

@MainActor enum HistoryDeletionConfirmation {
    static func makeAlert(count: Int) -> NSAlert {
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = count == 1 ? "Delete this transcript?" : "Delete \(count) transcripts?"
        alert.informativeText = count == 1
            ? "Its audio and generated notes will also be permanently deleted. This can’t be undone."
            : "Their audio and generated notes will also be permanently deleted. This can’t be undone."
        alert.addButton(withTitle: "Cancel").keyEquivalent = "\u{1b}"
        alert.addButton(withTitle: "Delete").keyEquivalent = ""
        alert.window.defaultButtonCell = nil
        return alert
    }
}
