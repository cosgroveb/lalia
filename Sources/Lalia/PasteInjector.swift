import AppKit
import ApplicationServices
import Foundation

struct PasteboardRepresentation: Equatable { let type: NSPasteboard.PasteboardType; let data: Data }
struct PasteboardSnapshot: Equatable {
    let items: [[PasteboardRepresentation]]
    init(_ pasteboard: NSPasteboard) { items = pasteboard.pasteboardItems?.map { item in item.types.compactMap { type in item.data(forType: type).map { PasteboardRepresentation(type: type, data: $0) } } } ?? [] }
    func restore(to pasteboard: NSPasteboard) throws {
        pasteboard.clearContents()
        guard !items.isEmpty else { return }
        let restoredItems = try items.map { representations in
            let item = NSPasteboardItem()
            for representation in representations {
                guard item.setData(representation.data, forType: representation.type) else { throw PasteboardError.restoreFailed }
            }
            return item
        }
        let restored = pasteboard.writeObjects(restoredItems)
        guard restored else { throw PasteboardError.restoreFailed }
    }
}

enum PasteboardError: LocalizedError { case writeFailed, restoreFailed
    var errorDescription: String? {
        switch self {
        case .writeFailed: "Could not write the transcript to the clipboard."
        case .restoreFailed: "Could not restore the clipboard."
        }
    }
}

@MainActor final class PasteInjector: Injecting {
    func inject(_ text: String) async throws {
        guard !text.isEmpty else { return }
        let pasteboard = NSPasteboard.general
        let snapshot = PasteboardSnapshot(pasteboard)
        do {
            pasteboard.clearContents()
            guard pasteboard.setString(text, forType: .string) else { throw PasteboardError.writeFailed }
            let source = CGEventSource(stateID: .hidSystemState)
            guard let down = CGEvent(keyboardEventSource: source, virtualKey: 9, keyDown: true), let up = CGEvent(keyboardEventSource: source, virtualKey: 9, keyDown: false) else { throw CocoaError(.coderInvalidValue) }
            down.flags = .maskCommand; up.flags = .maskCommand; down.post(tap: .cghidEventTap); up.post(tap: .cghidEventTap)
            try await Task.sleep(for: .milliseconds(150))
        } catch {
            try snapshot.restore(to: pasteboard)
            throw error
        }
        try snapshot.restore(to: pasteboard)
    }
    func copy(_ text: String) throws {
        let pasteboard = NSPasteboard.general
        let snapshot = PasteboardSnapshot(pasteboard)
        pasteboard.clearContents()
        guard !text.isEmpty, pasteboard.setString(text, forType: .string) else {
            try snapshot.restore(to: pasteboard)
            throw PasteboardError.writeFailed
        }
    }
}
