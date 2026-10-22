import Carbon
import Foundation

enum Shortcut: UInt32 {
    case dictation = 1
    case recording = 2

    var keyCode: UInt32 { UInt32(self == .dictation ? kVK_ANSI_D : kVK_ANSI_R) }
    var label: String { self == .dictation ? "⇧⌘D" : "⇧⌘R" }
}

final class CarbonHotkey: @unchecked Sendable {
    let shortcut: Shortcut
    private var hotKey: EventHotKeyRef?
    private var handler: EventHandlerRef?
    private let pressed: @MainActor () -> Void
    private let released: @MainActor () -> Void
    @MainActor private var isPressed = false
    private static let signature: OSType = 0x4C414C49

    init(shortcut: Shortcut, pressed: @escaping @MainActor () -> Void, released: @escaping @MainActor () -> Void = {}) {
        self.shortcut = shortcut; self.pressed = pressed; self.released = released
    }
    func matches(_ id: EventHotKeyID) -> Bool { id.signature == Self.signature && id.id == shortcut.rawValue }
    @MainActor func dispatch(_ kind: UInt32, id: EventHotKeyID) {
        guard matches(id) else { return }
        switch kind {
        case UInt32(kEventHotKeyPressed) where !isPressed:
            isPressed = true; pressed()
        case UInt32(kEventHotKeyReleased) where isPressed:
            isPressed = false; released()
        default: break
        }
    }
    func register() throws {
        var types = [EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed)), EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyReleased))]
        let status = InstallEventHandler(GetApplicationEventTarget(), { _, event, userData in
            guard let event, let userData else { return OSStatus(eventNotHandledErr) }
            let owner = Unmanaged<CarbonHotkey>.fromOpaque(userData).takeUnretainedValue()
            var id = EventHotKeyID()
            guard GetEventParameter(event, UInt32(kEventParamDirectObject), UInt32(typeEventHotKeyID), nil,
                                    MemoryLayout<EventHotKeyID>.size, nil, &id) == noErr,
                  owner.matches(id) else { return OSStatus(eventNotHandledErr) }
            let kind = GetEventKind(event)
            let hotkeyID = id
            Task { @MainActor in
                owner.dispatch(kind, id: hotkeyID)
            }
            return noErr
        }, 2, &types, Unmanaged.passUnretained(self).toOpaque(), &handler)
        guard status == noErr else { throw NSError(domain: NSOSStatusErrorDomain, code: Int(status)) }
        let id = EventHotKeyID(signature: Self.signature, id: shortcut.rawValue)
        let registration = RegisterEventHotKey(shortcut.keyCode, UInt32(cmdKey | shiftKey), id, GetApplicationEventTarget(), 0, &hotKey)
        guard registration == noErr else {
            if let handler { RemoveEventHandler(handler); self.handler = nil }
            throw NSError(domain: NSOSStatusErrorDomain, code: Int(registration))
        }
    }
    deinit { if let hotKey { UnregisterEventHotKey(hotKey) }; if let handler { RemoveEventHandler(handler) } }
}
