import Carbon
import Foundation

func dispatchHotkeyEvent(_ kind: UInt32, pressed: () -> Void, released: () -> Void) {
    switch kind {
    case UInt32(kEventHotKeyPressed): pressed()
    case UInt32(kEventHotKeyReleased): released()
    default: break
    }
}

final class CarbonHotkey: @unchecked Sendable {
    private var hotKey: EventHotKeyRef?
    private var handler: EventHandlerRef?
    private let pressed: @MainActor () -> Void
    private let released: @MainActor () -> Void
    init(pressed: @escaping @MainActor () -> Void, released: @escaping @MainActor () -> Void) { self.pressed = pressed; self.released = released }
    func register() throws {
        var types = [EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed)), EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyReleased))]
        let status = InstallEventHandler(GetApplicationEventTarget(), { _, event, userData in
            guard let event, let userData else { return OSStatus(eventNotHandledErr) }
            let owner = Unmanaged<CarbonHotkey>.fromOpaque(userData).takeUnretainedValue()
            let kind = GetEventKind(event)
            Task { @MainActor in
                dispatchHotkeyEvent(kind, pressed: owner.pressed, released: owner.released)
            }
            return noErr
        }, 2, &types, Unmanaged.passUnretained(self).toOpaque(), &handler)
        guard status == noErr else { throw NSError(domain: NSOSStatusErrorDomain, code: Int(status)) }
        let id = EventHotKeyID(signature: OSType(0x4C414C49), id: 1)
        let registration = RegisterEventHotKey(UInt32(kVK_ANSI_D), UInt32(cmdKey | shiftKey), id, GetApplicationEventTarget(), 0, &hotKey)
        guard registration == noErr else { throw NSError(domain: NSOSStatusErrorDomain, code: Int(registration)) }
    }
    deinit { if let hotKey { UnregisterEventHotKey(hotKey) }; if let handler { RemoveEventHandler(handler) } }
}
