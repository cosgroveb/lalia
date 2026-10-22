import Carbon
import Testing
@testable import Lalia

@MainActor struct CarbonHotkeyTests {
    @Test func routesShortcutsIndependentlyAndIgnoresOtherEvents() {
        var dictated = 0
        var released = 0
        var toggled = 0
        let dictation = CarbonHotkey(shortcut: .dictation, pressed: { dictated += 1 }, released: { released += 1 })
        let recording = CarbonHotkey(shortcut: .recording, pressed: { toggled += 1 })
        let dictationID = EventHotKeyID(signature: 0x4C414C49, id: 1)
        let recordingID = EventHotKeyID(signature: 0x4C414C49, id: 2)
        for hotkey in [dictation, recording] {
            #expect(!hotkey.matches(EventHotKeyID(signature: 0, id: hotkey.shortcut.rawValue)))
            hotkey.dispatch(UInt32(kEventHotKeyPressed), id: EventHotKeyID(signature: 0, id: hotkey.shortcut.rawValue))
            hotkey.dispatch(UInt32(kEventClassKeyboard), id: dictationID)
            hotkey.dispatch(UInt32(kEventHotKeyPressed), id: dictationID)
            hotkey.dispatch(UInt32(kEventHotKeyReleased), id: dictationID)
        }
        #expect(dictated == 1)
        #expect(released == 1)
        #expect(toggled == 0)
        for hotkey in [dictation, recording] { hotkey.dispatch(UInt32(kEventHotKeyPressed), id: recordingID) }
        #expect(dictated == 1)
        #expect(released == 1)
        #expect(toggled == 1)
    }

    @Test func holdingShortcutTogglesOnceUntilReleased() {
        var toggled = 0
        let hotkey = CarbonHotkey(shortcut: .recording, pressed: { toggled += 1 })
        let id = EventHotKeyID(signature: 0x4C414C49, id: 2)
        hotkey.dispatch(UInt32(kEventHotKeyReleased), id: id)
        hotkey.dispatch(UInt32(kEventHotKeyPressed), id: id)
        hotkey.dispatch(UInt32(kEventHotKeyPressed), id: id)
        #expect(toggled == 1)
        hotkey.dispatch(UInt32(kEventHotKeyReleased), id: id)
        hotkey.dispatch(UInt32(kEventHotKeyReleased), id: id)
        #expect(toggled == 1)
        hotkey.dispatch(UInt32(kEventHotKeyPressed), id: id)
        #expect(toggled == 2)
    }
}
