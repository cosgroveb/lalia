import Carbon
import Testing
@testable import Lalia

struct CarbonHotkeyTests {
    @Test func dispatchesOnlyMatchingHotkeyEvents() {
        var pressed = 0
        var released = 0

        dispatchHotkeyEvent(UInt32(kEventHotKeyPressed), pressed: { pressed += 1 }, released: { released += 1 })
        dispatchHotkeyEvent(UInt32(kEventHotKeyReleased), pressed: { pressed += 1 }, released: { released += 1 })
        dispatchHotkeyEvent(UInt32(kEventClassKeyboard), pressed: { pressed += 1 }, released: { released += 1 })

        #expect(pressed == 1)
        #expect(released == 1)
    }
}
