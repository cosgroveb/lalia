import CoreGraphics
import Carbon
import IOKit.hidsystem
import Testing
@testable import Lalia

@MainActor struct OptionGestureMonitorTests {
    @Test func observesAllChordAndLifecycleInputs() {
        #expect(OptionGestureMonitor.observedEventTypes == [
            .flagsChanged, .keyDown, .keyUp, .leftMouseDown, .rightMouseDown, .otherMouseDown,
            .leftMouseUp, .rightMouseUp, .otherMouseUp, .scrollWheel
        ])
    }

    @Test func healthFailsClosedForPermissionSecureInputAndDisabledTap() {
        #expect(OptionGestureMonitor.healthFailure(permission: false, secureInput: false, tapEnabled: true) != nil)
        #expect(OptionGestureMonitor.healthFailure(permission: true, secureInput: true, tapEnabled: true) != nil)
        #expect(OptionGestureMonitor.healthFailure(permission: true, secureInput: false, tapEnabled: false) != nil)
        #expect(OptionGestureMonitor.healthFailure(permission: true, secureInput: false, tapEnabled: true) == nil)
    }

    @Test func physicalStateMismatchResynchronizesWithoutDisablingListener() {
        var listener = OptionGestureListenerState()
        _ = listener.enable(options: [], otherKeys: [], at: 0)
        #expect(listener.validatePhysicalState(options: [.left], otherKeys: [], at: 0.1) == nil)

        _ = listener.setCaptureState(available: false, active: true)
        let mismatch = listener.validatePhysicalState(options: [], otherKeys: [0], at: 1.3)
        #expect(mismatch?.availability == nil)
        #expect(listener.enabled)
        #expect(listener.trackedOptions.isEmpty)
        #expect(listener.trackedOtherKeys == [0])
        #expect(listener.receive(.otherKey(0, isDown: false, repeatEvent: false, at: 1)).isEmpty)
        #expect(listener.validatePhysicalState(options: [], otherKeys: [], at: 1.4)?.availability == nil)
        #expect(listener.enabled)
    }

    @Test func blockedOrdinaryTypingIsNotPendingGesture() {
        var listener = OptionGestureListenerState()
        _ = listener.enable(options: [], otherKeys: [], at: 0)
        _ = listener.receive(.otherKey(0, isDown: true, repeatEvent: false, at: 1))
        #expect(!listener.hasPendingGesture)
        #expect(listener.validatePhysicalState(options: [], otherKeys: [0], at: 1.1) == nil)
        #expect(listener.enabled)
    }

    @Test func idleLostKeyUpResynchronizesAndGesturesWorkAgain() {
        var listener = OptionGestureListenerState()
        _ = listener.enable(options: [], otherKeys: [], at: 0)
        _ = listener.receive(.otherKey(0, isDown: true, repeatEvent: false, at: 1))
        #expect(listener.validatePhysicalState(options: [], otherKeys: [], at: 1.1)?.availability == nil)
        #expect(listener.trackedOtherKeys.isEmpty)

        _ = listener.receive(.option(.left, isDown: true, at: 2))
        _ = listener.receive(.option(.left, isDown: false, at: 2.1))
        _ = listener.receive(.option(.left, isDown: true, at: 2.2))
        guard case .toggleRecording? = listener.receive(.option(.left, isDown: false, at: 2.3)).first else { Issue.record("Expected double tap after resync"); return }
    }

    @Test func idlePhysicalSnapshotIgnoresLateNativeKeyUpAndAllowsFreshHold() throws {
        var listener = OptionGestureListenerState()
        _ = listener.enable(options: [], otherKeys: [], at: 0)
        let down = try keyboardEvent(0, flags: 0)
        let up = try keyboardEvent(0, flags: 0)
        let downInput = OptionGestureMonitor.normalizedInputs(for: down, type: .keyDown)[0]
        let upInput = OptionGestureMonitor.normalizedInputs(for: up, type: .keyUp)[0]
        guard case let .otherKey(_, _, _, downAt) = downInput,
              case let .otherKey(_, _, _, upAt) = upInput else { Issue.record("Expected key events"); return }
        _ = listener.receive(downInput)
        let snapshotAt = max(downAt, upAt) + 0.01
        #expect(listener.validatePhysicalState(options: [], otherKeys: [], at: snapshotAt)?.availability == nil)
        #expect(listener.receive(upInput).isEmpty)

        _ = listener.receive(.option(.left, isDown: true, at: snapshotAt + 0.1))
        guard case let .beginDictation(id)? = listener.receive(.tick(snapshotAt + 0.35)).first else { Issue.record("Expected fresh hold"); return }
        #expect(listener.receive(.option(.left, isDown: false, at: snapshotAt + 0.4)) == [.finishDictation(id)])
    }

    @Test func mismatchCancelsGestureButPreservesCaptureAvailability() {
        var listener = OptionGestureListenerState()
        _ = listener.enable(options: [], otherKeys: [], at: 0)
        _ = listener.setCaptureState(available: true, active: true)
        _ = listener.receive(.option(.left, isDown: true, at: 1))
        guard case let .beginDictation(id)? = listener.receive(.tick(1.25)).first else { Issue.record("Expected active gesture"); return }
        let mismatch = listener.validatePhysicalState(options: [], otherKeys: [0], at: 1.3)
        #expect(mismatch?.actions == [.cancelDictation(id)])
        #expect(listener.validatePhysicalState(options: [], otherKeys: [], at: 1.4)?.availability == nil)
        #expect(listener.enabled)
        #expect(listener.receive(.otherKey(0, isDown: false, repeatEvent: false, at: 2)).isEmpty)
        _ = listener.receive(.option(.left, isDown: true, at: 3))
        guard case .beginDictation? = listener.receive(.tick(3.25)).first else { Issue.record("Expected gesture after rearm") ; return }
    }

    @Test func ordinaryTypingAndOptionChordDoNotStopActiveSession() {
        var listener = OptionGestureListenerState()
        _ = listener.enable(options: [], otherKeys: [], at: 0)
        _ = listener.setCaptureState(available: false, active: true)
        let inputs: [GestureInput] = [
            .option(.left, isDown: true, at: 1),
            .otherKey(12, isDown: true, repeatEvent: false, at: 1.1),
            .otherKey(12, isDown: false, repeatEvent: false, at: 1.2),
            .option(.left, isDown: false, at: 1.3)
        ]
        for input in inputs { #expect(listener.receive(input).isEmpty) }
        #expect(listener.validatePhysicalState(options: [], otherKeys: [], at: 1.1) == nil)
        #expect(listener.enabled)
    }

    @Test func listenerFailureDisablesActionsAndRequiresExplicitEnableAndAllUp() {
        var listener = OptionGestureListenerState()
        let unavailable = listener.lose("tap creation failed")
        #expect(unavailable.availability == false)
        #expect(listener.receive(.option(.left, isDown: true, at: 0)).isEmpty)

        let enabled = listener.enable(options: [.left], otherKeys: [], at: 1)
        #expect(enabled.availability == true)
        _ = listener.receive(.option(.right, isDown: true, at: 1.1))
        #expect(listener.receive(.tick(1.35)).isEmpty)
        _ = listener.receive(.option(.left, isDown: false, at: 1.4))
        _ = listener.receive(.option(.right, isDown: false, at: 1.5))
        _ = listener.receive(.option(.right, isDown: true, at: 1.6))
        guard case .beginDictation(_)? = listener.receive(.tick(1.85)).first else { Issue.record("Expected action after all-up rearm"); return }
    }

    @Test func permissionSecureInputSleepWakeAndStateMismatchShareFailClosedTransition() {
        for reason in ["Input Monitoring permission is unavailable.", "Secure Event Input is active.", "sleep/wake", "physical state mismatch", "tap disabled"] {
            var listener = OptionGestureListenerState()
            _ = listener.enable(options: [], otherKeys: [], at: 0)
            _ = listener.receive(.option(.left, isDown: true, at: 1))
            guard case let .beginDictation(id)? = listener.receive(.tick(1.25)).first else { Issue.record("Expected active capture"); return }
            let lost = listener.lose(reason)
            #expect(lost.availability == false)
            #expect(lost.actions == [.cancelDictation(id)])
            #expect(listener.receive(.option(.left, isDown: false, at: 1.3)).isEmpty)
        }
    }

    @Test func normalizesSideSpecificOptionFlagsAndDrivesRecognizer() throws {
        let down = try keyboardEvent(kVK_Option, flags: UInt64(NX_DEVICELALTKEYMASK))
        let inputs = OptionGestureMonitor.normalizedInputs(for: down, type: .flagsChanged, at: 1)
        #expect(inputs == [.option(.left, isDown: true, at: 1)])
        #expect(OptionGestureMonitor.normalizedInputs(for: down, type: .flagsChanged) == [
            .option(.left, isDown: true, at: Double(down.timestamp) / 1_000_000_000)
        ])
        let rightDown = try keyboardEvent(kVK_RightOption, flags: UInt64(NX_DEVICERALTKEYMASK))
        #expect(OptionGestureMonitor.normalizedInputs(for: rightDown, type: .flagsChanged, at: 2) == [.option(.right, isDown: true, at: 2)])
        let leftUpWithRightHeld = try keyboardEvent(kVK_Option, flags: UInt64(NX_DEVICERALTKEYMASK))
        #expect(OptionGestureMonitor.normalizedInputs(for: leftUpWithRightHeld, type: .flagsChanged, at: 2.5) == [.option(.left, isDown: false, at: 2.5)])
        let prelatchedCaps = try keyboardEvent(kVK_Option, flags: UInt64(NX_DEVICELALTKEYMASK) | CGEventFlags.maskAlphaShift.rawValue)
        let prelatchedInput = OptionGestureMonitor.normalizedInputs(for: prelatchedCaps, type: .flagsChanged, at: 3)
        #expect(prelatchedInput == [.option(.left, isDown: true, at: 3)])
        var prelatchedRecognizer = OptionGestureRecognizer()
        for input in prelatchedInput { _ = prelatchedRecognizer.consume(input) }
        #expect(prelatchedRecognizer.consume(.tick(3.25)).count == 1)
        var recognizer = OptionGestureRecognizer()
        for input in inputs { _ = recognizer.consume(input) }
        guard case let .beginDictation(id)? = recognizer.consume(.tick(1.25)).first else { Issue.record("Expected hold action"); return }

        let rightUp = try keyboardEvent(kVK_RightOption, flags: 0)
        #expect(OptionGestureMonitor.normalizedInputs(for: rightUp, type: .flagsChanged, at: 1.3) == [.option(.right, isDown: false, at: 1.3)])
        let leftUp = try keyboardEvent(kVK_Option, flags: 0)
        for input in OptionGestureMonitor.normalizedInputs(for: leftUp, type: .flagsChanged, at: 1.4) { #expect(recognizer.consume(input) == [.finishDictation(id)]) }
    }

    @Test func physicalOptionSnapshotMatchesDeviceFlagsAndRightOptionEvent() throws {
        let cases: [(UInt64, Set<OptionSide>)] = [
            (UInt64(NX_DEVICERALTKEYMASK), [.right]),
            (UInt64(NX_DEVICELALTKEYMASK) | UInt64(NX_DEVICERALTKEYMASK), [.left, .right]),
            (0, [])
        ]
        for (flags, expectedSides) in cases {
            #expect(OptionGestureMonitor.optionSides(in: flags) == expectedSides)
            let rightEvent = try keyboardEvent(kVK_RightOption, flags: flags)
            #expect(OptionGestureMonitor.normalizedInputs(for: rightEvent, type: .flagsChanged, at: 1) == [
                .option(.right, isDown: expectedSides.contains(.right), at: 1)
            ])
        }
    }

    @Test func normalizesModifierFamiliesNormalKeysAndRepeats() throws {
        let families: [(Int, Int?, CGEventFlags, UInt16)] = [
            (kVK_Shift, kVK_RightShift, .maskShift, 240),
            (kVK_Control, kVK_RightControl, .maskControl, 241),
            (kVK_Command, kVK_RightCommand, .maskCommand, 242),
            (kVK_Function, nil, .maskSecondaryFn, 243)
        ]
        for (left, right, flag, family) in families {
            let down = try keyboardEvent(left, flags: flag.rawValue)
            #expect(OptionGestureMonitor.normalizedInputs(for: down, type: .flagsChanged, at: 1) == [.otherKey(family, isDown: true, repeatEvent: false, at: 1)])
            if let right {
                let rightDown = try keyboardEvent(right, flags: flag.rawValue)
                #expect(OptionGestureMonitor.normalizedInputs(for: rightDown, type: .flagsChanged, at: 2) == [.otherKey(family, isDown: true, repeatEvent: false, at: 2)])
                let leftUp = try keyboardEvent(left, flags: flag.rawValue)
                #expect(OptionGestureMonitor.normalizedInputs(for: leftUp, type: .flagsChanged, at: 3) == [.otherKey(family, isDown: true, repeatEvent: false, at: 3)])
                let rightUp = try keyboardEvent(right, flags: 0)
                #expect(OptionGestureMonitor.normalizedInputs(for: rightUp, type: .flagsChanged, at: 4) == [.otherKey(family, isDown: false, repeatEvent: false, at: 4)])
            } else {
                let up = try keyboardEvent(left, flags: 0)
                #expect(OptionGestureMonitor.normalizedInputs(for: up, type: .flagsChanged, at: 4) == [.otherKey(family, isDown: false, repeatEvent: false, at: 4)])
            }
        }

        let repeatedKey = try keyboardEvent(0, flags: 0, repeatEvent: true)
        #expect(OptionGestureMonitor.normalizedInputs(for: repeatedKey, type: .keyDown, at: 5) == [.otherKey(0, isDown: true, repeatEvent: true, at: 5)])
        #expect(OptionGestureMonitor.normalizedInputs(for: repeatedKey, type: .keyUp, at: 6) == [.otherKey(0, isDown: false, repeatEvent: true, at: 6)])
    }

    @Test func normalizesMouseScrollAndTransientCapsLockChord() throws {
        let mouse = try keyboardEvent(0, flags: 0)
        mouse.setIntegerValueField(.mouseEventButtonNumber, value: 1)
        #expect(OptionGestureMonitor.normalizedInputs(for: mouse, type: .rightMouseDown, at: 1) == [.mouse(button: 257, isDown: true, at: 1)])
        #expect(OptionGestureMonitor.normalizedInputs(for: mouse, type: .rightMouseUp, at: 2) == [.mouse(button: 257, isDown: false, at: 2)])
        #expect(OptionGestureMonitor.normalizedInputs(for: mouse, type: .scrollWheel, at: 3) == [.mouse(button: nil, isDown: true, at: 3), .mouse(button: nil, isDown: false, at: 3)])

        let option = try keyboardEvent(kVK_Option, flags: UInt64(NX_DEVICELALTKEYMASK))
        var recognizer = OptionGestureRecognizer()
        _ = recognizer.consume(OptionGestureMonitor.normalizedInputs(for: option, type: .flagsChanged, at: 1)[0])
        guard case let .beginDictation(id)? = recognizer.consume(.tick(1.25)).first else { Issue.record("Expected hold action"); return }
        let caps = try keyboardEvent(kVK_CapsLock, flags: CGEventFlags.maskAlphaShift.rawValue)
        let actions = OptionGestureMonitor.normalizedInputs(for: caps, type: .flagsChanged, at: 1.3).flatMap { recognizer.consume($0) }
        #expect(actions == [.cancelDictation(id)])

        let latchedCaps = try keyboardEvent(kVK_CapsLock, flags: CGEventFlags.maskAlphaShift.rawValue)
        #expect(OptionGestureMonitor.normalizedInputs(for: latchedCaps, type: .flagsChanged, at: 4) == [
            .otherKey(UInt16(kVK_CapsLock), isDown: true, repeatEvent: false, at: 4),
            .otherKey(UInt16(kVK_CapsLock), isDown: false, repeatEvent: false, at: 4)
        ])
    }

    @Test func actionDeliveryIsFifoAndInvalidationDropsQueuedActions() async {
        let result = DeliveryTestResult()
        let queue = GestureActionDelivery(isHealthy: { result.healthy }, onUnhealthy: { result.failures += 1 }, deliver: { result.delivered.append($0) })
        let id = UUID()
        queue.send([.beginDictation(id), .finishDictation(id)])
        queue.invalidate()
        await drainMainQueue()
        #expect(result.delivered.isEmpty)
        queue.send([.toggleRecording(id)])
        await drainMainQueue()
        #expect(result.delivered == [.toggleRecording(id)])
        result.healthy = false
        queue.send([.finishDictation(id)])
        await drainMainQueue()
        #expect(result.delivered == [.toggleRecording(id)])
        #expect(result.failures == 1)
    }

    @Test func invalidatingQueuedFinishCancelsAlreadyDeliveredBegin() async {
        let result = DeliveryTestResult()
        let queue = GestureActionDelivery(isHealthy: { result.healthy }, onUnhealthy: {}, deliver: { result.delivered.append($0) })
        var listener = OptionGestureListenerState()
        _ = listener.enable(options: [], otherKeys: [], at: 0)
        _ = listener.setCaptureState(available: true, active: true)
        _ = listener.receive(.option(.left, isDown: true, at: 1))
        let begin = listener.receive(.tick(1.25))
        queue.send(begin)
        await drainMainQueue()
        guard case let .beginDictation(id)? = begin.first else { Issue.record("Expected begin"); return }
        let finish = listener.receive(.option(.left, isDown: false, at: 1.3))
        queue.send(finish)
        let mismatch = listener.validatePhysicalState(options: [], otherKeys: [0], at: 1.4)
        #expect(mismatch?.actions.isEmpty == true)
        queue.invalidate()
        await drainMainQueue()
        #expect(result.delivered == [.beginDictation(id), .cancelDictation(id)])
    }
}

@MainActor private final class DeliveryTestResult {
    var delivered: [GestureAction] = []
    var healthy = true
    var failures = 0
}

private func keyboardEvent(_ keyCode: Int, flags: UInt64, repeatEvent: Bool = false) throws -> CGEvent {
    let event = try #require(CGEvent(keyboardEventSource: nil, virtualKey: CGKeyCode(keyCode), keyDown: true))
    event.setIntegerValueField(.keyboardEventKeycode, value: Int64(keyCode))
    event.setIntegerValueField(.keyboardEventAutorepeat, value: repeatEvent ? 1 : 0)
    event.flags = CGEventFlags(rawValue: flags)
    return event
}

@MainActor private func drainMainQueue() async {
    await withCheckedContinuation { continuation in DispatchQueue.main.async { continuation.resume() } }
}
