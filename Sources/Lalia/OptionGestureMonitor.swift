@preconcurrency import AppKit
@preconcurrency import CoreGraphics
@preconcurrency import ApplicationServices
import Carbon
import Foundation
import IOKit.hidsystem

private let leftOptionDeviceMask = UInt64(NX_DEVICELALTKEYMASK)
private let rightOptionDeviceMask = UInt64(NX_DEVICERALTKEYMASK)
private let modifierShift: UInt16 = 240
private let modifierControl: UInt16 = 241
private let modifierCommand: UInt16 = 242
private let modifierFunction: UInt16 = 243

struct OptionGestureListenerTransition {
    var actions: [GestureAction] = []
    var availability: Bool?
    var message: String?
}

struct OptionGestureListenerState {
    private var recognizer = OptionGestureRecognizer()
    private(set) var enabled = false
    var hasPendingGesture: Bool { recognizer.hasPendingGesture }
    private(set) var captureActive = false
    private var physicalStateTime: TimeInterval?
    private var needsPhysicalStateCheck: Bool {
        hasPendingGesture || captureActive || !trackedOptions.isEmpty || !trackedOtherKeys.isEmpty
    }
    var holdDeadline: TimeInterval? { recognizer.holdDeadline }
    var trackedOptions: Set<OptionSide> { recognizer.trackedOptions }
    var trackedOtherKeys: Set<UInt16> { recognizer.trackedOtherKeys }

    mutating func enable(options: Set<OptionSide>, otherKeys: Set<UInt16>, at: TimeInterval) -> OptionGestureListenerTransition {
        enabled = true
        physicalStateTime = nil
        let actions = recognizer.consume(.seed(options: options, otherKeys: otherKeys, at: at))
            + recognizer.consume(.availability(true))
        return OptionGestureListenerTransition(actions: actions, availability: true)
    }

    mutating func lose(_ message: String) -> OptionGestureListenerTransition {
        enabled = false
        let actions = recognizer.consume(.reset) + recognizer.consume(.availability(false))
        return OptionGestureListenerTransition(actions: actions, availability: false, message: message)
    }

    mutating func receive(_ input: GestureInput) -> [GestureAction] {
        guard enabled, physicalStateTime.map({ (input.timestamp ?? -.infinity) >= $0 }) ?? true else { return [] }
        return recognizer.consume(input)
    }

    mutating func setCaptureState(available: Bool, active: Bool) -> [GestureAction] {
        captureActive = active
        return recognizer.consume(.captureAvailability(available))
    }

    mutating func validatePhysicalState(options: Set<OptionSide>, otherKeys: Set<UInt16>, at: TimeInterval) -> OptionGestureListenerTransition? {
        guard needsPhysicalStateCheck,
              options != trackedOptions || otherKeys != trackedOtherKeys else { return nil }
        let actions = recognizer.consume(.reset) + recognizer.consume(.seed(options: options, otherKeys: otherKeys, at: at))
        physicalStateTime = at
        return OptionGestureListenerTransition(actions: actions)
    }
}

@MainActor final class OptionGestureMonitor {
    static let observedEventTypes: [CGEventType] = [
        .flagsChanged, .keyDown, .keyUp, .leftMouseDown, .rightMouseDown, .otherMouseDown,
        .leftMouseUp, .rightMouseUp, .otherMouseUp, .scrollWheel
    ]

    static func healthFailure(permission: Bool, secureInput: Bool, tapEnabled: Bool) -> String? {
        if !permission { return "Input Monitoring is off." }
        if secureInput { return "Secure input is active." }
        if !tapEnabled { return "Gesture listener stopped." }
        return nil
    }

    var onAction: (@MainActor (GestureAction) -> Void)?
    var onAvailability: (@MainActor (Bool, String?) -> Void)?

    private var tap: CFMachPort?
    private var source: CFRunLoopSource?
    private var timer: Timer?
    private var holdTimer: Timer?
    private var observers: [NSObjectProtocol] = []
    private var listener = OptionGestureListenerState()
    private lazy var actionDelivery = GestureActionDelivery(
        isHealthy: { [weak self] in self?.isListenerHealthy ?? false },
        onUnhealthy: { [weak self] in self?.loseListener("Gestures paused.") },
        deliver: { [weak self] in self?.onAction?($0) }
    )
    private var enabled = false
    private var lastHealthy = true
    private var eventTime: TimeInterval { ProcessInfo.processInfo.systemUptime }
    private var isListenerHealthy: Bool {
        guard enabled, let tap else { return false }
        return Self.healthFailure(permission: CGPreflightListenEventAccess(), secureInput: IsSecureEventInputEnabled(), tapEnabled: CGEvent.tapIsEnabled(tap: tap)) == nil
    }

    static func normalizedInputs(for event: CGEvent, type: CGEventType, at suppliedTime: TimeInterval? = nil) -> [GestureInput] {
        let at = suppliedTime ?? Double(event.timestamp) / 1_000_000_000
        switch type {
        case .flagsChanged:
            let code = UInt16(event.getIntegerValueField(.keyboardEventKeycode))
            if code == UInt16(kVK_Option) { return [.option(.left, isDown: optionSides(in: event.flags.rawValue).contains(.left), at: at)] }
            if code == UInt16(kVK_RightOption) { return [.option(.right, isDown: optionSides(in: event.flags.rawValue).contains(.right), at: at)] }
            if code == UInt16(kVK_CapsLock) {
                return [.otherKey(code, isDown: true, repeatEvent: false, at: at), .otherKey(code, isDown: false, repeatEvent: false, at: at)]
            }
            if let modifier = Self.modifierEvent(code: code, flags: event.flags) {
                return [.otherKey(modifier.code, isDown: modifier.down, repeatEvent: false, at: at)]
            }
            return []
        case .keyDown, .keyUp:
            let code = UInt16(event.getIntegerValueField(.keyboardEventKeycode))
            return [.otherKey(code, isDown: type == .keyDown, repeatEvent: event.getIntegerValueField(.keyboardEventAutorepeat) != 0, at: at)]
        case .leftMouseDown, .rightMouseDown, .otherMouseDown:
            let button = mouseButton(event: event, type: type)
            return [.mouse(button: 256 + button, isDown: true, at: at)]
        case .leftMouseUp, .rightMouseUp, .otherMouseUp:
            let button = mouseButton(event: event, type: type)
            return [.mouse(button: 256 + button, isDown: false, at: at)]
        case .scrollWheel:
            return [.mouse(button: nil, isDown: true, at: at), .mouse(button: nil, isDown: false, at: at)]
        default: return []
        }
    }

    static func optionSides(in flags: UInt64) -> Set<OptionSide> {
        var sides = Set<OptionSide>()
        if flags & leftOptionDeviceMask != 0 { sides.insert(.left) }
        if flags & rightOptionDeviceMask != 0 { sides.insert(.right) }
        return sides
    }

    func start(requestPermission: Bool = false) {
        if requestPermission { _ = CGRequestListenEventAccess() }
        guard CGPreflightListenEventAccess() else {
            setAvailable(false, "Input Monitoring is off."); return
        }
        guard !IsSecureEventInputEnabled() else {
            setAvailable(false, "Secure input is active."); return
        }
        guard tap == nil else { checkHealth(); return }
        let mask = Self.observedEventTypes.reduce(CGEventMask(0)) { $0 | (1 << $1.rawValue) }
        guard let created = CGEvent.tapCreate(tap: .cgSessionEventTap, place: .headInsertEventTap,
                                              options: .listenOnly, eventsOfInterest: CGEventMask(mask),
                                              callback: optionGestureTapCallback,
                                              userInfo: Unmanaged.passUnretained(self).toOpaque()) else {
            setAvailable(false, "Gesture listener unavailable."); return
        }
        tap = created
        source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, created, 0)
        if let source { CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes) }
        CGEvent.tapEnable(tap: created, enable: true)
        enabled = true
        timer = Timer(timeInterval: OptionGestureRecognizer.stateCheckInterval, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.checkHealth() }
        }
        if let timer { RunLoop.main.add(timer, forMode: .common) }
        let center = NSWorkspace.shared.notificationCenter
        for name in [NSWorkspace.willSleepNotification, NSWorkspace.sessionDidResignActiveNotification] {
            observers.append(center.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.loseListener("Gestures paused.") }
            })
        }
        observers.append(center.addObserver(forName: NSWorkspace.didWakeNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.loseListener("Gestures paused.") }
        })
        setAvailable(true, nil)
    }

    func stop() {
        actionDelivery.invalidate()
        tearDown()
        apply(listener.lose("Gesture listening stopped."))
    }

    func setCaptureState(available: Bool, active: Bool) {
        dispatch(listener.setCaptureState(available: available, active: active))
        scheduleHoldTimer()
    }

    private func tearDown() {
        timer?.invalidate(); timer = nil
        holdTimer?.invalidate(); holdTimer = nil
        for observer in observers { NSWorkspace.shared.notificationCenter.removeObserver(observer) }
        observers.removeAll()
        if let source { CFRunLoopRemoveSource(CFRunLoopGetMain(), source, .commonModes) }
        if let tap { CFMachPortInvalidate(tap) }
        source = nil; tap = nil; enabled = false
    }

    fileprivate func receive(_ event: CGEvent, type: CGEventType) -> Unmanaged<CGEvent>? {
        guard enabled, let tap else {
            loseListener("Gesture listener unavailable.")
            return Unmanaged.passUnretained(event)
        }
        if let failure = Self.healthFailure(permission: CGPreflightListenEventAccess(), secureInput: IsSecureEventInputEnabled(), tapEnabled: CGEvent.tapIsEnabled(tap: tap)) {
            loseListener(failure)
            return Unmanaged.passUnretained(event)
        }
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            loseListener("Gesture listener stopped.")
        } else {
            for input in Self.normalizedInputs(for: event, type: type) {
                dispatch(listener.receive(input))
            }
        }
        scheduleHoldTimer()
        return Unmanaged.passUnretained(event)
    }

    private func physicalState() -> (options: Set<OptionSide>, keys: Set<UInt16>) {
        let heldOptions = Self.optionSides(in: CGEventSource.flagsState(.combinedSessionState).rawValue)
        var heldKeys = physicalOtherKeys()
        heldKeys.formUnion((0..<5).compactMap { index in
            guard let button = CGMouseButton(rawValue: UInt32(index)), CGEventSource.buttonState(.combinedSessionState, button: button) else { return nil }
            return UInt16(256 + index)
        })
        return (heldOptions, heldKeys)
    }

    private func checkHealth() {
        guard enabled else { return }
        guard let tap else {
            loseListener("Gesture listener unavailable."); return
        }
        if let failure = Self.healthFailure(permission: CGPreflightListenEventAccess(), secureInput: IsSecureEventInputEnabled(), tapEnabled: CGEvent.tapIsEnabled(tap: tap)) {
            loseListener(failure); return
        }
        let actual = physicalState()
        if let transition = listener.validatePhysicalState(options: actual.options, otherKeys: actual.keys, at: eventTime) {
            actionDelivery.invalidate()
            apply(transition)
        }
    }

    private func scheduleHoldTimer() {
        holdTimer?.invalidate(); holdTimer = nil
        guard let deadline = listener.holdDeadline else { return }
        let delay = max(0, deadline - eventTime)
        let timer = Timer(timeInterval: delay, repeats: false) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.checkHealth()
                guard self.enabled else { return }
                self.dispatch(self.listener.receive(.tick(deadline)))
                self.scheduleHoldTimer()
            }
        }
        holdTimer = timer
        RunLoop.main.add(timer, forMode: .common)
    }

    private func keyDown(_ code: Int64) -> Bool { CGEventSource.keyState(.combinedSessionState, key: CGKeyCode(code)) }

    private func physicalOtherKeys() -> Set<UInt16> {
        let excluded = [kVK_Option, kVK_RightOption, kVK_Shift, kVK_RightShift, kVK_Control, kVK_RightControl,
                        kVK_Command, kVK_RightCommand, kVK_Function, kVK_CapsLock]
        var keys = Set((0..<128).filter { !excluded.contains($0) && keyDown(Int64($0)) }.map(UInt16.init))
        if [kVK_Shift, kVK_RightShift].contains(where: { keyDown(Int64($0)) }) { keys.insert(modifierShift) }
        if [kVK_Control, kVK_RightControl].contains(where: { keyDown(Int64($0)) }) { keys.insert(modifierControl) }
        if [kVK_Command, kVK_RightCommand].contains(where: { keyDown(Int64($0)) }) { keys.insert(modifierCommand) }
        if keyDown(Int64(kVK_Function)) { keys.insert(modifierFunction) }
        return keys
    }

    private static func modifierEvent(code: UInt16, flags: CGEventFlags) -> (code: UInt16, down: Bool)? {
        let family: UInt16
        let mask: UInt64
        switch Int(code) {
        case kVK_Shift, kVK_RightShift: family = modifierShift; mask = CGEventFlags.maskShift.rawValue
        case kVK_Control, kVK_RightControl: family = modifierControl; mask = CGEventFlags.maskControl.rawValue
        case kVK_Command, kVK_RightCommand: family = modifierCommand; mask = CGEventFlags.maskCommand.rawValue
        case kVK_Function: family = modifierFunction; mask = CGEventFlags.maskSecondaryFn.rawValue
        default: return nil
        }
        return (family, flags.rawValue & mask != 0)
    }

    private static func mouseButton(event: CGEvent, type: CGEventType) -> UInt16 {
        switch type {
        case .leftMouseDown, .leftMouseUp: 0
        case .rightMouseDown, .rightMouseUp: 1
        default: UInt16(event.getIntegerValueField(.mouseEventButtonNumber))
        }
    }

    private func loseListener(_ message: String) {
        guard enabled || lastHealthy else { return }
        handleListenerLoss(listener.lose(message))
    }

    private func handleListenerLoss(_ transition: OptionGestureListenerTransition) {
        actionDelivery.invalidate()
        tearDown()
        lastHealthy = false
        apply(transition)
    }

    private func setAvailable(_ value: Bool, _ message: String?) {
        if value {
            let state = physicalState()
            apply(listener.enable(options: state.options, otherKeys: state.keys, at: eventTime))
        } else {
            apply(listener.lose(message ?? "Gesture listener unavailable."))
        }
    }

    private func apply(_ transition: OptionGestureListenerTransition) {
        dispatch(transition.actions)
        guard let available = transition.availability else { return }
        lastHealthy = available
        onAvailability?(available, transition.message)
    }

    private func dispatch(_ actions: [GestureAction]) {
        actionDelivery.send(actions)
    }
}

@MainActor final class GestureActionDelivery {
    private let isHealthy: @MainActor () -> Bool
    private let onUnhealthy: @MainActor () -> Void
    private let deliver: @MainActor (GestureAction) -> Void
    private var generation = UUID()
    private var activeDictationID: UUID?

    init(isHealthy: @escaping @MainActor () -> Bool, onUnhealthy: @escaping @MainActor () -> Void,
         deliver: @escaping @MainActor (GestureAction) -> Void) {
        self.isHealthy = isHealthy; self.onUnhealthy = onUnhealthy; self.deliver = deliver
    }

    func invalidate() {
        generation = UUID()
        guard let id = activeDictationID else { return }
        activeDictationID = nil
        deliver(.cancelDictation(id))
    }

    func send(_ actions: [GestureAction]) {
        let generation = generation
        for action in actions {
            DispatchQueue.main.async { [weak self] in
                MainActor.assumeIsolated {
                    guard let self, self.generation == generation else { return }
                    if case let .beginDictation(id) = action {
                        guard self.isHealthy() else {
                            self.invalidate()
                            self.onUnhealthy()
                            return
                        }
                        self.activeDictationID = id
                        self.deliver(action)
                    } else if case let .cancelDictation(id) = action {
                        guard self.activeDictationID == id else { return }
                        self.activeDictationID = nil
                        self.deliver(action)
                    } else if case let .finishDictation(id) = action {
                        guard self.isHealthy() else {
                            self.invalidate()
                            self.onUnhealthy()
                            return
                        }
                        guard self.activeDictationID == id else { return }
                        self.activeDictationID = nil
                        self.deliver(action)
                    } else if self.isHealthy() {
                        self.deliver(action)
                    } else {
                        self.invalidate()
                        self.onUnhealthy()
                    }
                }
            }
        }
    }
}

private func optionGestureTapCallback(proxy: CGEventTapProxy, type: CGEventType, event: CGEvent, userInfo: UnsafeMutableRawPointer?) -> Unmanaged<CGEvent>? {
    guard let userInfo else { return Unmanaged.passUnretained(event) }
    let monitor = Unmanaged<OptionGestureMonitor>.fromOpaque(userInfo).takeUnretainedValue()
    return MainActor.assumeIsolated { monitor.receive(event, type: type) }
}
