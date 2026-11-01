import AppKit

struct SessionLifecycleState {
    enum Event { case sleep, wake, lock, unlock, resign, activate }
    private var awake = true
    private var unlocked = true
    private var active = true
    var usable: Bool { awake && unlocked && active }

    mutating func receive(_ event: Event) -> SessionInterruption? {
        switch event {
        case .sleep:
            guard awake else { return nil }
            awake = false; return .sleep
        case .wake: awake = true
        case .lock:
            guard unlocked else { return nil }
            unlocked = false; return .lock
        case .unlock: unlocked = true
        case .resign:
            guard active else { return nil }
            active = false; return .userSwitch
        case .activate: active = true
        }
        return nil
    }
}

@MainActor final class SessionLifecycleMonitor {
    var onInterruption: ((SessionInterruption) -> Void)?
    var onUsabilityChanged: ((Bool) -> Void)?
    private var state = SessionLifecycleState()
    private var observing = false
    private var workspaceObservers: [NSObjectProtocol] = []
    private var lockObservers: [NSObjectProtocol] = []

    func start() {
        guard !observing else { return }
        observing = true
        for (name, event) in [
            (NSWorkspace.willSleepNotification, SessionLifecycleState.Event.sleep),
            (NSWorkspace.didWakeNotification, .wake),
            (NSWorkspace.sessionDidResignActiveNotification, .resign),
            (NSWorkspace.sessionDidBecomeActiveNotification, .activate)
        ] {
            workspaceObservers.append(NSWorkspace.shared.notificationCenter.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.receive(event) }
            })
        }
        // macOS has no documented screen-lock notification. Keep these established
        // distributed names separate from display sleep and Secure Input.
        for (name, event) in [
            ("com.apple.screenIsLocked", SessionLifecycleState.Event.lock),
            ("com.apple.screenIsUnlocked", .unlock)
        ] {
            lockObservers.append(DistributedNotificationCenter.default().addObserver(forName: Notification.Name(name), object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.receive(event) }
            })
        }
    }

    func receive(_ event: SessionLifecycleState.Event) {
        guard observing else { return }
        let wasUsable = state.usable
        let reason = state.receive(event)
        if wasUsable != state.usable { onUsabilityChanged?(state.usable) }
        if let reason { onInterruption?(reason) }
    }

    func stop() {
        observing = false
        for observer in workspaceObservers { NSWorkspace.shared.notificationCenter.removeObserver(observer) }
        for observer in lockObservers { DistributedNotificationCenter.default().removeObserver(observer) }
        workspaceObservers.removeAll(); lockObservers.removeAll()
    }
}
