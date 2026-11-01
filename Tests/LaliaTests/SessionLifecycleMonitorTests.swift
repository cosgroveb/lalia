import Testing
@testable import Lalia

@MainActor struct SessionLifecycleMonitorTests {
    @Test func wakeBeforeUnlockRemainsUnavailable() {
        var state = SessionLifecycleState()
        #expect(state.receive(.lock) == .lock)
        #expect(state.receive(.sleep) == .sleep)
        #expect(state.receive(.wake) == nil)
        #expect(!state.usable)
        #expect(state.receive(.unlock) == nil)
        #expect(state.usable)
    }

    @Test func switchingUsersAndLockHaveIndependentState() {
        var state = SessionLifecycleState()
        #expect(state.receive(.resign) == .userSwitch)
        #expect(state.receive(.lock) == .lock)
        #expect(state.receive(.unlock) == nil)
        #expect(!state.usable)
        #expect(state.receive(.activate) == nil)
        #expect(state.usable)
        #expect(state.receive(.activate) == nil)
    }

    @Test func repeatedEventsAndStoppedMonitorDoNotDeliver() {
        let monitor = SessionLifecycleMonitor()
        var reasons: [SessionInterruption] = []
        monitor.onInterruption = { reasons.append($0) }
        monitor.start()
        monitor.receive(.lock)
        monitor.receive(.lock)
        #expect(reasons == [.lock])
        monitor.stop()
        monitor.receive(.sleep)
        #expect(reasons == [.lock])
    }
}
