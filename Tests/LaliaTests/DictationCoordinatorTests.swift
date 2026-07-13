import Foundation
import Testing
@testable import Lalia

struct DictationCoordinatorTests {
    @Test @MainActor func enablesOnlyAfterAuthorizationAndPreparation() async {
        let authorizer = TestAuthorizer(status: TestAuthorizer.granted, requestedStatus: TestAuthorizer.granted, suspendsRequest: true)
        let transcriber = TestTranscriber()
        let coordinator = makeCoordinator(transcriber: transcriber, authorizer: authorizer)

        #expect(!coordinator.isDictationEnabled)
        let task = Task { await coordinator.enableDictation() }
        await Task.yield()
        #expect(coordinator.phase == .preparing)
        #expect(!coordinator.isDictationEnabled)
        authorizer.resumeRequest()
        await task.value

        #expect(transcriber.prepareCalls == 1)
        #expect(coordinator.phase == .idle)
        #expect(coordinator.isDictationEnabled)
    }

    @Test @MainActor func disablesWhileRetryingEnablement() async {
        let authorizer = TestAuthorizer(status: TestAuthorizer.granted, requestedStatus: TestAuthorizer.granted, suspendsRequest: true)
        let coordinator = makeCoordinator(authorizer: authorizer)

        let firstAttempt = Task { await coordinator.enableDictation() }
        await Task.yield()
        authorizer.resumeRequest()
        await firstAttempt.value
        #expect(coordinator.isDictationEnabled)

        let retry = Task { await coordinator.enableDictation() }
        await Task.yield()
        #expect(coordinator.phase == .preparing)
        #expect(!coordinator.isDictationEnabled)
        authorizer.resumeRequest()
        await retry.value
        #expect(coordinator.isDictationEnabled)
    }

    @Test @MainActor func ignoresReleaseDuringPreparation() async {
        let authorizer = TestAuthorizer(status: TestAuthorizer.granted, requestedStatus: TestAuthorizer.granted, suspendsRequest: true)
        let recorder = TestRecorder()
        let coordinator = makeCoordinator(recorder: recorder, authorizer: authorizer)

        let task = Task { await coordinator.enableDictation() }
        await Task.yield()
        coordinator.hotkeyReleased()
        #expect(recorder.stopCalls == 0)
        authorizer.resumeRequest()
        await task.value
    }

    @Test @MainActor func remainsNeedingPermissionWhenAuthorizationIsDenied() async {
        let denied = AuthorizationStatus(microphoneGranted: false, speechGranted: true, accessibilityGranted: true)
        let transcriber = TestTranscriber()
        let coordinator = makeCoordinator(transcriber: transcriber, authorizer: TestAuthorizer(status: denied, requestedStatus: denied))

        await coordinator.enableDictation()

        #expect(!coordinator.isDictationEnabled)
        #expect(coordinator.phase == .needsPermission)
        #expect(transcriber.prepareCalls == 0)
    }

    @Test @MainActor func returnsToIdleWhenPreparationFails() async {
        let recorder = TestRecorder()
        let coordinator = makeCoordinator(recorder: recorder, transcriber: TestTranscriber(prepareError: TestError.failed), authorizer: TestAuthorizer(status: TestAuthorizer.granted, requestedStatus: TestAuthorizer.granted))

        await coordinator.enableDictation()
        coordinator.hotkeyPressed()

        #expect(!coordinator.isDictationEnabled)
        #expect(coordinator.phase == .idle)
        #expect(recorder.startCalls == 0)
    }

    @Test @MainActor func disablesWhenPermissionIsRevokedBeforeHotkeyPress() async {
        let authorizer = TestAuthorizer(status: TestAuthorizer.granted, requestedStatus: TestAuthorizer.granted)
        let recorder = TestRecorder()
        let coordinator = await makeReadyCoordinator(recorder: recorder, authorizer: authorizer)
        authorizer.status = AuthorizationStatus(microphoneGranted: true, speechGranted: false, accessibilityGranted: true)

        coordinator.hotkeyPressed()

        #expect(!coordinator.isDictationEnabled)
        #expect(coordinator.phase == .needsPermission)
        #expect(coordinator.message == "Microphone, Speech, and Accessibility permissions are required.")
        #expect(recorder.startCalls == 0)
    }

    @Test @MainActor func ignoresEnablementWhileDictationIsActive() async {
        let authorizer = TestAuthorizer(status: TestAuthorizer.granted, requestedStatus: TestAuthorizer.granted)
        let recorder = TestRecorder()
        let coordinator = await makeReadyCoordinator(recorder: recorder, authorizer: authorizer)

        coordinator.hotkeyPressed()
        await coordinator.enableDictation()

        #expect(coordinator.phase == .recording)
        #expect(authorizer.requestCalls == 1)
        coordinator.hotkeyReleased()
        await waitForIdle(coordinator)
    }

    @Test @MainActor func gatesOverlappingPressesAndReleases() async {
        let recorder = TestRecorder()
        let coordinator = await makeReadyCoordinator(recorder: recorder)

        coordinator.hotkeyPressed()
        coordinator.hotkeyPressed()
        #expect(recorder.startCalls == 1)
        #expect(coordinator.phase == .recording)

        coordinator.hotkeyReleased()
        coordinator.hotkeyReleased()
        await waitForIdle(coordinator)
        #expect(recorder.stopCalls == 1)
        #expect(recorder.discardCalls == 1)
    }

    @Test @MainActor func discardsWhenRecorderStartFails() async {
        let recorder = TestRecorder(startError: TestError.failed)
        let coordinator = await makeReadyCoordinator(recorder: recorder)

        coordinator.hotkeyPressed()

        #expect(recorder.discardCalls == 1)
        #expect(coordinator.phase == .idle)
    }

    @Test @MainActor func discardsWhenRecorderStopFails() async {
        let recorder = TestRecorder(stopError: TestError.failed)
        let coordinator = await makeReadyCoordinator(recorder: recorder)

        coordinator.hotkeyPressed()
        coordinator.hotkeyReleased()

        #expect(recorder.discardCalls == 1)
        #expect(coordinator.phase == .idle)
    }

    @Test @MainActor func discardsWhenTranscriptionFails() async {
        let recorder = TestRecorder()
        let transcriber = TestTranscriber(transcriptionError: TestError.failed)
        let coordinator = await makeReadyCoordinator(recorder: recorder, transcriber: transcriber)

        coordinator.hotkeyPressed()
        coordinator.hotkeyReleased()
        await waitForIdle(coordinator)

        #expect(recorder.discardCalls == 1)
        #expect(coordinator.lastTranscript == nil)
    }

    @Test @MainActor func doesNotInjectEmptyOutput() async {
        let recorder = TestRecorder()
        let injector = TestInjector()
        let coordinator = await makeReadyCoordinator(recorder: recorder, transcriber: TestTranscriber(output: " \n\t "), injector: injector)

        coordinator.hotkeyPressed()
        coordinator.hotkeyReleased()
        await waitForIdle(coordinator)

        #expect(injector.injected == [])
        #expect(coordinator.lastTranscript == nil)
        #expect(coordinator.message == "No speech recognized.")
        #expect(recorder.discardCalls == 1)
    }

    @Test @MainActor func doesNotInjectWhenAccessibilityIsDenied() async {
        let authorizer = TestAuthorizer(status: TestAuthorizer.granted, requestedStatus: TestAuthorizer.granted)
        let recorder = TestRecorder()
        let injector = TestInjector()
        let coordinator = await makeReadyCoordinator(recorder: recorder, injector: injector, authorizer: authorizer)

        coordinator.hotkeyPressed()
        authorizer.status = AuthorizationStatus(microphoneGranted: true, speechGranted: true, accessibilityGranted: false)
        coordinator.hotkeyReleased()
        await waitForIdle(coordinator)

        #expect(injector.injected == [])
        #expect(coordinator.lastTranscript == "spoken text")
        #expect(coordinator.message == "Accessibility permission is required to paste.")
        #expect(recorder.discardCalls == 1)
    }

    @Test @MainActor func retainsTranscriptWhenInjectionFails() async {
        let recorder = TestRecorder()
        let injector = TestInjector(injectionError: TestError.failed)
        let coordinator = await makeReadyCoordinator(recorder: recorder, injector: injector)

        coordinator.hotkeyPressed()
        coordinator.hotkeyReleased()
        await waitForIdle(coordinator)

        #expect(injector.injected == ["spoken text"])
        #expect(coordinator.lastTranscript == "spoken text")
        #expect(recorder.discardCalls == 1)
    }

    @Test @MainActor func completesFullPipelineAndCopiesLastTranscript() async {
        let recorder = TestRecorder()
        let injector = TestInjector()
        let coordinator = await makeReadyCoordinator(recorder: recorder, injector: injector)

        coordinator.hotkeyPressed()
        coordinator.hotkeyReleased()
        await waitForIdle(coordinator)
        coordinator.copyLastTranscript()

        #expect(injector.injected == ["spoken text"])
        #expect(injector.copied == ["spoken text"])
        #expect(recorder.discardCalls == 1)
        #expect(coordinator.phase == .idle)
    }

    @Test @MainActor func showsCopyFailure() async {
        let injector = TestInjector(copyError: TestError.failed)
        let coordinator = await makeReadyCoordinator(injector: injector)

        coordinator.hotkeyPressed()
        coordinator.hotkeyReleased()
        await waitForIdle(coordinator)
        coordinator.copyLastTranscript()

        #expect(coordinator.message == "Failed.")
    }
}

@MainActor private func makeReadyCoordinator(
    recorder: TestRecorder = TestRecorder(),
    transcriber: TestTranscriber = TestTranscriber(),
    injector: TestInjector = TestInjector(),
    authorizer: TestAuthorizer = TestAuthorizer(status: TestAuthorizer.granted, requestedStatus: TestAuthorizer.granted)
) async -> DictationCoordinator {
    let coordinator = makeCoordinator(recorder: recorder, transcriber: transcriber, injector: injector, authorizer: authorizer)
    await coordinator.enableDictation()
    return coordinator
}

@MainActor private func makeCoordinator(
    recorder: TestRecorder = TestRecorder(),
    transcriber: TestTranscriber = TestTranscriber(),
    injector: TestInjector = TestInjector(),
    authorizer: TestAuthorizer
) -> DictationCoordinator {
    DictationCoordinator(recorder: recorder, transcriber: transcriber, injector: injector, authorizer: authorizer)
}

@MainActor private func waitForIdle(_ coordinator: DictationCoordinator) async {
    for _ in 0..<20 where coordinator.phase != .idle { await Task.yield() }
}

private enum TestError: LocalizedError { case failed; var errorDescription: String? { "Failed." } }

private final class TestRecorder: Recording {
    var startCalls = 0; var stopCalls = 0; var discardCalls = 0
    let startError: Error?; let stopError: Error?
    init(startError: Error? = nil, stopError: Error? = nil) { self.startError = startError; self.stopError = stopError }
    func start() throws { startCalls += 1; if let startError { throw startError } }
    func stop() throws -> URL { stopCalls += 1; if let stopError { throw stopError }; return URL(fileURLWithPath: "/tmp/lalia-test.wav") }
    func discard() { discardCalls += 1 }
}

@MainActor private final class TestTranscriber: Transcribing {
    var prepareCalls = 0
    let output: String; let prepareError: Error?; let transcriptionError: Error?
    init(output: String = "spoken text", prepareError: Error? = nil, transcriptionError: Error? = nil) { self.output = output; self.prepareError = prepareError; self.transcriptionError = transcriptionError }
    func prepare() async throws { prepareCalls += 1; if let prepareError { throw prepareError } }
    func transcribe(_: URL) async throws -> String { if let transcriptionError { throw transcriptionError }; return output }
}

@MainActor private final class TestInjector: Injecting {
    var injected: [String] = []; var copied: [String] = []
    let injectionError: Error?; let copyError: Error?
    init(injectionError: Error? = nil, copyError: Error? = nil) { self.injectionError = injectionError; self.copyError = copyError }
    func inject(_ text: String) async throws { injected.append(text); if let injectionError { throw injectionError } }
    func copy(_ text: String) throws { copied.append(text); if let copyError { throw copyError } }
}

@MainActor private final class TestAuthorizer: Authorizing {
    static let granted = AuthorizationStatus(microphoneGranted: true, speechGranted: true, accessibilityGranted: true)
    var status: AuthorizationStatus; let requestedStatus: AuthorizationStatus; let suspendsRequest: Bool; var requestCalls = 0
    private var continuation: CheckedContinuation<Void, Never>?
    init(status: AuthorizationStatus, requestedStatus: AuthorizationStatus, suspendsRequest: Bool = false) { self.status = status; self.requestedStatus = requestedStatus; self.suspendsRequest = suspendsRequest }
    func currentStatus() -> AuthorizationStatus { status }
    func requestAll() async -> AuthorizationStatus { requestCalls += 1; if suspendsRequest { await withCheckedContinuation { continuation = $0 } }; status = requestedStatus; return status }
    func resumeRequest() { continuation?.resume(); continuation = nil }
}
