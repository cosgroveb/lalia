import AVFoundation
import Foundation
import IOKit.pwr_mgt
import Testing
@testable import Lalia

@MainActor struct DictationCoordinatorTests {
    @Test func sessionPreventsIdleSleepAndReleasesOnEveryExit() async throws {
        for ending in ["stop", "start failure", "capture failure", "stop failure", "quit"] {
            let test = CoordinatorFixture()
            if ending == "start failure" { test.capture.startError = TestError.failed }
            if ending == "stop failure" { test.capture.stopError = TestError.failed }
            test.coordinator.startSession()
            await test.coordinator.waitForOperation()
            let id = try #require(test.coordinator.sessions.first).id
            if ending == "start failure" {
                #expect(try sleepAssertions(for: id).isEmpty)
                continue
            }
            #expect(try sleepAssertions(for: id) == Set(["PreventUserIdleDisplaySleep", "PreventUserIdleSystemSleep"]))
            switch ending {
            case "quit": await test.coordinator.shutdown()
            case "capture failure": test.capture.fail?(TestError.failed)
            default: test.coordinator.stopSession()
            }
            await test.coordinator.waitForOperation()
            #expect(try sleepAssertions(for: id).isEmpty)
        }
    }
    @Test func enablesOfflineAndGatesOverlappingHotkeys() async throws {
        let test = CoordinatorFixture()
        test.transcriber.prepareError = TestError.failed
        await test.coordinator.enableDictation()
        #expect(test.coordinator.isDictationEnabled)
        #expect(test.transcriber.prepareCalls == 0)
        test.coordinator.hotkeyPressed(); test.coordinator.hotkeyPressed()
        #expect(test.recorder.startCalls == 1)
        await test.coordinator.enableDictation()
        #expect(test.authorizer.requestCalls == 1)
        test.coordinator.hotkeyReleased(); test.coordinator.hotkeyReleased()
        await test.coordinator.waitForOperation()
        #expect(test.recorder.stopCalls == 1)
        let saved = try #require(test.coordinator.sessions.first)
        #expect(saved.segments[0].complete)
        #expect(FileManager.default.fileExists(atPath: try test.storage.store.file(saved.id, saved.segments[0].filename).path))
    }
    @Test func authorizationDenialAndRevocationPreventDictation() async {
        let test = CoordinatorFixture()
        test.authorizer.status = AuthorizationStatus(microphoneGranted: false, accessibilityGranted: true)
        await test.coordinator.enableDictation()
        #expect(test.coordinator.phase == .needsPermission)
        #expect(!test.coordinator.isDictationEnabled)
        test.coordinator.hotkeyReleased()
        #expect(test.recorder.stopCalls == 0)
        test.authorizer.status = TestAuthorizer.granted
        await test.coordinator.enableDictation()
        test.authorizer.status = AuthorizationStatus(microphoneGranted: true, accessibilityGranted: false)
        test.coordinator.hotkeyPressed()
        #expect(test.coordinator.phase == .needsPermission)
        #expect(test.recorder.startCalls == 0)
    }
    @Test func ignoresReleaseAndReenableDuringPermissionRequest() async {
        let test = CoordinatorFixture()
        test.authorizer.suspend = true
        let task = Task { await test.coordinator.enableDictation() }
        while !test.authorizer.waiting { await Task.yield() }
        #expect(test.coordinator.phase == .preparing)
        test.coordinator.hotkeyReleased()
        await test.coordinator.enableDictation()
        #expect(test.authorizer.requestCalls == 1)
        #expect(test.recorder.stopCalls == 0)
        test.authorizer.resume()
        await task.value
        #expect(test.coordinator.isDictationEnabled)
    }
    @Test func startStopAndTranscriptionFailuresRetainAudio() async throws {
        for failure in ["start", "stop", "transcription"] {
            let test = CoordinatorFixture()
            if failure == "start" { test.recorder.startError = TestError.failed }
            if failure == "stop" { test.recorder.stopError = TestError.failed }
            if failure == "transcription" { test.transcriber.transcriptionError = TestError.failed }
            await test.coordinator.enableDictation()
            test.coordinator.hotkeyPressed(); test.coordinator.hotkeyReleased()
            await test.coordinator.waitForOperation()
            #expect(test.coordinator.phase == .idle)
            #expect(test.injector.injected.isEmpty)
            let record = try #require(test.coordinator.sessions.first)
            let source = record.segments[0]
            let file = try test.storage.store.file(record.id, source.complete ? source.filename : source.partialFilename)
            #expect(FileManager.default.fileExists(atPath: file.path))
            test.transcriber.transcriptionError = nil
            test.coordinator.retry(record.id)
            await test.coordinator.waitForOperation()
            #expect(try test.storage.store.read(record.id).segments[0].complete)
            #expect(try test.storage.store.read(record.id).transcriptionComplete)
        }
    }
    @Test func preservesPasteSafeguardsAndCopy() async {
        for failure in ["empty", "permission", "paste", "none"] {
            let test = CoordinatorFixture()
            await test.coordinator.enableDictation()
            test.coordinator.hotkeyPressed()
            if failure == "empty" { test.transcriber.output = " \n\t" }
            if failure == "permission" { test.authorizer.status = AuthorizationStatus(microphoneGranted: true, accessibilityGranted: false) }
            if failure == "paste" { test.injector.injectionError = TestError.failed }
            test.coordinator.hotkeyReleased()
            await test.coordinator.waitForOperation()
            if failure == "empty" {
                #expect(test.coordinator.lastTranscript == nil)
                #expect(test.injector.injected.isEmpty)
            } else {
                #expect(test.coordinator.lastTranscript == "spoken text")
                test.coordinator.copyLastTranscript()
                #expect(test.injector.copied == ["spoken text"])
                #expect(test.injector.injected == (failure == "permission" ? [] : ["spoken text"]))
            }
        }
    }
    @Test func sessionsDoNotRequireAccessibilityOrServerAndQuitFinalizes() async throws {
        let test = CoordinatorFixture()
        test.authorizer.status = AuthorizationStatus(microphoneGranted: true, accessibilityGranted: false)
        test.transcriber.prepareError = TestError.failed
        test.coordinator.startSession()
        await test.coordinator.waitForOperation()
        #expect(test.coordinator.phase == .recordingSession)
        test.coordinator.hotkeyPressed()
        test.coordinator.startSession()
        #expect(test.recorder.startCalls == 0)
        #expect(test.capture.startCalls == 1)
        #expect(test.transcriber.prepareCalls == 0)
        await test.coordinator.shutdown()
        #expect(test.capture.stopCalls == 1)
        #expect(try test.storage.store.history().records[0].segments[0].complete)
        #expect(test.transcriber.prepareCalls == 0)
    }
    @Test func recordingToggleSavesTranscribesAndStartsANewSession() async throws {
        let test = CoordinatorFixture()
        test.authorizer.status = AuthorizationStatus(microphoneGranted: true, accessibilityGranted: false)
        for count in 1...2 {
            test.coordinator.toggleRecording()
            #expect(test.coordinator.phase == .preparing)
            test.coordinator.toggleRecording()
            await test.coordinator.waitForOperation()
            #expect(test.coordinator.phase == .recordingSession)
            #expect(test.capture.startCalls == count)
            test.coordinator.hotkeyReleased()
            #expect(test.coordinator.phase == .recordingSession)
            test.coordinator.toggleRecording()
            #expect(test.coordinator.phase == .stopping)
            test.coordinator.toggleRecording()
            await test.coordinator.waitForOperation()
            #expect(test.capture.stopCalls == count)
            #expect(test.coordinator.phase == .idle)
        }
        #expect(test.coordinator.sessions.count == 2)
        #expect(test.coordinator.sessions.allSatisfy { $0.captureState == .complete && $0.transcriptionComplete })
        #expect(test.transcriber.transcriptionCalls == 2)
        #expect(test.injector.injected.isEmpty)
        #expect(test.recorder.startCalls == 0)
    }
    @Test func recordingToggleDoesNotInterruptDictation() async {
        let test = CoordinatorFixture()
        await test.coordinator.enableDictation()
        test.coordinator.hotkeyPressed()
        test.coordinator.toggleRecording()
        #expect(test.coordinator.phase == .recording)
        #expect(test.capture.startCalls == 0)
        #expect(test.recorder.stopCalls == 0)
        test.coordinator.hotkeyReleased()
        await test.coordinator.waitForOperation()
        #expect(test.injector.injected == ["spoken text"])
    }
    @Test func captureDenialAndUnexpectedFailureAreVisible() async throws {
        let denied = CoordinatorFixture()
        denied.capture.startError = TestError.failed
        denied.coordinator.startSession()
        await denied.coordinator.waitForOperation()
        #expect(denied.coordinator.phase == .idle)
        #expect(denied.coordinator.sessions[0].captureState == .failed)
        let failed = CoordinatorFixture()
        failed.coordinator.startSession()
        await failed.coordinator.waitForOperation()
        failed.capture.fail?(TestError.failed)
        await failed.coordinator.waitForOperation()
        #expect(failed.coordinator.sessions[0].captureState == .failed)
        #expect(failed.coordinator.sessions[0].segments[0].complete)
        #expect(failed.coordinator.message.contains("Failed."))
    }
    @Test func retryPreservesCompletedResponsesAndNotesStaySeparate() async throws {
        let test = CoordinatorFixture()
        let saved = try savedAudio(test.storage.store)
        try AudioJobPreparer().prepare(session: saved.id, store: test.storage.store)
        let first = try test.storage.store.read(saved.id).jobs[0]
        var second = first; second.id = "second"; second.start = 2
        try test.storage.store.update(saved.id) { $0.jobs.append(second) }
        test.transcriber.failCall = 2
        test.coordinator.retry(saved.id)
        await test.coordinator.waitForOperation()
        #expect(test.transcriber.transcriptionCalls == 2)
        #expect(try test.storage.store.read(saved.id).jobs.map(\.state) == [.complete, .failed])
        test.transcriber.failCall = nil
        test.coordinator.retry(saved.id)
        await test.coordinator.waitForOperation()
        #expect(test.transcriber.transcriptionCalls == 3)
        let transcript = try test.storage.store.read(saved.id).transcript
        #expect(try test.storage.store.read(saved.id).plainTranscript == "spoken text spoken text")
        test.transcriber.notesError = TestError.failed
        test.coordinator.generateNotes(saved.id)
        await test.coordinator.waitForOperation()
        #expect(try test.storage.store.read(saved.id).transcript == transcript)
        #expect(try test.storage.store.read(saved.id).notes == nil)
        test.transcriber.notesError = nil
        test.coordinator.generateNotes(saved.id)
        await test.coordinator.waitForOperation()
        #expect(try test.storage.store.read(saved.id).notes == "notes")
        #expect(try test.storage.store.read(saved.id).transcript == transcript)
    }
    @Test func cancellationKeepsFilesAndExcludesCaptureDuringInference() async throws {
        let test = CoordinatorFixture()
        let saved = try savedAudio(test.storage.store)
        test.transcriber.suspend = true
        test.coordinator.retry(saved.id)
        while !test.transcriber.waiting { await Task.yield() }
        test.coordinator.toggleRecording(); test.coordinator.hotkeyPressed(); test.coordinator.generateNotes(saved.id)
        #expect(test.capture.startCalls == 0)
        #expect(test.recorder.startCalls == 0)
        #expect(test.coordinator.phase == .transcribing)
        test.coordinator.cancelProcessing()
        test.transcriber.resume()
        await test.coordinator.waitForOperation()
        #expect(test.coordinator.message.contains("cancelled"))
        #expect(try test.storage.store.read(saved.id).jobs[0].state == .failed)
        #expect(try test.storage.store.read(saved.id).segments[0].complete)
    }
    @Test func copyFailuresAndHotkeyRegistrationRemainVisible() async {
        let test = CoordinatorFixture()
        await test.coordinator.enableDictation()
        test.coordinator.hotkeyPressed(); test.coordinator.hotkeyReleased()
        await test.coordinator.waitForOperation()
        test.injector.copyError = TestError.failed
        test.coordinator.copyLastTranscript()
        test.coordinator.hotkeyRegistrationFailed(TestError.failed, shortcut: .dictation)
        test.coordinator.hotkeyRegistrationFailed(TestError.failed, shortcut: .recording)
        #expect(test.coordinator.message == "Failed.")
        #expect(test.coordinator.hotkeyError?.contains("Failed.") == true)
        #expect(test.coordinator.hotkeyError?.contains("⇧⌘D") == true)
        #expect(test.coordinator.hotkeyError?.contains("⇧⌘R") == true)
    }
}

private enum TestError: LocalizedError { case failed; var errorDescription: String? { "Failed." } }

private func sleepAssertions(for session: UUID) throws -> Set<String> {
    var assertions: Unmanaged<CFDictionary>?
    #expect(IOPMCopyAssertionsByProcess(&assertions) == kIOReturnSuccess)
    let processes = try #require(assertions).takeRetainedValue() as NSDictionary
    let owned = processes[NSNumber(value: ProcessInfo.processInfo.processIdentifier)] as? [[String: Any]] ?? []
    return Set(owned.compactMap { entry in
        guard (entry[kIOPMAssertionNameKey] as? String)?.contains(session.uuidString) == true,
              (entry[kIOPMAssertionLevelKey] as? Int) == Int(kIOPMAssertionLevelOn) else { return nil }
        return entry[kIOPMAssertionTypeKey] as? String
    })
}

@MainActor private final class CoordinatorFixture {
    let storage = TemporaryStore()
    let recorder = TestRecorder()
    let capture = TestCapture()
    let transcriber = TestTranscriber()
    let injector = TestInjector()
    let authorizer = TestAuthorizer()
    lazy var coordinator = DictationCoordinator(recorder: recorder, transcriber: transcriber, injector: injector, authorizer: authorizer, store: storage.store, capture: capture)
}

private final class TestRecorder: Recording {
    var startCalls = 0; var stopCalls = 0
    var startError: Error?; var stopError: Error?
    func start(at url: URL) throws {
        startCalls += 1
        let buffer = try pcm(rate: 16_000, channels: 1, frames: 16_000)
        let file = try AVAudioFile(forWriting: url, settings: recorderSettings)
        try file.write(from: buffer)
        if let startError { throw startError }
    }
    func stop() throws { stopCalls += 1; if let stopError { throw stopError } }
    func close() {}
}

@MainActor private final class TestCapture: SessionCapturing {
    var startCalls = 0; var stopCalls = 0
    var startError: Error?
    var stopError: Error?
    var fail: (@MainActor @Sendable (Error) -> Void)?
    var writer: SourceAudioWriter?
    func start(session: UUID, store: SessionStore, failed: @escaping @MainActor @Sendable (Error) -> Void) async throws {
        startCalls += 1; fail = failed
        if let startError { throw startError }
        writer = SourceAudioWriter(store: store, session: session, origin: 0)
        try writer?.append(pcm(frames: 48_000), source: .microphone, timestamp: 0)
    }
    func stop() async throws {
        stopCalls += 1; try writer?.finish(); writer = nil
        if let stopError { throw stopError }
    }
}

@MainActor private final class TestTranscriber: Transcribing {
    var prepareCalls = 0; var transcriptionCalls = 0
    var output = "spoken text"
    var prepareError: Error?; var transcriptionError: Error?; var notesError: Error?
    var failCall: Int?
    var suspend = false
    var waiting = false
    private var continuation: CheckedContinuation<Void, Never>?
    func prepare() async throws { prepareCalls += 1; if let prepareError { throw prepareError } }
    func transcribe(_: URL) async throws -> Transcription {
        transcriptionCalls += 1
        if suspend { waiting = true; await withCheckedContinuation { continuation = $0 }; waiting = false }
        if let transcriptionError { throw transcriptionError }
        if failCall == transcriptionCalls { throw TestError.failed }
        return Transcription(text: output)
    }
    func resume() { continuation?.resume(); continuation = nil }
    func notes(for _: String) async throws -> String { if let notesError { throw notesError }; return "notes" }
}

@MainActor private final class TestInjector: Injecting {
    var injected: [String] = []; var copied: [String] = []
    var injectionError: Error?; var copyError: Error?
    func inject(_ text: String) async throws { injected.append(text); if let injectionError { throw injectionError } }
    func copy(_ text: String) throws { copied.append(text); if let copyError { throw copyError } }
}

@MainActor private final class TestAuthorizer: Authorizing {
    static let granted = AuthorizationStatus(microphoneGranted: true, accessibilityGranted: true)
    var status = granted
    var requestCalls = 0
    var suspend = false
    var waiting = false
    private var continuation: CheckedContinuation<Void, Never>?
    func currentStatus() -> AuthorizationStatus { status }
    func requestAll() async -> AuthorizationStatus {
        requestCalls += 1
        if suspend { waiting = true; await withCheckedContinuation { continuation = $0 }; waiting = false }
        return status
    }
    func requestMicrophone() async -> Bool { status.microphoneGranted }
    func resume() { continuation?.resume(); continuation = nil }
}
