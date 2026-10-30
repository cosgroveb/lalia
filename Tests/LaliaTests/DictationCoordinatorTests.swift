import AVFoundation
import Foundation
import IOKit.pwr_mgt
import Testing
@testable import Lalia

@Suite(.serialized) @MainActor struct DictationCoordinatorTests {
    @Test func transcriptionAllowsNewCaptureAndDoesNotChangeItsPhase() async throws {
        for session in [false, true] {
            let test = CoordinatorFixture(); let saved = try savedAudio(test.storage.store)
            await test.coordinator.enableDictation()
            test.transcriber.suspend = true
            test.coordinator.reprocess(saved.id)
            while !test.transcriber.waiting { await Task.yield() }
            if session { test.coordinator.startSession() }
            else { test.beginGestureDictation() }
            for _ in 0..<20 { await Task.yield() }
            let expected: Phase = session ? .recordingSession : .recording
            #expect(test.coordinator.phase == expected)
            test.transcriber.resume()
            await test.coordinator.waitForOperation()
            #expect(test.coordinator.phase == expected)
            #expect(try test.storage.store.read(saved.id).transcriptionComplete)
            await test.coordinator.shutdown()
        }
    }
    @Test func queuesDictationWithoutPastingAndKeepsResultsWithTheirRecordings() async throws {
        let test = CoordinatorFixture(); let saved = try savedAudio(test.storage.store)
        await test.coordinator.enableDictation()
        test.transcriber.suspend = true
        test.coordinator.reprocess(saved.id)
        while !test.transcriber.waiting { await Task.yield() }
        test.beginGestureDictation(); test.finishGestureDictation()
        let dictation = try #require(test.coordinator.sessions.first { $0.kind == .dictation })
        #expect(test.coordinator.processing[dictation.id] == "Queued")
        test.coordinator.retry(dictation.id)
        #expect(test.coordinator.processing.count == 2)
        #expect(test.transcriber.transcriptionCalls == 0)
        test.coordinator.startSession()
        while test.coordinator.phase == .preparing { await Task.yield() }
        test.transcriber.output = "first recording"
        test.transcriber.resume()
        while test.transcriber.transcriptionCalls == 0 { await Task.yield() }
        #expect(try test.storage.store.read(saved.id).transcript.contains("first recording"))
        #expect(try !test.storage.store.read(dictation.id).transcriptionComplete)
        #expect(test.coordinator.phase == .recordingSession)
        test.transcriber.output = "queued dictation"
        test.transcriber.resume()
        await test.coordinator.waitForOperation()
        #expect(test.coordinator.processing.isEmpty)
        #expect(test.coordinator.phase == .recordingSession)
        #expect(try test.storage.store.read(dictation.id).plainTranscript == "queued dictation")
        #expect(try test.storage.store.read(saved.id).transcript.contains("first recording"))
        #expect(test.injector.injected.isEmpty)
        #expect(test.injector.copied.isEmpty)
        test.coordinator.copyLastTranscript()
        #expect(test.injector.copied == ["queued dictation"])
        test.coordinator.copyTranscript(saved.id)
        #expect(test.injector.copied.last?.contains("first recording") == true)
        await test.coordinator.shutdown()
    }
    @Test func newCaptureInvalidatesPendingPasteAndLaterImmediateDictationCanPaste() async throws {
        let test = CoordinatorFixture()
        await test.coordinator.enableDictation()
        test.transcriber.suspend = true
        test.beginGestureDictation(); test.finishGestureDictation()
        while !test.transcriber.waiting { await Task.yield() }
        test.beginGestureDictation()
        test.transcriber.resume()
        await test.coordinator.waitForOperation()
        #expect(test.coordinator.phase == .recording)
        #expect(test.injector.injected.isEmpty)
        #expect(test.injector.copied.isEmpty)
        test.transcriber.suspend = false
        test.transcriber.output = "new dictation"
        test.finishGestureDictation()
        await test.coordinator.waitForOperation()
        #expect(test.injector.injected == ["new dictation"])
        #expect(test.coordinator.lastTranscript == "new dictation")
    }
    @Test func cancelsQueuedWorkWithoutCancellingRunningRequestAndCanRetry() async throws {
        let test = CoordinatorFixture()
        let first = try savedAudio(test.storage.store); let second = try savedAudio(test.storage.store)
        test.transcriber.suspend = true
        test.coordinator.reprocess(first.id)
        while !test.transcriber.waiting { await Task.yield() }
        test.coordinator.reprocess(second.id)
        test.coordinator.cancelProcessing(second.id)
        #expect(test.coordinator.processing[second.id] == nil)
        #expect(test.coordinator.canCancel)
        #expect(try test.storage.store.read(second.id).transcriptionError?.contains("cancelled before starting") == true)
        test.transcriber.suspend = false; test.transcriber.resume()
        await test.coordinator.waitForOperation()
        #expect(test.transcriber.sessionCalls == 1)
        #expect(try test.storage.store.read(first.id).transcriptionComplete)
        test.coordinator.reprocess(second.id)
        await test.coordinator.waitForOperation()
        #expect(test.transcriber.sessionCalls == 2)
        #expect(try test.storage.store.read(second.id).transcriptionComplete)
        #expect(try test.storage.store.read(second.id).transcriptionError == nil)
    }
    @Test func shutdownSavesCaptureAndCancelsRunningAndQueuedWork() async throws {
        let test = CoordinatorFixture()
        let first = try savedAudio(test.storage.store); let second = try savedAudio(test.storage.store)
        test.transcriber.suspend = true
        test.coordinator.reprocess(first.id)
        while !test.transcriber.waiting { await Task.yield() }
        test.coordinator.reprocess(second.id)
        test.coordinator.startSession()
        while test.coordinator.phase == .preparing { await Task.yield() }
        let captured = try #require(test.coordinator.sessions.first { $0.captureState == .recording })
        let shutdown = Task { await test.coordinator.shutdown() }
        while test.capture.stopCalls == 0 { await Task.yield() }
        #expect(try test.storage.store.read(captured.id).captureState == .complete)
        #expect(test.coordinator.processing[second.id] == nil)
        test.transcriber.resume()
        await shutdown.value
        #expect(test.coordinator.processing.isEmpty)
        #expect(test.transcriber.sessionCalls == 1)
        #expect(try test.storage.store.read(first.id).analysis?.attempt?.state == .failed)
        #expect(try test.storage.store.read(second.id).transcriptionError?.contains("cancelled") == true)
        #expect(try test.storage.store.read(captured.id).analysis?.attempt == nil)
        #expect(try sleepAssertions(for: captured.id).isEmpty)
        let recovered = try test.storage.store.history(recover: true).records
        #expect(recovered.allSatisfy { $0.segments.allSatisfy(\.complete) })
    }
    @Test func cancellingCurrentRequestAdvancesQueueWithoutCancellingNextRequest() async throws {
        let test = CoordinatorFixture()
        let first = try savedAudio(test.storage.store); let second = try savedAudio(test.storage.store)
        test.transcriber.suspend = true
        test.coordinator.reprocess(first.id)
        while !test.transcriber.waiting { await Task.yield() }
        test.coordinator.reprocess(second.id)
        test.coordinator.cancelProcessing(first.id)
        test.transcriber.resume()
        while test.transcriber.sessionCalls < 2 { await Task.yield() }
        #expect(try test.storage.store.read(first.id).analysis?.attempt?.state == .failed)
        #expect(test.coordinator.processing[first.id] == nil)
        #expect(test.coordinator.processing[second.id] != nil)
        test.transcriber.resume()
        await test.coordinator.waitForOperation()
        #expect(try test.storage.store.read(second.id).transcriptionComplete)
        #expect(test.coordinator.processing.isEmpty)
    }
    @Test func notesFinishDuringCaptureAndEditsStillMakeThemStale() async throws {
        let test = CoordinatorFixture(); let saved = try savedAudio(test.storage.store)
        test.coordinator.reprocess(saved.id); await test.coordinator.waitForOperation()
        test.transcriber.suspend = true
        test.coordinator.generateNotes(saved.id)
        while !test.transcriber.waiting { await Task.yield() }
        test.coordinator.startSession()
        while test.coordinator.phase == .preparing { await Task.yield() }
        test.coordinator.editTranscript(saved.id) { try $0.rename("S01", to: "Brian") }
        test.transcriber.resume()
        await test.coordinator.waitForOperation()
        #expect(try test.storage.store.read(saved.id).notes == "notes")
        #expect(try test.storage.store.read(saved.id).notesStale)
        #expect(test.coordinator.phase == .recordingSession)
        await test.coordinator.shutdown()
    }
    @Test func stopsPlaybackSynchronouslyBeforeEitherCaptureStarts() async throws {
        for session in [false, true] {
            let test = CoordinatorFixture()
            let events = PlaybackCaptureEvents()
            test.playback.onStop = { events.values.append("stop playback") }
            test.recorder.onStart = { events.values.append("start dictation") }
            test.capture.onStart = { events.values.append("start session") }
            if session {
                test.coordinator.startSession()
                await test.coordinator.waitForOperation()
            } else {
                await test.coordinator.enableDictation()
                test.beginGestureDictation()
            }
            #expect(events.values == ["stop playback", session ? "start session" : "start dictation"])
            await test.coordinator.shutdown()
        }
    }
    @Test func firstSpeakerUpgradeCanRestoreLegacyTranscriptAndToggleBack() async throws {
        let test = CoordinatorFixture(); let saved = try savedAudio(test.storage.store)
        try AudioJobPreparer().prepare(session: saved.id, store: test.storage.store)
        try test.storage.store.update(saved.id) {
            $0.jobs[0].state = .complete; $0.jobs[0].response = Transcription(text: "original source transcript")
        }
        let legacy = try test.storage.store.read(saved.id).transcript
        test.coordinator.reprocess(saved.id); await test.coordinator.waitForOperation()
        let speakerText = try test.storage.store.read(saved.id).transcript
        #expect(speakerText != legacy)
        #expect(try test.storage.store.read(saved.id).analysis?.previousIsLegacy == true)
        test.coordinator.generateNotes(saved.id); await test.coordinator.waitForOperation()
        test.coordinator.restoreTranscript(saved.id)
        #expect(try test.storage.store.read(saved.id).transcript == legacy)
        #expect(try test.storage.store.read(saved.id).notesStale)
        let calls = test.transcriber.sessionCalls
        test.coordinator.retry(saved.id); await test.coordinator.waitForOperation()
        #expect(test.transcriber.sessionCalls == calls)
        let exported = test.storage.store.root.appending(path: "restored.txt")
        test.coordinator.export(saved.id, to: exported)
        #expect(try String(contentsOf: exported, encoding: .utf8) == legacy)
        test.coordinator.copyLastTranscript()
        #expect(test.injector.copied.last == "original source transcript")
        test.coordinator.restoreTranscript(saved.id)
        #expect(try test.storage.store.read(saved.id).transcript == speakerText)
        #expect(try !test.storage.store.read(saved.id).notesStale)
        #expect(test.coordinator.lastTranscript == speakerText)
    }
    @Test func speakerUpgradeRetainsEmptyAndPartialLegacyResultsAcrossReload() async throws {
        for outcome in ["empty", "whitespace", "silence", "partial"] {
            let test = CoordinatorFixture(); let store = test.storage.store
            let saved = try savedAudio(store, amplitude: outcome == "silence" ? 0 : 0.1)
            try AudioJobPreparer().prepare(session: saved.id, store: store)
            try store.update(saved.id) {
                if outcome != "silence" {
                    $0.jobs[0].state = .complete
                    $0.jobs[0].response = Transcription(text: outcome == "partial" ? "retained words" : outcome == "whitespace" ? " \n " : "")
                }
                if outcome == "partial" {
                    $0.jobs.append(TranscriptionJob(id: "failed", source: .system, filename: "failed.wav", start: 0, frames: 16_000, state: .failed))
                }
            }
            let legacy = try store.read(saved.id)
            #expect(legacy.transcriptionComplete == (outcome != "partial"))
            test.coordinator.reprocess(saved.id); await test.coordinator.waitForOperation()
            let reopened = SessionStore(root: store.root)
            let replacement = try reopened.read(saved.id)
            let active = try #require(replacement.analysis?.active)
            #expect(replacement.analysis?.previousIsLegacy == true)
            test.coordinator.restoreTranscript(saved.id)
            let restored = try reopened.read(saved.id)
            #expect(restored.analysis?.active == nil)
            #expect(restored.transcript == legacy.transcript)
            #expect(restored.transcriptionComplete == legacy.transcriptionComplete)
            try reopened.restoreTranscript(saved.id)
            let toggled = try SessionStore(root: store.root).read(saved.id)
            #expect(toggled.analysis?.active?.id == active.id)
            #expect(toggled.analysis?.previousIsLegacy == true)
            try reopened.restoreTranscript(saved.id)
            #expect(try store.read(saved.id).analysis?.active == nil)
            #expect(try store.read(saved.id).transcript == legacy.transcript)
        }
    }
    @Test func copyLastTranscriptReflectsEditsOnlyToItsOwnRecording() async throws {
        let test = CoordinatorFixture(); let older = try savedAudio(test.storage.store)
        test.coordinator.reprocess(older.id); await test.coordinator.waitForOperation()
        let last = try savedAudio(test.storage.store)
        test.coordinator.reprocess(last.id); await test.coordinator.waitForOperation()
        test.coordinator.editTranscript(last.id) { try $0.rename("S01", to: "Brian") }
        test.coordinator.copyLastTranscript()
        #expect(test.injector.copied.last?.contains("Brian") == true)
        let copied = test.injector.copied.last
        test.coordinator.editTranscript(older.id) { try $0.rename("S01", to: "Older person") }
        test.coordinator.copyLastTranscript()
        #expect(test.injector.copied.last == copied)
        test.coordinator.reprocess(last.id); await test.coordinator.waitForOperation()
        test.coordinator.restoreTranscript(last.id)
        #expect(test.coordinator.lastTranscript == copied)
    }
    @Test func deadlineWarnsStopsOnceAndCapsAudio() async throws {
        let test = CoordinatorFixture(); test.recordingLimit = 0.2; test.warningSeconds = 0.15
        test.coordinator.startSession(); await test.coordinator.waitForOperation()
        try await Task.sleep(for: .milliseconds(85))
        #expect(test.coordinator.recordingWarning)
        try await Task.sleep(for: .milliseconds(150))
        test.coordinator.stopSession(); test.capture.fail?(TestError.failed)
        await test.coordinator.waitForOperation()
        #expect(test.capture.stopCalls == 1)
        #expect(test.coordinator.phase == .idle)
        let record = try #require(test.coordinator.sessions.first)
        #expect(record.endedAtLimit == true)
        #expect(record.duration <= 0.2)
        #expect(record.transcriptionComplete)
        #expect(test.transcriber.sessionCalls == 1)
    }
    @Test func userStopCancelsDeadlineAndSilenceSkipsInference() async throws {
        let test = CoordinatorFixture(); test.recordingLimit = 0.5; test.capture.amplitude = 0
        test.coordinator.startSession(); await test.coordinator.waitForOperation()
        test.coordinator.stopSession(); await test.coordinator.waitForOperation()
        try await Task.sleep(for: .milliseconds(550))
        #expect(test.capture.stopCalls == 1)
        #expect(test.transcriber.sessionCalls == 0)
        #expect(test.coordinator.sessions.first?.transcriptionComplete == true)
        #expect(test.coordinator.sessions.first?.endedAtLimit != true)
    }
    @Test func rejectedReplacementRetainsRawCorrectionsAndRetriesFrozenProfile() async throws {
        let test = CoordinatorFixture()
        test.coordinator.startSession(); await test.coordinator.waitForOperation()
        test.coordinator.stopSession(); await test.coordinator.waitForOperation()
        let id = try #require(test.coordinator.sessions.first).id
        test.coordinator.editTranscript(id) { try $0.rename("S01", to: "Brian") }
        let before = try test.storage.store.read(id).transcript
        test.transcriber.sessionResponse = Data("{broken response".utf8)
        test.coordinator.reprocess(id); await test.coordinator.waitForOperation()
        #expect(try test.storage.store.read(id).transcript == before)
        #expect(try test.storage.store.read(id).analysis?.attempt?.rawResponse == test.transcriber.sessionResponse)
        let frozen = try #require(test.storage.store.read(id).analysis?.attempt?.profile)
        #expect(frozen.prompt == SessionTranscriptionProfile.officialPrompt)
        test.transcriber.profile.model = "changed-model"
        test.transcriber.profile.prompt = "changed prompt"
        test.transcriber.sessionResponse = nil
        test.coordinator.retry(id); await test.coordinator.waitForOperation()
        #expect(test.transcriber.requestedProfiles.last == frozen)
        #expect(try test.storage.store.read(id).analysis?.previous?.transcript == before)
        test.coordinator.restoreTranscript(id)
        #expect(try test.storage.store.read(id).transcript == before)
        let calls = test.transcriber.sessionCalls
        test.coordinator.retry(id); await test.coordinator.waitForOperation()
        #expect(test.transcriber.sessionCalls == calls)
        #expect(test.injector.injected.isEmpty)
        #expect(test.transcriber.transcriptionCalls == 0)
    }
    @Test func savedSessionRetryPreservesAbsentPrompt() async throws {
        let test = CoordinatorFixture(); let saved = try savedAudio(test.storage.store)
        var old = SessionTranscriptionProfile(); old.prompt = nil
        try test.storage.store.update(saved.id) {
            $0.analysis = SessionAnalysis(attempt: .init(profile: old, state: .failed))
        }
        #expect(try test.storage.store.read(saved.id).analysis?.attempt?.profile.prompt == nil)
        test.coordinator.retry(saved.id); await test.coordinator.waitForOperation()
        #expect(test.transcriber.requestedProfiles.count == 1)
        #expect(test.transcriber.requestedProfiles.first?.prompt == nil)
        #expect(try test.storage.store.read(saved.id).analysis?.active?.profile.prompt == nil)
    }
    @Test func sessionCancellationAndMissingModelRetainAudioAndActiveResult() async throws {
        let test = CoordinatorFixture(); let saved = try savedAudio(test.storage.store)
        test.transcriber.prepareError = TestError.failed
        test.coordinator.reprocess(saved.id); await test.coordinator.waitForOperation()
        #expect(try test.storage.store.read(saved.id).analysis?.attempt?.state == .failed)
        #expect(test.transcriber.sessionCalls == 0)
        test.transcriber.prepareError = nil
        test.coordinator.retry(saved.id); await test.coordinator.waitForOperation()
        let original = try test.storage.store.read(saved.id).transcript
        test.transcriber.suspend = true
        test.coordinator.reprocess(saved.id)
        while !test.transcriber.waiting { await Task.yield() }
        #expect(try sleepAssertions(for: saved.id) == Set(["PreventUserIdleSystemSleep"]))
        test.coordinator.cancelProcessing(); test.transcriber.resume()
        await test.coordinator.waitForOperation()
        let record = try test.storage.store.read(saved.id)
        #expect(record.transcript == original)
        #expect(record.analysis?.attempt?.state == .failed)
        #expect(test.coordinator.message.contains("oMLX may still be processing"))
        #expect(try sleepAssertions(for: saved.id).isEmpty)
        #expect(FileManager.default.fileExists(atPath: try test.storage.store.file(saved.id, record.segments[0].filename).path))
    }
    @Test func notesUseCorrectedNamesAndBecomeStaleOnFurtherEdit() async throws {
        let test = CoordinatorFixture(); let saved = try savedAudio(test.storage.store)
        test.coordinator.reprocess(saved.id); await test.coordinator.waitForOperation()
        test.coordinator.editTranscript(saved.id) { try $0.rename("S01", to: "Brian") }
        test.coordinator.generateNotes(saved.id); await test.coordinator.waitForOperation()
        #expect(test.transcriber.notesInput?.contains("Brian") == true)
        #expect(try !test.storage.store.read(saved.id).notesStale)
        test.coordinator.editTranscript(saved.id) { try $0.assign($0.turns[0].id, to: nil) }
        #expect(try test.storage.store.read(saved.id).notesStale)
        #expect(try test.storage.store.read(saved.id).notes == "notes")
    }
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
    @Test func enablesOfflineAndGatesOverlappingGestures() async throws {
        let test = CoordinatorFixture()
        test.transcriber.prepareError = TestError.failed
        await test.coordinator.enableDictation()
        #expect(test.coordinator.isDictationEnabled)
        #expect(test.transcriber.prepareCalls == 0)
        test.beginGestureDictation(); test.beginGestureDictation()
        #expect(test.recorder.startCalls == 1)
        await test.coordinator.enableDictation()
        #expect(test.authorizer.requestCalls == 1)
        test.finishGestureDictation(); test.finishGestureDictation()
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
        test.finishGestureDictation()
        #expect(test.recorder.stopCalls == 0)
        test.authorizer.status = TestAuthorizer.granted
        await test.coordinator.enableDictation()
        test.authorizer.status = AuthorizationStatus(microphoneGranted: true, accessibilityGranted: false)
        test.beginGestureDictation()
        #expect(test.coordinator.phase == .needsPermission)
        #expect(test.recorder.startCalls == 0)
    }
    @Test func ignoresReleaseAndReenableDuringPermissionRequest() async {
        let test = CoordinatorFixture()
        test.authorizer.suspend = true
        let task = Task { await test.coordinator.enableDictation() }
        while !test.authorizer.waiting { await Task.yield() }
        #expect(test.coordinator.phase == .preparing)
        test.finishGestureDictation()
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
            test.beginGestureDictation(); test.finishGestureDictation()
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
    @Test func rejectedMOSSResultRetainsDictationAudioAndRetryCompletes() async throws {
        let test = CoordinatorFixture()
        await test.coordinator.enableDictation()
        test.transcriber.transcriptionError = LaliaError("MOSS completion could not be verified: invalid token usage.")
        test.beginGestureDictation(); test.finishGestureDictation()
        await test.coordinator.waitForOperation()
        let failed = try #require(test.coordinator.sessions.first)
        let audio = try test.storage.store.file(failed.id, failed.segments[0].filename)
        #expect(try test.storage.store.read(failed.id).jobs.first?.state == .failed)
        #expect(FileManager.default.fileExists(atPath: audio.path))
        #expect(test.injector.injected.isEmpty)
        test.transcriber.transcriptionError = nil
        test.coordinator.retry(failed.id)
        await test.coordinator.waitForOperation()
        #expect(try test.storage.store.read(failed.id).transcriptionComplete)
        #expect(test.injector.injected.isEmpty)
        #expect(try test.storage.store.read(failed.id).plainTranscript == "spoken text")
        #expect(FileManager.default.fileExists(atPath: audio.path))
    }
    @Test func preservesPasteSafeguardsAndCopy() async {
        for failure in ["empty", "permission", "paste", "none"] {
            let test = CoordinatorFixture()
            await test.coordinator.enableDictation()
            test.beginGestureDictation()
            if failure == "empty" { test.transcriber.output = " \n\t" }
            if failure == "permission" { test.authorizer.status = AuthorizationStatus(microphoneGranted: true, accessibilityGranted: false) }
            if failure == "paste" { test.injector.injectionError = TestError.failed }
            test.finishGestureDictation()
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
        test.beginGestureDictation()
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
            test.finishGestureDictation()
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
        #expect(test.transcriber.sessionCalls == 2)
        #expect(test.injector.injected.isEmpty)
        #expect(test.recorder.startCalls == 0)
    }
    @Test func recordingToggleDoesNotInterruptDictation() async {
        let test = CoordinatorFixture()
        await test.coordinator.enableDictation()
        test.beginGestureDictation()
        test.coordinator.toggleRecording()
        #expect(test.coordinator.phase == .recording)
        #expect(test.capture.startCalls == 0)
        #expect(test.recorder.stopCalls == 0)
        test.finishGestureDictation()
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
        try test.storage.store.update(saved.id) { $0.kind = .dictation }
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
    @Test func cancellationKeepsFilesWithoutStoppingNewCapture() async throws {
        let test = CoordinatorFixture()
        let saved = try savedAudio(test.storage.store)
        test.transcriber.suspend = true
        test.coordinator.retry(saved.id)
        while !test.transcriber.waiting { await Task.yield() }
        test.coordinator.toggleRecording(); test.beginGestureDictation(); test.coordinator.generateNotes(saved.id)
        for _ in 0..<20 { await Task.yield() }
        #expect(test.capture.startCalls == 1)
        #expect(test.recorder.startCalls == 0)
        #expect(test.coordinator.phase == .recordingSession)
        test.coordinator.cancelProcessing()
        test.transcriber.resume()
        await test.coordinator.waitForOperation()
        #expect(test.coordinator.phase == .recordingSession)
        #expect(test.capture.stopCalls == 0)
        #expect(try test.storage.store.read(saved.id).transcriptionError?.contains("cancelled") == true)
        #expect(try test.storage.store.read(saved.id).jobs[0].state == .failed)
        #expect(try test.storage.store.read(saved.id).segments[0].complete)
        await test.coordinator.shutdown()
    }
    @Test func copyFailuresRemainVisible() async {
        let test = CoordinatorFixture()
        await test.coordinator.enableDictation()
        test.beginGestureDictation(); test.finishGestureDictation()
        await test.coordinator.waitForOperation()
        test.injector.copyError = TestError.failed
        test.coordinator.copyLastTranscript()
        #expect(test.coordinator.message == "Failed.")
    }

    @Test func gestureCancellationRetainsAudioAndIgnoresLateRelease() async throws {
        let test = CoordinatorFixture()
        await test.coordinator.enableDictation()
        test.coordinator.setGestureAvailability(true)
        let id = UUID()
        test.coordinator.gestureBegan(id)
        #expect(test.coordinator.phase == .recording)
        test.coordinator.gestureCancelled(id)
        test.coordinator.gestureFinished(id)
        await test.coordinator.waitForOperation()
        let record = try #require(test.coordinator.sessions.first { $0.kind == .dictation })
        #expect(record.captureState == .complete)
        #expect(record.captureError == "Dictation cancelled. Audio was saved without transcription.")
        #expect(record.transcriptionComplete == false)
        #expect(test.transcriber.transcriptionCalls == 0)
        #expect(test.injector.injected.isEmpty)
        #expect(test.coordinator.message.contains("saved without transcription"))
    }

    @Test func listenerLossStopsSessionWithoutSubmittingAnalysis() async throws {
        let test = CoordinatorFixture()
        test.coordinator.setGestureAvailability(true)
        test.coordinator.gestureToggle(UUID())
        for _ in 0..<20 { await Task.yield() }
        #expect(test.coordinator.phase == .recordingSession)
        test.coordinator.setGestureAvailability(false, message: "listener lost")
        await test.coordinator.waitForOperation()
        #expect(test.coordinator.phase == .idle)
        #expect(test.capture.stopCalls == 1)
        #expect(test.transcriber.sessionCalls == 0)
        #expect(test.coordinator.sessions.first?.captureError?.contains("Gesture monitoring was lost") == true)
    }

    @Test func listenerLossDuringSessionStartCannotResumeCapture() async {
        let test = CoordinatorFixture()
        test.capture.suspend = true
        test.coordinator.startSession()
        while !test.capture.waiting { await Task.yield() }
        test.coordinator.setGestureAvailability(false, message: "listener lost")
        test.capture.resume()
        await test.coordinator.waitForOperation()
        #expect(test.coordinator.phase == .idle)
        #expect(test.capture.stopCalls == 1)
        #expect(test.transcriber.sessionCalls == 0)
    }

    @Test func listenerLossWhileMicrophonePermissionIsPendingCannotStartSession() async {
        let test = CoordinatorFixture()
        test.authorizer.suspendMicrophone = true
        test.coordinator.startSession()
        while !test.authorizer.microphoneWaiting { await Task.yield() }
        test.coordinator.setGestureAvailability(false, message: "listener lost")
        test.authorizer.resumeMicrophone()
        await test.coordinator.waitForOperation()
        #expect(test.capture.startCalls == 0)
        #expect(test.coordinator.sessions.isEmpty)
        #expect(test.coordinator.phase == .idle)
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
    let playback = TestPlayback()
    var recordingLimit = SessionTranscriptionProfile.maximumDuration
    var warningSeconds = 300.0
    private var gestureID = UUID()
    lazy var coordinator: DictationCoordinator = {
        let value = DictationCoordinator(recorder: recorder, transcriber: transcriber, injector: injector, authorizer: authorizer, store: storage.store, capture: capture, playback: playback, recordingLimit: recordingLimit, warningSeconds: warningSeconds)
        value.setGestureAvailability(true)
        return value
    }()
    func beginGestureDictation() { coordinator.gestureBegan(gestureID) }
    func finishGestureDictation() { coordinator.gestureFinished(gestureID); gestureID = UUID() }
}

private final class PlaybackCaptureEvents { var values: [String] = [] }
@MainActor private final class TestPlayback: PlaybackStopping {
    var onStop: (() -> Void)?
    func stop() { onStop?() }
}

private final class TestRecorder: Recording {
    var startCalls = 0; var stopCalls = 0
    var startError: Error?; var stopError: Error?
    var onStart: (() -> Void)?
    func start(at url: URL) throws {
        onStart?()
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
    var amplitude: Float = 0.1
    var suspend = false
    var waiting = false
    private var continuation: CheckedContinuation<Void, Never>?
    var onStart: (() -> Void)?
    func start(session: UUID, store: SessionStore, maximumDuration: Double, failed: @escaping @MainActor @Sendable (Error) -> Void) async throws {
        onStart?()
        startCalls += 1; fail = failed
        if suspend { waiting = true; await withCheckedContinuation { continuation = $0 }; waiting = false }
        if let startError { throw startError }
        writer = SourceAudioWriter(store: store, session: session, origin: 0, maximumDuration: maximumDuration)
        try writer?.append(pcm(frames: 48_000, amplitude: amplitude), source: .microphone, timestamp: 0)
    }
    func stop() async throws {
        stopCalls += 1; try writer?.finish(); writer = nil
        if let stopError { throw stopError }
    }
    func resume() { continuation?.resume(); continuation = nil }
}

@MainActor private final class TestTranscriber: Transcribing {
    var prepareCalls = 0; var transcriptionCalls = 0
    var sessionCalls = 0
    var profile = SessionTranscriptionProfile()
    var requestedProfiles: [SessionTranscriptionProfile] = []
    var sessionResponse: Data?
    var notesInput: String?
    var output = "spoken text"
    var prepareError: Error?; var transcriptionError: Error?; var notesError: Error?
    var failCall: Int?
    var suspend = false
    var waiting = false
    private var continuation: CheckedContinuation<Void, Never>?
    func prepare() async throws { prepareCalls += 1; if let prepareError { throw prepareError } }
    func transcribe(_ file: URL, duration: Double) async throws -> Transcription {
        transcriptionCalls += 1
        if suspend { waiting = true; await withCheckedContinuation { continuation = $0 }; waiting = false }
        if let transcriptionError { throw transcriptionError }
        if failCall == transcriptionCalls { throw TestError.failed }
        return Transcription(text: output, segments: [.init(start: 0, end: duration, text: output)])
    }
    func resume() { continuation?.resume(); continuation = nil }
    func sessionProfile() throws -> SessionTranscriptionProfile { profile }
    func prepareSession(profile: SessionTranscriptionProfile) async throws { try await prepare() }
    func transcribeSession(_ file: URL, profile: SessionTranscriptionProfile) async throws -> Data {
        sessionCalls += 1; requestedProfiles.append(profile)
        if suspend { waiting = true; await withCheckedContinuation { continuation = $0 }; waiting = false }
        if let transcriptionError { throw transcriptionError }
        if let sessionResponse { return sessionResponse }
        let audio = try AVAudioFile(forReading: file)
        let duration = Double(audio.length) / audio.processingFormat.sampleRate
        return try JSONEncoder().encode(Transcription(text: "[0][S01]\(output)[\(duration)]",
            segments: [.init(start: 0, end: duration, text: "[S01] \(output)", speakerID: "S01")],
            usage: .init(type: "tokens", inputTokens: 100, outputTokens: 10, totalTokens: 110)))
    }
    func notes(for transcript: String) async throws -> String {
        notesInput = transcript
        if suspend { waiting = true; await withCheckedContinuation { continuation = $0 }; waiting = false }
        if let notesError { throw notesError }
        return "notes"
    }
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
    var suspendMicrophone = false
    var microphoneWaiting = false
    private var microphoneContinuation: CheckedContinuation<Void, Never>?
    func currentStatus() -> AuthorizationStatus { status }
    func requestAll() async -> AuthorizationStatus {
        requestCalls += 1
        if suspend { waiting = true; await withCheckedContinuation { continuation = $0 }; waiting = false }
        return status
    }
    func requestMicrophone() async -> Bool {
        if suspendMicrophone { microphoneWaiting = true; await withCheckedContinuation { microphoneContinuation = $0 }; microphoneWaiting = false }
        return status.microphoneGranted
    }
    func requestListeningAccess() async -> Bool { status.listeningAccessGranted }
    func resume() { continuation?.resume(); continuation = nil }
    func resumeMicrophone() { microphoneContinuation?.resume(); microphoneContinuation = nil }
}
