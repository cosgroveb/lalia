import AVFoundation
import Testing
@testable import Lalia

struct SessionInterruptionTests {
    @Test func combinedCaptureFailuresPreserveManualRecovery() {
        let stream = CaptureFailure(kind: .stream, underlying: LaliaError("stream"))
        let permission = CaptureFailure(kind: .permission, underlying: LaliaError("permission"))
        let storage = CaptureFailure(kind: .storage, underlying: LaliaError("storage"))
        for (first, second, expected) in [
            (permission, stream, CaptureFailure.Kind.permission),
            (stream, permission, .permission),
            (permission, storage, .storage),
            (storage, permission, .storage),
            (storage, stream, .storage),
            (stream, storage, .storage)
        ] {
            #expect((CaptureFailure.preferred(first, second) as? CaptureFailure)?.kind == expected)
        }
        #expect((CaptureFailure.preferred(nil, stream) as? CaptureFailure)?.kind == .stream)
        #expect((CaptureFailure.preferred(permission, LaliaError("unreadable audio")) as? CaptureFailure)?.kind == .permission)
    }

    @Test @MainActor func unreadableLastSegmentKeepsEarlierAudioEligible() async throws {
        let fixture = TemporaryStore(); let store = fixture.store
        let record = try store.create(kind: .session)
        let writer = SourceAudioWriter(store: store, session: record.id, origin: 0)
        try writer.append(pcm(frames: 48_000), source: .microphone, timestamp: 0)
        try writer.finish()
        try writer.append(pcm(frames: 48_000), source: .microphone, timestamp: 1)
        let segment = try #require(store.read(record.id).segments.last)
        try Data().write(to: store.file(record.id, segment.partialFilename), options: .atomic)
        let output = CaptureOutput(writer: writer, failed: { _ in })
        do {
            try await output.finish()
            Issue.record("Expected unreadable final audio")
        } catch {
            #expect((error as? CaptureFailure)?.kind != .storage)
        }
        try store.update(record.id) { $0.captureState = .failed }
        _ = try store.recover(record.id)
        let input = try SessionAudioPreparer().prepare(session: record.id, store: store, allowIncomplete: true)
        #expect(input.incomplete == true)
        #expect(input.frames >= 16_000)
    }

    @Test func actualManifestWriteFailureIsStorageFailure() throws {
        let fixture = TemporaryStore(); let store = fixture.store
        let record = try store.create(kind: .session)
        let writer = SourceAudioWriter(store: store, session: record.id, origin: 0)
        let directory = store.directory(record.id)
        try FileManager.default.setAttributes([.posixPermissions: 0o500], ofItemAtPath: directory.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path) }
        do {
            try writer.append(pcm(frames: 48_000), source: .microphone, timestamp: 0)
            Issue.record("Expected unwritable manifest directory")
        } catch {
            #expect((error as? CaptureFailure)?.kind == .storage)
        }
    }

    @Test @MainActor func selectedMicrophoneDisconnectHonorsObservationLifetime() async throws {
        let center = NotificationCenter()
        let delivery = DisconnectDelivery()
        let capture = SessionCapture(notificationCenter: center,
            disconnectedDeviceID: { $0.object as? String }, deliverDisconnect: { delivery.pending.append($0) })
        var failures: [String] = []
        capture.observeMicrophone("selected") { failures.append($0.localizedDescription) }
        center.post(name: AVCaptureDevice.wasDisconnectedNotification, object: "other")
        for callback in delivery.pending { callback() }
        delivery.pending.removeAll()
        #expect(failures.isEmpty)
        center.post(name: AVCaptureDevice.wasDisconnectedNotification, object: "selected")
        let selected = try #require(delivery.pending.last)
        selected()
        #expect(failures == ["The microphone disconnected."])
        delivery.pending.removeAll()
        try await capture.stop()
        center.post(name: AVCaptureDevice.wasDisconnectedNotification, object: "selected")
        #expect(delivery.pending.isEmpty)
        capture.observeMicrophone("selected") { failures.append($0.localizedDescription) }
        selected()
        #expect(failures.count == 1)
        center.post(name: AVCaptureDevice.wasDisconnectedNotification, object: "selected")
        let current = try #require(delivery.pending.last)
        current()
        #expect(failures.count == 2)
        try await capture.stop()
    }

    @Test func policyPreservesManualRecoveryExceptions() {
        for cause in [SessionInterruption.sleep, .lock, .userSwitch, .captureFailure, .limit] {
            #expect(cause.endsSession)
            #expect(cause.attemptsTranscription)
            #expect(cause.noticeTitle != nil)
        }
        for cause in [SessionInterruption.storageFailure, .audioPermission, .quit] {
            #expect(cause.endsSession)
            #expect(!cause.attemptsTranscription)
            #expect(cause.noticeTitle == nil)
        }
        #expect(!SessionInterruption.keyboardUnavailable.endsSession)
    }
}

private final class DisconnectDelivery: @unchecked Sendable {
    var pending: [@MainActor @Sendable () -> Void] = []
}
