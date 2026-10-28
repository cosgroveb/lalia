import AVFoundation
import Foundation
import Testing
@testable import Lalia

struct SessionStoreTests {
    @Test func earlierLongAnalysisSurvivesReducedRecordingLimit() throws {
        let fixture = TemporaryStore(); let store = fixture.store
        let record = try savedAudio(store, seconds: 2)
        let transcript = try SpeakerTranscript(attempt: speakerAttempt(), duration: 5_400)
        try store.update(record.id) {
            $0.analysis = SessionAnalysis(input: .init(filename: "session-mix.wav", frames: 5_400 * 16_000, silence: false), active: transcript)
        }
        let reopened = try SessionStore(root: store.root).read(record.id)
        #expect(reopened.analysis?.input?.duration == 5_400)
        #expect(reopened.transcript == transcript.transcript)
    }
    @Test func interruptedAnalysisRecoverySurfacesRetryReasonInHistory() throws {
        let fixture = TemporaryStore(); let store = fixture.store
        for state: SessionAnalysis.Attempt.State in [.preparing, .running] {
            let record = try savedAudio(store, seconds: 2)
            let active = try SpeakerTranscript(attempt: speakerAttempt(), duration: 2)
            try store.update(record.id) {
                $0.analysis = SessionAnalysis(attempt: .init(profile: active.profile, state: state), active: active)
            }
            let recovered = try store.history(recover: true).records.first { $0.id == record.id }
            #expect(recovered?.analysis?.attempt?.state == .failed)
            #expect(recovered?.transcriptionError == recovered?.analysis?.attempt?.error)
            #expect(recovered?.transcriptionError?.contains("interrupted") == true)
            #expect(recovered?.transcript == active.transcript)
        }
    }
    @Test func restoringTranscriptClearsFailedReplacementError() throws {
        let fixture = TemporaryStore(); let store = fixture.store
        let record = try savedAudio(store, seconds: 2)
        let previous = try SpeakerTranscript(attempt: speakerAttempt(), duration: 2)
        let active = try SpeakerTranscript(attempt: speakerAttempt(), duration: 2)
        try store.update(record.id) {
            $0.analysis = SessionAnalysis(attempt: .init(profile: active.profile, state: .failed, error: "replacement failed"), active: active, previous: previous)
            $0.transcriptionError = "replacement failed"
            $0.captureError = "retained capture warning"
        }
        try store.restoreTranscript(record.id)
        let restored = try store.read(record.id)
        #expect(restored.analysis?.active?.id == previous.id)
        #expect(restored.analysis?.attempt == nil)
        #expect(restored.transcriptionError == nil)
        #expect(restored.captureError == "retained capture warning")
    }
    @Test func migratesSchemaOneOnWriteWithPrivateOriginalBackup() throws {
        let fixture = TemporaryStore(); let store = fixture.store
        var record = try savedAudio(store)
        record.schemaVersion = 1
        let original = try JSONEncoder().encode(record)
        try store.write(original, to: store.file(record.id, "session.json"))
        #expect(try store.read(record.id).schemaVersion == 1)
        #expect(!FileManager.default.fileExists(atPath: try store.file(record.id, "session-v1.backup.json").path))
        try store.update(record.id) { $0.notes = "preserved" }
        #expect(try store.read(record.id).schemaVersion == 2)
        let backup = try store.file(record.id, "session-v1.backup.json")
        #expect(try Data(contentsOf: backup) == original)
        #expect(try FileManager.default.attributesOfItem(atPath: backup.path)[.posixPermissions] as? Int == 0o600)
        try store.update(record.id) { $0.notes = "changed" }
        #expect(try Data(contentsOf: backup) == original)
    }
    @Test func preservesCorrectionsDuringFailureRecoveryRestoreAndExport() throws {
        let fixture = TemporaryStore(); let store = fixture.store
        let record = try savedAudio(store, seconds: 2)
        let input = try SessionAudioPreparer().prepare(session: record.id, store: store)
        let attempt = try speakerAttempt()
        let accepted = try SpeakerTranscript(attempt: attempt, duration: input.duration)
        try store.update(record.id) { $0.analysis?.active = accepted; $0.notes = "notes"; $0.notesRevision = accepted.revision }
        try store.editTranscript(record.id) { try $0.rename("S01", to: "Brian") }
        let corrected = try store.read(record.id)
        #expect(corrected.notesStale)
        let before = corrected.transcript
        try store.update(record.id) { $0.analysis?.attempt = SessionAnalysis.Attempt(profile: attempt.profile, state: .running) }
        let recovered = try store.recover(record.id)
        #expect(recovered.analysis?.attempt?.state == .failed)
        #expect(recovered.transcript == before)
        let replacement = try SpeakerTranscript(attempt: speakerAttempt(), duration: input.duration)
        try store.update(record.id) { $0.analysis?.previous = corrected.analysis?.active; $0.analysis?.active = replacement }
        try store.restoreTranscript(record.id)
        let reopened = try SessionStore(root: store.root).read(record.id)
        #expect(reopened.transcript == before)
        #expect(reopened.analysis?.active?.turns.map(\.id) == accepted.turns.map(\.id))
        #expect(reopened.notes == "notes")
        let output = store.root.appending(path: "speakers.txt")
        try store.export(record.id, to: output)
        #expect(try String(contentsOf: output, encoding: .utf8) == before)
        try store.update(record.id) { $0.analysis?.input?.filename = "../outside.wav" }
        #expect(throws: Error.self) { try store.read(record.id) }
    }
    @Test func rejectsBrokenTurnCorrectionReferences() throws {
        let fixture = TemporaryStore(); let store = fixture.store
        let record = try savedAudio(store, seconds: 2)
        let accepted = try SpeakerTranscript(attempt: speakerAttempt(), duration: 2)
        try store.update(record.id) { $0.analysis = SessionAnalysis(active: accepted) }
        #expect(throws: Error.self) { try store.editTranscript(record.id) { $0.assignments[UUID().uuidString] = "S01" } }
        #expect(try store.read(record.id).analysis?.active?.assignments.isEmpty == true)
    }
    @Test func rejectsUnrepresentableTimesWithoutRemovingRecordings() throws {
        let fixture = TemporaryStore(); let store = fixture.store
        for invalid in ["source start", "source duration", "source end", "job start", "job end"] {
            let record = try savedAudio(store)
            try store.update(record.id) { record in
                switch invalid {
                case "source start": record.segments[0].start = 1e20
                case "source duration": record.segments[0].sampleRate = 1e-300
                case "source end":
                    record.segments[0].start = Double(Int.max).nextDown
                    record.segments[0].frames = 32_000_000
                default:
                    record.jobs = [TranscriptionJob(id: "invalid", source: .microphone, filename: "invalid.wav",
                                                   start: invalid == "job start" ? 1e20 : Double(Int.max).nextDown, frames: 9_600_000)]
                }
            }
            #expect(throws: Error.self) { try store.read(record.id) }
            #expect(FileManager.default.fileExists(atPath: try store.file(record.id, record.segments[0].filename).path))
        }
        let history = try store.history(recover: true)
        #expect(history.records.isEmpty)
        #expect(history.diagnostics.count == 5)
    }

    @Test func timestampFormattingHandlesInvalidInputs() {
        for value in [Double.nan, .infinity, -.infinity, -1, 1e20, Double(Int.max)] {
            #expect(SessionRecord.timestamp(value) == "Unknown")
        }
        #expect(SessionRecord.timestamp(90_061.5) == "25:01:01")
        #expect(SessionRecord.timestamp(0) == "00:00:00")
        let largest = Double(Int.max).nextDown
        #expect(SessionRecord.timestamp(largest).hasPrefix(String(Int(largest) / 3600)))
    }

    @Test func recoveredAudioAddsJobsWithoutReplacingCompletedRanges() throws {
        let fixture = TemporaryStore(); let store = fixture.store
        let record = try savedAudio(store)
        let middle = AudioSegment(id: UUID(), source: .microphone, filename: "middle.caf", start: 1, sampleRate: 16_000, channels: 1)
        let last = AudioSegment(id: UUID(), source: .microphone, filename: "last.caf", start: 2, sampleRate: 16_000, channels: 1)
        try store.update(record.id) { $0.segments += [middle, last]; $0.captureState = .failed }
        for segment in [middle, last] {
            let buffer = try pcm(rate: 16_000, channels: 1, frames: 16_000)
            let file = try AVAudioFile(forWriting: store.file(record.id, segment.partialFilename), settings: buffer.format.settings)
            try file.write(from: buffer)
            file.close()
        }
        try store.finishSegment(record.id, segmentID: last.id)
        try AudioJobPreparer().prepare(session: record.id, store: store)
        try store.update(record.id) { record in
            for index in record.jobs.indices { record.jobs[index].state = .complete; record.jobs[index].response = Transcription(text: "saved \(index)") }
            record.notes = "earlier notes"
        }
        let original = try store.read(record.id)
        #expect(!original.transcriptionComplete)
        _ = try store.history(recover: true)
        #expect(try !store.read(record.id).jobsPrepared)
        try AudioJobPreparer().prepare(session: record.id, store: store)
        let recovered = try store.read(record.id)
        #expect(recovered.jobs.count == 3)
        #expect(recovered.jobs.reduce(0) { $0 + $1.frames } == 48_000)
        #expect(recovered.jobs.filter { $0.state == .complete }.map(\.id) == original.jobs.map(\.id))
        #expect(recovered.jobs.filter { $0.state == .complete }.map(\.response) == original.jobs.map(\.response))
        #expect(recovered.notes == "earlier notes")
        #expect(recovered.notesError != nil)
    }
    @Test func savesReopensAndRetainsPrivateFiles() throws {
        let fixture = TemporaryStore(); let store = fixture.store
        let record = try savedAudio(store)
        let reopened = SessionStore(root: store.root)
        #expect(try reopened.read(record.id).segments[0].frames == 16_000)
        for file in [store.root, store.directory(record.id)] {
            #expect(try FileManager.default.attributesOfItem(atPath: file.path)[.posixPermissions] as? Int == 0o700)
        }
        for name in ["session.json", record.segments[0].filename] {
            #expect(try FileManager.default.attributesOfItem(atPath: store.file(record.id, name).path)[.posixPermissions] as? Int == 0o600)
        }
    }
    @Test func recoversPartialFilesAndInterruptedJobsWithoutDeletingOriginals() throws {
        let fixture = TemporaryStore(); let store = fixture.store
        let record = try store.create(kind: .session)
        let segment = AudioSegment(id: UUID(), source: .microphone, filename: "saved.caf", start: 1, sampleRate: 16_000, channels: 1)
        try store.update(record.id) { $0.segments.append(segment) }
        let partial = try store.file(record.id, segment.partialFilename)
        do {
            let buffer = try pcm(rate: 16_000, channels: 1, frames: 16_000)
            let file = try AVAudioFile(forWriting: partial, settings: buffer.format.settings)
            try file.write(from: buffer)
        }
        let history = try store.history(recover: true)
        #expect(history.diagnostics.isEmpty)
        #expect(history.records[0].captureState == .interrupted)
        #expect(history.records[0].segments[0].complete)
        #expect(history.records[0].segments[0].start == 1)
        try AudioJobPreparer().prepare(session: record.id, store: store)
        try store.update(record.id) { $0.jobs[0].state = .running }
        #expect(try store.history(recover: true).records[0].jobs[0].state == .pending)
        #expect(FileManager.default.fileExists(atPath: try store.file(record.id, segment.filename).path))
    }
    @Test func keepsCorruptSessionsAndReportsMissingAudio() throws {
        let fixture = TemporaryStore(); let store = fixture.store
        let bad = try store.create(kind: .session)
        let good = try savedAudio(store)
        try Data("invalid".utf8).write(to: store.file(bad.id, "session.json"))
        try FileManager.default.removeItem(at: store.file(good.id, good.segments[0].filename))
        let history = try store.history(recover: true)
        #expect(history.diagnostics.count == 2)
        #expect(history.records.map(\.id) == [good.id])
        #expect(FileManager.default.fileExists(atPath: store.directory(bad.id).path))
    }
    @Test func rejectsTraversalAndExternalSymlinks() throws {
        let fixture = TemporaryStore(); let store = fixture.store
        let record = try store.create(kind: .session)
        for name in ["../secret", "/tmp/audio", "sub/audio", "..", ""] { #expect(throws: Error.self) { try store.file(record.id, name) } }
        try FileManager.default.createSymbolicLink(atPath: store.directory(record.id).appending(path: "link").path, withDestinationPath: "/tmp/outside")
        #expect(throws: Error.self) { try store.file(record.id, "link") }
    }
    @Test func exportsChronologicalSourceLabelsAndKeepsRepeatedWords() throws {
        let fixture = TemporaryStore(); let store = fixture.store
        let record = try store.create(kind: .session)
        try store.update(record.id) {
            $0.jobs = [
                TranscriptionJob(id: "system", source: .system, filename: "system.wav", start: 5, frames: 32_000, state: .complete, response: Transcription(text: "yes yes")),
                TranscriptionJob(id: "mic", source: .microphone, filename: "mic.wav", start: 0, frames: 32_000, state: .complete, response: Transcription(text: "begin")),
            ]
        }
        let export = store.root.appending(path: "export.txt")
        try store.export(record.id, to: export)
        let text = try String(contentsOf: export, encoding: .utf8)
        #expect(text == "[00:00:00–00:00:02] Microphone (approximate)\nbegin\n\n[00:00:05–00:00:07] Mac audio (approximate)\nyes yes")
    }
}
