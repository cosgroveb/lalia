import AVFoundation
import Foundation

struct SessionRecord: Codable, Identifiable, Sendable {
    enum Kind: String, Codable { case dictation, session }
    enum CaptureState: String, Codable { case recording, complete, interrupted, failed }
    var schemaVersion = 2
    var id: UUID
    var kind: Kind
    var createdAt: Date
    var captureState: CaptureState = .recording
    var captureError: String?
    var segments: [AudioSegment] = []
    var jobs: [TranscriptionJob] = []
    var jobsPrepared = false
    var transcriptionError: String?
    var notes: String?
    var notesError: String?
    var notesRevision: UUID?
    var analysis: SessionAnalysis?
    var endedAtLimit: Bool?
    var notesStale: Bool { notes != nil && notesRevision != analysis?.active?.revision }
    var transcriptionComplete: Bool {
        if analysis?.active != nil { return segments.allSatisfy { $0.complete } }
        return legacyTranscriptionComplete
    }
    private var legacyTranscriptionComplete: Bool {
        jobsPrepared && segments.allSatisfy { $0.complete } && jobs.allSatisfy { $0.state == .complete || $0.state == .silence }
    }
    var hasLegacyTranscript: Bool { legacyTranscriptionComplete || !blocks.isEmpty }
    var duration: Double { segments.map { $0.start + $0.duration }.max() ?? 0 }
    var transcript: String {
        if let active = analysis?.active { return active.transcript }
        return blocks.map { "[\(Self.timestamp($0.start))–\(Self.timestamp($0.end))] \($0.source.label) (approximate)\n\($0.text)" }.joined(separator: "\n\n")
    }
    var plainTranscript: String { analysis?.active?.transcript ?? blocks.map(\.text).joined(separator: " ") }
    struct Block { var source: AudioSource; var start: Double; var end: Double; var text: String }
    var blocks: [Block] {
        jobs.flatMap { job -> [Block] in
            guard job.state == .complete, let response = job.response else { return [] }
            if let segments = response.segments, !segments.isEmpty {
                return segments.compactMap { segment in
                    guard let text = trimmedTranscript(segment.text) else { return nil }
                    return Block(source: job.source, start: job.start + segment.start, end: job.start + segment.end, text: text)
                }
            }
            guard let text = trimmedTranscript(response.text) else { return [] }
            return [Block(source: job.source, start: job.start, end: job.start + job.duration, text: text)]
        }.sorted { $0.start == $1.start ? $0.source.rawValue < $1.source.rawValue : $0.start < $1.start }
    }
    static func timestamp(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0, seconds < Double(Int.max) else { return "Unknown" }
        let value = Int(seconds)
        return String(format: "%02ld:%02ld:%02ld", value / 3600, value / 60 % 60, value % 60)
    }
}

enum AudioSource: String, Codable, CaseIterable, Sendable {
    case microphone, system
    var label: String { self == .microphone ? "Microphone" : "Mac audio" }
}

struct AudioSegment: Codable, Identifiable, Sendable {
    var id: UUID
    var source: AudioSource
    var filename: String
    var start: Double
    var sampleRate: Double
    var channels: UInt32
    var frames: Int64 = 0
    var complete = false
    var error: String?
    var duration: Double { Double(frames) / sampleRate }
    var partialFilename: String {
        let url = URL(fileURLWithPath: filename)
        return url.deletingPathExtension().lastPathComponent + ".partial." + url.pathExtension
    }
}

struct TranscriptionJob: Codable, Identifiable, Sendable {
    enum State: String, Codable { case pending, running, complete, failed, silence }
    var id: String
    var source: AudioSource
    var filename: String
    var start: Double
    var frames: Int64
    var state: State = .pending
    var response: Transcription?
    var error: String?
    var duration: Double { Double(frames) / 16_000 }
}

// The capture queue and main actor share the store. A transaction reloads and saves
// under one lock so a completed segment cannot overwrite a concurrent status update.
final class SessionStore: @unchecked Sendable {
    let root: URL
    private let lock = NSRecursiveLock()
    private let manager = FileManager.default
    init(root: URL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appending(path: "Lalia/Sessions")) {
        self.root = root
    }

    func directory(_ id: UUID) -> URL { root.appending(path: id.uuidString, directoryHint: .isDirectory) }

    func file(_ id: UUID, _ name: String) throws -> URL {
        guard !name.isEmpty, name != ".", name != "..", !name.contains("/"), !name.contains("\\"), !name.contains("\0") else {
            throw LaliaError("Unsafe session filename: \(name)")
        }
        let directory = directory(id).standardizedFileURL
        let file = directory.appending(path: name)
        guard (try? manager.destinationOfSymbolicLink(atPath: directory.path)) == nil,
              (try? manager.destinationOfSymbolicLink(atPath: file.path)) == nil else {
            throw LaliaError("Session paths cannot be symbolic links.")
        }
        guard file.resolvingSymlinksInPath().deletingLastPathComponent() == directory.resolvingSymlinksInPath() else {
            throw LaliaError("Session files cannot link outside their recording directory.")
        }
        return file
    }

    func create(kind: SessionRecord.Kind) throws -> SessionRecord {
        lock.lock(); defer { lock.unlock() }
        let record = SessionRecord(id: UUID(), kind: kind, createdAt: Date())
        try manager.createDirectory(at: root, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        try manager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: root.path)
        try manager.createDirectory(at: directory(record.id), withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        try save(record)
        return record
    }

    func read(_ id: UUID) throws -> SessionRecord {
        lock.lock(); defer { lock.unlock() }
        let record = try JSONDecoder().decode(SessionRecord.self, from: Data(contentsOf: file(id, "session.json")))
        guard [1, 2].contains(record.schemaVersion), record.id == id else { throw LaliaError("Unsupported or mismatched session manifest.") }
        for segment in record.segments {
            _ = try file(id, segment.filename)
            _ = try file(id, segment.partialFilename)
            guard segment.sampleRate.isFinite, segment.sampleRate > 0,
                  segment.channels > 0, segment.frames >= 0,
                  validTimeRange(start: segment.start, duration: segment.duration) else { throw LaliaError("Invalid source audio metadata.") }
        }
        for job in record.jobs {
            _ = try file(id, job.filename)
            guard job.frames > 0, job.frames <= 600 * 16_000,
                  validTimeRange(start: job.start, duration: job.duration) else { throw LaliaError("Invalid transcription job metadata.") }
            if let response = job.response { _ = try response.validated(duration: job.duration) }
            if job.state == .complete && job.response == nil { throw LaliaError("Completed transcription has no response.") }
        }
        guard Set(record.segments.map(\.id)).count == record.segments.count,
              Set(record.jobs.map(\.id)).count == record.jobs.count else { throw LaliaError("Duplicate session identifiers.") }
        if let analysis = record.analysis {
            if let input = analysis.input {
                _ = try file(id, input.filename)
                guard input.frames > 0, input.frames <= Int64(SessionTranscriptionProfile.maximumDuration * 16_000) else { throw LaliaError("Invalid mixed audio metadata.") }
            }
            if let attempt = analysis.attempt { try attempt.profile.validate() }
            let duration = analysis.input?.duration ?? record.duration
            if let active = analysis.active { try active.validate(duration: duration, silence: analysis.input?.silence == true) }
            if let previous = analysis.previous { try previous.validate(duration: duration, silence: analysis.input?.silence == true) }
            if analysis.attempt?.state == .complete, analysis.active?.id != analysis.attempt?.id {
                throw LaliaError("Completed analysis has no matching transcript.")
            }
            if analysis.previousIsLegacy == true, analysis.previous != nil || !record.hasLegacyTranscript {
                throw LaliaError("Previous source transcript is unavailable or ambiguous.")
            }
        }
        return record
    }

    private func validTimeRange(start: Double, duration: Double) -> Bool {
        let end = start + duration
        return start.isFinite && start >= 0 && start < Double(Int.max)
            && duration.isFinite && duration >= 0 && duration < Double(Int.max)
            && end.isFinite && end < Double(Int.max)
    }

    @discardableResult func update(_ id: UUID, _ change: (inout SessionRecord) throws -> Void) throws -> SessionRecord {
        lock.lock(); defer { lock.unlock() }
        var record = try read(id)
        try change(&record)
        try save(record)
        return record
    }

    private func save(_ record: SessionRecord) throws {
        var record = record
        if record.schemaVersion == 1 {
            let backup = try file(record.id, "session-v1.backup.json")
            if !manager.fileExists(atPath: backup.path) {
                try write(Data(contentsOf: file(record.id, "session.json")), to: backup)
            }
            record.schemaVersion = 2
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try write(encoder.encode(record), to: file(record.id, "session.json"))
    }

    func write(_ data: Data, to url: URL) throws {
        try data.write(to: url, options: .atomic)
        try manager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }

    func privateFile(_ file: URL) throws {
        guard manager.createFile(atPath: file.path, contents: nil, attributes: [.posixPermissions: 0o600]) else {
            throw CocoaError(.fileWriteUnknown)
        }
    }

    func finishSegment(_ id: UUID, segmentID: UUID) throws {
        let record = try read(id)
        guard let segment = record.segments.first(where: { $0.id == segmentID }) else { throw LaliaError("Missing source segment.") }
        let partial = try file(id, segment.partialFilename)
        let final = try file(id, segment.filename)
        let source = manager.fileExists(atPath: final.path) ? final : partial
        let audio = try AVAudioFile(forReading: source)
        let frames = audio.length
        guard frames > 0 else { throw LaliaError("Recording contains no audio frames.") }
        let sampleRate = audio.processingFormat.sampleRate
        let channels = audio.processingFormat.channelCount
        if source == partial { try manager.moveItem(at: partial, to: final) }
        try manager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: final.path)
        try update(id) { record in
            guard let index = record.segments.firstIndex(where: { $0.id == segmentID }) else { return }
            record.segments[index].frames = frames
            record.segments[index].sampleRate = sampleRate
            record.segments[index].channels = channels
            record.segments[index].complete = true
            record.segments[index].error = nil
            if !segment.complete {
                record.jobsPrepared = false
                record.analysis?.input = nil
                if record.notes != nil { record.notesError = "Recovered audio changed the transcript. Generate notes again after transcription." }
            }
        }
    }

    func history(recover: Bool = false) throws -> (records: [SessionRecord], diagnostics: [String]) {
        guard manager.fileExists(atPath: root.path) else { return ([], []) }
        var records: [SessionRecord] = []
        var diagnostics: [String] = []
        for directory in try manager.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) {
            guard let id = UUID(uuidString: directory.lastPathComponent) else { continue }
            do {
                var record = try read(id)
                if recover { record = try self.recover(id) }
                for segment in record.segments where segment.complete {
                    if !manager.fileExists(atPath: try file(id, segment.filename).path) {
                        diagnostics.append("\(id): missing recording \(segment.filename)")
                    }
                }
                records.append(record)
            } catch { diagnostics.append("\(id): \(error.localizedDescription)") }
        }
        return (records.sorted { $0.createdAt > $1.createdAt }, diagnostics)
    }

    @discardableResult func recover(_ id: UUID) throws -> SessionRecord {
        let record = try read(id)
        for segment in record.segments where !segment.complete {
            do { try finishSegment(id, segmentID: segment.id) }
            catch {
                try update(id) { record in
                    if let index = record.segments.firstIndex(where: { $0.id == segment.id }) {
                        record.segments[index].error = "Unfinished audio retained: \(error.localizedDescription)"
                    }
                }
            }
        }
        return try update(id) { record in
            if record.captureState == .recording {
                record.captureState = .interrupted
                record.captureError = "Capture was interrupted. Recovered audio is available for Retry."
            }
            for index in record.jobs.indices where record.jobs[index].state == .running {
                record.jobs[index].state = .pending
            }
            if record.analysis?.attempt?.state == .running || record.analysis?.attempt?.state == .preparing {
                let failure = "Analysis was interrupted. Saved audio and previous transcript are retained. Retry may wait for an earlier oMLX request."
                record.analysis?.attempt?.state = .failed
                record.analysis?.attempt?.error = failure
                record.transcriptionError = failure
            }
        }
    }

    func editTranscript(_ id: UUID, _ edit: (inout SpeakerTranscript) throws -> Void) throws {
        try update(id) { record in
            guard var active = record.analysis?.active else { throw LaliaError("No speaker transcript is available.") }
            try edit(&active)
            try active.validate(duration: record.analysis?.input?.duration ?? record.duration, silence: record.analysis?.input?.silence == true)
            record.analysis?.active = active
        }
    }

    func restoreTranscript(_ id: UUID) throws {
        try update(id) { record in
            guard var analysis = record.analysis else { throw LaliaError("No previous transcript is available.") }
            let active = analysis.active
            if analysis.previousIsLegacy == true {
                analysis.active = nil
                analysis.previous = active
                analysis.previousIsLegacy = false
            } else if let previous = analysis.previous {
                analysis.active = previous
                analysis.previous = active
                analysis.previousIsLegacy = active == nil && record.hasLegacyTranscript
            } else { throw LaliaError("No previous transcript is available.") }
            // Restoring selects an older result and clears the latest attempt status.
            analysis.attempt = nil
            record.analysis = analysis
            record.transcriptionError = nil
        }
    }

    func export(_ id: UUID, to url: URL) throws { try write(Data(read(id).transcript.utf8), to: url) }
}
