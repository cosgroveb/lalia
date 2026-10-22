import AVFoundation
import Foundation
import Testing
@testable import Lalia

final class TemporaryStore: @unchecked Sendable {
    let store: SessionStore
    init() { store = SessionStore(root: FileManager.default.temporaryDirectory.appending(path: "LaliaTests-\(UUID())")) }
    deinit { try? FileManager.default.removeItem(at: store.root) }
}

func pcm(rate: Double = 48_000, channels: AVAudioChannelCount = 2, frames: Int, amplitude: Float = 0.1) throws -> AVAudioPCMBuffer {
    let format = try #require(AVAudioFormat(standardFormatWithSampleRate: rate, channels: channels))
    let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)))
    buffer.frameLength = buffer.frameCapacity
    for channel in 0..<Int(channels) {
        for frame in 0..<frames { buffer.floatChannelData![channel][frame] = amplitude * sin(Float(frame) * 0.1) }
    }
    return buffer
}

func savedAudio(_ store: SessionStore, seconds: Double = 1, amplitude: Float = 0.1) throws -> SessionRecord {
    let record = try store.create(kind: .session)
    let writer = SourceAudioWriter(store: store, session: record.id, origin: 100)
    try writer.append(pcm(rate: 16_000, channels: 1, frames: Int(seconds * 16_000), amplitude: amplitude), source: .microphone, timestamp: 100)
    try writer.finish()
    return try store.update(record.id) { $0.captureState = .complete }
}

struct AudioTests {
    @Test func writeFailureRetainsCompletedAudio() throws {
        let fixture = TemporaryStore(); let store = fixture.store
        let record = try store.create(kind: .session)
        let writer = SourceAudioWriter(store: store, session: record.id, origin: 0, rotation: 1)
        try writer.append(pcm(rate: 16_000, channels: 1, frames: 16_000), source: .microphone, timestamp: 0)
        let saved = try store.read(record.id)
        let source = try store.file(record.id, saved.segments[0].filename)
        let original = try Data(contentsOf: source)
        let manifest = try store.file(record.id, "session.json")
        let backup = try Data(contentsOf: manifest)
        try FileManager.default.removeItem(at: manifest)
        try FileManager.default.createDirectory(at: manifest, withIntermediateDirectories: false)
        #expect(throws: Error.self) { try writer.append(pcm(frames: 48_000), source: .system, timestamp: 1) }
        #expect(try Data(contentsOf: source) == original)
        try FileManager.default.removeItem(at: manifest)
        try store.write(backup, to: manifest)
        #expect(try store.history(recover: true).records[0].segments[0].complete)
    }
    @Test func rotatesInsideBuffersWithoutLosingFrames() throws {
        let fixture = TemporaryStore(); let store = fixture.store
        let record = try store.create(kind: .session)
        let writer = SourceAudioWriter(store: store, session: record.id, origin: 100, rotation: 1)
        try writer.append(pcm(frames: 120_000), source: .microphone, timestamp: 100.25)
        try writer.finish()
        let saved = try store.read(record.id)
        #expect(saved.segments.map(\.frames) == [48_000, 48_000, 24_000])
        #expect(saved.segments.map(\.start) == [0.25, 1.25, 2.25])
        #expect(saved.segments.allSatisfy { $0.complete })
        let values = try saved.segments.flatMap { segment -> [Float] in
            let file = try AVAudioFile(forReading: store.file(record.id, segment.filename))
            let buffer = try #require(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)))
            try file.read(into: buffer)
            return Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
        }
        #expect(values.count == 120_000)
        #expect(abs(values[48_001] - 0.1 * sin(Float(48_001) * 0.1)) < 0.000001)
    }

    @Test func preservesMixedSourcesGapsFormatChangesAndConverterTail() throws {
        let fixture = TemporaryStore(); let store = fixture.store
        let record = try store.create(kind: .session)
        let writer = SourceAudioWriter(store: store, session: record.id, origin: 100, rotation: 0.1)
        try writer.append(pcm(rate: 44_100, channels: 1, frames: 44_100), source: .microphone, timestamp: 100)
        try writer.append(pcm(frames: 48_000), source: .system, timestamp: 100.5)
        try writer.append(pcm(rate: 44_100, channels: 1, frames: 22_050), source: .microphone, timestamp: 102)
        try writer.append(pcm(rate: 48_000, channels: 1, frames: 48_000), source: .microphone, timestamp: 102.5)
        try writer.finish()
        try store.update(record.id) { $0.captureState = .complete }
        let systemFrames = try store.read(record.id).segments.filter { $0.source == .system }.map(\.frames)
        #expect(systemFrames == Array(repeating: 4800, count: 10))
        try AudioJobPreparer().prepare(session: record.id, store: store)
        let jobs = try store.read(record.id).jobs
        #expect(jobs.map(\.start) == [0, 2, 2.5, 0.5])
        #expect(jobs.map(\.frames) == [16_000, 8_000, 16_000, 16_000])
        #expect(jobs.allSatisfy { $0.state == .pending })
    }

    @Test func preservesQuietSpeechAndSkipsOnlyDigitalSilence() throws {
        let fixture = TemporaryStore(); let store = fixture.store
        let quiet = try savedAudio(store, amplitude: 0.0001)
        let silence = try savedAudio(store, amplitude: 0)
        try AudioJobPreparer().prepare(session: quiet.id, store: store)
        try AudioJobPreparer().prepare(session: silence.id, store: store)
        #expect(try store.read(quiet.id).jobs.first?.state == .pending)
        #expect(try store.read(silence.id).jobs.first?.state == .silence)
    }

    @Test func boundsTenMinuteUploadsAndConservesFramesAcrossStorageFiles() throws {
        let fixture = TemporaryStore(); let store = fixture.store
        let record = try store.create(kind: .session)
        let writer = SourceAudioWriter(store: store, session: record.id, origin: 0)
        let buffer = try pcm(rate: 16_000, channels: 1, frames: 16_000)
        for second in 0..<610 { try writer.append(buffer, source: .microphone, timestamp: Double(second)) }
        try writer.finish()
        try store.update(record.id) { $0.captureState = .complete }
        try AudioJobPreparer().prepare(session: record.id, store: store)
        let saved = try store.read(record.id)
        #expect(saved.jobs.count == 2)
        #expect(saved.jobs.reduce(0) { $0 + $1.frames } == 610 * 16_000)
        #expect(saved.jobs[1].start == saved.jobs[0].duration)
        for job in saved.jobs {
            #expect(job.duration <= 600)
            #expect(try store.file(record.id, job.filename).resourceValues(forKeys: [.fileSizeKey]).fileSize! <= 20_000_000)
        }
        try AudioJobPreparer().prepare(session: record.id, store: store)
        #expect(try store.read(record.id).jobs.map(\.id) == saved.jobs.map(\.id))
    }
}
