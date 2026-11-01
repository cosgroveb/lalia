import AVFoundation
import Foundation
import Testing
@testable import Lalia

struct SessionAudioPreparerTests {
    @Test func interruptedInputUsesCompleteSegmentsAndKeepsOriginals() throws {
        let fixture = TemporaryStore(); let store = fixture.store; let record = try store.create(kind: .session)
        let writer = SourceAudioWriter(store: store, session: record.id, origin: 0)
        try writer.append(pcm(rate: 16_000, channels: 1, frames: 16_000), source: .microphone, timestamp: 2)
        try writer.finish()
        let original = try #require(store.read(record.id).segments.first)
        let bytes = try Data(contentsOf: store.file(record.id, original.filename))
        try store.update(record.id) {
            $0.captureState = .failed
            $0.segments.append(AudioSegment(id: UUID(), source: .system, filename: "broken.caf", start: 0, sampleRate: 16_000, channels: 1))
        }
        #expect(throws: Error.self) { try SessionAudioPreparer().prepare(session: record.id, store: store) }
        let input = try SessionAudioPreparer().prepare(session: record.id, store: store, allowIncomplete: true)
        #expect(input.incomplete == true)
        #expect(input.frames == 48_000)
        let samples = try values(input, session: record.id, store: store)
        #expect(samples[..<32_000].allSatisfy { $0 == 0 })
        #expect(samples[32_000...].contains { $0 != 0 })
        #expect(try Data(contentsOf: store.file(record.id, original.filename)) == bytes)
        #expect(throws: Error.self) { try SessionAudioPreparer().prepare(session: record.id, store: store) }
    }

    @Test func interruptedInputWithNoUsableAudioFails() throws {
        let fixture = TemporaryStore(); let record = try fixture.store.create(kind: .session)
        try fixture.store.update(record.id) { $0.captureState = .failed }
        #expect(throws: Error.self) { try SessionAudioPreparer().prepare(session: record.id, store: fixture.store, allowIncomplete: true) }
    }

    private func values(_ input: SessionAnalysis.Input, session: UUID, store: SessionStore) throws -> [Float] {
        let file = try AVAudioFile(forReading: store.file(session, input.filename))
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)))
        try file.read(into: buffer)
        return Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
    }
    @Test func alignsResampledTracksGapsAndFormatChanges() throws {
        let fixture = TemporaryStore(); let store = fixture.store; let record = try store.create(kind: .session)
        let writer = SourceAudioWriter(store: store, session: record.id, origin: 0, rotation: 0.1)
        try writer.append(pcm(rate: 44_100, channels: 1, frames: 44_100), source: .microphone, timestamp: 0)
        try writer.append(pcm(rate: 48_000, frames: 24_000), source: .system, timestamp: 1.5)
        try writer.append(pcm(rate: 16_000, channels: 1, frames: 16_000), source: .microphone, timestamp: 2)
        try writer.finish(); try store.update(record.id) { $0.captureState = .complete }
        let originals = try store.read(record.id).segments.map { try Data(contentsOf: store.file(record.id, $0.filename)) }
        let input = try SessionAudioPreparer().prepare(session: record.id, store: store)
        #expect(input.frames == 48_000)
        let samples = try values(input, session: record.id, store: store)
        #expect(samples[16_000..<24_000].allSatisfy { $0 == 0 })
        #expect(samples[24_000..<32_000].contains { $0 != 0 })
        #expect(samples[32_000..<48_000].contains { $0 != 0 })
        #expect(try SessionAudioPreparer().prepare(session: record.id, store: store) == input)
        #expect(try store.read(record.id).segments.map { try Data(contentsOf: store.file(record.id, $0.filename)) } == originals)
    }
    @Test func silentTrackDoesNotAttenuateQuietSpeechAndOverlapDoesNotClip() throws {
        for amplitude: Float in [0, 0.0001, 0.8] {
            let fixture = TemporaryStore(); let store = fixture.store; let record = try store.create(kind: .session)
            let writer = SourceAudioWriter(store: store, session: record.id, origin: 0)
            let source = try pcm(rate: 16_000, channels: 1, frames: 16_000, amplitude: amplitude)
            try writer.append(source, source: .microphone, timestamp: 0)
            try writer.append(pcm(rate: 16_000, channels: 1, frames: 16_000, amplitude: amplitude == 0.8 ? 0.8 : 0), source: .system, timestamp: 0)
            try writer.finish(); try store.update(record.id) { $0.captureState = .complete }
            let input = try SessionAudioPreparer().prepare(session: record.id, store: store)
            let samples = try values(input, session: record.id, store: store)
            #expect(input.silence == (amplitude == 0))
            #expect(samples.allSatisfy { abs($0) <= 1 })
            if amplitude == 0.0001 {
                #expect(samples.contains { $0 != 0 })
                #expect(abs(samples[20] - source.floatChannelData![0][20]) < 1.0 / 32768)
            }
            if amplitude == 0.8 { #expect(samples.map { abs($0) }.max()! > 0.99) }
        }
    }
    @Test func capTrimsActualFramesAndRejectsLongLegacyWithoutCropping() throws {
        let fixture = TemporaryStore(); let store = fixture.store; let record = try store.create(kind: .session)
        let writer = SourceAudioWriter(store: store, session: record.id, origin: 0, maximumDuration: 0.25)
        try writer.append(pcm(rate: 16_000, channels: 1, frames: 16_000), source: .microphone, timestamp: 0)
        try writer.append(pcm(rate: 16_000, channels: 1, frames: 16_000), source: .system, timestamp: 0.2)
        try writer.append(pcm(frames: 48_000), source: .microphone, timestamp: 1)
        try writer.finish(); try store.update(record.id) { $0.captureState = .complete }
        #expect(try store.read(record.id).duration <= 0.25)
        #expect(try SessionAudioPreparer().prepare(session: record.id, store: store).frames == 4_000)
        try store.update(record.id) { $0.segments[0].start = 3_600; $0.analysis?.input = nil }
        #expect(throws: Error.self) { try SessionAudioPreparer().prepare(session: record.id, store: store) }
        #expect(try store.read(record.id).segments[0].frames == 4_000)
    }
}
