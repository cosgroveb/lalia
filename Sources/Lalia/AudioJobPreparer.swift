import AVFoundation
import Foundation

struct AudioJobPreparer: Sendable {
    var maximumSeconds: Double = 600
    private let rate = 16_000.0

    func prepare(session id: UUID, store: SessionStore) throws {
        let record = try store.read(id)
        if record.jobsPrepared, try record.jobs.allSatisfy({ job in
            if job.state == .complete || job.state == .silence { return true }
            return FileManager.default.fileExists(atPath: try store.file(id, job.filename).path)
        }) { return }
        guard record.captureState != .recording else { throw LaliaError("Stop recording before transcription.") }
        guard record.segments.contains(where: \.complete) else { throw LaliaError("No recoverable audio is available.") }
        var expectedJobs: Set<String> = []
        for source in AudioSource.allCases {
            let segments = record.segments.filter { $0.source == source && $0.complete }.sorted { $0.start < $1.start }
            var runs: [[AudioSegment]] = []
            for segment in segments {
                let savedBoundary = record.jobs.contains { $0.id.hasPrefix("\(source.rawValue)-\(segment.id)-") }
                if let previous = runs.last?.last,
                   !savedBoundary,
                   previous.sampleRate == segment.sampleRate, previous.channels == segment.channels,
                   abs(previous.start + previous.duration - segment.start) <= 1.5 / segment.sampleRate {
                    runs[runs.count - 1].append(segment)
                } else { runs.append([segment]) }
            }
            for run in runs {
                try Task.checkCancellation()
                let normalized = try store.file(id, "converted-\(run[0].id).caf")
                defer { try? FileManager.default.removeItem(at: normalized) }
                try normalize(run, session: id, store: store, to: normalized)
                let input = try AVAudioFile(forReading: normalized)
                var start: Int64 = 0
                while start < input.length {
                    try Task.checkCancellation()
                    let saved = record.jobs.first {
                        $0.id.hasPrefix("\(source.rawValue)-\(run[0].id)-\(start)-")
                    }
                    let end = try saved.map { start + $0.frames } ?? boundary(input, start: start)
                    guard end <= input.length else { throw LaliaError("Source audio is shorter than its saved transcription range.") }
                    let jobID = saved?.id ?? "\(source.rawValue)-\(run[0].id)-\(start)-\(end)"
                    expectedJobs.insert(jobID)
                    let existing = try store.read(id).jobs.first { $0.id == jobID }
                    if existing == nil {
                        var job = TranscriptionJob(id: jobID, source: source, filename: "\(jobID).wav",
                                                   start: run[0].start + Double(start) / rate, frames: end - start)
                        let output = try store.file(id, job.filename)
                        let silent = try writeWAV(input, from: start, to: end, output: output, store: store)
                        if silent { job.state = .silence }
                        try store.update(id) { $0.jobs.append(job) }
                    } else if let existing, !FileManager.default.fileExists(atPath: try store.file(id, existing.filename).path) {
                        _ = try writeWAV(input, from: start, to: end, output: store.file(id, existing.filename), store: store)
                    }
                    start = end
                }
            }
        }
        try store.update(id) { record in
            guard Set(record.jobs.map(\.id)) == expectedJobs else { throw LaliaError("Saved transcription ranges do not match source audio.") }
            record.jobsPrepared = true
        }
    }

    // One converter spans contiguous storage files. Its end-of-stream pass drains
    // the resampling tail before any request boundaries are chosen.
    private func normalize(_ segments: [AudioSegment], session: UUID, store: SessionStore, to destination: URL) throws {
        let urls = try segments.map { try store.file(session, $0.filename) }
        for (segment, url) in zip(segments, urls) {
            let file = try AVAudioFile(forReading: url)
            guard file.length == segment.frames, file.processingFormat.sampleRate == segment.sampleRate,
                  file.processingFormat.channelCount == segment.channels else { throw LaliaError("Source audio no longer matches its saved metadata.") }
        }
        let reader = try ContiguousAudioReader(urls: urls)
        guard let format = AVAudioFormat(standardFormatWithSampleRate: rate, channels: 1),
              let converter = AVAudioConverter(from: reader.format, to: format),
              let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4096) else { throw LaliaError("Unsupported audio conversion.") }
        try store.privateFile(destination)
        let file = try AVAudioFile(forWriting: destination, settings: format.settings)
        defer { file.close() }
        var conversionError: NSError?
        while true {
            try Task.checkCancellation()
            let status = converter.convert(to: output, error: &conversionError) { count, status in
                do {
                    if let buffer = try reader.next(count) { status.pointee = .haveData; return buffer }
                    status.pointee = .endOfStream
                } catch { reader.error = error; status.pointee = .endOfStream }
                return nil
            }
            if let error = reader.error { throw error }
            if let conversionError { throw conversionError }
            if output.frameLength > 0 { try file.write(from: output) }
            if status == .endOfStream { break }
            guard status != .error, status != .inputRanDry else { throw LaliaError("Audio conversion ended before its source.") }
        }
    }

    private func boundary(_ file: AVAudioFile, start: Int64) throws -> Int64 {
        let limit = min(file.length, start + Int64(maximumSeconds * rate))
        guard limit < file.length else { return limit }
        let searchStart = max(start + 1, limit - Int64(5 * rate))
        let count = AVAudioFrameCount(limit - searchStart)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: count) else { throw LaliaError("Cannot allocate boundary buffer.") }
        file.framePosition = searchStart
        try file.read(into: buffer, frameCount: count)
        guard let samples = buffer.floatChannelData?[0] else { throw LaliaError("Boundary audio must use float PCM.") }
        let window = min(1600, Int(count))
        var best = Double.infinity
        var result = limit
        for offset in stride(from: 0, through: Int(count) - window, by: window) {
            var energy = 0.0
            for index in offset..<(offset + window) { energy += Double(samples[index]) * Double(samples[index]) }
            if energy < best { best = energy; result = searchStart + Int64(offset + window / 2) }
        }
        return result
    }

    private func writeWAV(_ input: AVAudioFile, from start: Int64, to end: Int64, output: URL, store: SessionStore) throws -> Bool {
        let partial = output.deletingPathExtension().appendingPathExtension("partial.wav")
        try store.privateFile(partial)
        do {
            var file: AVAudioFile? = try AVAudioFile(forWriting: partial, settings: recorderSettings)
            guard let buffer = AVAudioPCMBuffer(pcmFormat: input.processingFormat, frameCapacity: 4096) else { throw LaliaError("Cannot allocate WAV buffer.") }
            input.framePosition = start
            var silence = true
            while input.framePosition < end {
                try Task.checkCancellation()
                try input.read(into: buffer, frameCount: AVAudioFrameCount(min(4096, end - input.framePosition)))
                guard buffer.frameLength > 0, let samples = buffer.floatChannelData?[0] else { throw LaliaError("Unexpected end of converted audio.") }
                if silence { silence = (0..<Int(buffer.frameLength)).allSatisfy { samples[$0] == 0 } }
                try file?.write(from: buffer)
            }
            file?.close()
            file = nil
            if FileManager.default.fileExists(atPath: output.path) { try FileManager.default.removeItem(at: output) }
            try FileManager.default.moveItem(at: partial, to: output)
            return silence
        } catch {
            // This is disposable derived audio. Originals and completed jobs remain.
            try? FileManager.default.removeItem(at: partial)
            throw error
        }
    }
}

// AVAudioConverter calls its input block synchronously on the conversion thread.
private final class ContiguousAudioReader: @unchecked Sendable {
    let format: AVAudioFormat
    private let urls: [URL]
    private var index = 0
    private var file: AVAudioFile
    var error: Error?
    init(urls: [URL]) throws {
        self.urls = urls
        file = try AVAudioFile(forReading: urls[0])
        format = file.processingFormat
    }
    func next(_ count: AVAudioPacketCount) throws -> AVAudioPCMBuffer? {
        while file.framePosition == file.length {
            index += 1
            guard index < urls.count else { return nil }
            file = try AVAudioFile(forReading: urls[index])
            guard format.isEqual(file.processingFormat) else { throw LaliaError("Source format changed within a contiguous audio range.") }
        }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: min(count, 4096)) else { throw LaliaError("Cannot allocate conversion buffer.") }
        try file.read(into: buffer)
        return buffer
    }
}
