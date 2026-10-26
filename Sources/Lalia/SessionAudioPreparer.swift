import AVFoundation
import Foundation

struct SessionAudioPreparer: Sendable {
    func prepare(session id: UUID, store: SessionStore) throws -> SessionAnalysis.Input {
        let record = try store.read(id)
        guard record.captureState != .recording else { throw LaliaError("Stop recording before transcription.") }
        guard record.duration <= SessionTranscriptionProfile.maximumDuration else {
            throw LaliaError("Speaker labels support recordings up to 90 minutes. This recording and its transcript are unchanged.")
        }
        if let input = record.analysis?.input, FileManager.default.fileExists(atPath: try store.file(id, input.filename).path) {
            let file = try AVAudioFile(forReading: store.file(id, input.filename))
            guard file.length == input.frames, file.processingFormat.sampleRate == 16_000, file.processingFormat.channelCount == 1 else {
                throw LaliaError("Mixed audio no longer matches its saved metadata.")
            }
            return input
        }
        guard record.segments.contains(where: \.complete) else { throw LaliaError("No recoverable audio is available.") }
        guard record.segments.allSatisfy(\.complete) else { throw LaliaError("Some captured audio could not be recovered. Original files are retained.") }
        let frames = Int64((record.duration * 16_000).rounded())
        guard frames > 0, frames <= Int64(SessionTranscriptionProfile.maximumDuration * 16_000),
              let format = AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4096) else { throw LaliaError("Invalid session audio duration or format.") }
        var temporary: [URL] = []
        defer { for file in temporary { try? FileManager.default.removeItem(at: file) } }
        var runs: [(start: Int64, file: AVAudioFile)] = []
        for source in AudioSource.allCases {
            var groups: [[AudioSegment]] = []
            for segment in record.segments.filter({ $0.source == source }).sorted(by: { $0.start < $1.start }) {
                if let last = groups.last?.last,
                   last.sampleRate == segment.sampleRate, last.channels == segment.channels,
                   abs(last.start + last.duration - segment.start) <= 1.5 / segment.sampleRate {
                    groups[groups.count - 1].append(segment)
                } else { groups.append([segment]) }
            }
            for group in groups {
                try Task.checkCancellation()
                let url = try store.file(id, "mix-\(UUID()).caf")
                temporary.append(url)
                try AudioJobPreparer().normalize(group, session: id, store: store, to: url)
                runs.append((Int64((group[0].start * 16_000).rounded()), try AVAudioFile(forReading: url)))
            }
        }
        let summed = try store.file(id, "mix-\(UUID()).caf")
        temporary.append(summed)
        try store.privateFile(summed)
        let sum = try AVAudioFile(forWriting: summed, settings: format.settings)
        var peak: Float = 0
        var position: Int64 = 0
        while position < frames {
            try Task.checkCancellation()
            let count = Int(min(4096, frames - position))
            var samples = [Float](repeating: 0, count: count)
            for run in runs where run.start < position + Int64(count) && run.start + run.file.length > position {
                let from = max(position, run.start)
                let end = min(position + Int64(count), run.start + run.file.length)
                run.file.framePosition = from - run.start
                try run.file.read(into: buffer, frameCount: AVAudioFrameCount(end - from))
                guard buffer.frameLength == end - from, let values = buffer.floatChannelData?[0] else { throw LaliaError("Unexpected end of aligned audio.") }
                for index in 0..<Int(buffer.frameLength) { samples[Int(from - position) + index] += values[index] }
            }
            buffer.frameLength = AVAudioFrameCount(count)
            for index in 0..<count {
                guard samples[index].isFinite else { throw LaliaError("Source audio contains nonfinite samples.") }
                peak = max(peak, abs(samples[index]))
                buffer.floatChannelData![0][index] = samples[index]
            }
            try sum.write(from: buffer)
            position += Int64(count)
        }
        sum.close()
        for run in runs { run.file.close() }
        let input = try AVAudioFile(forReading: summed)
        defer { input.close() }
        let partial = try store.file(id, "session-mix.partial.wav")
        temporary.append(partial)
        try store.privateFile(partial)
        let output = try AVAudioFile(forWriting: partial, settings: recorderSettings)
        // A constant gain preserves relative loudness and sole-track samples. Reduce
        // only when the aligned sum would clip, without a noise gate or AGC.
        let gain = 1 / max(1, peak)
        while input.framePosition < frames {
            try Task.checkCancellation()
            try input.read(into: buffer, frameCount: AVAudioFrameCount(min(4096, frames - input.framePosition)))
            guard buffer.frameLength > 0 else { throw LaliaError("Unexpected end of mixed audio.") }
            for index in 0..<Int(buffer.frameLength) { buffer.floatChannelData![0][index] *= gain }
            try output.write(from: buffer)
        }
        output.close()
        let destination = try store.file(id, "session-mix.wav")
        // rename atomically replaces a disposable derived mix, never an original.
        guard rename(partial.path, destination.path) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        let saved = SessionAnalysis.Input(filename: destination.lastPathComponent, frames: frames, silence: peak == 0)
        try store.update(id) { record in
            if record.analysis == nil { record.analysis = SessionAnalysis() }
            record.analysis?.input = saved
        }
        return saved
    }
}
