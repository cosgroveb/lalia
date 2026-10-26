import AVFoundation
import CoreMedia
import ScreenCaptureKit

// Used only on the serial capture queue (and synchronously with synthetic test buffers).
final class SourceAudioWriter {
    private struct Track {
        var segment: AudioSegment
        var file: AVAudioFile?
        var format: AVAudioFormat
        var frames: Int64 = 0
        var end: Double { segment.start + Double(frames) / format.sampleRate }
    }
    private let store: SessionStore
    private let session: UUID
    private let origin: Double
    private let rotation: Double
    private let maximumDuration: Double
    private var tracks: [AudioSource: Track] = [:]

    init(store: SessionStore, session: UUID, origin: Double, rotation: Double = 60, maximumDuration: Double = SessionTranscriptionProfile.maximumDuration) {
        self.store = store; self.session = session; self.origin = origin; self.rotation = rotation
        self.maximumDuration = maximumDuration
    }

    func append(_ buffer: AVAudioPCMBuffer, source: AudioSource, timestamp: Double) throws {
        guard buffer.frameLength > 0 else { return }
        let start = timestamp - origin
        guard start.isFinite, start >= 0, buffer.format.sampleRate > 0 else { throw LaliaError("Invalid capture timestamp or audio format.") }
        guard start < maximumDuration else { return }
        let length = min(Int64(buffer.frameLength), Int64(((maximumDuration - start) * buffer.format.sampleRate).rounded(.down)))
        if let track = tracks[source], (!track.format.isEqual(buffer.format) || abs(track.end - start) > 1.5 / buffer.format.sampleRate) {
            try finish(source)
        }
        var offset: AVAudioFrameCount = 0
        while offset < length {
            if tracks[source] == nil {
                let id = UUID()
                let segment = AudioSegment(id: id, source: source, filename: "\(source.rawValue)-\(id).caf",
                                           start: start + Double(offset) / buffer.format.sampleRate,
                                           sampleRate: buffer.format.sampleRate, channels: buffer.format.channelCount)
                try store.update(session) { $0.segments.append(segment) }
                let url = try store.file(session, segment.partialFilename)
                try store.privateFile(url)
                var settings = buffer.format.settings
                settings[AVLinearPCMIsNonInterleaved] = false
                let file = try AVAudioFile(forWriting: url, settings: settings,
                                           commonFormat: buffer.format.commonFormat, interleaved: buffer.format.isInterleaved)
                tracks[source] = Track(segment: segment, file: file, format: buffer.format)
            }
            guard var track = tracks[source], let file = track.file else { throw LaliaError("Capture file is unavailable.") }
            let room = Int64((rotation * buffer.format.sampleRate).rounded()) - track.frames
            let remaining = Int64(((maximumDuration - track.end) * buffer.format.sampleRate).rounded(.down))
            guard remaining > 0 else { break }
            let count = AVAudioFrameCount(min(length - Int64(offset), room, remaining))
            let part = try Self.slice(buffer, offset: offset, count: count)
            try file.write(from: part)
            track.frames += Int64(count)
            tracks[source] = track
            offset += count
            if track.frames >= Int64((rotation * buffer.format.sampleRate).rounded()) { try finish(source) }
        }
    }

    func finish() throws {
        var firstError: Error?
        for source in AudioSource.allCases {
            do { try finish(source) } catch { if firstError == nil { firstError = error } }
        }
        if let firstError { throw firstError }
    }

    private func finish(_ source: AudioSource) throws {
        guard var track = tracks.removeValue(forKey: source) else { return }
        track.file?.close()
        track.file = nil
        try store.finishSegment(session, segmentID: track.segment.id)
    }

    static func slice(_ buffer: AVAudioPCMBuffer, offset: AVAudioFrameCount, count: AVAudioFrameCount) throws -> AVAudioPCMBuffer {
        guard let part = AVAudioPCMBuffer(pcmFormat: buffer.format, frameCapacity: count) else { throw LaliaError("Cannot allocate audio buffer.") }
        part.frameLength = count
        let stride = Int(buffer.format.streamDescription.pointee.mBytesPerFrame)
        let input = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: buffer.audioBufferList))
        let output = UnsafeMutableAudioBufferListPointer(part.mutableAudioBufferList)
        for index in input.indices {
            guard let from = input[index].mData, let to = output[index].mData else { throw LaliaError("Missing PCM samples.") }
            memcpy(to, from.advanced(by: Int(offset) * stride), Int(count) * stride)
        }
        return part
    }
}

private final class CaptureOutput: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    let queue = DispatchQueue(label: "com.bcosgrove.Lalia.audio")
    private let writer: SourceAudioWriter
    private let failed: @MainActor @Sendable (Error) -> Void
    private var stopped = false
    init(writer: SourceAudioWriter, failed: @escaping @MainActor @Sendable (Error) -> Void) {
        self.writer = writer; self.failed = failed
    }
    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of outputType: SCStreamOutputType) {
        guard !stopped, sampleBuffer.isValid, outputType == .audio || outputType == .microphone else { return }
        do {
            guard let description = sampleBuffer.formatDescription else { throw LaliaError("Captured audio has no format.") }
            let format = AVAudioFormat(cmAudioFormatDescription: description)
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(sampleBuffer.numSamples)) else {
                throw LaliaError("ScreenCaptureKit returned an unsupported audio format.")
            }
            buffer.frameLength = buffer.frameCapacity
            let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(sampleBuffer, at: 0, frameCount: Int32(buffer.frameLength), into: buffer.mutableAudioBufferList)
            guard status == noErr else { throw LaliaError("Cannot copy captured audio (\(status)).") }
            try writer.append(buffer, source: outputType == .audio ? .system : .microphone,
                              timestamp: sampleBuffer.presentationTimeStamp.seconds)
        } catch { fail(error) }
    }
    func stream(_ stream: SCStream, didStopWithError error: Error) { queue.async { self.fail(error) } }
    private func fail(_ error: Error) {
        guard !stopped else { return }
        stopped = true
        Task { @MainActor in failed(error) }
    }
    func finish() async throws {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                self.stopped = true
                do { try self.writer.finish(); continuation.resume() }
                catch { continuation.resume(throwing: error) }
            }
        }
    }
}

@MainActor final class SessionCapture: SessionCapturing {
    private var stream: SCStream?
    private var output: CaptureOutput?
    func start(session: UUID, store: SessionStore, maximumDuration: Double, failed: @escaping @MainActor @Sendable (Error) -> Void) async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        guard let display = content.displays.first else { throw LaliaError("Mac audio capture requires an available display.") }
        let filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.captureMicrophone = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 48_000
        config.channelCount = 2
        let writer = SourceAudioWriter(store: store, session: session, origin: CMClockGetTime(CMClockGetHostTimeClock()).seconds, maximumDuration: maximumDuration)
        let output = CaptureOutput(writer: writer, failed: failed)
        let stream = SCStream(filter: filter, configuration: config, delegate: output)
        try stream.addStreamOutput(output, type: .audio, sampleHandlerQueue: output.queue)
        try stream.addStreamOutput(output, type: .microphone, sampleHandlerQueue: output.queue)
        self.stream = stream; self.output = output
        do { try await stream.startCapture() }
        catch {
            self.stream = nil; self.output = nil
            do { try await output.finish() } catch { throw LaliaError("Capture failed and could not finalize audio: \(error.localizedDescription)") }
            throw error
        }
    }
    func stop() async throws {
        let stream = self.stream
        let output = self.output
        self.stream = nil; self.output = nil
        var stopError: Error?
        do { try await stream?.stopCapture() } catch { stopError = error }
        do { try await output?.finish() } catch { if stopError == nil { stopError = error } }
        if let stopError { throw stopError }
    }
}
