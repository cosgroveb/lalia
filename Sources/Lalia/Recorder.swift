import AVFoundation
import Foundation

nonisolated(unsafe) let recorderSettings: [String: Any] = [
    AVFormatIDKey: kAudioFormatLinearPCM,
    AVSampleRateKey: 16_000.0,
    AVNumberOfChannelsKey: 1,
    AVLinearPCMBitDepthKey: 16,
    AVLinearPCMIsFloatKey: false,
]

final class Recorder: Recording {
    private var recorder: AVAudioRecorder?
    func start(at file: URL) throws {
        let recorder = try AVAudioRecorder(url: file, settings: recorderSettings)
        self.recorder = recorder
        guard recorder.prepareToRecord(), recorder.record() else { throw CocoaError(.fileWriteUnknown) }
    }
    func stop() throws {
        guard let recorder else { throw CocoaError(.fileNoSuchFile) }
        recorder.stop()
        self.recorder = nil
    }
    func close() { recorder?.stop(); recorder = nil }
    deinit { recorder?.stop() }
}
