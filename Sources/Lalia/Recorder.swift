import AVFoundation
import Foundation

nonisolated(unsafe) let recorderSettings: [String: Any] = [
    AVFormatIDKey: kAudioFormatLinearPCM,
    AVSampleRateKey: 16_000.0,
    AVNumberOfChannelsKey: 1,
    AVLinearPCMBitDepthKey: 16,
    AVLinearPCMIsFloatKey: false,
]

final class RecordingFileStore {
    private let directory: URL
    private var current: URL?
    init(directory: URL = FileManager.default.temporaryDirectory.appending(path: "Lalia", directoryHint: .isDirectory)) { self.directory = directory }
    func allocate() throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appending(path: "\(UUID().uuidString).wav")
        current = url
        return url
    }
    func removeCurrent() {
        guard let current else { return }
        try? FileManager.default.removeItem(at: current)
        self.current = nil
    }
    deinit { removeCurrent() }
}

final class Recorder: Recording {
    private let files = RecordingFileStore()
    private var recorder: AVAudioRecorder?
    func start() throws {
        let url = try files.allocate()
        let recorder = try AVAudioRecorder(url: url, settings: recorderSettings)
        recorder.prepareToRecord()
        guard recorder.record() else { throw CocoaError(.fileWriteUnknown) }
        self.recorder = recorder
    }
    func stop() throws -> URL {
        guard let recorder else { throw CocoaError(.fileNoSuchFile) }
        recorder.stop()
        self.recorder = nil
        return recorder.url
    }
    func discard() { recorder?.stop(); recorder = nil; files.removeCurrent() }
}
