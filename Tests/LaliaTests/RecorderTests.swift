import AVFoundation
import Foundation
import Testing
@testable import Lalia

struct RecorderTests {
    @Test func allocatesUniqueWavFilesAndRemovesThem() throws {
        let store = RecordingFileStore(directory: FileManager.default.temporaryDirectory.appending(path: "LaliaTests-\(UUID().uuidString)"))
        let first = try store.allocate()
        let second = try store.allocate()
        #expect(first.pathExtension == "wav")
        #expect(first.deletingLastPathComponent() == second.deletingLastPathComponent())
        #expect(first != second)
        try Data("fixture".utf8).write(to: second)
        store.removeCurrent()
        #expect(!FileManager.default.fileExists(atPath: second.path))
        store.removeCurrent()
    }

    @Test func usesSpeechFriendlyPCMSettings() {
        #expect(recorderSettings[AVSampleRateKey] as? Double == 16_000)
        #expect(recorderSettings[AVNumberOfChannelsKey] as? Int == 1)
        #expect(recorderSettings[AVLinearPCMBitDepthKey] as? Int == 16)
        #expect(recorderSettings[AVLinearPCMIsFloatKey] as? Bool == false)
    }

    @Test func removesCurrentFileWhenDeinitialized() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "LaliaTests-\(UUID().uuidString)")
        let file: URL
        do {
            let store = RecordingFileStore(directory: directory)
            file = try store.allocate()
            try Data("fixture".utf8).write(to: file)
            #expect(FileManager.default.fileExists(atPath: file.path))
        }
        #expect(!FileManager.default.fileExists(atPath: file.path))
    }
}
