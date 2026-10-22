import AVFoundation
import Foundation
import Testing
@testable import Lalia

struct RecorderTests {
    @Test func usesSpeechFriendlyPCMSettings() {
        #expect(recorderSettings[AVSampleRateKey] as? Double == 16_000)
        #expect(recorderSettings[AVNumberOfChannelsKey] as? Int == 1)
        #expect(recorderSettings[AVLinearPCMBitDepthKey] as? Int == 16)
        #expect(recorderSettings[AVLinearPCMIsFloatKey] as? Bool == false)
    }
    @Test func recorderCleanupDoesNotDeleteOwnedFiles() throws {
        let fixture = TemporaryStore()
        let record = try savedAudio(fixture.store)
        let file = try fixture.store.file(record.id, record.segments[0].filename)
        do { let recorder = Recorder(); recorder.close() }
        #expect(FileManager.default.fileExists(atPath: file.path))
    }
}
