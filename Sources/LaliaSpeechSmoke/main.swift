import AVFoundation
import AVFAudio
import Foundation
import Speech

enum SmokeError: LocalizedError { case unavailable, locale, noSpeech, recorder
    var errorDescription: String? { switch self { case .unavailable: "Speech transcription is unavailable."; case .locale: "Current locale is unsupported."; case .noSpeech: "No speech recognized."; case .recorder: "Could not record from the default microphone." } }
}

func skip(_ message: String) -> Never { fputs("SKIP: \(message)\n", stderr); exit(77) }
func fail(_ error: Error) -> Never { fputs("ERROR: \(error.localizedDescription)\n", stderr); exit(1) }

@main struct LaliaSpeechSmoke {
    static func main() async {
        guard CommandLine.arguments.count == 2 else { fputs("usage: LaliaSpeechSmoke fixture.wav\n", stderr); exit(2) }
        guard AVAudioApplication.shared.recordPermission == .granted else { skip("microphone permission is not granted") }
        guard SFSpeechRecognizer.authorizationStatus() == .authorized else { skip("Speech authorization is not granted") }
        guard AVCaptureDevice.default(for: .audio) != nil else { skip("no default audio input") }
        do {
            guard SpeechTranscriber.isAvailable else { throw SmokeError.unavailable }
            guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: .current) else { throw SmokeError.locale }
            let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)
            guard await AssetInventory.status(forModules: [transcriber]) == .installed else { skip("Speech assets are not installed") }

            let recording = FileManager.default.temporaryDirectory.appending(path: "LaliaSpeechSmoke-\(UUID().uuidString).wav")
            defer { try? FileManager.default.removeItem(at: recording) }
            let recorder = try AVAudioRecorder(url: recording, settings: [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: 16_000.0,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false,
            ])
            guard recorder.prepareToRecord(), recorder.record() else { throw SmokeError.recorder }
            try await Task.sleep(for: .milliseconds(100))
            recorder.stop()
            let attributes = try FileManager.default.attributesOfItem(atPath: recording.path)
            guard let size = attributes[.size] as? NSNumber, size.intValue > 44 else { throw SmokeError.recorder }

            let audio = try AVAudioFile(forReading: URL(fileURLWithPath: CommandLine.arguments[1]))
            let analyzer = try await SpeechAnalyzer(inputAudioFile: audio, modules: [transcriber], options: .init(priority: .userInitiated, modelRetention: .processLifetime), finishAfterFile: true)
            var results = ""; for try await result in transcriber.results where result.isFinal { results += String(result.text.characters) }; _ = analyzer
            guard !results.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw SmokeError.noSpeech }
            print(results)
        } catch { fail(error) }
    }
}
