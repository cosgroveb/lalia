import AVFoundation
import Foundation
import Speech

@MainActor final class NativeTranscriber: Transcribing {
    private var locale: Locale?
    func prepare() async throws {
        guard SpeechTranscriber.isAvailable else { throw LaliaError.unavailableSpeech }
        guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: .current) else { throw LaliaError.unsupportedLocale }
        let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)
        if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) { try await request.downloadAndInstall() }
        guard await AssetInventory.status(forModules: [transcriber]) == .installed else { throw LaliaError.assetsNotInstalled }
        self.locale = locale
    }
    func transcribe(_ file: URL) async throws -> String {
        guard let locale else { throw LaliaError.assetsNotInstalled }
        let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)
        let audioFile = try AVAudioFile(forReading: file)
        let options = SpeechAnalyzer.Options(priority: .userInitiated, modelRetention: .processLifetime)
        let analyzer = try await SpeechAnalyzer(inputAudioFile: audioFile, modules: [transcriber], options: options, finishAfterFile: true)
        var phrases: [String] = []
        for try await result in transcriber.results where result.isFinal { phrases.append(String(result.text.characters)) }
        _ = analyzer
        return joinedFinalPhrases(phrases)
    }
}
