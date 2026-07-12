import Foundation

protocol Recording: AnyObject { func start() throws; func stop() throws -> URL; func discard() }
@MainActor protocol Transcribing: AnyObject { func prepare() async throws; func transcribe(_ file: URL) async throws -> String }
@MainActor protocol Injecting: AnyObject { func inject(_ text: String) async throws; func copy(_ text: String) throws }

struct AuthorizationStatus: Equatable {
    let microphoneGranted: Bool
    let speechGranted: Bool
    let accessibilityGranted: Bool
    var ready: Bool { microphoneGranted && speechGranted && accessibilityGranted }
}

@MainActor protocol Authorizing: AnyObject {
    func currentStatus() -> AuthorizationStatus
    func requestAll() async -> AuthorizationStatus
}

enum Phase: Equatable { case needsPermission, preparing, idle, recording, transcribing, pasting }

func trimmedTranscript(_ text: String) -> String? {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    return trimmed.isEmpty ? nil : trimmed
}

func joinedFinalPhrases(_ phrases: [String]) -> String { phrases.joined() }

enum LaliaError: LocalizedError {
    case unavailableSpeech, unsupportedLocale, assetsNotInstalled
    case noSpeech
    var errorDescription: String? {
        switch self {
        case .unavailableSpeech: "Speech transcription is unavailable."
        case .unsupportedLocale: "Current locale is unsupported for on-device Speech."
        case .assetsNotInstalled: "Speech assets are not installed."
        case .noSpeech: "No speech recognized."
        }
    }
}
