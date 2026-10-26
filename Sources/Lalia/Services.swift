import Foundation

protocol Recording: AnyObject { func start(at file: URL) throws; func stop() throws; func close() }
@MainActor protocol Transcribing: AnyObject {
    func prepare() async throws
    func transcribe(_ file: URL) async throws -> Transcription
    func sessionProfile() throws -> SessionTranscriptionProfile
    func prepareSession(profile: SessionTranscriptionProfile) async throws
    func transcribeSession(_ file: URL, profile: SessionTranscriptionProfile) async throws -> Data
    func notes(for transcript: String) async throws -> String
}
@MainActor protocol Injecting: AnyObject { func inject(_ text: String) async throws; func copy(_ text: String) throws }
@MainActor protocol PlaybackStopping: AnyObject { func stop() }
@MainActor protocol SessionCapturing: AnyObject {
    func start(session: UUID, store: SessionStore, maximumDuration: Double, failed: @escaping @MainActor @Sendable (Error) -> Void) async throws
    func stop() async throws
}

struct AuthorizationStatus: Equatable {
    let microphoneGranted: Bool
    let accessibilityGranted: Bool
    var ready: Bool { microphoneGranted && accessibilityGranted }
}

@MainActor protocol Authorizing: AnyObject {
    func currentStatus() -> AuthorizationStatus
    func requestAll() async -> AuthorizationStatus
    func requestMicrophone() async -> Bool
}

enum Phase: Equatable { case needsPermission, preparing, idle, recording, recordingSession, stopping, transcribing, pasting, summarizing }

func trimmedTranscript(_ text: String) -> String? {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    return trimmed.isEmpty ? nil : trimmed
}

struct LaliaError: LocalizedError, Sendable {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

struct Transcription: Codable, Equatable, Sendable {
    struct Segment: Codable, Equatable, Sendable {
        var start: Double
        var end: Double
        var text: String
        var speakerID: String?
        enum CodingKeys: String, CodingKey { case start, end, text; case speakerID = "speaker_id" }
    }
    var text: String
    var segments: [Segment]?
    var usage: Usage?

    struct Usage: Codable, Equatable, Sendable {
        var type: String
        var inputTokens: Int
        var outputTokens: Int
        var totalTokens: Int
        enum CodingKeys: String, CodingKey {
            case type
            case inputTokens = "input_tokens", outputTokens = "output_tokens", totalTokens = "total_tokens"
        }
    }

    func validated(duration: Double) throws -> Transcription {
        if let segments {
            for segment in segments {
                guard segment.start.isFinite, segment.end.isFinite,
                      segment.start >= 0, segment.end > segment.start,
                      segment.end <= duration + 0.1 else {
                    throw LaliaError("oMLX returned invalid transcript time ranges.")
                }
            }
        }
        return self
    }
}
