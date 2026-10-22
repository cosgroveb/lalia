import Foundation

protocol Recording: AnyObject { func start(at file: URL) throws; func stop() throws; func close() }
@MainActor protocol Transcribing: AnyObject {
    func prepare() async throws
    func transcribe(_ file: URL) async throws -> Transcription
    func notes(for transcript: String) async throws -> String
}
@MainActor protocol Injecting: AnyObject { func inject(_ text: String) async throws; func copy(_ text: String) throws }
@MainActor protocol SessionCapturing: AnyObject {
    func start(session: UUID, store: SessionStore, failed: @escaping @MainActor @Sendable (Error) -> Void) async throws
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
    }
    var text: String
    var segments: [Segment]?

    func validated(duration: Double) throws -> Transcription {
        if let segments {
            var previous = 0.0
            for segment in segments {
                guard segment.start.isFinite, segment.end.isFinite,
                      segment.start >= previous, segment.end > segment.start,
                      segment.end <= duration + 0.1 else {
                    throw LaliaError("oMLX returned invalid transcript time ranges.")
                }
                previous = segment.end
            }
        }
        return self
    }
}
