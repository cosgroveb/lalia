import Foundation

enum SessionInterruption: Equatable {
    case keyboardUnavailable, sleep, lock, userSwitch, captureFailure, limit
    case storageFailure, audioPermission, quit, stop

    var endsSession: Bool { self != .keyboardUnavailable }
    var attemptsTranscription: Bool {
        switch self {
        case .sleep, .lock, .userSwitch, .captureFailure, .limit, .stop: true
        default: false
        }
    }
    var allowsPartialTranscript: Bool {
        switch self {
        case .sleep, .lock, .userSwitch, .captureFailure: true
        default: false
        }
    }
    var noticeTitle: String? {
        switch self {
        case .sleep: "Transcribing stopped when your Mac slept"
        case .lock: "Transcribing stopped when you locked your Mac"
        case .userSwitch: "Transcribing stopped when you switched users"
        case .captureFailure: "Transcribing stopped because audio capture failed"
        case .limit: "Transcribing stopped at the 60-minute limit"
        default: nil
        }
    }
    func merging(_ next: Self) -> Self {
        if !attemptsTranscription { return self }
        if !next.attemptsTranscription && next.endsSession { return next }
        if self == .captureFailure, [.sleep, .lock, .userSwitch].contains(next) { return next }
        return self
    }
}

struct SessionInterruptionNotice: Identifiable {
    let id: UUID
    var reason: SessionInterruption
    var title: String { reason.noticeTitle ?? "Transcribing stopped" }
    let message = "Start transcribing again if you wish to continue."
}

struct CaptureFailure: LocalizedError {
    enum Kind { case storage, permission, stream }
    let kind: Kind
    let underlying: Error
    var errorDescription: String? { underlying.localizedDescription }
    static func preferred(_ first: Error?, _ second: Error) -> Error {
        func priority(_ error: Error) -> Int {
            switch (error as? CaptureFailure)?.kind {
            case .storage: 2
            case .permission: 1
            default: 0
            }
        }
        guard let first else { return second }
        return priority(first) >= priority(second) ? first : second
    }
    var interruption: SessionInterruption {
        switch kind {
        case .storage: .storageFailure
        case .permission: .audioPermission
        case .stream: .captureFailure
        }
    }
}
