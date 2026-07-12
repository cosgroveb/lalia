import AVFAudio
import ApplicationServices
import Foundation
import Speech

@MainActor final class SystemAuthorizer: Authorizing {
    func currentStatus() -> AuthorizationStatus {
        AuthorizationStatus(
            microphoneGranted: AVAudioApplication.shared.recordPermission == .granted,
            speechGranted: SFSpeechRecognizer.authorizationStatus() == .authorized,
            accessibilityGranted: AXIsProcessTrusted()
        )
    }
    func requestAll() async -> AuthorizationStatus {
        if AVAudioApplication.shared.recordPermission == .undetermined { _ = await AVAudioApplication.requestRecordPermission() }
        if SFSpeechRecognizer.authorizationStatus() == .notDetermined {
            _ = await withCheckedContinuation { continuation in SFSpeechRecognizer.requestAuthorization { continuation.resume(returning: $0) } }
        }
        if !AXIsProcessTrusted() { AXIsProcessTrustedWithOptions(["AXTrustedCheckOptionPrompt": true] as CFDictionary) }
        return currentStatus()
    }
}
