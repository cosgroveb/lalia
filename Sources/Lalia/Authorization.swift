import AVFAudio
import ApplicationServices
import Foundation

@MainActor final class SystemAuthorizer: Authorizing {
    func currentStatus() -> AuthorizationStatus {
        AuthorizationStatus(
            microphoneGranted: AVAudioApplication.shared.recordPermission == .granted,
            accessibilityGranted: AXIsProcessTrusted()
        )
    }
    func requestMicrophone() async -> Bool {
        if AVAudioApplication.shared.recordPermission == .undetermined { return await AVAudioApplication.requestRecordPermission() }
        return currentStatus().microphoneGranted
    }
    func requestAll() async -> AuthorizationStatus {
        _ = await requestMicrophone()
        if !AXIsProcessTrusted() { AXIsProcessTrustedWithOptions(["AXTrustedCheckOptionPrompt": true] as CFDictionary) }
        return currentStatus()
    }
}
