import Combine
import Foundation
import os

@MainActor final class DictationCoordinator: ObservableObject {
    @Published private(set) var phase: Phase = .needsPermission
    @Published private(set) var message = "Enable Dictation to get started."
    @Published private(set) var lastTranscript: String?
    @Published private(set) var hotkeyError: String?
    @Published private(set) var isDictationEnabled = false
    private let recorder: Recording
    private let transcriber: Transcribing
    private let injector: Injecting
    private let authorizer: Authorizing
    private let logger = Logger(subsystem: "com.bcosgrove.Lalia", category: "dictation")
    private let clock = ContinuousClock()
    private var recordingStartedAt: ContinuousClock.Instant?
    private var dictationStartedAt: ContinuousClock.Instant?
    init(recorder: Recording, transcriber: Transcribing, injector: Injecting, authorizer: Authorizing) { self.recorder = recorder; self.transcriber = transcriber; self.injector = injector; self.authorizer = authorizer }
    func startupRefresh() { if authorizer.currentStatus().ready { Task { await enableDictation() } } else { phase = .needsPermission; message = "Permissions required." } }
    func enableDictation() async {
        guard phase != .preparing, phase != .recording, phase != .transcribing, phase != .pasting else { return }
        isDictationEnabled = false
        phase = .preparing; message = "Checking permissions…"
        guard (await authorizer.requestAll()).ready else { phase = .needsPermission; message = "Microphone, Speech, and Accessibility permissions are required."; return }
        do { message = "Preparing Speech…"; try await transcriber.prepare(); isDictationEnabled = true; phase = .idle; message = "Ready — hold ⇧⌘D to dictate." }
        catch { phase = .idle; message = error.localizedDescription }
    }
    func hotkeyPressed() {
        guard phase == .idle, isDictationEnabled else { return }
        guard authorizer.currentStatus().ready else {
            isDictationEnabled = false
            phase = .needsPermission
            message = "Microphone, Speech, and Accessibility permissions are required."
            return
        }
        dictationStartedAt = clock.now
        do { try recorder.start(); recordingStartedAt = clock.now; phase = .recording; message = "Recording…" } catch { recorder.discard(); phase = .idle; message = error.localizedDescription; logTotalDuration() }
    }
    func hotkeyReleased() {
        guard phase == .recording else { return }
        let file: URL
        do { file = try recorder.stop(); logRecordingDuration() } catch { logRecordingDuration(); recorder.discard(); phase = .idle; message = error.localizedDescription; logTotalDuration(); return }
        phase = .transcribing; message = "Transcribing…"
        Task {
            let transcribingStartedAt = clock.now
            defer { self.logTotalDuration(); recorder.discard(); phase = .idle }
            do {
                let text: String
                do { text = try await transcriber.transcribe(file) }
                catch { logTranscriptionDuration(since: transcribingStartedAt); throw error }
                logTranscriptionDuration(since: transcribingStartedAt)
                guard let transcript = trimmedTranscript(text) else { message = "No speech recognized."; return }
                lastTranscript = transcript
                guard authorizer.currentStatus().accessibilityGranted else { message = "Accessibility permission is required to paste."; return }
                phase = .pasting; message = "Pasting…"
                let pastingStartedAt = clock.now
                do { try await injector.inject(transcript) }
                catch { logger.error("Paste duration: \(String(describing: self.clock.now - pastingStartedAt), privacy: .public)"); throw error }
                logger.info("Paste duration: \(String(describing: self.clock.now - pastingStartedAt), privacy: .public)")
                message = "Ready — hold ⇧⌘D to dictate."
            } catch { logger.error("Dictation failed: \(error.localizedDescription, privacy: .public)"); message = error.localizedDescription }
        }
    }
    func copyLastTranscript() { if let lastTranscript { do { try injector.copy(lastTranscript); message = "Copied last transcript." } catch { message = error.localizedDescription } } }
    func hotkeyRegistrationFailed(_ error: Error) { hotkeyError = "Could not register ⇧⌘D: \(error.localizedDescription)" }
    private func logRecordingDuration() { if let recordingStartedAt { logger.info("Recording duration: \(String(describing: self.clock.now - recordingStartedAt), privacy: .public)"); self.recordingStartedAt = nil } }
    private func logTranscriptionDuration(since start: ContinuousClock.Instant) { logger.info("Transcription duration: \(String(describing: self.clock.now - start), privacy: .public)") }
    private func logTotalDuration() { if let dictationStartedAt { logger.info("Total dictation duration: \(String(describing: self.clock.now - dictationStartedAt), privacy: .public)"); self.dictationStartedAt = nil } }
}
