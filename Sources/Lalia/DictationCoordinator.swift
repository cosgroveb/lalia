import Combine
import Foundation

@MainActor final class DictationCoordinator: ObservableObject {
    @Published private(set) var phase: Phase = .needsPermission
    @Published private(set) var message = "Press ⇧⌘R to record, or enable dictation."
    @Published private(set) var lastTranscript: String?
    @Published private(set) var hotkeyError: String?
    @Published private(set) var isDictationEnabled = false
    @Published private(set) var sessions: [SessionRecord] = []
    @Published private(set) var diagnostics: [String] = []
    let store: SessionStore
    private let recorder: Recording
    private let capture: SessionCapturing
    private let transcriber: Transcribing
    private let injector: Injecting
    private let authorizer: Authorizing
    private var activeSession: UUID?
    private var dictationSegment: UUID?
    private var operation: Task<Void, Never>?
    private var quitting = false
    private var startingFailure: Error?
    var available: Bool { !quitting && (phase == .idle || phase == .needsPermission) }
    var canCancel: Bool { phase == .transcribing || phase == .summarizing }

    init(recorder: Recording, transcriber: Transcribing, injector: Injecting, authorizer: Authorizing,
         store: SessionStore = SessionStore(), capture: SessionCapturing = SessionCapture()) {
        self.recorder = recorder; self.transcriber = transcriber; self.injector = injector
        self.authorizer = authorizer; self.store = store; self.capture = capture
    }
    func startupRefresh() {
        refreshHistory(recover: true)
        if authorizer.currentStatus().ready { isDictationEnabled = true; phase = .idle; message = "Ready. Hold ⇧⌘D to dictate." }
    }
    func refreshHistory(recover: Bool = false) {
        do { let history = try store.history(recover: recover); sessions = history.records; diagnostics = history.diagnostics }
        catch { diagnostics = [error.localizedDescription] }
    }
    func enableDictation() async {
        guard available else { return }
        isDictationEnabled = false
        phase = .preparing; message = "Checking permissions…"
        guard (await authorizer.requestAll()).ready else {
            phase = .needsPermission; message = "Microphone and Accessibility permissions are required for dictation."; return
        }
        isDictationEnabled = true; phase = .idle; message = "Ready. Hold ⇧⌘D to dictate."
    }
    func hotkeyPressed() {
        guard available, isDictationEnabled else { return }
        guard authorizer.currentStatus().ready else {
            isDictationEnabled = false; phase = .needsPermission
            message = "Microphone and Accessibility permissions are required for dictation."
            return
        }
        do {
            let session = try store.create(kind: .dictation)
            activeSession = session.id
            let segment = AudioSegment(id: UUID(), source: .microphone, filename: "microphone.wav", start: 0, sampleRate: 16_000, channels: 1)
            dictationSegment = segment.id
            try store.update(session.id) { $0.segments.append(segment) }
            let file = try store.file(session.id, segment.partialFilename)
            try store.privateFile(file)
            try recorder.start(at: file)
            phase = .recording; message = "Recording dictation…"
        } catch {
            recorder.close()
            recordCaptureFailure(error)
            activeSession = nil; dictationSegment = nil; phase = .idle
        }
        refreshHistory()
    }
    func hotkeyReleased() {
        guard phase == .recording else { return }
        finishDictation(transcribe: true)
    }
    private func finishDictation(transcribe: Bool) {
        guard let id = activeSession, let segment = dictationSegment else { return }
        do {
            try recorder.stop()
            try store.finishSegment(id, segmentID: segment)
            try store.update(id) { $0.captureState = .complete }
            activeSession = nil; dictationSegment = nil
            if transcribe { beginTranscription(id, paste: true) }
            else { phase = .idle }
        } catch {
            recorder.close()
            recordCaptureFailure(error)
            activeSession = nil; dictationSegment = nil; phase = .idle
        }
        refreshHistory()
    }
    func startSession() {
        guard available else { return }
        phase = .preparing; message = "Starting microphone and Mac audio…"
        startingFailure = nil
        operation = Task {
            guard await authorizer.requestMicrophone() else {
                phase = .idle; message = "Microphone permission is required to start a session."; return
            }
            do {
                try Task.checkCancellation()
                let session = try store.create(kind: .session)
                activeSession = session.id
                try await capture.start(session: session.id, store: store) { [weak self] error in
                    self?.captureFailed(error)
                }
                if let startingFailure {
                    phase = .stopping
                    await finishSession(transcribe: false, failure: startingFailure)
                } else { phase = .recordingSession; message = "Recording microphone and Mac audio…" }
            } catch {
                recordCaptureFailure(error)
                activeSession = nil; phase = .idle
            }
            refreshHistory()
        }
    }
    func stopSession() {
        guard phase == .recordingSession else { return }
        phase = .stopping; message = "Saving recording…"
        operation = Task { await finishSession(transcribe: true) }
    }
    func toggleRecording() {
        if phase == .recordingSession { stopSession() }
        else { startSession() }
    }
    private func captureFailed(_ error: Error) {
        guard activeSession != nil, phase == .recordingSession || phase == .preparing else { return }
        if phase == .preparing { startingFailure = error; return }
        phase = .stopping
        operation = Task { await finishSession(transcribe: false, failure: error) }
    }
    private func finishSession(transcribe: Bool, failure: Error? = nil) async {
        guard let id = activeSession else { return }
        var failure = failure
        do { try await capture.stop() } catch { if failure == nil { failure = error } }
        if let failure { recordCaptureFailure(failure) }
        else {
            do { try store.update(id) { $0.captureState = .complete } }
            catch { failure = error; recordCaptureFailure(error) }
        }
        activeSession = nil; phase = .idle
        refreshHistory()
        if failure == nil, transcribe { await transcribeSession(id, paste: false) }
    }
    private func recordCaptureFailure(_ error: Error) {
        message = "Capture failed: \(error.localizedDescription) Saved audio is retained."
        if let id = activeSession {
            do { try store.update(id) { $0.captureState = .failed; $0.captureError = error.localizedDescription } }
            catch { message += " Could not save status: \(error.localizedDescription)" }
        }
    }
    func retry(_ id: UUID) {
        guard available else { return }
        beginTranscription(id, paste: false)
    }
    private func beginTranscription(_ id: UUID, paste: Bool) {
        phase = .transcribing; message = "Preparing saved audio…"
        operation = Task { await transcribeSession(id, paste: paste) }
    }
    private func transcribeSession(_ id: UUID, paste: Bool) async {
        phase = .transcribing; message = "Preparing saved audio…"
        defer { phase = .idle; refreshHistory() }
        do {
            try store.update(id) { $0.transcriptionError = nil }
            let store = store
            let preparation = Task.detached {
                try store.recover(id)
                try AudioJobPreparer().prepare(session: id, store: store)
            }
            try await withTaskCancellationHandler { try await preparation.value } onCancel: { preparation.cancel() }
            try Task.checkCancellation()
            var record = try store.read(id)
            let pending = record.jobs.filter { $0.state != .complete && $0.state != .silence }
            if !pending.isEmpty { try await transcriber.prepare() }
            for job in pending {
                try Task.checkCancellation()
                message = "Transcribing \(job.source.label)…"
                try store.update(id) { record in
                    if let index = record.jobs.firstIndex(where: { $0.id == job.id }) { record.jobs[index].state = .running; record.jobs[index].error = nil }
                }
                do {
                    let result = try await transcriber.transcribe(store.file(id, job.filename)).validated(duration: job.duration)
                    try Task.checkCancellation()
                    try store.update(id) { record in
                        if let index = record.jobs.firstIndex(where: { $0.id == job.id }) {
                            record.jobs[index].response = result; record.jobs[index].state = .complete; record.jobs[index].error = nil
                        }
                    }
                } catch {
                    try store.update(id) { record in
                        if let index = record.jobs.firstIndex(where: { $0.id == job.id }) {
                            record.jobs[index].state = .failed; record.jobs[index].error = error.localizedDescription
                        }
                    }
                    throw error
                }
                refreshHistory()
            }
            record = try store.read(id)
            guard let transcript = trimmedTranscript(record.plainTranscript) else { message = "No speech recognized."; return }
            lastTranscript = transcript
            if paste {
                guard authorizer.currentStatus().accessibilityGranted else { message = "Accessibility permission is required to paste."; return }
                phase = .pasting; message = "Pasting…"
                try await injector.inject(transcript)
                message = "Ready. Hold ⇧⌘D to dictate."
            } else { message = "Transcript saved." }
        } catch {
            message = Task.isCancelled ? "Processing cancelled. Saved audio and completed transcripts are retained." : error.localizedDescription
            do { let failure = message; try store.update(id) { $0.transcriptionError = failure } }
            catch { message += " Could not save status: \(error.localizedDescription)" }
        }
    }
    func generateNotes(_ id: UUID) {
        guard available else { return }
        do {
            let record = try store.read(id)
            guard record.transcriptionComplete, !record.transcript.isEmpty else { throw LaliaError("Complete transcription before generating notes.") }
            phase = .summarizing; message = "Generating notes…"
            operation = Task {
                defer { phase = .idle; refreshHistory() }
                do {
                    let notes = try await transcriber.notes(for: record.transcript)
                    try Task.checkCancellation()
                    try store.update(id) { $0.notes = notes; $0.notesError = nil }
                    message = "Notes saved."
                } catch {
                    message = error.localizedDescription
                    do { try store.update(id) { $0.notesError = error.localizedDescription } }
                    catch { message += " Could not save status: \(error.localizedDescription)" }
                }
            }
        } catch { message = error.localizedDescription }
    }
    func cancelProcessing() { if canCancel { operation?.cancel() } }
    func waitForOperation() async { await operation?.value }
    func shutdown() async {
        quitting = true
        if phase == .preparing { await operation?.value }
        if phase == .recording { finishDictation(transcribe: false) }
        if phase == .recordingSession { phase = .stopping; await finishSession(transcribe: false) }
        operation?.cancel()
        await operation?.value
    }
    func export(_ id: UUID, to url: URL) {
        do { try store.export(id, to: url); message = "Transcript exported." }
        catch { message = error.localizedDescription }
    }
    func copyLastTranscript() { if let lastTranscript { do { try injector.copy(lastTranscript); message = "Copied last transcript." } catch { message = error.localizedDescription } } }
    func hotkeyRegistrationFailed(_ error: Error, shortcut: Shortcut) {
        let failure = "Could not register \(shortcut.label): \(error.localizedDescription)"
        hotkeyError = hotkeyError.map { "\($0)\n\(failure)" } ?? failure
    }
}
