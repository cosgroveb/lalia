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
    @Published private(set) var recordingElapsed = 0.0
    @Published private(set) var recordingWarning = false
    let store: SessionStore
    private let recorder: Recording
    private let capture: SessionCapturing
    private let transcriber: Transcribing
    private let injector: Injecting
    private let authorizer: Authorizing
    private let playback: PlaybackStopping
    private var activeSession: UUID?
    private var lastTranscriptSession: UUID?
    private var dictationSegment: UUID?
    private var operation: Task<Void, Never>?
    private var quitting = false
    private var startingFailure: Error?
    private var recordingActivity: RecordingActivity?
    private var deadline: Task<Void, Never>?
    let recordingLimit: Double
    private let warningSeconds: Double
    var recordingRemaining: Double { max(0, recordingLimit - recordingElapsed) }
    var available: Bool { !quitting && (phase == .idle || phase == .needsPermission) }
    var canCancel: Bool { phase == .transcribing || phase == .summarizing }

    init(recorder: Recording, transcriber: Transcribing, injector: Injecting, authorizer: Authorizing,
         store: SessionStore = SessionStore(), capture: SessionCapturing = SessionCapture(),
         playback: PlaybackStopping = SessionPlayback(),
         recordingLimit: Double = SessionTranscriptionProfile.maximumDuration, warningSeconds: Double = 300) {
        self.recorder = recorder; self.transcriber = transcriber; self.injector = injector
        self.authorizer = authorizer; self.store = store; self.capture = capture
        self.playback = playback
        self.recordingLimit = recordingLimit; self.warningSeconds = warningSeconds
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
            playback.stop()
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
                try store.update(session.id) { $0.analysis = SessionAnalysis() }
                activeSession = session.id
                recordingActivity = RecordingActivity(session: session.id)
                playback.stop()
                let started = ContinuousClock.now
                try await capture.start(session: session.id, store: store, maximumDuration: recordingLimit) { [weak self] error in
                    self?.captureFailed(error)
                }
                if let startingFailure {
                    phase = .stopping
                    await finishSession(transcribe: false, failure: startingFailure)
                } else {
                    phase = .recordingSession; message = "Recording microphone and Mac audio…"
                    startDeadline(at: started)
                }
            } catch {
                recordingActivity = nil
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
    private func startDeadline(at start: ContinuousClock.Instant) {
        deadline?.cancel()
        recordingElapsed = 0; recordingWarning = false
        deadline = Task { [weak self] in
            while !Task.isCancelled {
                guard let self, self.phase == .recordingSession else { return }
                let duration = start.duration(to: .now).components
                self.recordingElapsed = min(self.recordingLimit, Double(duration.seconds) + Double(duration.attoseconds) / 1e18)
                self.recordingWarning = self.recordingRemaining <= self.warningSeconds
                if self.recordingElapsed >= self.recordingLimit {
                    if let id = self.activeSession {
                        do { try self.store.update(id) { $0.endedAtLimit = true } }
                        catch { self.message = "Could not save recording limit status: \(error.localizedDescription)" }
                    }
                    self.stopSession()
                    return
                }
                if self.recordingWarning { self.message = "Recording ends in \(SessionRecord.timestamp(self.recordingRemaining))." }
                let next = self.recordingWarning ? self.recordingRemaining : self.recordingRemaining - self.warningSeconds
                do { try await Task.sleep(for: .seconds(min(1, max(0.001, next)))) }
                catch { return }
            }
        }
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
        deadline?.cancel(); deadline = nil
        var failure = failure
        do { try await capture.stop() } catch { if failure == nil { failure = error } }
        recordingActivity = nil
        if let failure { recordCaptureFailure(failure) }
        else {
            do { try store.update(id) { $0.captureState = .complete } }
            catch { failure = error; recordCaptureFailure(error) }
        }
        activeSession = nil; phase = .idle
        refreshHistory()
        if failure == nil, transcribe { await analyzeSession(id, reprocess: false) }
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
        if (try? store.read(id).analysis) != nil { beginAnalysis(id, reprocess: false); return }
        beginTranscription(id, paste: false)
    }
    func reprocess(_ id: UUID) {
        guard available else { return }
        beginAnalysis(id, reprocess: true)
    }
    private func beginAnalysis(_ id: UUID, reprocess: Bool) {
        phase = .transcribing; message = "Preparing conversation audio…"
        operation = Task { await analyzeSession(id, reprocess: reprocess) }
    }
    private func analyzeSession(_ id: UUID, reprocess: Bool) async {
        phase = .transcribing; message = "Preparing conversation audio…"
        let activity = ProcessInfo.processInfo.beginActivity(options: [.idleSystemSleepDisabled], reason: "Lalia analysis \(id)")
        defer { ProcessInfo.processInfo.endActivity(activity); phase = .idle; refreshHistory() }
        do {
            var record = try store.recover(id)
            if !reprocess, record.analysis?.attempt?.state == .complete || (record.analysis?.attempt == nil && record.transcriptionComplete) {
                message = "Transcript already saved. Use Reprocess to replace it."; return
            }
            let profile = try (!reprocess ? record.analysis?.attempt?.profile : nil) ?? transcriber.sessionProfile()
            let attempt = SessionAnalysis.Attempt(profile: profile)
            try store.update(id) { record in
                if record.analysis == nil { record.analysis = SessionAnalysis() }
                record.analysis?.attempt = attempt
                record.transcriptionError = nil
            }
            let store = store
            let preparation = Task.detached { try SessionAudioPreparer().prepare(session: id, store: store) }
            let input = try await withTaskCancellationHandler { try await preparation.value } onCancel: { preparation.cancel() }
            try Task.checkCancellation()
            var accepted: SpeakerTranscript
            if input.silence { accepted = SpeakerTranscript(silenceFor: attempt) }
            else {
                try await transcriber.prepareSession(profile: profile)
                try Task.checkCancellation()
                message = "Transcribing conversation with speaker labels…"
                try store.update(id) { $0.analysis?.attempt?.state = .running }
                let raw = try await transcriber.transcribeSession(store.file(id, input.filename), profile: profile)
                // Save even a rejected response before interpreting model output.
                try store.update(id) { $0.analysis?.attempt?.rawResponse = raw }
                try Task.checkCancellation()
                record = try store.read(id)
                guard let saved = record.analysis?.attempt else { throw LaliaError("Missing analysis attempt.") }
                accepted = try SpeakerTranscript(attempt: saved, duration: input.duration)
            }
            try Task.checkCancellation()
            message = "Saving speaker transcript…"
            try store.update(id) { record in
                let active = record.analysis?.active
                let previousIsLegacy = active == nil && record.hasLegacyTranscript
                record.analysis?.previous = active
                record.analysis?.previousIsLegacy = previousIsLegacy
                record.analysis?.active = accepted
                record.analysis?.attempt?.state = .complete
                record.analysis?.attempt?.error = nil
            }
            lastTranscript = trimmedTranscript(accepted.transcript)
            lastTranscriptSession = id
            message = accepted.turns.isEmpty ? "No speech recognized." : "Speaker transcript saved."
        } catch {
            message = Task.isCancelled ? "Processing cancelled. Audio and previous transcript are retained. oMLX may still be processing; Retry may wait for that request." : error.localizedDescription
            do {
                let failure = message
                try store.update(id) { $0.analysis?.attempt?.state = .failed; $0.analysis?.attempt?.error = failure; $0.transcriptionError = failure }
            } catch { message += " Could not save status: \(error.localizedDescription)" }
        }
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
            lastTranscriptSession = id
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
                    try store.update(id) { $0.notes = notes; $0.notesError = nil; $0.notesRevision = record.analysis?.active?.revision }
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
        deadline?.cancel(); deadline = nil
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
    func editTranscript(_ id: UUID, _ edit: (inout SpeakerTranscript) throws -> Void) {
        do { try store.editTranscript(id, edit); try refreshLastTranscript(id); message = "Speaker correction saved."; refreshHistory() }
        catch { message = error.localizedDescription }
    }
    func restoreTranscript(_ id: UUID) {
        guard available else { return }
        do { try store.restoreTranscript(id); try refreshLastTranscript(id); message = "Previous transcript restored."; refreshHistory() }
        catch { message = error.localizedDescription }
    }
    private func refreshLastTranscript(_ id: UUID) throws {
        if lastTranscriptSession == id { lastTranscript = trimmedTranscript(try store.read(id).plainTranscript) }
    }
    func copyLastTranscript() { if let lastTranscript { do { try injector.copy(lastTranscript); message = "Copied last transcript." } catch { message = error.localizedDescription } } }
    func hotkeyRegistrationFailed(_ error: Error, shortcut: Shortcut) {
        let failure = "Could not register \(shortcut.label): \(error.localizedDescription)"
        hotkeyError = hotkeyError.map { "\($0)\n\(failure)" } ?? failure
    }
}

private final class RecordingActivity {
    private let token: NSObjectProtocol
    init(session: UUID) {
        // ScreenCaptureKit also stops audio when its display goes to sleep.
        token = ProcessInfo.processInfo.beginActivity(
            options: [.idleDisplaySleepDisabled, .idleSystemSleepDisabled],
            reason: "Lalia recording \(session.uuidString)"
        )
    }
    deinit { ProcessInfo.processInfo.endActivity(token) }
}
