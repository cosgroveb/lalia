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
    @Published private(set) var processing: [UUID: String] = [:]
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
    private var captureOperation: Task<Void, Never>?
    private enum Work {
        case transcription(paste: Bool, generation: UUID?), analysis(reprocess: Bool), notes
    }
    private struct ProcessingJob { let session: UUID; let work: Work }
    private var queue: [ProcessingJob] = []
    private var currentJob: ProcessingJob?
    private var processingOperation: Task<Void, Never>?
    private var captureGeneration = UUID()
    private var quitting = false
    private var startingFailure: Error?
    private var recordingActivity: RecordingActivity?
    private var deadline: Task<Void, Never>?
    let recordingLimit: Double
    private let warningSeconds: Double
    var recordingRemaining: Double { max(0, recordingLimit - recordingElapsed) }
    var available: Bool { !quitting && (phase == .idle || phase == .needsPermission) }
    var canCancel: Bool { currentJob != nil }
    func canProcess(_ id: UUID) -> Bool { !quitting && id != activeSession && processing[id] == nil }

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
        captureGeneration = UUID()
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
            activeSession = nil; dictationSegment = nil; phase = .idle
            if transcribe { beginTranscription(id, paste: true) }
        } catch {
            recorder.close()
            recordCaptureFailure(error)
            activeSession = nil; dictationSegment = nil; phase = .idle
        }
        refreshHistory()
    }
    func startSession() {
        guard available else { return }
        captureGeneration = UUID()
        phase = .preparing; message = "Starting microphone and Mac audio…"
        startingFailure = nil
        captureOperation = Task {
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
        captureOperation = Task { await finishSession(transcribe: true) }
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
        captureOperation = Task { await finishSession(transcribe: false, failure: error) }
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
        if failure == nil, transcribe { beginAnalysis(id, reprocess: false) }
    }
    private func recordCaptureFailure(_ error: Error) {
        message = "Capture failed: \(error.localizedDescription) Saved audio is retained."
        if let id = activeSession {
            do { try store.update(id) { $0.captureState = .failed; $0.captureError = error.localizedDescription } }
            catch { message += " Could not save status: \(error.localizedDescription)" }
        }
    }
    func retry(_ id: UUID) {
        guard canProcess(id) else { return }
        if (try? store.read(id).analysis) != nil { beginAnalysis(id, reprocess: false); return }
        beginTranscription(id, paste: false)
    }
    func reprocess(_ id: UUID) {
        guard canProcess(id) else { return }
        beginAnalysis(id, reprocess: true)
    }
    private func beginAnalysis(_ id: UUID, reprocess: Bool) {
        enqueue(id, work: .analysis(reprocess: reprocess))
    }
    private func analyzeSession(_ id: UUID, reprocess: Bool) async {
        processingMessage("Preparing conversation audio…", for: id)
        let activity = ProcessInfo.processInfo.beginActivity(options: [.idleSystemSleepDisabled], reason: "Lalia analysis \(id)")
        defer { ProcessInfo.processInfo.endActivity(activity); refreshHistory() }
        do {
            try Task.checkCancellation()
            var record = try store.recover(id)
            if !reprocess, record.analysis?.attempt?.state == .complete || (record.analysis?.attempt == nil && record.transcriptionComplete) {
                processingMessage("Transcript already saved. Use Reprocess to replace it.", for: id); return
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
                processingMessage("Transcribing conversation with speaker labels…", for: id)
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
            processingMessage("Saving speaker transcript…", for: id)
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
            processingMessage(accepted.turns.isEmpty ? "No speech recognized." : "Speaker transcript saved.", for: id)
        } catch {
            let failure = Task.isCancelled ? "Processing cancelled. Audio and previous transcript are retained. oMLX may still be processing; Retry may wait for that request." : error.localizedDescription
            processingMessage(failure, for: id)
            do {
                try store.update(id) { $0.analysis?.attempt?.state = .failed; $0.analysis?.attempt?.error = failure; $0.transcriptionError = failure }
            } catch { processingMessage("\(failure) Could not save status: \(error.localizedDescription)", for: id) }
        }
    }
    private func beginTranscription(_ id: UUID, paste: Bool) {
        let generation = paste && currentJob == nil && queue.isEmpty ? captureGeneration : nil
        enqueue(id, work: .transcription(paste: paste, generation: generation))
    }
    private func transcribeSession(_ id: UUID, paste: Bool, generation: UUID?) async {
        processingMessage("Preparing saved audio…", for: id)
        defer { refreshHistory() }
        do {
            try Task.checkCancellation()
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
                processingMessage("Transcribing \(job.source.label)…", for: id)
                try store.update(id) { record in
                    if let index = record.jobs.firstIndex(where: { $0.id == job.id }) { record.jobs[index].state = .running; record.jobs[index].error = nil }
                }
                do {
                    let result = try await transcriber.transcribe(store.file(id, job.filename), duration: job.duration).validated(duration: job.duration)
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
            guard let transcript = trimmedTranscript(record.plainTranscript) else { processingMessage("No speech recognized.", for: id); return }
            lastTranscript = transcript
            lastTranscriptSession = id
            if paste {
                guard generation == captureGeneration, available else {
                    processingMessage("Dictation saved. Use Copy Last Transcript or Recording History.", for: id); return
                }
                guard authorizer.currentStatus().accessibilityGranted else { processingMessage("Accessibility permission is required to paste.", for: id); return }
                processingMessage("Pasting…", for: id)
                try await injector.inject(transcript)
                processingMessage("Ready. Hold ⇧⌘D to dictate.", for: id)
            } else { processingMessage("Transcript saved.", for: id) }
        } catch {
            let failure = Task.isCancelled ? "Processing cancelled. Saved audio and completed transcripts are retained." : error.localizedDescription
            processingMessage(failure, for: id)
            do { try store.update(id) { $0.transcriptionError = failure } }
            catch { processingMessage("\(failure) Could not save status: \(error.localizedDescription)", for: id) }
        }
    }
    func generateNotes(_ id: UUID) {
        enqueue(id, work: .notes)
    }
    private func summarize(_ id: UUID) async {
        defer { refreshHistory() }
        do {
            try Task.checkCancellation()
            let record = try store.read(id)
            guard record.transcriptionComplete, !record.transcript.isEmpty else { throw LaliaError("Complete transcription before generating notes.") }
            processingMessage("Generating notes…", for: id)
            let notes = try await transcriber.notes(for: record.transcript)
            try Task.checkCancellation()
            try store.update(id) { $0.notes = notes; $0.notesError = nil; $0.notesRevision = record.analysis?.active?.revision }
            processingMessage("Notes saved.", for: id)
        } catch {
            let failure = Task.isCancelled ? "Notes cancelled. Saved audio and transcripts are retained." : error.localizedDescription
            processingMessage(failure, for: id)
            do { try store.update(id) { $0.notesError = failure } }
            catch { processingMessage("\(failure) Could not save status: \(error.localizedDescription)", for: id) }
        }
    }
    private func enqueue(_ id: UUID, work: Work) {
        guard canProcess(id) else { return }
        processing[id] = "Queued"
        queue.append(ProcessingJob(session: id, work: work))
        if available {
            if case .notes = work { message = "Notes queued." }
            else { message = "Transcription queued. Audio is saved." }
        }
        startNextProcessing()
    }
    private func startNextProcessing() {
        guard !quitting, currentJob == nil, !queue.isEmpty else { return }
        let job = queue.removeFirst()
        currentJob = job
        processingOperation = Task {
            switch job.work {
            case .transcription(let paste, let generation): await transcribeSession(job.session, paste: paste, generation: generation)
            case .analysis(let reprocess): await analyzeSession(job.session, reprocess: reprocess)
            case .notes: await summarize(job.session)
            }
            processing.removeValue(forKey: job.session)
            currentJob = nil; processingOperation = nil
            startNextProcessing()
        }
    }
    private func processingMessage(_ value: String, for id: UUID) {
        processing[id] = value
        if available { message = value }
    }
    func cancelProcessing(_ id: UUID? = nil) {
        guard let id = id ?? currentJob?.session else { return }
        if currentJob?.session == id { processingOperation?.cancel(); return }
        guard let index = queue.firstIndex(where: { $0.session == id }) else { return }
        let job = queue.remove(at: index)
        processing.removeValue(forKey: id)
        let failure = "Processing cancelled before starting. Saved audio and transcripts are retained."
        do {
            try store.update(id) {
                if case .notes = job.work { $0.notesError = failure }
                else { $0.transcriptionError = failure }
            }
        } catch { if available { message = "Could not save cancellation: \(error.localizedDescription)" } }
        refreshHistory()
    }
    func waitForOperation() async {
        await captureOperation?.value
        while let processingOperation { await processingOperation.value }
    }
    func shutdown() async {
        quitting = true
        deadline?.cancel(); deadline = nil
        if phase == .preparing { await captureOperation?.value }
        if phase == .recording { finishDictation(transcribe: false) }
        if phase == .recordingSession { phase = .stopping; await finishSession(transcribe: false) }
        captureOperation?.cancel()
        await captureOperation?.value
        for job in queue { cancelProcessing(job.session) }
        processingOperation?.cancel()
        await processingOperation?.value
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
        guard canProcess(id) else { return }
        do { try store.restoreTranscript(id); try refreshLastTranscript(id); message = "Previous transcript restored."; refreshHistory() }
        catch { message = error.localizedDescription }
    }
    private func refreshLastTranscript(_ id: UUID) throws {
        if lastTranscriptSession == id { lastTranscript = trimmedTranscript(try store.read(id).plainTranscript) }
    }
    func copyTranscript(_ id: UUID) {
        do {
            guard let transcript = trimmedTranscript(try store.read(id).plainTranscript) else { throw LaliaError("No transcript to copy.") }
            try injector.copy(transcript); message = "Copied transcript."
        } catch { message = error.localizedDescription }
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
