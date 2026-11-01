import AVFoundation
import SwiftUI

@main enum LaliaMain {
    @MainActor static func main() async {
        if CommandLine.arguments.count > 1 {
            do {
                let arguments = CommandLine.arguments
                guard (arguments.count == 3 || (arguments.count == 5 && arguments[3] == "--config")),
                      ["--transcribe-file", "--transcribe-session-file"].contains(arguments[1]) else {
                    throw LaliaError("Usage: Lalia --transcribe-file|--transcribe-session-file AUDIO_FILE [--config CONFIG_FILE]")
                }
                let config = try arguments.count == 5 ? OMLXConfiguration.load(from: URL(fileURLWithPath: arguments[4])) : OMLXConfiguration.load()
                try await transcribeFile(URL(fileURLWithPath: arguments[2]), configuration: config, session: arguments[1] == "--transcribe-session-file")
                exit(0)
            } catch { FileHandle.standardError.write(Data("\(error.localizedDescription)\n".utf8)); exit(1) }
        }
        LaliaApp.main()
    }

    // The file probe uses the same conversion, job limits, and HTTP client as the app.
    // Its temporary derived files never enter the user's recording history.
    @MainActor static func transcribeFile(_ file: URL, configuration: OMLXConfiguration, session: Bool = false) async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "LaliaSmoke-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let store = SessionStore(root: root)
        let record = try store.create(kind: session ? .session : .dictation)
        let audio = try AVAudioFile(forReading: file)
        let segment = AudioSegment(id: UUID(), source: .microphone, filename: "fixture.\(file.pathExtension)", start: 0,
                                   sampleRate: audio.processingFormat.sampleRate, channels: audio.processingFormat.channelCount,
                                   frames: audio.length, complete: true)
        try FileManager.default.copyItem(at: file, to: store.file(record.id, segment.filename))
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: store.file(record.id, segment.filename).path)
        try store.update(record.id) { $0.segments = [segment]; $0.captureState = .complete }
        let client = OMLXClient(configuration: .success(configuration))
        if session {
            let input = try SessionAudioPreparer().prepare(session: record.id, store: store)
            var attempt = SessionAnalysis.Attempt(profile: try client.sessionProfile())
            if input.silence { print("No speech recognized."); return }
            try await client.prepareSession(profile: attempt.profile)
            attempt.rawResponse = try await client.transcribeSession(store.file(record.id, input.filename), profile: attempt.profile)
            let transcript = try SpeakerTranscript(attempt: attempt, duration: input.duration)
            print(transcript.turns.isEmpty ? "No speech recognized." : transcript.transcript)
            return
        }
        try AudioJobPreparer().prepare(session: record.id, store: store)
        try await client.prepare()
        for job in try store.read(record.id).jobs where job.state != .silence {
            let result = try await client.transcribe(store.file(record.id, job.filename), duration: job.duration).validated(duration: job.duration)
            try store.update(record.id) { record in
                if let index = record.jobs.firstIndex(where: { $0.id == job.id }) { record.jobs[index].response = result; record.jobs[index].state = .complete }
            }
        }
        let result = try store.read(record.id)
        print(result.transcript.isEmpty ? "No speech recognized." : result.transcript)
    }
}
@MainActor final class AppDelegate: NSObject, NSApplicationDelegate {
    var coordinator: DictationCoordinator?
    var gestureMonitor: OptionGestureMonitor?
    private var terminating = false
    private let lifecycle = SessionLifecycleMonitor()
    private var noticeScheduled = false
    private var noticeAlert: NSAlert?

    func connect(coordinator: DictationCoordinator, gestures: OptionGestureMonitor) {
        self.coordinator = coordinator
        gestureMonitor = gestures
        lifecycle.onUsabilityChanged = { usable in
            gestures.setSessionUsable(usable)
            coordinator.setSessionUsable(usable)
        }
        lifecycle.onInterruption = { coordinator.interrupt($0) }
        coordinator.onInterruptionNotice = { [weak self] in self?.presentInterruptionNotice() }
        coordinator.onCaptureAvailabilityChanged = { [weak self] available, active in
            gestures.setCaptureState(available: available, active: active)
            if active, let alert = self?.noticeAlert {
                alert.window.orderOut(nil)
                self?.noticeAlert = nil
            }
        }
        lifecycle.start()
    }

    private func presentInterruptionNotice() {
        guard !terminating, !noticeScheduled, noticeAlert == nil else { return }
        noticeScheduled = true
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.noticeScheduled = false
            guard !self.terminating, let notice = self.coordinator?.takeInterruptionNotice() else { return }
            let alert = NSAlert()
            alert.alertStyle = .informational
            alert.messageText = notice.title
            alert.informativeText = notice.message
            let dismiss = alert.addButton(withTitle: "OK")
            dismiss.target = self
            dismiss.action = #selector(self.dismissInterruptionNotice)
            self.noticeAlert = alert
            NSApp.activate()
            alert.window.makeKeyAndOrderFront(nil)
        }
    }

    @objc private func dismissInterruptionNotice() {
        noticeAlert?.window.orderOut(nil)
        noticeAlert = nil
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard !terminating else { return .terminateLater }
        terminating = true
        lifecycle.stop()
        dismissInterruptionNotice()
        Task {
            await coordinator?.shutdown()
            gestureMonitor?.stop()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}

struct LaliaApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var coordinator: DictationCoordinator
    @StateObject private var playback: SessionPlayback
    private let gestures: OptionGestureMonitor
    init() {
        let playback = SessionPlayback()
        let coordinator = DictationCoordinator(recorder: Recorder(), transcriber: OMLXClient(), injector: PasteInjector(), authorizer: SystemAuthorizer(), playback: playback)
        _coordinator = StateObject(wrappedValue: coordinator)
        _playback = StateObject(wrappedValue: playback)
        let gestures = OptionGestureMonitor()
        self.gestures = gestures
        delegate.gestureMonitor = gestures
        gestures.onAction = { action in
            switch action {
            case let .beginDictation(id): coordinator.gestureBegan(id)
            case let .finishDictation(id): coordinator.gestureFinished(id)
            case let .cancelDictation(id): coordinator.gestureCancelled(id)
            case let .toggleRecording(id): coordinator.gestureToggle(id)
            }
        }
        gestures.onAvailability = { available, message in coordinator.setGestureAvailability(available, message: message) }
        delegate.connect(coordinator: coordinator, gestures: gestures)
        coordinator.startupRefresh()
        gestures.start()
        delegate.coordinator = coordinator
    }
    var body: some Scene {
        MenuBarExtra {
            LaliaMenu(coordinator: coordinator, gestures: gestures)
                .onAppear { delegate.coordinator = coordinator }
        } label: {
            HStack(spacing: 3) {
                Image(systemName: "waveform")
                if !coordinator.gestureAvailable { Image(systemName: "exclamationmark.triangle") }
                if coordinator.phase == .recording || coordinator.phase == .recordingSession { Image(systemName: "record.circle.fill") }
            }
            .help(coordinator.shortcutStatus)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Lalia. \(coordinator.phase == .recording ? "Dictating." : coordinator.phase == .recordingSession ? "Transcribing." : "Not capturing audio.") \(coordinator.shortcutStatus)")
        }
        Window("Lalia Transcripts", id: "history") {
            HistoryView(coordinator: coordinator, playback: playback)
                .onAppear { delegate.coordinator = coordinator }
        }
        .defaultSize(width: 900, height: 600)
    }
}

private struct LaliaMenu: View {
    @ObservedObject var coordinator: DictationCoordinator
    let gestures: OptionGestureMonitor
    @Environment(\.openWindow) private var openWindow
    @State private var setupMessage: String?
    var body: some View {
        Text(coordinator.message)
        if !coordinator.isDictationEnabled {
            Button("Enable Dictation") { Task { await coordinator.enableDictation() } }.disabled(!coordinator.available)
        }
        if !coordinator.gestureAvailable && !CGPreflightListenEventAccess() {
            Button("Allow Keyboard Shortcuts") {
                Task {
                    if await coordinator.enableGestures() { gestures.start() }
                }
            }
        }
        if !coordinator.gestureAvailable { Text(coordinator.shortcutStatus) }
        RecordingButton(coordinator: coordinator)
        if !coordinator.processing.isEmpty { Text("Preparing \(coordinator.processing.count) transcripts") }
        if coordinator.canCancel { Button("Cancel Current Processing") { coordinator.cancelProcessing() } }
        Button("Transcript History") { coordinator.refreshHistory(); openWindow(id: "history"); NSApp.activate() }
        Button("Copy Last Transcript") { coordinator.copyLastTranscript() }.disabled(coordinator.lastTranscript == nil)
        Divider()
        Button("Check Runtime Setup") {
            Task { setupMessage = await OMLXClient().setupCheck() }
        }
        Button("Copy Setup Command") {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString("brew install cosgroveb/tap/lalia-runtime && lalia-runtime setup", forType: .string)
        }
        if let setupMessage { Text(setupMessage).textSelection(.enabled) }
        Divider()
        Button("About Lalia") { NSApp.orderFrontStandardAboutPanel(options: [:]); NSApp.activate() }
        Button("Quit") { NSApplication.shared.terminate(nil) }
    }
}

struct RecordingButton: View {
    @ObservedObject var coordinator: DictationCoordinator
    var body: some View {
        if coordinator.phase == .recordingSession {
            Text("\(SessionRecord.timestamp(coordinator.recordingElapsed)) elapsed · \(SessionRecord.timestamp(coordinator.recordingRemaining)) remaining")
                .foregroundStyle(coordinator.recordingWarning ? .orange : .secondary)
        }
        Button(coordinator.phase == .recordingSession ? "Stop Transcribing" : "Start Transcribing") {
            coordinator.toggleRecording()
        }
        .disabled(coordinator.phase != .recordingSession && !coordinator.available)
        .help("Double-tap Right Option to start or stop transcribing microphone and Mac audio for up to 60 minutes.")
    }
}
