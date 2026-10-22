import AVFoundation
import SwiftUI
import UniformTypeIdentifiers

@main enum LaliaMain {
    @MainActor static func main() async {
        if CommandLine.arguments.count > 1 {
            do {
                let arguments = CommandLine.arguments
                guard (arguments.count == 3 || (arguments.count == 5 && arguments[3] == "--config")), arguments[1] == "--transcribe-file" else {
                    throw LaliaError("Usage: Lalia --transcribe-file AUDIO_FILE [--config CONFIG_FILE]")
                }
                let config = try arguments.count == 5 ? OMLXConfiguration.load(from: URL(fileURLWithPath: arguments[4])) : OMLXConfiguration.load()
                try await transcribeFile(URL(fileURLWithPath: arguments[2]), configuration: config)
                exit(0)
            } catch { FileHandle.standardError.write(Data("\(error.localizedDescription)\n".utf8)); exit(1) }
        }
        LaliaApp.main()
    }

    // The file probe uses the same conversion, job limits, and HTTP client as the app.
    // Its temporary derived files never enter the user's recording history.
    @MainActor static func transcribeFile(_ file: URL, configuration: OMLXConfiguration) async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "LaliaSmoke-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let store = SessionStore(root: root)
        let record = try store.create(kind: .dictation)
        let audio = try AVAudioFile(forReading: file)
        let segment = AudioSegment(id: UUID(), source: .microphone, filename: "fixture.\(file.pathExtension)", start: 0,
                                   sampleRate: audio.processingFormat.sampleRate, channels: audio.processingFormat.channelCount,
                                   frames: audio.length, complete: true)
        try FileManager.default.copyItem(at: file, to: store.file(record.id, segment.filename))
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: store.file(record.id, segment.filename).path)
        try store.update(record.id) { $0.segments = [segment]; $0.captureState = .complete }
        try AudioJobPreparer().prepare(session: record.id, store: store)
        let client = OMLXClient(configuration: .success(configuration))
        try await client.prepare()
        for job in try store.read(record.id).jobs where job.state != .silence {
            let result = try await client.transcribe(store.file(record.id, job.filename)).validated(duration: job.duration)
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
    private var terminating = false
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard !terminating else { return .terminateLater }
        terminating = true
        Task { await coordinator?.shutdown(); sender.reply(toApplicationShouldTerminate: true) }
        return .terminateLater
    }
}

struct LaliaApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var coordinator: DictationCoordinator
    private let hotkeys: [CarbonHotkey]
    init() {
        let coordinator = DictationCoordinator(recorder: Recorder(), transcriber: OMLXClient(), injector: PasteInjector(), authorizer: SystemAuthorizer())
        _coordinator = StateObject(wrappedValue: coordinator)
        hotkeys = [
            CarbonHotkey(shortcut: .dictation, pressed: { coordinator.hotkeyPressed() }, released: { coordinator.hotkeyReleased() }),
            CarbonHotkey(shortcut: .recording, pressed: { coordinator.toggleRecording() }),
        ]
        coordinator.startupRefresh()
        for hotkey in hotkeys {
            do { try hotkey.register() } catch { coordinator.hotkeyRegistrationFailed(error, shortcut: hotkey.shortcut) }
        }
        delegate.coordinator = coordinator
    }
    var body: some Scene {
        MenuBarExtra("Lalia", systemImage: coordinator.phase == .recording || coordinator.phase == .recordingSession ? "record.circle.fill" : "waveform") {
            LaliaMenu(coordinator: coordinator)
                .onAppear { delegate.coordinator = coordinator }
        }
        Window("Lalia Recordings", id: "history") {
            HistoryView(coordinator: coordinator)
                .onAppear { delegate.coordinator = coordinator }
        }
        .defaultSize(width: 900, height: 600)
    }
}

private struct LaliaMenu: View {
    @ObservedObject var coordinator: DictationCoordinator
    @Environment(\.openWindow) private var openWindow
    var body: some View {
        Text(coordinator.message)
        if let hotkeyError = coordinator.hotkeyError { Text(hotkeyError) }
        if !coordinator.isDictationEnabled {
            Button("Enable Dictation") { Task { await coordinator.enableDictation() } }.disabled(!coordinator.available)
        }
        RecordingButton(coordinator: coordinator)
        if coordinator.canCancel { Button("Cancel Processing") { coordinator.cancelProcessing() } }
        Button("Recording History") { coordinator.refreshHistory(); openWindow(id: "history"); NSApp.activate() }
        Button("Copy Last Transcript") { coordinator.copyLastTranscript() }.disabled(coordinator.lastTranscript == nil)
        Divider()
        Button("Quit") { NSApplication.shared.terminate(nil) }
    }
}

private struct RecordingButton: View {
    @ObservedObject var coordinator: DictationCoordinator
    var body: some View {
        Button(coordinator.phase == .recordingSession ? "Stop Recording (⇧⌘R)" : "Start Recording (⇧⌘R)") {
            coordinator.toggleRecording()
        }
        .disabled(coordinator.phase != .recordingSession && !coordinator.available)
        .help("Record microphone and Mac audio until stopped, then transcribe.")
    }
}

private struct HistoryView: View {
    @ObservedObject var coordinator: DictationCoordinator
    @State private var selection: UUID?
    var body: some View {
        NavigationSplitView {
            List(coordinator.sessions, selection: $selection) { session in
                VStack(alignment: .leading) {
                    Text(session.createdAt.formatted(date: .abbreviated, time: .shortened))
                    Text("\(session.kind.rawValue.capitalized) · \(session.captureState.rawValue) · \(SessionRecord.timestamp(session.duration))")
                        .font(.caption).foregroundStyle(.secondary)
                }.tag(session.id)
            }
        } detail: {
            if let session = coordinator.sessions.first(where: { $0.id == selection }) {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        Text(session.createdAt.formatted()).font(.title2)
                        if let error = session.captureError { Text(error).foregroundStyle(.red) }
                        if let error = session.transcriptionError { Text(error).foregroundStyle(.red) }
                        ForEach(session.segments.filter { $0.error != nil }) { segment in Text(segment.error ?? "").foregroundStyle(.red) }
                        ForEach(session.jobs.filter { $0.error != nil }) { job in Text(job.error ?? "").foregroundStyle(.red) }
                        HStack {
                            Button("Retry Transcription") { coordinator.retry(session.id) }.disabled(!coordinator.available || session.captureState == .recording)
                            Button("Export TXT") { export(session) }.disabled(session.transcript.isEmpty)
                            Button("Reveal Recordings") { NSWorkspace.shared.open(coordinator.store.directory(session.id)) }
                        }
                        Text(session.transcriptionComplete ? "Transcript" : "Transcript (incomplete)").font(.headline)
                        Text(session.transcript.isEmpty ? "No transcript yet." : session.transcript).textSelection(.enabled)
                        Button("Generate Notes") { coordinator.generateNotes(session.id) }
                            .disabled(!coordinator.available || !session.transcriptionComplete || session.transcript.isEmpty)
                        if let error = session.notesError { Text(error).foregroundStyle(.red) }
                        if let notes = session.notes { Text("Generated notes").font(.headline); Text(notes).textSelection(.enabled) }
                    }.frame(maxWidth: .infinity, alignment: .leading).padding()
                }
            } else { Text("Select a recording.") }
        }
        .toolbar { RecordingButton(coordinator: coordinator) }
        .safeAreaInset(edge: .bottom) {
            VStack(alignment: .leading) {
                Text(coordinator.message)
                ForEach(coordinator.diagnostics, id: \.self) { Text($0).foregroundStyle(.red) }
            }.font(.caption).frame(maxWidth: .infinity, alignment: .leading).padding()
        }
    }
    private func export(_ session: SessionRecord) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.plainText]
        panel.nameFieldStringValue = "Lalia-\(session.id).txt"
        if panel.runModal() == .OK, let url = panel.url { coordinator.export(session.id, to: url) }
    }
}
