import SwiftUI
import UniformTypeIdentifiers

struct HistoryView: View {
    @ObservedObject var coordinator: DictationCoordinator
    @State private var selection: UUID?
    @ObservedObject var playback: SessionPlayback
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
                        if session.endedAtLimit == true { Text("Recording stopped at the 60-minute limit.").foregroundStyle(.secondary) }
                        if let status = coordinator.processing[session.id] {
                            Text(status).foregroundStyle(.secondary)
                            Button("Cancel Processing") { coordinator.cancelProcessing(session.id) }
                        }
                        ForEach(session.segments.filter { $0.error != nil }) { segment in Text(segment.error ?? "").foregroundStyle(.red) }
                        ForEach(session.jobs.filter { $0.error != nil }) { job in Text(job.error ?? "").foregroundStyle(.red) }
                        HStack {
                            Button("Retry Transcription") { coordinator.retry(session.id) }.disabled(!coordinator.canProcess(session.id) || session.captureState == .recording)
                            if session.kind == .session {
                                Button(session.analysis?.active == nil ? "Add Speaker Labels" : "Reprocess Speaker Labels") { coordinator.reprocess(session.id) }
                                    .disabled(!coordinator.canProcess(session.id) || session.captureState == .recording || session.duration > SessionTranscriptionProfile.maximumDuration)
                            }
                            Button("Export TXT") { export(session) }.disabled(session.transcript.isEmpty)
                            Button("Copy Transcript") { coordinator.copyTranscript(session.id) }.disabled(session.transcript.isEmpty)
                            Button("Reveal Recordings") { NSWorkspace.shared.open(coordinator.store.directory(session.id)) }
                        }
                        if session.duration > SessionTranscriptionProfile.maximumDuration {
                            Text("Speaker labels support recordings up to 60 minutes. Original audio and existing transcripts remain available.").foregroundStyle(.secondary)
                        }
                        if session.analysis?.previous != nil || session.analysis?.previousIsLegacy == true {
                            Button("Restore Previous Transcript") { playback.stop(); coordinator.restoreTranscript(session.id) }.disabled(!coordinator.canProcess(session.id))
                        }
                        Text(session.transcriptionComplete ? "Transcript" : "Transcript (incomplete)").font(.headline)
                        if let transcript = session.analysis?.active {
                            Text("Transcripts may omit speech or include incorrect text and speaker labels. Saved audio is retained.")
                                .font(.caption).foregroundStyle(.secondary)
                            SpeakerControls(coordinator: coordinator, session: session.id, transcript: transcript)
                                .id(transcript.id)
                            if playback.playing { Button("Stop Playback") { playback.stop() } }
                            if let error = playback.error { Text(error).foregroundStyle(.red) }
                            ForEach(transcript.turns) { turn in
                                VStack(alignment: .leading, spacing: 6) {
                                    HStack {
                                        Text("\(SessionRecord.timestamp(turn.start))–\(SessionRecord.timestamp(turn.end))").monospacedDigit().foregroundStyle(.secondary)
                                        if turn.isEvent { Text("Sound").italic() }
                                        else {
                                            Menu(transcript.label(for: turn)) {
                                                Button("Unassigned") { coordinator.editTranscript(session.id) { try $0.assign(turn.id, to: nil) } }
                                                ForEach(transcript.speakerNames.keys.sorted(), id: \.self) { speaker in
                                                    Button(transcript.speakerNames[speaker] ?? speaker) { coordinator.editTranscript(session.id) { try $0.assign(turn.id, to: speaker) } }
                                                }
                                                Button("New Speaker") {
                                                    coordinator.editTranscript(session.id) {
                                                        let speaker = try $0.addSpeaker(named: $0.nextSpeakerName)
                                                        try $0.assign(turn.id, to: speaker)
                                                    }
                                                }
                                            }.fixedSize()
                                        }
                                        if let input = session.analysis?.input {
                                            Button("Play from Here") {
                                                playback.play(session: session.id, filename: input.filename, store: coordinator.store, from: turn.start)
                                            }.disabled([.preparing, .recording, .recordingSession, .stopping].contains(coordinator.phase))
                                        }
                                    }.font(.caption)
                                    Text(turn.text).textSelection(.enabled)
                                }
                            }
                            if transcript.turns.isEmpty { Text("No speech recognized.") }
                        } else { Text(session.transcript.isEmpty ? "No transcript yet." : session.transcript).textSelection(.enabled) }
                        Button("Generate Notes") { coordinator.generateNotes(session.id) }
                            .disabled(!coordinator.canProcess(session.id) || !session.transcriptionComplete || session.transcript.isEmpty)
                        if let error = session.notesError { Text(error).foregroundStyle(.red) }
                        if session.notesStale { Text("Notes are stale after transcript changes. Generate notes again.").foregroundStyle(.orange) }
                        if let notes = session.notes { Text("Generated notes").font(.headline); Text(notes).textSelection(.enabled) }
                    }.frame(maxWidth: .infinity, alignment: .leading).padding()
                }
            } else { Text("Select a recording.") }
        }
        .toolbar { RecordingButton(coordinator: coordinator) }
        .onChange(of: selection) { playback.stop() }
        .onDisappear { playback.stop() }
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

private struct SpeakerControls: View {
    @ObservedObject var coordinator: DictationCoordinator
    var session: UUID
    var transcript: SpeakerTranscript
    @State private var newName = ""
    var body: some View {
        DisclosureGroup("Speakers") {
            ForEach(transcript.speakerNames.keys.sorted(), id: \.self) { speaker in
                SpeakerNameRow(coordinator: coordinator, session: session, speaker: speaker, names: transcript.speakerNames)
            }
            HStack {
                TextField("New speaker name", text: $newName)
                Button("Add Speaker") {
                    coordinator.editTranscript(session) { _ = try $0.addSpeaker(named: newName) }
                    newName = ""
                }.disabled(trimmedTranscript(newName) == nil)
            }
        }
    }
}

private struct SpeakerNameRow: View {
    @ObservedObject var coordinator: DictationCoordinator
    var session: UUID
    var speaker: String
    var names: [String: String]
    @State private var name = ""
    var body: some View {
        HStack {
            TextField("Speaker name", text: $name).onSubmit(rename)
            Button("Rename", action: rename).disabled(trimmedTranscript(name) == nil || name == names[speaker])
            Menu("Merge Into") {
                ForEach(names.keys.filter { $0 != speaker }.sorted(), id: \.self) { destination in
                    Button(names[destination] ?? destination) { coordinator.editTranscript(session) { try $0.merge(speaker, into: destination) } }
                }
            }.disabled(names.count < 2)
        }
        .onAppear { name = names[speaker] ?? "" }
        .onChange(of: names[speaker]) { name = names[speaker] ?? "" }
    }
    private func rename() { coordinator.editTranscript(session) { try $0.rename(speaker, to: name) } }
}
