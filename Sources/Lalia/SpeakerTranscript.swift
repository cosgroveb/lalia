import Foundation

struct SessionTranscriptionProfile: Codable, Equatable, Sendable {
    static let maximumDuration = 3_600.0
    // Keep analyses saved by the earlier 90-minute build readable.
    static let maximumSavedDuration = 5_400.0
    // OpenMOSS inference_utils.py DEFAULT_PROMPT at 79c675f0670cc2f824bcbcdd98263704f40f1f7e.
    static let officialPrompt = "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。"
    var baseURL = "http://127.0.0.1:8000/v1"
    var model = "MOSS-Transcribe-Diarize"
    var language = "en"
    var prompt: String? = officialPrompt // Synthesized decoding keeps absent saved prompts nil.
    var maximumTokens = 49_152
    var contextTokens = 131_072
    var maximumUploadBytes = 250_000_000
    var timeout = 2_400.0

    func validate() throws {
        _ = try OMLXConfiguration(baseURL: baseURL, transcriptionModel: model, language: language).endpoint("models")
        guard maximumTokens > 0, maximumTokens < contextTokens, contextTokens <= 131_072,
              maximumUploadBytes > 0, maximumUploadBytes <= 250_000_000,
              timeout.isFinite, timeout > 0, timeout <= 2_400 else { throw LaliaError("Invalid saved session request profile.") }
    }
}

struct SessionAnalysis: Codable, Sendable {
    struct Input: Codable, Equatable, Sendable {
        var filename: String
        var frames: Int64
        var silence: Bool
        var duration: Double { Double(frames) / 16_000 }
    }
    struct Attempt: Codable, Identifiable, Sendable {
        enum State: String, Codable { case preparing, running, complete, failed }
        var id = UUID()
        var profile: SessionTranscriptionProfile
        var state: State = .preparing
        var error: String?
        var rawResponse: Data?
    }
    var input: Input?
    var attempt: Attempt?
    var active: SpeakerTranscript?
    var previous: SpeakerTranscript?
    var previousIsLegacy: Bool?
}

struct SpeakerTranscript: Codable, Identifiable, Sendable {
    struct Turn: Codable, Identifiable, Sendable {
        var id = UUID()
        var start: Double
        var end: Double
        var text: String
        var speakerID: String?
        var isEvent: Bool
    }
    var id: UUID
    var revision = UUID()
    var profile: SessionTranscriptionProfile
    var rawResponse: Data
    var turns: [Turn]
    var speakerNames: [String: String]
    // Empty assignment explicitly means unassigned; absent means the model's label.
    var assignments: [String: String] = [:]

    init(silenceFor attempt: SessionAnalysis.Attempt) {
        id = attempt.id; profile = attempt.profile; rawResponse = Data(); turns = []; speakerNames = [:]
    }

    init(attempt: SessionAnalysis.Attempt, duration: Double) throws {
        try attempt.profile.validate()
        guard duration.isFinite, duration > 0, duration <= SessionTranscriptionProfile.maximumSavedDuration,
              let raw = attempt.rawResponse else { throw LaliaError("Session analysis has no valid saved audio or response.") }
        let response = try JSONDecoder().decode(Transcription.self, from: raw).validated(duration: duration)
        guard let usage = response.usage, usage.type == "tokens", usage.inputTokens > 0,
              usage.outputTokens >= 0, usage.outputTokens < attempt.profile.maximumTokens,
              usage.outputTokens > 0 || trimmedTranscript(response.text) == nil,
              usage.inputTokens <= attempt.profile.contextTokens - attempt.profile.maximumTokens,
              usage.totalTokens == usage.inputTokens + usage.outputTokens else {
            throw LaliaError("MOSS completion could not be verified: missing, inconsistent, exhausted, or out-of-context token usage. Audio and the previous transcript are retained.")
        }
        let segments = response.segments ?? []
        try Self.reconcile(response.text, segments: segments)
        id = attempt.id; profile = attempt.profile; rawResponse = raw
        turns = segments.enumerated().sorted {
            $0.element.start == $1.element.start ? $0.offset < $1.offset : $0.element.start < $1.element.start
        }.compactMap { _, segment in
            let text = Self.content(segment)
            guard !text.isEmpty else { return nil }
            return Turn(start: segment.start, end: segment.end, text: text,
                        speakerID: Self.knownSpeaker(segment.speakerID), isEvent: Self.events.contains(text.lowercased()))
        }
        speakerNames = [:]
        for turn in turns where !turn.isEvent {
            if let speaker = turn.speakerID, speakerNames[speaker] == nil {
                speakerNames[speaker] = "Speaker \(speakerNames.count + 1)"
            }
        }
    }

    private static let events: Set<String> = ["[environmental sounds]", "[human sounds]", "[silence]", "[music]"]

    private static func knownSpeaker(_ id: String?) -> String? {
        guard let id, id.range(of: #"^S\d+$"#, options: .regularExpression) != nil else { return nil }
        return id
    }

    static func content(_ segment: Transcription.Segment) -> String {
        let text = segment.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if let speaker = segment.speakerID, text.hasPrefix("[\(speaker)]") {
            return String(text.dropFirst(speaker.count + 2)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return text
    }

    private static func reconcile(_ text: String, segments: [Transcription.Segment]) throws {
        let raw = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if raw.isEmpty, segments.allSatisfy({ content($0).isEmpty }) { return }
        let expression = try NSRegularExpression(pattern: #"\[(\d+(?:\.\d+)?)\]\[(S\d+)\](.*?)\[(\d+(?:\.\d+)?)\]"#, options: [.dotMatchesLineSeparators])
        let string = raw as NSString
        let matches = expression.matches(in: raw, range: NSRange(location: 0, length: string.length))
        if matches.isEmpty, segments.count == 1, knownSpeaker(segments[0].speakerID) == nil,
           content(segments[0]) == raw,
           raw.range(of: #"\[(?:-?\d|S(?:\d|\]|$))"#, options: .regularExpression) == nil { return }
        guard matches.count == segments.count, !matches.isEmpty else {
            throw LaliaError("MOSS raw output does not match its timed passages. The rejected response is retained.")
        }
        var end = 0
        for (match, segment) in zip(matches, segments) {
            let gap = string.substring(with: NSRange(location: end, length: match.range.location - end))
            guard trimmedTranscript(gap) == nil,
                  Double(string.substring(with: match.range(at: 1))) == segment.start,
                  Double(string.substring(with: match.range(at: 4))) == segment.end,
                  segment.speakerID == string.substring(with: match.range(at: 2)),
                  content(segment) == string.substring(with: match.range(at: 3)).trimmingCharacters(in: .whitespacesAndNewlines) else {
                throw LaliaError("MOSS raw output does not match its timed passages. The rejected response is retained.")
            }
            end = NSMaxRange(match.range)
        }
        guard trimmedTranscript(string.substring(from: end)) == nil else {
            throw LaliaError("MOSS left unmatched text after its timed passages. The rejected response is retained.")
        }
    }

    func speaker(for turn: Turn) -> String? {
        if let value = assignments[turn.id.uuidString] { return value.isEmpty ? nil : value }
        return turn.speakerID
    }
    func label(for turn: Turn) -> String {
        if turn.isEvent { return "Sound" }
        return speaker(for: turn).flatMap { speakerNames[$0] } ?? "Unassigned"
    }
    var transcript: String {
        turns.map { "[\(SessionRecord.timestamp($0.start))–\(SessionRecord.timestamp($0.end))] \(label(for: $0))\n\($0.text)" }.joined(separator: "\n\n")
    }
    var nextSpeakerName: String {
        var number = 1
        let names = Set(speakerNames.values)
        while names.contains("Speaker \(number)") { number += 1 }
        return "Speaker \(number)"
    }
    mutating func rename(_ speaker: String, to name: String) throws {
        guard speakerNames[speaker] != nil, let name = trimmedTranscript(name) else { throw LaliaError("Enter a speaker name.") }
        guard speakerNames[speaker] != name else { return }
        speakerNames[speaker] = name; revision = UUID()
    }
    @discardableResult mutating func addSpeaker(named name: String) throws -> String {
        guard let name = trimmedTranscript(name) else { throw LaliaError("Enter a speaker name.") }
        let id = UUID().uuidString
        speakerNames[id] = name; revision = UUID()
        return id
    }
    mutating func assign(_ turn: UUID, to speaker: String?) throws {
        guard turns.contains(where: { $0.id == turn && !$0.isEvent }), speaker == nil || speakerNames[speaker!] != nil else {
            throw LaliaError("Unknown passage or speaker.")
        }
        guard let passage = turns.first(where: { $0.id == turn }), self.speaker(for: passage) != speaker else { return }
        assignments[turn.uuidString] = speaker ?? ""; revision = UUID()
    }
    mutating func merge(_ source: String, into destination: String) throws {
        guard source != destination, speakerNames[source] != nil, speakerNames[destination] != nil else { throw LaliaError("Choose two different speakers.") }
        for turn in turns where !turn.isEvent && speaker(for: turn) == source { assignments[turn.id.uuidString] = destination }
        speakerNames.removeValue(forKey: source); revision = UUID()
    }
    func validate(duration: Double, silence: Bool = false) throws {
        try profile.validate()
        if silence, rawResponse.isEmpty, turns.isEmpty, speakerNames.isEmpty, assignments.isEmpty { return }
        let original = try SpeakerTranscript(attempt: .init(id: id, profile: profile, state: .complete, rawResponse: rawResponse), duration: duration)
        guard turns.count == original.turns.count, Set(turns.map(\.id)).count == turns.count,
              zip(turns, original.turns).allSatisfy({ $0.start == $1.start && $0.end == $1.end && $0.text == $1.text && $0.speakerID == $1.speakerID && $0.isEvent == $1.isEvent }),
              speakerNames.allSatisfy({ !$0.key.isEmpty && trimmedTranscript($0.value) != nil }),
              turns.allSatisfy({ $0.isEvent || speaker(for: $0) == nil || speakerNames[speaker(for: $0)!] != nil }),
              assignments.allSatisfy({ key, value in turns.contains { $0.id.uuidString == key && !$0.isEvent } && (value.isEmpty || speakerNames[value] != nil) }) else {
            throw LaliaError("Invalid saved speaker transcript or correction references.")
        }
    }
}
