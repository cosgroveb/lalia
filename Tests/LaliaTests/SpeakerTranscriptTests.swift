import Foundation
import Testing
@testable import Lalia

func speakerAttempt(_ response: Transcription? = nil) throws -> SessionAnalysis.Attempt {
    let response = response ?? Transcription(text: "[0][S01]Hello.[1][0.5][S02]Yes yes.[2]",
        segments: [.init(start: 0, end: 1, text: "[S01] Hello.", speakerID: "S01"),
                   .init(start: 0.5, end: 2, text: "[S02] Yes yes.", speakerID: "S02")],
        usage: .init(type: "tokens", inputTokens: 100, outputTokens: 20, totalTokens: 120))
    return SessionAnalysis.Attempt(profile: SessionTranscriptionProfile(), rawResponse: try JSONEncoder().encode(response))
}

struct SpeakerTranscriptTests {
    @Test func mergeLeavesSoundEventsUnassigned() throws {
        let response = Transcription(text: "[0][S01]Hello.[0.5][0.5][S01][Human Sounds][1][1][S02]Yes.[2]",
            segments: [.init(start: 0, end: 0.5, text: "[S01] Hello.", speakerID: "S01"),
                       .init(start: 0.5, end: 1, text: "[S01] [Human Sounds]", speakerID: "S01"),
                       .init(start: 1, end: 2, text: "[S02] Yes.", speakerID: "S02")],
            usage: .init(type: "tokens", inputTokens: 100, outputTokens: 20, totalTokens: 120))
        var transcript = try SpeakerTranscript(attempt: speakerAttempt(response), duration: 2)
        let event = transcript.turns[1]
        try transcript.merge("S01", into: "S02")
        #expect(transcript.assignments[event.id.uuidString] == nil)
        #expect(transcript.label(for: event) == "Sound")
        #expect(transcript.speaker(for: transcript.turns[0]) == "S02")
        try transcript.validate(duration: 2)
    }
    @Test func endRoundingKeepsOriginalTimestampsAndText() throws {
        let response = Transcription(text: "[0.99][S01]late[1.05]",
            segments: [.init(start: 0.99, end: 1.05, text: "[S01] late", speakerID: "S01")],
            usage: .init(type: "tokens", inputTokens: 100, outputTokens: 4, totalTokens: 104))
        let attempt = try speakerAttempt(response)
        let transcript = try SpeakerTranscript(attempt: attempt, duration: 1)
        #expect(try response.validated(duration: 1) == response)
        #expect(transcript.rawResponse == attempt.rawResponse)
        #expect(transcript.turns[0].start == 0.99)
        #expect(transcript.turns[0].end == 1.05)
        #expect(transcript.turns[0].text == "late")
        try transcript.validate(duration: 1)
        for range in [(1.01, 1.05), (0.99, 1.11)] {
            var invalid = response
            invalid.segments?[0].start = range.0
            invalid.segments?[0].end = range.1
            #expect(throws: Error.self) { try invalid.validated(duration: 1) }
            #expect(throws: Error.self) { try SpeakerTranscript(attempt: speakerAttempt(invalid), duration: 1) }
        }
    }
    @Test func earlierLongAnalysesRemainReadable() throws {
        let transcript = try SpeakerTranscript(attempt: speakerAttempt(), duration: 5_400)
        try transcript.validate(duration: 5_400)
        #expect(SessionTranscriptionProfile.maximumDuration == 3_600)
    }
    @Test func newSpeakerDefaultNameIsUnusedAfterMerge() throws {
        var transcript = try SpeakerTranscript(attempt: speakerAttempt(), duration: 2)
        let third = try transcript.addSpeaker(named: transcript.nextSpeakerName)
        #expect(transcript.speakerNames[third] == "Speaker 3")
        try transcript.merge("S02", into: "S01")
        #expect(transcript.nextSpeakerName == "Speaker 2")
        let new = try transcript.addSpeaker(named: transcript.nextSpeakerName)
        #expect(transcript.speakerNames[new] == "Speaker 2")
        #expect(Set(transcript.speakerNames.values).count == transcript.speakerNames.count)
    }
    @Test func keepsOverlapStartZeroUnicodeAndMatchingMarkerOnly() throws {
        let response = Transcription(text: "[0][S01]नमस्ते 🫠 [S02] is quoted.[1][0.5][S02]Yes yes.[2]",
            segments: [.init(start: 0, end: 1, text: "[S01] नमस्ते 🫠 [S02] is quoted.", speakerID: "S01"),
                       .init(start: 0.5, end: 2, text: "[S02] Yes yes.", speakerID: "S02")],
            usage: .init(type: "tokens", inputTokens: 100, outputTokens: 20, totalTokens: 120))
        let transcript = try SpeakerTranscript(attempt: speakerAttempt(response), duration: 2)
        #expect(transcript.turns.map(\.start) == [0, 0.5])
        #expect(transcript.turns[0].text == "नमस्ते 🫠 [S02] is quoted.")
        #expect(transcript.transcript.contains("Speaker 2\nYes yes."))
        #expect(SpeakerTranscript.content(.init(start: 0, end: 1, text: "[S02] spoken", speakerID: "S01")) == "[S02] spoken")
    }
    @Test func plainTranscriptJoinsSpeechInTimeOrderAndOmitsEvents() throws {
        let response = Transcription(text: "[1][S02]second[2][0][S01]first[1][2][S02][Music][3]",
            segments: [.init(start: 1, end: 2, text: "[S02] second", speakerID: "S02"),
                       .init(start: 0, end: 1, text: "[S01] first", speakerID: "S01"),
                       .init(start: 2, end: 3, text: "[S02] [Music]", speakerID: "S02")],
            usage: .init(type: "tokens", inputTokens: 100, outputTokens: 20, totalTokens: 120))
        let transcript = try SpeakerTranscript(attempt: speakerAttempt(response), duration: 3)
        #expect(transcript.plainTranscript == "first second")
    }
    @Test func rejectsMissingExhaustedInconsistentAndImpossibleUsage() throws {
        let good = try JSONDecoder().decode(Transcription.self, from: #require(speakerAttempt().rawResponse))
        let usages: [Transcription.Usage?] = [nil,
            .init(type: "tokens", inputTokens: 100, outputTokens: 49_152, totalTokens: 49_252),
            .init(type: "tokens", inputTokens: 100, outputTokens: 0, totalTokens: 100),
            .init(type: "tokens", inputTokens: 0, outputTokens: 20, totalTokens: 20),
            .init(type: "tokens", inputTokens: 100, outputTokens: -1, totalTokens: 99),
            .init(type: "tokens", inputTokens: 100, outputTokens: 20, totalTokens: 121),
            .init(type: "tokens", inputTokens: 90_000, outputTokens: 20, totalTokens: 90_020),
            .init(type: "seconds", inputTokens: 100, outputTokens: 20, totalTokens: 120)]
        for usage in usages {
            var response = good; response.usage = usage
            #expect(throws: Error.self) { try SpeakerTranscript(attempt: speakerAttempt(response), duration: 2) }
        }
    }
    @Test func missingUsageSuggestsRuntimeButOrdinaryMalformedOutputDoesNot() throws {
        let good = try #require(speakerAttempt().rawResponse)
        var json = try #require(JSONSerialization.jsonObject(with: good) as? [String: Any])
        json.removeValue(forKey: "usage")
        let missingUsage = SessionAnalysis.Attempt(profile: SessionTranscriptionProfile(), rawResponse: try JSONSerialization.data(withJSONObject: json))
        do {
            _ = try SpeakerTranscript(attempt: missingUsage, duration: 2)
            Issue.record("MOSS response without usage unexpectedly succeeded.")
        } catch {
            #expect(error.localizedDescription.contains("token usage is missing"))
            #expect(error.localizedDescription.contains("lalia-runtime setup"))
        }

        var response = try JSONDecoder().decode(Transcription.self, from: good)
        response.text = "unmatched model output"
        do {
            _ = try SpeakerTranscript(attempt: speakerAttempt(response), duration: 2)
            Issue.record("Malformed MOSS output unexpectedly succeeded.")
        } catch {
            #expect(error.localizedDescription.contains("raw output"))
            #expect(!error.localizedDescription.contains("lalia-runtime setup"))
        }
    }
    @Test func rejectsUnmatchedRawTextAndMalformedTimingsWithoutDroppingSpeech() throws {
        let good = try JSONDecoder().decode(Transcription.self, from: #require(speakerAttempt().rawResponse))
        for raw in ["discarded " + good.text, good.text + " discarded", good.text + "[2][S01]unfinished", "[0][S01]different[1]"] {
            var response = good; response.text = raw
            #expect(throws: Error.self) { try SpeakerTranscript(attempt: speakerAttempt(response), duration: 2) }
        }
        for range in [(0.0, 0.0), (-1, 1), (0, 3)] {
            var response = good; response.segments?[0].start = range.0; response.segments?[0].end = range.1
            #expect(throws: Error.self) { try SpeakerTranscript(attempt: speakerAttempt(response), duration: 2) }
        }
    }
    @Test func retainsUnassignedSpeechAndExactEventsAndAcceptsEmptyEOS() throws {
        let usage = Transcription.Usage(type: "tokens", inputTokens: 100, outputTokens: 4, totalTokens: 104)
        let plain = Transcription(text: "I am unassigned.", segments: [.init(start: 0, end: 1, text: "I am unassigned.")], usage: usage)
        let unknown = try SpeakerTranscript(attempt: speakerAttempt(plain), duration: 1)
        #expect(unknown.label(for: unknown.turns[0]) == "Unassigned")
        var unknownID = plain; unknownID.segments?[0].speakerID = "unknown"
        #expect(try SpeakerTranscript(attempt: speakerAttempt(unknownID), duration: 1).speakerNames.isEmpty)
        for raw in ["[0][S01]unfinished", "[0]lost timing[1]", "[0", "[S"] {
            let malformed = Transcription(text: raw, segments: [.init(start: 0, end: 1, text: raw)], usage: usage)
            #expect(throws: Error.self) { try SpeakerTranscript(attempt: speakerAttempt(malformed), duration: 1) }
        }
        let event = Transcription(text: "[0][S01][Human Sounds][1]", segments: [.init(start: 0, end: 1, text: "[S01] [Human Sounds]", speakerID: "S01")], usage: usage)
        let sounds = try SpeakerTranscript(attempt: speakerAttempt(event), duration: 1)
        #expect(sounds.turns[0].isEvent)
        #expect(sounds.speakerNames.isEmpty)
        let unassignedEvent = Transcription(text: "[Silence]", segments: [.init(start: 0, end: 1, text: "[Silence]")], usage: usage)
        #expect(try SpeakerTranscript(attempt: speakerAttempt(unassignedEvent), duration: 1).turns[0].isEvent)
        let empty = Transcription(text: "", segments: [.init(start: 0, end: 1, text: "")], usage: .init(type: "tokens", inputTokens: 100, outputTokens: 0, totalTokens: 100))
        #expect(try SpeakerTranscript(attempt: speakerAttempt(empty), duration: 1).turns.isEmpty)
    }
    @Test func correctionsKeepRawAndStableTurnsAndRenderEveryConsumer() throws {
        var transcript = try SpeakerTranscript(attempt: speakerAttempt(), duration: 2)
        let raw = transcript.rawResponse; let ids = transcript.turns.map(\.id); let revision = transcript.revision
        try transcript.rename("S01", to: "Speaker 1")
        try transcript.assign(ids[0], to: "S01")
        #expect(transcript.revision == revision)
        try transcript.rename("S01", to: "Brian")
        let other = try transcript.addSpeaker(named: "Guest")
        try transcript.assign(ids[1], to: other)
        #expect(transcript.transcript.contains("Guest\nYes yes."))
        try transcript.merge(other, into: "S01")
        #expect(transcript.transcript.components(separatedBy: "Brian").count == 3)
        try transcript.assign(ids[0], to: nil)
        #expect(transcript.label(for: transcript.turns[0]) == "Unassigned")
        #expect(transcript.revision != revision)
        #expect(transcript.rawResponse == raw)
        #expect(transcript.turns.map(\.id) == ids)
        try transcript.validate(duration: 2)
        #expect(throws: Error.self) { try transcript.assign(UUID(), to: "S01") }
        #expect(throws: Error.self) { try transcript.assign(ids[0], to: "missing") }
        #expect(throws: Error.self) { try transcript.rename("S01", to: " ") }
    }
}
