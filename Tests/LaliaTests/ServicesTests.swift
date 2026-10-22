import Testing
@testable import Lalia

struct ServicesTests {
    @Test func trimsTranscript() {
        #expect(trimmedTranscript(" \n hello \t") == "hello")
        #expect(trimmedTranscript(" \n\t ") == nil)
    }
    @Test func rejectsInventedOrOverlappingTimeRanges() throws {
        for range in [(start: -1.0, end: 1.0), (start: 2.0, end: 1.0), (start: 0.0, end: 12.0)] {
            let response = Transcription(text: "text", segments: [.init(start: range.start, end: range.end, text: "text")])
            #expect(throws: Error.self) { try response.validated(duration: 10) }
        }
        let valid = Transcription(text: "text", segments: [.init(start: 0, end: 10, text: "text")])
        #expect(try valid.validated(duration: 10) == valid)
    }
}
