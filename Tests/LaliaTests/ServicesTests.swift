import Testing
@testable import Lalia

struct ServicesTests {
    @Test func trimsTranscript() {
        #expect(trimmedTranscript(" \n hello \t") == "hello")
        #expect(trimmedTranscript(" \n\t ") == nil)
    }

    @Test func joinsFinalPhrasesInOrder() {
        #expect(joinedFinalPhrases(["first ", "second"]) == "first second")
    }
}
