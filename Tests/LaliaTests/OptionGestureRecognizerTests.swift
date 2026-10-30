import Testing
@testable import Lalia

struct OptionGestureRecognizerTests {
    @Test func holdBeginsAndReleaseFinishesWithSameIdentity() {
        var recognizer = OptionGestureRecognizer()
        #expect(recognizer.consume(.option(.left, isDown: true, at: 0)).isEmpty)
        let begin = recognizer.consume(.tick(0.250))
        guard case let .beginDictation(id) = begin.first else { Issue.record("Expected begin action"); return }
        #expect(recognizer.consume(.option(.left, isDown: false, at: 0.4)) == [.finishDictation(id)])
    }

    @Test func sameSideDoubleTapTogglesOnlyOnSecondRelease() {
        var recognizer = OptionGestureRecognizer()
        _ = recognizer.consume(.option(.right, isDown: true, at: 0))
        _ = recognizer.consume(.option(.right, isDown: false, at: 0.1))
        #expect(recognizer.consume(.option(.right, isDown: true, at: 0.4)).isEmpty)
        #expect(recognizer.consume(.option(.right, isDown: false, at: 0.5)).count == 1)
    }

    @Test func oppositeSidesNeverCompleteDoubleTapAndChordCancelsHold() {
        var recognizer = OptionGestureRecognizer()
        _ = recognizer.consume(.option(.left, isDown: true, at: 0))
        _ = recognizer.consume(.option(.left, isDown: false, at: 0.1))
        _ = recognizer.consume(.option(.right, isDown: true, at: 0.2))
        #expect(recognizer.consume(.option(.right, isDown: false, at: 0.3)).isEmpty)
        _ = recognizer.consume(.option(.left, isDown: true, at: 1))
        let action = recognizer.consume(.tick(1.25))
        guard case let .beginDictation(id) = action.first else { Issue.record("Expected begin action"); return }
        #expect(recognizer.consume(.otherKey(8, isDown: true, repeatEvent: false, at: 1.3)) == [.cancelDictation(id)])
        #expect(recognizer.consume(.option(.left, isDown: false, at: 1.4)).isEmpty)
    }

    @Test func exactBoundariesAreInclusiveAndBlockedRequiresAllUp() {
        var recognizer = OptionGestureRecognizer()
        _ = recognizer.consume(.option(.left, isDown: true, at: 0))
        _ = recognizer.consume(.option(.left, isDown: false, at: 0.1))
        _ = recognizer.consume(.option(.left, isDown: true, at: 0.4))
        #expect(recognizer.consume(.option(.left, isDown: false, at: 0.65)).isEmpty)
        _ = recognizer.consume(.option(.left, isDown: true, at: 1))
        _ = recognizer.consume(.otherKey(4, isDown: true, repeatEvent: false, at: 1.1))
        #expect(recognizer.consume(.option(.left, isDown: false, at: 1.2)).isEmpty)
        _ = recognizer.consume(.otherKey(4, isDown: false, repeatEvent: false, at: 1.3))
        #expect(recognizer.consume(.option(.left, isDown: true, at: 1.4)).isEmpty)
    }

    @Test func exactHoldThresholdWinsOverTap() {
        var recognizer = OptionGestureRecognizer()
        _ = recognizer.consume(.option(.left, isDown: true, at: 0))
        guard case let .beginDictation(id)? = recognizer.consume(.tick(0.250)).first else { Issue.record("Expected hold action"); return }
        #expect(recognizer.consume(.option(.left, isDown: false, at: 0.250)) == [.finishDictation(id)])
    }

    @Test func slowTripleAndRepeatedTapsDoNotToggleAgain() {
        var recognizer = OptionGestureRecognizer()
        _ = recognizer.consume(.option(.left, isDown: true, at: 0))
        _ = recognizer.consume(.option(.left, isDown: false, at: 0.1))
        _ = recognizer.consume(.option(.left, isDown: true, at: 0.5))
        #expect(recognizer.consume(.option(.left, isDown: false, at: 0.6)).isEmpty)
        _ = recognizer.consume(.option(.left, isDown: true, at: 0.8))
        _ = recognizer.consume(.option(.left, isDown: true, at: 0.85))
        #expect(recognizer.consume(.option(.left, isDown: false, at: 0.9)).count == 1)
        _ = recognizer.consume(.option(.left, isDown: true, at: 1.1))
        #expect(recognizer.consume(.option(.left, isDown: false, at: 1.2)).isEmpty)
    }

    @Test func preheldKeysAndAvailabilityLossRequireAllUp() {
        var recognizer = OptionGestureRecognizer()
        _ = recognizer.consume(.seed(options: [], otherKeys: [12], at: 0))
        #expect(recognizer.consume(.option(.right, isDown: true, at: 0.1)).isEmpty)
        _ = recognizer.consume(.option(.right, isDown: false, at: 0.2))
        _ = recognizer.consume(.otherKey(12, isDown: false, repeatEvent: false, at: 0.3))
        #expect(recognizer.consume(.availability(false)).isEmpty)
        #expect(recognizer.consume(.option(.right, isDown: true, at: 0.4)).isEmpty)
        _ = recognizer.consume(.option(.right, isDown: false, at: 0.5))
        _ = recognizer.consume(.availability(true))
        #expect(recognizer.consume(.option(.right, isDown: true, at: 0.6)).isEmpty)
    }

    @Test func resetCancelsHeldCaptureAndOutOfOrderInputBlocks() {
        var recognizer = OptionGestureRecognizer()
        _ = recognizer.consume(.option(.left, isDown: true, at: 1))
        guard case let .beginDictation(id)? = recognizer.consume(.tick(1.25)).first else { Issue.record("Expected begin action"); return }
        #expect(recognizer.consume(.reset) == [.cancelDictation(id)])
        #expect(recognizer.consume(.option(.left, isDown: false, at: 0.9)).isEmpty)
    }

    @Test func captureReadinessChangeInvalidatesCandidateButLeavesSessionToggleAvailable() {
        var recognizer = OptionGestureRecognizer()
        _ = recognizer.consume(.captureAvailability(true))
        _ = recognizer.consume(.option(.left, isDown: true, at: 0))
        _ = recognizer.consume(.option(.left, isDown: false, at: 0.1))
        _ = recognizer.consume(.captureAvailability(false))
        _ = recognizer.consume(.option(.left, isDown: true, at: 0.2))
        #expect(recognizer.consume(.option(.left, isDown: false, at: 0.3)).isEmpty)
        _ = recognizer.consume(.option(.left, isDown: true, at: 0.4))
        #expect(recognizer.consume(.option(.left, isDown: false, at: 0.5)).count == 1)
        _ = recognizer.consume(.option(.left, isDown: true, at: 0.6))
        #expect(recognizer.consume(.tick(0.85)).isEmpty)
        #expect(recognizer.consume(.option(.left, isDown: false, at: 0.9)).isEmpty)
    }
}
