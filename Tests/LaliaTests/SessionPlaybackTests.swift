import Foundation
import Testing
@testable import Lalia

@MainActor struct SessionPlaybackTests {
    @Test func stopClearsPreviousPlaybackFailure() throws {
        let fixture = TemporaryStore(); let store = fixture.store
        let record = try store.create(kind: .session)
        let playback = SessionPlayback()
        playback.play(session: record.id, filename: "missing.wav", store: store, from: 0)
        #expect(playback.error != nil)
        playback.stop()
        #expect(playback.error == nil)
        #expect(!playback.playing)
    }
    @Test func reportsAudioSymlinkCreatedAfterHistoryLoads() throws {
        let fixture = TemporaryStore(); let store = fixture.store
        let record = try savedAudio(store)
        _ = try SessionAudioPreparer().prepare(session: record.id, store: store)
        let session = try #require(store.history().records.first)
        let input = try #require(session.analysis?.input)
        let file = try store.file(session.id, input.filename)
        try FileManager.default.removeItem(at: file)
        try FileManager.default.createSymbolicLink(at: file, withDestinationURL: store.file(session.id, session.segments[0].filename))
        let playback = SessionPlayback()
        playback.play(session: session.id, filename: input.filename, store: store, from: 0)
        #expect(playback.error == "Session paths cannot be symbolic links.")
        #expect(!playback.playing)
    }
}
