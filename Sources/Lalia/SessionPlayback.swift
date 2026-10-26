import AVFoundation
import Combine

@MainActor final class SessionPlayback: NSObject, ObservableObject, AVAudioPlayerDelegate, PlaybackStopping {
    @Published private(set) var playing = false
    @Published private(set) var error: String?
    private var player: AVAudioPlayer?

    func play(session: UUID, filename: String, store: SessionStore, from seconds: Double) {
        stop()
        do {
            let file = try store.file(session, filename)
            let player = try AVAudioPlayer(contentsOf: file)
            guard seconds.isFinite, seconds >= 0, seconds < player.duration else { throw LaliaError("Passage is outside the saved audio.") }
            player.delegate = self
            player.currentTime = seconds
            guard player.play() else { throw LaliaError("Could not play saved audio.") }
            self.player = player; playing = true; error = nil
        } catch { self.error = error.localizedDescription }
    }
    func stop() { player?.stop(); player = nil; playing = false; error = nil }
    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        let finished = ObjectIdentifier(player)
        Task { @MainActor [weak self] in
            guard let self, self.player.map(ObjectIdentifier.init) == finished else { return }
            self.player = nil; self.playing = false
        }
    }
}
