import SwiftUI

@main struct LaliaApp: App {
    @StateObject private var coordinator: DictationCoordinator
    private let hotkey: CarbonHotkey
    init() {
        let coordinator = DictationCoordinator(recorder: Recorder(), transcriber: NativeTranscriber(), injector: PasteInjector(), authorizer: SystemAuthorizer())
        _coordinator = StateObject(wrappedValue: coordinator)
        hotkey = CarbonHotkey(pressed: { coordinator.hotkeyPressed() }, released: { coordinator.hotkeyReleased() })
        coordinator.startupRefresh()
        do { try hotkey.register() } catch { coordinator.hotkeyRegistrationFailed(error) }
    }
    var body: some Scene {
        MenuBarExtra("Lalia", systemImage: "waveform") {
            Text(coordinator.message)
            if let hotkeyError = coordinator.hotkeyError { Text(hotkeyError) }
            if !coordinator.isDictationEnabled {
                Button("Enable Dictation") { Task { await coordinator.enableDictation() } }
                    .disabled(coordinator.phase == .preparing || coordinator.phase == .recording || coordinator.phase == .transcribing || coordinator.phase == .pasting)
            }
            Button("Copy Last Transcript") { coordinator.copyLastTranscript() }.disabled(coordinator.lastTranscript == nil)
            Divider()
            Button("Quit") { NSApplication.shared.terminate(nil) }
        }
    }
}
