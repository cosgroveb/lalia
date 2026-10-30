import Foundation

enum OptionSide: Hashable { case left, right }

enum GestureInput: Equatable {
    case option(OptionSide, isDown: Bool, at: TimeInterval)
    case otherKey(UInt16, isDown: Bool, repeatEvent: Bool, at: TimeInterval)
    case mouse(button: UInt16?, isDown: Bool, at: TimeInterval)
    case tick(TimeInterval)
    case reset
    case availability(Bool)
    case captureAvailability(Bool)
    case seed(options: Set<OptionSide>, otherKeys: Set<UInt16>, at: TimeInterval)
}

enum GestureAction: Equatable {
    case beginDictation(UUID)
    case finishDictation(UUID)
    case cancelDictation(UUID)
    case toggleRecording(UUID)
}

struct OptionGestureRecognizer {
    static let holdDelay: TimeInterval = 0.250
    static let doubleTapGap: TimeInterval = 0.300
    static let stateCheckInterval: TimeInterval = 0.100

    private enum State {
        case idle
        case pressed(OptionSide, TimeInterval, UUID)
        case awaitSecond(OptionSide, TimeInterval)
        case secondPressed(OptionSide, TimeInterval, UUID)
        case holding(OptionSide, UUID?)
        case blocked
    }

    private var state: State = .idle
    private var options = Set<OptionSide>()
    private var otherKeys = Set<UInt16>()
    private var available = true
    private var captureAvailable = true
    private var lastTime: TimeInterval = -.infinity
    var trackedOptions: Set<OptionSide> { options }
    var trackedOtherKeys: Set<UInt16> { otherKeys }
    var hasPendingGesture: Bool {
        if case .idle = state { return false }
        return true
    }
    var holdDeadline: TimeInterval? {
        switch state {
        case let .pressed(_, started, _), let .secondPressed(_, started, _): started + Self.holdDelay
        default: nil
        }
    }

    mutating func consume(_ input: GestureInput) -> [GestureAction] {
        let timestamp: TimeInterval
        switch input {
        case let .option(_, _, at), let .otherKey(_, _, _, at), let .mouse(_, _, at), let .tick(at), let .seed(_, _, at): timestamp = at
        case .reset, .availability, .captureAvailability: timestamp = lastTime
        }
        if timestamp < lastTime {
            return reset(blocked: true)
        }
        lastTime = timestamp
        switch input {
        case let .availability(value):
            available = value
            return invalidate(blocked: !value || !allUp)
        case let .captureAvailability(value):
            guard captureAvailable != value else { return [] }
            captureAvailable = value
            switch state {
            case .pressed, .awaitSecond, .secondPressed:
                return invalidate(blocked: !allUp)
            default: return []
            }
        case let .seed(seedOptions, seedKeys, _):
            options = seedOptions; otherKeys = seedKeys
            return invalidate(blocked: !allUp)
        case .reset:
            return reset(blocked: true)
        case let .mouse(button, down, _):
            if let button {
                if down { otherKeys.insert(button) } else { otherKeys.remove(button) }
            }
            if down || !allUp { return invalidate(blocked: true) }
            if case .blocked = state { state = .idle }
            return []
        case let .otherKey(code, down, repeatEvent, _):
            if down { otherKeys.insert(code) } else { otherKeys.remove(code) }
            if down { return invalidate(blocked: true) }
            if repeatEvent { return invalidate(blocked: !allUp) }
            if case .blocked = state, allUp { state = .idle }
            else if !options.isEmpty { return invalidate(blocked: !allUp) }
            return []
        case let .option(side, down, at):
            if down {
                if options.contains(side) { return [] }
                options.insert(side)
                guard options.count == 1, otherKeys.isEmpty, available else { return invalidate(blocked: true) }
                switch state {
                case let .awaitSecond(previous, released) where previous == side && at - released <= Self.doubleTapGap + 1e-9:
                    let id = UUID(); state = .secondPressed(side, at, id); return []
                default:
                    let id = UUID(); state = .pressed(side, at, id); return []
                }
            }
            guard options.contains(side) else { return reset(blocked: true) }
            options.remove(side)
            switch state {
            case let .pressed(pressedSide, started, _) where pressedSide == side:
                if at - started >= Self.holdDelay { state = .idle; return [] }
                state = .awaitSecond(side, at); return []
            case let .secondPressed(pressedSide, started, id) where pressedSide == side:
                if at - started < Self.holdDelay { state = .idle; return [.toggleRecording(id)] }
                state = .idle; return []
            case let .holding(heldSide, id) where heldSide == side:
                state = .idle
                return id.map { [.finishDictation($0)] } ?? []
            case .blocked:
                if allUp { state = .idle }
                return []
            default:
                return reset(blocked: !allUp)
            }
        case let .tick(at):
            switch state {
            case let .pressed(side, started, id) where at - started >= Self.holdDelay:
                guard options == [side], otherKeys.isEmpty, available else { return invalidate(blocked: true) }
                guard captureAvailable else { state = .holding(side, nil); return [] }
                state = .holding(side, id); return [.beginDictation(id)]
            case let .secondPressed(side, started, _) where at - started >= Self.holdDelay:
                guard options == [side], otherKeys.isEmpty, available else { return invalidate(blocked: true) }
                guard captureAvailable else { state = .holding(side, nil); return [] }
                let id = UUID(); state = .holding(side, id); return [.beginDictation(id)]
            default: return []
            }
        }
    }

    private var allUp: Bool { options.isEmpty && otherKeys.isEmpty }
    private mutating func invalidate(blocked: Bool) -> [GestureAction] {
        reset(blocked: blocked)
    }
    private mutating func reset(blocked: Bool) -> [GestureAction] {
        var actions: [GestureAction] = []
        if case let .holding(_, id?) = state { actions.append(.cancelDictation(id)) }
        state = blocked || !allUp ? .blocked : .idle
        return actions
    }
}
