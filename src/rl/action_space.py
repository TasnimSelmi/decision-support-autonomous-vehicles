from enum import IntEnum


class DrivingAction(IntEnum):
    CONTINUE = 0
    SLOW_DOWN = 1
    STOP = 2
    TURN_LEFT = 3
    TURN_RIGHT = 4
    CORRECT_LANE = 5


ACTION_NAMES = {
    DrivingAction.CONTINUE: "continue",
    DrivingAction.SLOW_DOWN: "slow_down",
    DrivingAction.STOP: "stop",
    DrivingAction.TURN_LEFT: "turn_left",
    DrivingAction.TURN_RIGHT: "turn_right",
    DrivingAction.CORRECT_LANE: "correct_lane",
}


def action_to_name(action: int | DrivingAction) -> str:
    return ACTION_NAMES[DrivingAction(action)]


def name_to_action(name: str) -> DrivingAction:
    normalized = name.strip().lower()

    for action, action_name in ACTION_NAMES.items():
        if action_name == normalized:
            return action

    raise ValueError(f"Unknown action name: {name}")