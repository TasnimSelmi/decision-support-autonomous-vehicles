from enum import IntEnum
import carla


class DrivingAction(IntEnum):
    CONTINUE = 0
    SLOW_DOWN = 1
    BRAKE = 2
    TURN_LEFT = 3
    TURN_RIGHT = 4
    CORRECT_LANE = 5


ACTION_NAMES = {
    0: "continue",
    1: "slow_down",
    2: "brake",
    3: "turn_left",
    4: "turn_right",
    5: "correct_lane",
}


def action_to_name(action: int | DrivingAction) -> str:
    return ACTION_NAMES.get(int(action), str(action))


def action_to_carla_control(action_id: int) -> carla.VehicleControl:
    action = DrivingAction(action_id)

    if action == DrivingAction.CONTINUE:
        return carla.VehicleControl(throttle=0.45, brake=0.0, steer=0.0)

    if action == DrivingAction.SLOW_DOWN:
        return carla.VehicleControl(throttle=0.20, brake=0.25, steer=0.0)

    if action == DrivingAction.BRAKE:
        return carla.VehicleControl(throttle=0.0, brake=0.85, steer=0.0)

    if action == DrivingAction.TURN_LEFT:
        return carla.VehicleControl(throttle=0.25, brake=0.0, steer=-0.35)

    if action == DrivingAction.TURN_RIGHT:
        return carla.VehicleControl(throttle=0.25, brake=0.0, steer=0.35)

    if action == DrivingAction.CORRECT_LANE:
        return carla.VehicleControl(throttle=0.25, brake=0.0, steer=0.15)

    return carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)