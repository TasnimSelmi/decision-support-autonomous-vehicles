from src.rl.action_space import DrivingAction
from src.rl.schemas import DrivingState


def compute_reward(state: DrivingState, action: DrivingAction) -> float:
    reward = 0.0

    # 1. Collision risk: stopping is strongly preferred.
    if state.collision_risk == 1:
        if action == DrivingAction.STOP:
            reward += 12.0
        else:
            reward -= 18.0

    # 2. Pedestrian: stopping is mandatory.
    if state.pedestrian_presence == 1:
        if action == DrivingAction.STOP:
            reward += 15.0
        else:
            reward -= 22.0

    # 3. High hazard: slow down or stop.
    if state.hazard_level == 2:
        if action == DrivingAction.STOP:
            reward += 9.0
        elif action == DrivingAction.SLOW_DOWN:
            reward += 8.0
        elif action == DrivingAction.CONTINUE:
            reward -= 12.0

    # 4. Medium hazard: slow down is better than stopping.
    if state.hazard_level == 1:
        if action == DrivingAction.SLOW_DOWN:
            reward += 6.0
        elif action == DrivingAction.STOP:
            reward += 2.0
        elif action == DrivingAction.CONTINUE:
            reward -= 5.0

    # 5. Obstacle: slow/stop/avoid are allowed.
    if state.obstacle_presence == 1:
        if action == DrivingAction.SLOW_DOWN:
            reward += 6.0
        elif action == DrivingAction.STOP:
            reward += 5.0
        elif action in (DrivingAction.TURN_LEFT, DrivingAction.TURN_RIGHT):
            reward += 3.0
        elif action == DrivingAction.CONTINUE:
            reward -= 8.0

    # 6. Lane issue: correct lane is preferred.
    if state.lane_unsafe == 1:
        if action == DrivingAction.CORRECT_LANE:
            reward += 10.0
        elif action == DrivingAction.CONTINUE:
            reward -= 8.0
        elif action in (DrivingAction.TURN_LEFT, DrivingAction.TURN_RIGHT):
            reward += 2.0
        else:
            reward -= 3.0

    # 7. Speed: high speed + any risk should reduce aggression.
    if state.speed_level == 2 and has_any_risk(state):
        if action == DrivingAction.STOP:
            reward += 5.0
        elif action == DrivingAction.SLOW_DOWN:
            reward += 7.0
        elif action == DrivingAction.CONTINUE:
            reward -= 8.0

    # 8. Safe state: continue, don't over-stop.
    if is_safe_state(state):
        if action == DrivingAction.CONTINUE:
            reward += 10.0
        elif action == DrivingAction.STOP:
            reward -= 8.0
        elif action == DrivingAction.SLOW_DOWN:
            reward -= 3.0
        elif action in (
            DrivingAction.TURN_LEFT,
            DrivingAction.TURN_RIGHT,
            DrivingAction.CORRECT_LANE,
        ):
            reward -= 5.0

    # 9. Penalize unnecessary turning if lane is already safe and no obstacle.
    if (
        state.lane_unsafe == 0
        and state.obstacle_presence == 0
        and action in (DrivingAction.TURN_LEFT, DrivingAction.TURN_RIGHT)
    ):
        reward -= 4.0

    # 10. Small step penalty.
    reward -= 0.1

    return reward


def is_safe_state(state: DrivingState) -> bool:
    return (
        state.hazard_level == 0
        and state.obstacle_presence == 0
        and state.collision_risk == 0
        and state.lane_unsafe == 0
        and state.pedestrian_presence == 0
    )


def has_any_risk(state: DrivingState) -> bool:
    return (
        state.hazard_level > 0
        or state.obstacle_presence == 1
        or state.collision_risk == 1
        or state.lane_unsafe == 1
        or state.pedestrian_presence == 1
    )