import numpy as np

class RewardComponent:
    def __init__(self, name, weight=1.0):
        self.name = name
        self.weight = weight

    def compute(self, env) -> float:
        raise NotImplementedError

class TargetProgressReward(RewardComponent):
    def compute(self, env) -> float:
        drone_position = env.drone.state.position
        distance_to_target = np.linalg.norm(drone_position - env.target_position)

        if not hasattr(env, "prev_distance_to_target") or env.prev_distance_to_target is None:
            env.prev_distance_to_target = distance_to_target

        improvement = env.prev_distance_to_target - distance_to_target
        env.prev_distance_to_target = distance_to_target

        return improvement / 100  # 缩放一下数值

class ObstaclePenalty(RewardComponent):
    def compute(self, env) -> float:
        min_distance_to_obstacle = env.drone.get_closest_obstacle_distance()

        if min_distance_to_obstacle < env.safe_distance:
            return -(env.safe_distance - min_distance_to_obstacle) / env.safe_distance
        else:
            return 0.0

class HeadingAlignmentReward(RewardComponent):
    def compute(self, env) -> float:
        drone_position = env.drone.state.position
        drone_velocity = env.drone.state.linear_velocity
        target_position = env.target_position

        if np.linalg.norm(drone_velocity) > 1e-6:
            velocity_dir = drone_velocity / np.linalg.norm(drone_velocity)
            direction_to_target = (target_position - drone_position) / np.linalg.norm(target_position - drone_position)
            cos_theta = np.clip(np.dot(velocity_dir, direction_to_target), -1.0, 1.0)
            heading_reward = cos_theta  # cos越接近1，飞得越对
        else:
            heading_reward = 0.0  # 静止不给奖励

        # 可以调权重，在组件管理里也可以再调
        return heading_reward
