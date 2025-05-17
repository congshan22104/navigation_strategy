"""
env/low_level_env.py
"""

import gym
from gym import spaces
import numpy as np
import random
import logging
from sim.world import World
from env.wrappers.reward_wrapper import (
    TargetProgressReward,
    NonlinearCollisionPenalty,
    DirectionReward,
    VelocityReward,
    LinearCollisionPenalty,
    TerminalReward,
    SphericalDirectionReward,
    DistanceToObstacleReward,
    NonlinearSphericalDirectionReward,
    ImageNonlinearCollisionPenalty,
    ImageNonlinearCollisionPenalty2,
    ImageLinearCollisionPenalty,
    CosineSphericalDirectionReward,
    CosineSphericalDirectionReward2,
    InterpolationSphericalDirectionReward,
    TanhSphericalDirectionReward
)

import pybullet as p


class NavigationEnv(gym.Env):
    def __init__(self, env_params):
        super().__init__()
        self.env_params = env_params
        self._init_simulation()
        self._init_obs_space()
        self._init_action_space()
        self._init_reward()

    def _init_simulation(self):
        scene_region = self.env_params['scene']['region']
        obstacle_params = self.env_params['scene']['obstacle']
        drone_params = self.env_params['drone']
        scene_type = self.env_params['scene'].get('type', 'random')
        voxel_size = self.env_params['scene'].get('voxel', {}).get('size', None)
        building_path = self.env_params.get('world', {}).get('building_path', '')

        self.sim = World(
            use_gui=self.env_params['use_gui'],
            scene_type=scene_type,
            scene_region=scene_region,
            obstacle_params=obstacle_params,
            drone_params=drone_params,
            voxel_size=voxel_size,
            building_path=building_path
        )

    def _init_action_space(self, mode: str = "adjust"):
        """
        根据动作控制模式初始化动作空间。

        参数:
            mode (str): 'cartesian', 'spherical', 或 'adjust'
        """
        if mode == "cartesian":
            self.action_space = spaces.Box(
                low=np.array([-15.0, -15.0, -15.0], dtype=np.float32),
                high=np.array([15.0, 15.0, 15.0], dtype=np.float32),
                dtype=np.float32
            )

        elif mode == "spherical":
            self.action_space = spaces.Box(
                low=np.array([0.0, 0.0, -np.pi], dtype=np.float32),       # v ∈ [0, 15], θ ∈ [0, π], φ ∈ [-π, π]
                high=np.array([15.0, np.pi, np.pi], dtype=np.float32),
                dtype=np.float32
            )

        elif mode == "adjust":
            self.action_space = spaces.Box(
                low=np.array([-np.pi/12, -np.pi/12], dtype=np.float32),  # v_abs, Δθ, Δφ
                high=np.array([np.pi/12, np.pi/12], dtype=np.float32),
                dtype=np.float32
            )
        else:
            raise ValueError(f"Unsupported control mode: '{mode}'")

    def _init_obs_space(self):
        features = self.env_params['observation']['features']
        state_dims = 0

        # 统计总共的 state 向量维度
        if "position" in features:
            state_dims += 3
        if "velocity" in features:
            state_dims += 3
        if "spherical_velocity" in features:
            state_dims += 3
        if "orientation" in features:
            state_dims += 3
        if "target" in features:
            state_dims += 3
        if "target_relative_position" in features:
            state_dims += 3  # r, theta, phi
        if "spherical_direction_error" in features:
            state_dims += 2  # theta, phi

        self.observation_space = spaces.Dict({
            "depth_image": spaces.Box(low=0, high=1, shape=(224, 224), dtype=np.float32),
            "state": spaces.Box(low=-1, high=1, shape=(state_dims,), dtype=np.float32)
        })

    def _init_reward(self):
        reward_params = self.env_params["reward"]
        self.reward_components = []
        active = reward_params["active_components"]  # dict: {name: weight}

        if "target_progress_reward" in active:
            self.reward_components.append(TargetProgressReward("target_progress", active["target_progress_reward"]))

        if "nonlinear_collision_penalty" in active:
            self.reward_components.append(NonlinearCollisionPenalty("nonlinear_collision_penalty", active["nonlinear_collision_penalty"]))

        if "linear_collision_penalty" in active:
            self.reward_components.append(LinearCollisionPenalty("linear_collision_penalty", active["linear_collision_penalty"]))

        if "distance_to_obstacle_reward" in active:
            self.reward_components.append(DistanceToObstacleReward("distance_to_obstacle_reward", active["distance_to_obstacle_reward"]))

        if "direction_reward" in active:
            self.reward_components.append(DirectionReward("direction_reward", active["direction_reward"]))

        if "spherical_direction_reward" in active:
            self.reward_components.append(SphericalDirectionReward("spherical_direction_reward", active["spherical_direction_reward"]))

        if "nonlinear_spherical_direction_reward" in active:
            self.reward_components.append(NonlinearSphericalDirectionReward("nonlinear_spherical_direction_reward", active["nonlinear_spherical_direction_reward"]))

        if "velocity_reward" in active:
            self.reward_components.append(VelocityReward("velocity_reward", active["velocity_reward"]))

        if "image_nonlinear_collision_penalty" in active:
            self.reward_components.append(
                ImageNonlinearCollisionPenalty("image_nonlinear_collision_penalty", active["image_nonlinear_collision_penalty"])
            )
        
        if "image_nonlinear_collision_penalty_2" in active:
            self.reward_components.append(
                ImageNonlinearCollisionPenalty2("image_nonlinear_collision_penalty_2", active["image_nonlinear_collision_penalty_2"])
            )

        if "image_linear_collision_penalty" in active:
            self.reward_components.append(
                ImageLinearCollisionPenalty("image_linear_collision_penalty", active["image_linear_collision_penalty"])
            )

        if "cosine_spherical_direction_reward" in active:
            self.reward_components.append(
                CosineSphericalDirectionReward("cosine_spherical_direction_reward", active["cosine_spherical_direction_reward"])
            )
        
        if "cosine_spherical_direction_reward_2" in active:
            self.reward_components.append(
                CosineSphericalDirectionReward2("cosine_spherical_direction_reward_2", active["cosine_spherical_direction_reward_2"])
            )

        if "tanh_spherical_direction_reward" in active:
            self.reward_components.append(
                TanhSphericalDirectionReward("tanh_spherical_direction_reward", active["tanh_spherical_direction_reward"])
            )
        
        if "interpolation_spherical_direction_reward" in active:
            self.reward_components.append(
                InterpolationSphericalDirectionReward("interpolation_spherical_direction_reward", active["interpolation_spherical_direction_reward"])
            )


        if "terminal_reward" in active:
            arrival_reward = reward_params["extra_rewards"]["arrival_reward"]
            collision_penalty = reward_params["extra_rewards"]["collision_penalty"]
            self.reward_components.append(TerminalReward("terminal_reward", active["terminal_reward"], arrival_reward, collision_penalty))

    def reset(self):
        # 1. 重置仿真环境
        logging.info("仿真环境重置 ...")
        self.sim.reset()
        # 2. 重置计数器
        self.step_count = 0
        # 3. 重置初始位置、目标位置
        self.sim.drone.target_position = self.generate_target_positions()
        # 初始化每个组件的回合累计奖励
        self.episode_total_reward = 0
        self.episode_component_rewards = {comp.name: 0.0 for comp in self.reward_components}
        # 3. 获取初始观测
        obs = self.get_obs()
        return obs
    
    def step(self, action:np.ndarray):
        self.step_count += 1

        # === 1. 施加动作并推进仿真 ===
        action = action.squeeze()
        velocity = self.compute_velocity_from_action(action)
        is_collided, nearest_info = self.sim.step(velocity)
        is_arrived = self.check_arrived()
        is_step_limited = self.step_count >= self.env_params['episode']['max_episode_timesteps']
        
        # === 2. 状态判断 ===
        done = is_collided or is_arrived or is_step_limited

        # === 3. 观测 ===
        obs = self.get_obs()

        # if is_collided:
        #     depth_min = np.min(obs['depth_image'])
        #     print(depth_min)
       # === 4. 奖励计算（基于奖励组件系统）===
        total_reward, component_rewards = self.get_reward(obs, is_arrived, is_collided)  # get_reward返回总奖励 + 子项奖励字典

        # === 6. 附加info返回（包括细粒度奖励统计） ===
        info = dict(
            step_count=self.step_count,
            done=done,
            total_reward=total_reward,
            collision=is_collided,
            arrival=is_arrived,
            step_limited=is_step_limited,
        )
        # === 加入每个 reward component 的 step 级奖励 ===
        for name, reward in component_rewards.items():
            info[f"reward/{name}"] = reward

        return obs, total_reward, done, info    

    def generate_target_positions(self):
        bounds = self.env_params['scene']['region']

        target_position = np.array([
            random.uniform(bounds['x_min'], bounds['x_max']),
            random.uniform(bounds['y_min'], bounds['y_max']),
            random.uniform(bounds['z_min'], bounds['z_max']),
        ])

        return target_position         

    def get_obs(self):
        """
        获取当前无人机的动态观测，根据需求选择观测特征。
        
        返回：
            np.array: 拼接后的观测数据
        """
        # 获取当前无人机的位置、速度、朝向和目标，根据需求拼接不同的特征
        observation = []
        observation_features = self.env_params['observation']['features']
        drone_state = self.sim.drone.state # 获取 DroneState 对象

        if "position" in observation_features:
            observation.extend(drone_state.position)  # 添加位置 [x, y, z]

        if "velocity" in observation_features:
            observation.extend(drone_state.linear_velocity)  # 添加线速度 [vx, vy, vz]

        if "spherical_velocity" in observation_features:
            velocity = drone_state.linear_velocity
            v = np.linalg.norm(velocity)
            if v < 1e-6:
                theta = 0.0  # 默认方向
                phi = 0.0
            else:
                theta = np.arccos(velocity[2] / v)        # 极角 θ ∈ [0, π]
                phi = np.arctan2(velocity[1], velocity[0])  # 方位角 φ ∈ [-π, π]

            observation.extend([v, theta, phi])

        if "orientation" in observation_features:
            observation.extend(drone_state.euler)  # 添加朝向（欧拉角）[roll, pitch, yaw]
        
        if "target" in observation_features:
            observation.extend(self.sim.drone.target_position)  # 添加目标位置

        if "spherical_target_relative_position" in observation_features:
            # 获取当前位置和目标位置
            pos = drone_state.position
            target = self.sim.drone.target_position
            diff = np.array(target) - np.array(pos)  # 差向量 [dx, dy, dz]
            
            dx, dy, dz = diff
            r = np.linalg.norm(diff) + 1e-6  # 距离（防止除以 0）
            theta = np.arccos(dz / r)        # 极角
            phi = np.arctan2(dy, dx)         # 方位角
            
            observation.extend([r, theta, phi])
        
        if "spherical_direction_error" in observation_features:
            pos = np.array(drone_state.position)
            vel = np.array(drone_state.linear_velocity)
            target = np.array(self.sim.drone.target_position)

            # 当前速度单位向量
            if np.linalg.norm(vel) > 1e-6:
                velocity_dir = vel / np.linalg.norm(vel)
            else:
                velocity_dir = np.zeros(3)

            # 目标方向单位向量
            diff = target - pos
            if np.linalg.norm(diff) > 1e-6:
                target_dir = diff / np.linalg.norm(diff)
            else:
                target_dir = np.zeros(3)

            # --- 球坐标计算 ---
            def cartesian_to_spherical(vec):
                x, y, z = vec
                r = np.linalg.norm(vec)
                if r < 1e-6:
                    return 0.0, 0.0  # 默认方向
                theta = np.arccos(z / r)       # 极角 θ ∈ [0, π]
                phi = np.arctan2(y, x)         # 方位角 φ ∈ [-π, π]
                return theta, phi

            theta_v, phi_v = cartesian_to_spherical(velocity_dir)
            theta_t, phi_t = cartesian_to_spherical(target_dir)

            # 俯仰方向（极角）：
            delta_theta = theta_t - theta_v
            delta_theta_norm = (delta_theta + np.pi) / (2 * np.pi)   # ∈ [0, 1]

            # 偏航方向（方位角）：
            delta_phi = (phi_t - phi_v + np.pi) % (2 * np.pi) - np.pi
            delta_phi_norm = (delta_phi + np.pi) / (2 * np.pi)       # ∈ [0, 1]


            observation.extend([delta_theta_norm, delta_phi_norm])
                
        self_position = np.array(observation)

        # 获取深度图信息（前方障碍物距离）每个像素是一个浮点数，介于 [0,1] 之间
        # 靠近相机的物体 → 深度值接近0
        # 远离相机的物体 → 深度值接近1
        # 如果看向空无一物的地方，深度值趋近于 far
        # 是二维矩阵，比如 shape = (240, 320)
        depth_image = self.sim.drone.get_depth_image()

        # 拼接当前无人机的状态信息和深度图信息
        obs = {
            "depth_image": depth_image,       # shape: [224, 224]，可扩展通道
            "state": self_position          # shape: [n]
        }

        return obs
    
    def get_reward(self, obs, is_arrived, is_collided):
        """
        计算当前无人机的奖励值，组件化管理
        返回:
            total_reward: 综合奖励
            component_rewards: 每个子奖励项
        """
        total_reward = 0.0
        component_rewards = {}

        for component in self.reward_components:
            reward = component.compute(self, obs=obs, is_arrived=is_arrived, is_collided=is_collided)
            weighted_reward = reward * component.weight
            total_reward += weighted_reward
            component_rewards[component.name] = weighted_reward

        return total_reward, component_rewards

    def check_arrived(self,arrival_threshold=5.0):
        """
        检查是否到达目标点附近。

        参数：
            current_position: 当前无人机的位置 (x, y, z)
            target_position: 目标位置 (x, y, z)
            arrival_threshold: 到达目标的距离阈值
        
        返回：
            bool: 如果到达目标附近，返回 True；否则返回 False
        """
        distance_to_target = np.linalg.norm(np.array(self.sim.drone.state.position) - np.array(self.sim.drone.target_position))
        return distance_to_target <= arrival_threshold  # 如果距离小于阈值，认为到达目标

    def compute_velocity_from_action(self, action: np.ndarray, mode: str = "adjust"):
        """
        根据指定 mode 解释动作，并执行对应控制。

        参数:
            action (np.ndarray): 动作向量
            mode (str): 控制模式，可为 'cartesian', 'spherical', 'adjust'
        """
        if mode == "cartesian":
            new_velocity = np.array(action, dtype=np.float32)

        elif mode == "spherical":
            # 绝对球坐标 → 笛卡尔
            v, theta, phi = action
            vx = v * np.sin(theta) * np.cos(phi)
            vy = v * np.sin(theta) * np.sin(phi)
            vz = v * np.cos(theta)
            new_velocity = np.array([vx, vy, vz], dtype=np.float32)

        elif mode == "adjust":
            delta_theta = action[0]
            delta_phi   = action[1]

            # 目标速度设定
            v_horiz = 15.0  # 水平速度
            v_vert = 5.0    # 垂直速度

            current_v = np.array(self.sim.drone.state.linear_velocity)
            norm = np.linalg.norm(current_v)

            if norm < 1e-3:
                theta = np.pi / 2
                phi = 0.0
            else:
                theta = np.arccos(current_v[2] / norm)
                phi = np.arctan2(current_v[1], current_v[0])

            theta_new = np.clip(theta + delta_theta, 0, np.pi)
            phi_new = phi + delta_phi

            # 构造单位方向向量（方向确定，但模长与速度无关）
            vx_unit = np.sin(theta_new) * np.cos(phi_new)
            vy_unit = np.sin(theta_new) * np.sin(phi_new)
            vz_unit = np.cos(theta_new)

            # 归一化水平分量向量
            horiz_norm = np.linalg.norm([vx_unit, vy_unit])
            if horiz_norm < 1e-6:
                vx = 0.0
                vy = 0.0
            else:
                vx = v_horiz * (vx_unit / horiz_norm)
                vy = v_horiz * (vy_unit / horiz_norm)

            # 垂直速度直接设为固定模长（方向由 theta_new 决定）
            vz = v_vert * np.sign(vz_unit)

            new_velocity = np.array([vx, vy, vz], dtype=np.float32)

        else:
            raise ValueError(f"Unsupported action mode: '{mode}'. Expected 'cartesian', 'spherical', or 'adjust'.")
             
        return new_velocity


