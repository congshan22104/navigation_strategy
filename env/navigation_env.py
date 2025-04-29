"""
env/low_level_env.py
"""

import gym
from gym import spaces
import numpy as np
import random
import pybullet as p
import logging
import cv2

from typing import List, Tuple
from sim.single_drone_world_builder import SingleDroneWorldBuilder
from env.wrappers.reward_wrapper import TargetProgressReward, ObstaclePenalty, HeadingAlignmentReward


# logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class NavigationEnv(gym.Env):
    def __init__(self, env_params):
        super().__init__()
        
        self.env_params = env_params

        # ====== 读取基本仿真参数 ======
        self.max_episode_timesteps = env_params["episode"]["max_episode_timesteps"]

        self.sim = SingleDroneWorldBuilder(env_params)
        self.drone = self.sim.drone
        
        self.observation_features = ["position", "target"]

        # 动作空间：3维连续力控制
        self.action_space = spaces.Box(
            low=-20, high=20,
            shape=(3,),
            dtype=np.float32
        )

        # 观测空间：图像 + 状态
        self.observation_space = spaces.Dict({
            "depth_image": spaces.Box(low=0, high=1, shape=(224, 224), dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        })

        self.step_count = 0

        # [x_min, y_min, z_min], [x_max, y_max, z_max]
        self.bounds = {
            "x_min": env_params["region"]["x_min"],
            "x_max": env_params["region"]["x_max"],
            "y_min": env_params["region"]["y_min"],
            "y_max": env_params["region"]["y_max"],
            "z_min": env_params["region"]["z_min"],
            "z_max": env_params["region"]["z_max"],
        }

        self.safe_distance = 10  # 可从config拓展，目前写死

        self.target_position = self.generate_target_position()

        # ====== 读取奖励参数 ======
        reward_params = env_params["reward"]

        self.arrival_bonus = reward_params["extra_rewards"]["arrival_bonus"]
        self.collision_penalty = reward_params["extra_rewards"]["collision_penalty"]

        # ====== 初始化奖励组件列表（根据config选择）======
        self.episode_total_reward = 0
        active_components = reward_params["active_components"]
        self.reward_components = []

        if "target_progress" in active_components:
            self.reward_components.append(TargetProgressReward(name="target_progress", weight=1.0))
        if "obstacle_penalty" in active_components:
            self.reward_components.append(ObstaclePenalty(name="obstacle_penalty", weight=1.0))
        if "heading_alignment" in active_components:
            self.reward_components.append(HeadingAlignmentReward(name="heading_alignment", weight=0.5))
        logging.info(f"激活奖励组件: {[comp.name for comp in self.reward_components]}")



    def reset(self):
        # 1. 重置仿真环境
        logging.info("仿真环境重置 ...")
        self.sim.close()
        self.sim = SingleDroneWorldBuilder(env_params=self.env_params)
        self.drone = self.sim.drone
        # 2. 重置计数器
        self.step_count = 0
        self.target_position = self.generate_target_position()
        # 3. 获取初始观测
        obs = self.get_obs()
        # 每次reset时，清零
        self.step_count = 0
        self.episode_total_reward = 0
        return obs
    
    def step(self, action:np.ndarray):
        self.step_count += 1

        # === 1. 施加动作并推进仿真 ===
        force = action.reshape(-1, 3)
        is_collided, collision_info = self.sim.step_simulation(force)
        is_arrived = self.check_arrived()
        is_step_limited = self.step_count >= self.max_episode_timesteps

        # === 2. 状态判断 ===
        done = is_collided or is_arrived or is_step_limited

        # === 3. 观测 ===
        obs = self.get_obs()

       # === 4. 奖励计算（基于奖励组件系统）===
        total_reward, component_rewards = self.get_reward()  # get_reward返回总奖励 + 子项奖励字典

        # 到达目标，加额外奖励
        if is_arrived:
            total_reward += 100.0

        # 碰撞，额外惩罚
        elif is_collided:
            total_reward -= 100.0

        # === 5. 累加到当前回合的奖励统计 ===

        # 初始化episode统计
        if not hasattr(self, "episode_component_rewards"):
            self.episode_component_rewards = {k: 0.0 for k in component_rewards.keys()}

        # 累加各个子项的奖励
        for name, reward in component_rewards.items():
            self.episode_component_rewards[name] += reward

        self.episode_total_reward += total_reward  # 总奖励也累加

        # === 6. 附加info返回（包括细粒度奖励统计） ===
        info = dict(
            step_count=self.step_count,
            done = done,
            total_reward=total_reward,
            collision=is_collided,
            arrival=is_arrived,
            step_limited=is_step_limited,
        )

        # === 7. 如果done=True，加上整回合episode级别统计 ===
        if done:
            # 统一加上"episode/"前缀
            info.update({
                "episode/total_reward": self.episode_total_reward,
                "episode/step_count": self.step_count
            })
            
            # 各子奖励项也加到info里
            for name, reward_sum in self.episode_component_rewards.items():
                info[f"episode/{name}"] = reward_sum
            
            # === 在回合结束后，记录info到日志 ===
            logging.info(f"Episode finished: {info}")

        return obs, total_reward, done, info    

    def generate_target_position(self):
        """
        在地图范围内生成一个不目标位置
        返回：
            tuple: 生成的目标位置 (x, y, z)
        """

        # 随机生成一个目标位置
        target_position = np.array([
            random.uniform(self.bounds['x_min'], self.bounds['x_max']),
            random.uniform(self.bounds['y_min'], self.bounds['y_max']),
            random.uniform(self.bounds['z_min'], self.bounds['z_max'])
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
        observation_features = self.observation_features
        drone_state = self.drone.state  # 获取 DroneState 对象

        if "position" in observation_features:
            observation.extend(drone_state.position)  # 添加位置 [x, y, z]

        if "velocity" in observation_features:
            observation.extend(drone_state.linear_velocity)  # 添加线速度 [vx, vy, vz]

        if "orientation" in observation_features:
            observation.extend(drone_state.euler)  # 添加朝向（欧拉角）[roll, pitch, yaw]
        
        if "target" in observation_features:
            observation.extend(self.target_position)  # 添加目标位置

        self_position = np.array(observation)
        
        # 获取深度图信息（前方障碍物距离）每个像素是一个浮点数，介于 [0,1] 之间
        # 靠近相机的物体 → 深度值接近0
        # 远离相机的物体 → 深度值接近1
        # 如果看向空无一物的地方，深度值趋近于 far
        # 是二维矩阵，比如 shape = (240, 320)
        depth_image = self.drone.get_depth_image()
        resized_depth_image = cv2.resize(depth_image, (224,224), interpolation=cv2.INTER_LINEAR)

        # 拼接当前无人机的状态信息和深度图信息
        obs = {
            "depth_image": resized_depth_image.astype(np.uint8),       # shape: [1, 224, 224]，可扩展通道
            "state": self_position.astype(np.float32)          # shape: [n]
        }

        return obs
    

    def get_reward(self):
        """
        计算当前无人机的奖励值，组件化管理
        返回:
            total_reward: 综合奖励
            component_rewards: 每个子奖励项
        """
        total_reward = 0.0
        component_rewards = {}

        for component in self.reward_components:
            reward = component.compute(self)
            weighted_reward = reward * component.weight
            total_reward += weighted_reward
            component_rewards[component.name] = weighted_reward

        return total_reward, component_rewards


    def check_arrived(self,arrival_threshold=10.0):
        """
        检查是否到达目标点附近。

        参数：
            current_position: 当前无人机的位置 (x, y, z)
            target_position: 目标位置 (x, y, z)
            arrival_threshold: 到达目标的距离阈值
        
        返回：
            bool: 如果到达目标附近，返回 True；否则返回 False
        """
        distance_to_target = np.linalg.norm(np.array(self.drone.state.position) - np.array(self.target_position))
        return distance_to_target <= arrival_threshold  # 如果距离小于阈值，认为到达目标
