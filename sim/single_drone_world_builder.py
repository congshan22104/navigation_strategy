import pybullet as p
import pybullet_data
import numpy as np
import logging
from sim.drone_agent import DroneAgent


class SingleDroneWorldBuilder:
    def __init__(self, env_params=None):
        """
        :param env_params: YAML 里的 env_params 字典
        """
        logging.info("初始化 SingleDroneSim")

        # --------------- 读取配置 ----------------
        self.env_params = env_params

        # 按新结构读取scene字段
        scene_params = self.env_params["scene"]
        self.building_path = scene_params["building_path"]
        self.drone_urdf_path = scene_params["drone_urdf_path"]
        self.use_gui = scene_params["use_gui"]
        # --------------- 连接 PyBullet ------------
        try:
            self.physics_client = (
                p.connect(p.GUI) if self.use_gui else p.connect(p.DIRECT)
            )
            if self.use_gui:
                p.resetDebugVisualizerCamera(
                    cameraDistance=300.0,
                    cameraYaw=45,
                    cameraPitch=-60,
                    cameraTargetPosition=[0, 0, 100],
                )
                p.removeAllUserDebugItems()
            logging.info("成功连接 PyBullet, use_gui=%s", self.use_gui)
        except Exception as e:
            logging.error("无法连接 PyBullet: %s", e)
            raise

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # --------------- 加载地图 -----------------
        self.map_scale       = 1.0
        self.map_position    = [0, 0, 0]
        self.map_orientation = p.getQuaternionFromEuler([0, 0, 0])

        logging.info("加载地图模型 building_path=%s", self.building_path)
        try:
            self.map_visual_shape = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=self.building_path,
                meshScale=[self.map_scale] * 3,
                rgbaColor=[0.6, 0.6, 0.8, 0.3],  # 半透明效果
            )

            self.map_collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                fileName=self.building_path,
                meshScale=[self.map_scale] * 3,
                flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
            )

            self.map_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=self.map_collision_shape,
                baseVisualShapeIndex=self.map_visual_shape,
                basePosition=self.map_position,
                baseOrientation=self.map_orientation,
            )
            p.changeDynamics(self.map_id, -1, lateralFriction=0.8, restitution=0.5)
        except Exception as e:
            logging.error("地图加载失败: %s", e)
            raise

        # --------------- 加载无人机 -----------------
        drone_params = self.env_params["drone"]
        self.drone_init_pos = drone_params["drone_init_pos"]

        self.init_drone()

        # --------------- 轨迹绘制辅助 -------------
        self.drone_path = []  # 单无人机的轨迹列表
        self._last_drawn_idx = 0  # 上一次绘制的索引

    # ------------------------------------------------------------------ #
    #                         初 始 化 无 人 机                            #
    # ------------------------------------------------------------------ #
    def init_drone(self, global_scaling=10.0):
        """初始化单架无人机，封装为 DroneAgent 类"""
        self.drone = DroneAgent(
            index=0,
            team="blue",
            init_pos=self.drone_init_pos,
            urdf_path=self.drone_urdf_path,
            color=[0, 0, 1, 1],  # 蓝色
            global_scaling=global_scaling
        )

    def step_simulation(self, force, num_steps=60):
        """
        推进仿真 num_steps 个步长
        1. 施加外力
        2. p.stepSimulation()
        3. 记录当前位置 (满足“先记录再画线”要求)
        4. 增量绘制新线段
        """
        collision_detected = False  # 标记是否发生碰撞
        collisions = []  # 用于存储所有碰撞的详细信息

        for _ in range(num_steps):
            # ------ 1. 施加外力 -----
            self.drone.apply_force(force)  # 施加外力

            # ------ 2. 物理推进 ------
            p.stepSimulation()

            # ------ 3. 碰撞检查 ------
            # 检查碰撞
            collided, drone_collisions = self.drone.check_collision()
            if collided:
                logging.info(f"无人机 {self.drone.index} 与其他物体发生碰撞！")
                collision_detected = True  # 标记发生碰撞
                collisions.append({"drone_id": self.drone.index, "collisions": drone_collisions})
                break  # 立即跳出循环，不再继续仿真步骤
        
        # ------ 3. 参数更新 ------
        self.drone.update_state() 
        self.drone.update_path()  # 更新路径

        # ------ 4. 绘制轨迹 ------
        if self.use_gui:
            self.drone.draw_trajectory()  # 在所有步骤完成后统一绘制轨迹

        # 返回碰撞状态和详细信息
        if collision_detected:
            return True, collisions  # 返回碰撞信息
        else:
            return False, None  # 没有碰撞

    def clear_debug_trajectory(self):
        """
        清除 GUI 上所有调试线，并重置绘制索引
        """
        p.removeAllUserDebugItems()
        logging.info("已清除 GUI 轨迹")

    def close(self):
        try:
            self.clear_debug_trajectory()
            p.disconnect()
            logging.info("已断开 PyBullet 连接")
        except Exception as e:
            logging.error("关闭环境时出错: %s", e)
