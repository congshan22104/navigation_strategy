import numpy as np
import logging
import pybullet as p
from dataclasses import dataclass

@dataclass
class DroneState:
    position: np.ndarray
    orientation: np.ndarray
    euler: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray

class DroneAgent:
    """
    DroneAgent 类用于模拟一个无人机智能体，提供对无人机的基本控制、状态查询、碰撞检测、轨迹绘制等功能。

    方法:
    - `__init__(self, index, team, init_pos, urdf_path, color, global_scaling=10.0)`:
      初始化无人机智能体，包括设置编号、阵营、位置、加载模型等。
      
    - `apply_force(self, force)`:
      施加外力来影响无人机的运动状态。

    - `update_path(self, pos)`:
      更新无人机路径记录，追加当前位置到路径列表中。

    - `get_position(self)`:
      获取当前无人机的位置，返回一个 3D 坐标 [x, y, z]。

    - `get_orientation(self, euler=False)`:
      获取无人机的朝向，返回四元数或欧拉角。

    - `get_velocity(self)`:
      获取无人机的线速度和角速度。

    - `get_state(self)`:
      获取当前无人机的完整状态信息，包括位置、朝向、速度等。

    - `draw_safety_zone(self)`:
      绘制一个表示无人机安全区域的感知球，便于调试和可视化。

    - `draw_trajectory(self)`:
      增量绘制无人机的飞行轨迹，用于 GUI 可视化显示。

    - `check_collision(self)`:
      检测无人机是否与其他物体发生碰撞，判断其是否在碰撞阈值内，并返回碰撞信息。

    - `distance_to(self, other_agent)`:
      计算当前无人机与另一架无人机之间的欧几里得距离，支持多智能体间的交互。


    """

    def __init__(self, index, team, init_pos, urdf_path, color, global_scaling=10.0):
        """
        初始化单架无人机智能体

        参数:
        - index: 无人机的逻辑编号
        - team: 阵营 ("red" 或 "blue")
        - init_pos: 初始位置 [x, y, z]
        - urdf_path: URDF 文件路径
        - color: RGBA 颜色
        - global_scaling: 模型缩放系数
        """
        self.index = index
        self.team = team
        self.color = color
        self.init_pos = init_pos

        # 加载 URDF 模型
        ori = p.getQuaternionFromEuler([0, 0, 0])  # 初始朝向
        self.id = p.loadURDF(urdf_path, init_pos, ori, globalScaling=global_scaling)
        p.changeVisualShape(self.id, -1, rgbaColor=color)

        logging.info("[Init] %s #%d | ID=%d | Pos=%s", team.capitalize(), index, self.id, init_pos)

        # 设置相机视角矩阵和投影矩阵
        self.view_matrix = p.computeViewMatrix(cameraEyePosition=[0, 0, 10],
                                          cameraTargetPosition=[0, 0, 0],
                                          cameraUpVector=[0, 0, 1])
        
        self.projection_matrix = p.computeProjectionMatrixFOV(fov=90,
                                                        aspect=float(320) / float(240),
                                                        nearVal=0.1,
                                                        farVal=100.0)
        self.safety_radius = 2.0  # 安全距离(米)
        self.trajectory = [tuple(init_pos)]  # 初始路径
        self.state = self.get_state()
    
    def apply_force(self, force):
        """
        对无人机施加外力，影响无人机的运动

        参数:
        - force: 3D 向量(np.ndarray 或 list)
        """
        force = force.squeeze().tolist()
        pos, _ = p.getBasePositionAndOrientation(self.id)
        try:
            p.applyExternalForce(
                objectUniqueId=self.id,
                linkIndex=-1,
                forceObj=force,
                posObj=pos,
                flags=p.WORLD_FRAME
            )
        except Exception as e:
            logging.error("施加外力失败 [ID=%d]: %s", self.id, e)

    def update_path(self):
        """
        更新无人机路径，记录当前位置信息
        """
        self.trajectory.append(tuple(self.state.position))
    
    def update_state(self):
        """
        更新无人机状态
        """
        self.state = self.get_state()

    def get_position(self):
        """
        获取当前无人机的位置 [x, y, z]

        返回:
        - numpy 数组:无人机当前位置
        """
        pos, _ = p.getBasePositionAndOrientation(self.id)
        return np.array(pos)

    def get_orientation(self, euler=False):
        """
        获取当前无人机朝向，返回四元数或欧拉角

        参数:
        - euler: 是否返回欧拉角(默认返回四元数)

        返回:
        - 四元数或欧拉角(取决于 euler 参数)
        """
        _, ori = p.getBasePositionAndOrientation(self.id)
        return p.getEulerFromQuaternion(ori) if euler else ori

    def get_velocity(self):
        """
        获取无人机的线速度和角速度

        返回:
        - linear: 线速度 [vx, vy, vz]
        - angular: 角速度 [wx, wy, wz]
        """
        linear, angular = p.getBaseVelocity(self.id)
        return np.array(linear), np.array(angular)
    
    def get_state(self)-> DroneState:
        """
        获取无人机的完整状态，包括位置、朝向、速度等信息

        返回:
        - dict: 包含 position、orientation、euler、linear_velocity、angular_velocity
        """
        pos, ori = p.getBasePositionAndOrientation(self.id)
        linear, angular = p.getBaseVelocity(self.id)
        return DroneState(
            position=np.array(pos),
            orientation=np.array(ori),
            euler=np.array(p.getEulerFromQuaternion(ori)),
            linear_velocity=np.array(linear),
            angular_velocity=np.array(angular)
        )
    
    def draw_safety_zone(self):
        """
        绘制无人机的安全区域(感知球可视化，用于调试)
        """
        pos, _ = p.getBasePositionAndOrientation(self.id)
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.safety_radius,
            rgbaColor=[1, 0, 0, 0.15],
            specularColor=[0.4, 0.4, 0]
        )
        p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=pos)
    

    def draw_trajectory(self):
        """
        绘制当前轨迹段，然后清空轨迹，保留最后一点用于接下一段
        """
        path = self.trajectory
        if len(path) < 2:
            return  # 不足以成线，跳过

        # 逐段绘制
        for i in range(len(path) - 1):
            p.addUserDebugLine(
                path[i],
                path[i + 1],
                lineColorRGB=self.color,
                lineWidth=2,
                lifeTime=0  # 永久显示
            )

        # 保留最后一点，开始新轨迹
        self.trajectory = [path[-1]]

    
    def check_collision(self):
        """
        检测无人机是否发生碰撞，判断是否有物体靠近无人机

        返回:
        - collided: 是否发生碰撞(True/False)
        - collisions: 所有碰撞物体的详细信息 [{body_id, distance, position}]
        """
        collision_radius = 2.0  # 碰撞判定阈值
        contacts = p.getClosestPoints(bodyA=self.id, bodyB=-1, distance=collision_radius)

        collisions = []
        collided = False

        for c in contacts:
            other_id = c[2]
            if other_id == self.id:
                continue
            # 记录碰撞物体的信息
            collisions.append({
                "body_id": other_id,
                "distance": c[8],  # contactDistance
                "position": c[6],  # positionOnB
            })

            collided = True  # 一旦发生碰撞，标记为 True

        return collided, collisions
    
    def distance_to(self, other_agent):
        """
        计算当前无人机与另一架无人机之间的欧几里得距离

        参数:
        - other_agent: 另一架无人机实例

        返回:
        - float: 两者之间的欧几里得距离
        """
        pos_a = self.get_position()
        pos_b = other_agent.get_position()
        return np.linalg.norm(pos_a - pos_b)

    def check_obstacle(self, max_ray_length=30.0):
        """
        为无人机模拟雷达系统，检测前方20米内的障碍物。

        参数:
            drone: 当前无人机
            max_ray_length: 雷达最大射程，默认30米
        
        返回:
            float: 前方障碍物的实际距离。如果无障碍物，则返回最大射程值。
        """
        # 获取无人机的当前位置
        position = self.get_position()

        # 获取无人机的朝向方向，这里假设朝向沿着Z轴方向（如果需要，可以调整）
        # 假设朝向是单位向量（x, y, z），指向前方
        orientation = self.get_orientation(euler=True)
        forward_direction = np.array([np.cos(orientation[2]), np.sin(orientation[2]), 0])  # 在XY平面上的朝向
        
        # 计算射线的终点位置
        ray_to_position = position + forward_direction * max_ray_length

        # 发射射线，检测是否有障碍物
        hit, hit_position, hit_distance = p.rayTest(position, ray_to_position)
        
        # 判断雷达检测结果
        if hit_distance == -1:
            # 没有障碍物，返回最大射程值（表示没有障碍物）
            return max_ray_length
        else:
            # 返回实际的射线距离
            return hit_distance
      
    def get_depth_image(self, width=320, height=240):
      """
      获取深度图
      
      参数:
          view_matrix: 相机视角矩阵
          projection_matrix: 相机投影矩阵
          width: 图像宽度
          height: 图像高度
      
      返回:
          depth_image: 深度图（归一化为0-1范围）
      """
      # 获取图像信息
      img_arr = p.getCameraImage(width, height, viewMatrix=self.view_matrix, projectionMatrix=self.projection_matrix)
      
      # 获取深度图信息
      depth_image = np.array(img_arr[3])  # 深度图

      return depth_image

    def get_closest_obstacle_distance(self):
        depth_image = self.get_depth_image()
        near=0.1
        far=100.0
        depth = (far * near) / (far - (far - near) * depth_image)


        # 过滤掉背景像素（=1.0表示无深度）
        valid_depth = depth[depth_image < 0.999]
        
        if valid_depth.size == 0:
            return 100  # 没检测到物体

        return valid_depth.min()
