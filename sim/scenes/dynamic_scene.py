from sim.scenes.random_scene import RandomScene 
import numpy as np
import random

class VoxelizedRandomScene(RandomScene):
    def __init__(self, scene_size_x, scene_size_y, scene_size_z, num_obstacles,
                 min_radius, max_radius, min_height, max_height, voxel_size):
        super().__init__(scene_size_x, scene_size_y, scene_size_z,
                         num_obstacles, min_radius, max_radius,
                         min_height, max_height)
        self.voxel_size = voxel_size
        self.voxel_map = np.zeros((
            int(scene_size_x / voxel_size),
            int(scene_size_y / voxel_size),
            int(scene_size_z / voxel_size)
        ), dtype=np.uint8)

    def build_scene(self):
        self._generate_obstacles()
        self._voxelize_obstacles()

    def _generate_dynamic_obstacles(self):
        # 生成200个5x5x5的正方体
        cubes = []
        for i in range(200):
            # 随机位置
            pos = [random.uniform(-10, 10), random.uniform(-10, 10), 2]  # z高度2避免与地面重叠
            cube = p.loadURDF("cube.urdf", basePosition=pos, globalScaling=5)  # 加载正方体模型并设置大小

            # 设置不与其他物体发生碰撞
            p.setCollisionFilterGroupMask(cube, -1, 0, 0)  # 禁用与其他物体的碰撞

            cubes.append(cube)
