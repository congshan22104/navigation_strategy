# depth_feature_extractor.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym.spaces import Dict


class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Dict, features_dim=128, cfg=None):
        """
        使用原始 ResNet18 提取深度图特征并映射为 cfg["resnet_output_dim"]，
        同时通过 MLP 处理状态向量，并将两者拼接
        """
        super().__init__(observation_space, features_dim)
        self.cfg = cfg or {}

        # === 1. ResNet 分支 ===
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 输出: [B, 512, 1, 1]
        self.projector = nn.Linear(512, self.cfg["resnet_output_dim"])

        # === 2. MLP 处理状态向量 ===
        self.mlp = nn.Sequential(
            nn.Linear(self.cfg["state_dim"], 512),
            nn.ReLU(),
            nn.Linear(512, self.cfg["mlp_output_dim"]),
            nn.ReLU()
        )

        # === 3. 特征维度 = ResNet 输出 + MLP 输出 ===
        self._features_dim = self.cfg["resnet_output_dim"] + 64

        # 图像预处理器
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # 灰度图复制为 RGB
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def forward(self, observation):
        # --- 1. 处理深度图 ---
        batch_depth = []
        for depth_np in observation["depth_image"]:
            depth_img = depth_np.squeeze().cpu().numpy()  # [H, W]
            processed = self.preprocess_depth_to_3ch(depth_img)
            batch_depth.append(processed)
        x1 = torch.cat(batch_depth, dim=0).to(observation["depth_image"].device)  # [B, 3, 224, 224]

        features = self.backbone(x1)              # ResNet -> [B, 512, 1, 1]
        features = features.view(x1.size(0), -1)  # Flatten -> [B, 512]
        y1 = self.projector(features)             # Project to low-dim -> [B, resnet_output_dim]

        # --- 2. MLP 分支 ---
        y2 = self.mlp(observation["state"])       # MLP -> [B, 64]

        # --- 3. 拼接两个分支 ---
        fused = torch.cat([y1, y2], dim=1)
        return fused

    def preprocess_depth_to_3ch(self, depth_image: np.ndarray):
        """
        将已经归一化到 [0,1] 的单通道深度图扩展为 3 通道，并标准化到 [-1,1]
        输入: 2D numpy array [224, 224], 值在 [0, 1]
        输出: Tensor [1, 3, 224, 224]
        ResNet输入要求：
            - Tensor shape: [Batch_Size, 3, 224, 224]
            - dtype: torch.float32
            - Value range: 通常归一化到 [-1, 1]
            - 3通道 (RGB)，如果是灰度图需要复制成3通道
            - 输入尺寸固定为224x224（需要resize）
        """
        # 确保输入是 float32
        if not isinstance(depth_image, np.ndarray):
            raise TypeError("depth_image must be a numpy array")
        if depth_image.dtype != np.float32:
            depth_image = depth_image.astype(np.float32)

        # 扩展到3通道
        depth_3ch = np.stack([depth_image] * 3, axis=0)  # [3, 224, 224]

        # 转成Tensor
        depth_tensor = torch.from_numpy(depth_3ch)  # [3,224,224], float32, [0,1]

        # Normalize到 [-1, 1] （注意不是ImageNet mean/std）
        depth_tensor = depth_tensor * 2.0 - 1.0  # 线性变换 [0,1] → [-1,1]

        # 加 batch 维度
        depth_tensor = depth_tensor.unsqueeze(0)  # [1, 3, 224, 224]
        

        return depth_tensor
