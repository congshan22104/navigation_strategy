深度图：
    PyBullet 使用的是 OpenGL 风格的 z-buffer 深度图，这是一种 非线性映射。
    映射公式使得 近处对象的深度变化很敏感，而 远处对象的深度变化被极度压缩；
    因此，大多数场景中，远处的背景（例如几米外的障碍或地面）会被映射到深度值 非常接近 1.0；只有靠近摄像头的物体，其深度值才会明显小于 0.9；
    可视的范围目前是0-100米，这与实际不符合，但是不重要，先能规划出来路径再说
特征提取器：
    concat：将深度图从（240，320）转为（3，4）大小
    resent：利用resnet18与训练网络将深度图转为维度为25的tensor
