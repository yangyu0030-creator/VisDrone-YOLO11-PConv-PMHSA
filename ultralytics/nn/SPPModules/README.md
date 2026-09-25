# SPPModules

这是一个基于 YOLO 的 SPP（空间金字塔池化）模块复现与改进合集。本项目实现了多种注意力机制与池化模块的变体，用于提升目标检测模型的性能。

## 模块说明

*   **`ASPP.py`**: 空洞空间金字塔池化（Atrous Spatial Pyramid Pooling），通过不同膨胀率的空洞卷积捕获多尺度上下文信息。
*   **`DBSPPF.py`**: 密集连接空间金字塔池化（Dense Block SPPF），增强特征复用与梯度流动。
*   **`PMHSA.py`**: 并行多头自注意力模块（Parallel Multi-Head Self-Attention），用于增强全局特征建模能力。
*   **`SE_SPPF.py`**: 引入 SE（Squeeze-and-Excitation）通道注意力的 SPPF 模块。
*   **`SPPF_LSKA.py`**: 结合大核可分离注意力（LSKA）的 SPPF 模块，在降低计算量的同时扩大感受野。

## 环境依赖

*   Python 3.8+
*   PyTorch 1.10
*   其他依赖请根据实际运行情况安装（如 `numpy`, `torchvision` 等）。
