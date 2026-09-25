<div align="center">
  <h1>ConvModules</h1>
  <p>🚀 深度学习卷积模块复现与改进合集</p>
  
  <!-- 徽章 -->
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.10%2B-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</div>

---

## 项目简介

本项目收集并实现了多种轻量化、注意力机制与卷积变体模块，适用于计算机视觉任务（如目标检测、图像分类）。所有模块均基于 PyTorch 编写，方便直接集成到 YOLO 等网络架构中。

## 模块说明

| 模块文件 | 模块名称 | 简要说明 |
| :---: | :---: | :--- |
| `C3_Ghost.py` | C3_Ghost | 结合 GhostNet 思想的 C3 模块，极致轻量化 |
| `C3_TR.py` | C3_TR | 引入 Transformer 结构的 C3 模块 |
| `C3_X.py` | C3_X | 改进型 C3 模块（如 CSP 结构变体） |
| `FasterBlock.py` | FasterBlock | 快速卷积块（如 FasterNet 结构） |
| `FCM.py` | FCM | 特征上下文模块 (Feature Context Module) |
| `GHBlock.py` | GHBlock | Ghost 卷积块 |
| `LAE.py` | LAE | 局部注意力增强模块 |
| `MSBlock.py` | MSBlock | 多尺度卷积块 |
| `MSCAM.py` | MSCAM | 多尺度通道注意力模块 |
| `PConv.py` | PConv | 部分卷积（Partial Convolution），降低计算冗余 |
| `RFEM.py` | RFEM | 感受野增强模块 |
| `SCConv.py` | SCConv | 空间与通道重建卷积 |
| `SPDConv.py` | SPDConv | 空间到深度卷积 |

## 环境依赖

* Python 3.8+
* PyTorch 1.10+
* 其他依赖请根据实际情况安装（如 `numpy`, `torchvision` 等）