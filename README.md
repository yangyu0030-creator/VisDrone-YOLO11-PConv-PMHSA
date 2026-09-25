# VisDrone-YOLO11-PConv-PMHSA

基于 YOLO11s 改进的小目标检测模型，在 VisDrone2019-DET 数据集上达到 **mAP50 53.25%**，**mAP50-95 33.25%**。

## 改进点
1. **PConv（风车卷积）**：替换主干网络浅层卷积，增强小目标的方向性特征提取。
2. **PMHSA（多头自注意力）**：插入主干末端，增强全局上下文建模。

## 数据集
VisDrone2019-DET（[下载地址](https://github.com/VisDrone/VisDrone-Dataset)）
放置到 `datasets/VisDrone2019-DET/` 目录下，包含 `images/` 和 `labels/`。

## 环境配置
pip install ultralytics torch torchvision opencv-python pyyaml

## 训练
python train.py

## 验证
python val.py

## 结果
| 指标 | 数值 |
|------|------|
| mAP50 | 0.5325 |
| mAP50-95 | 0.3325 |
| P | 0.6296 |
| R | 0.5141 |

## 每类指标
| 类别 | mAP50 | mAP50-95 |
|------|-------|----------|
| car | 0.841 | 0.583 |
| bus | 0.741 | 0.560 |
| truck | 0.649 | 0.459 |
| van | 0.627 | 0.459 |
| pedestrian | 0.586 | 0.283 |
| motor | 0.551 | 0.276 |
| tricycle | 0.388 | 0.243 |
| people | 0.338 | 0.132 |
| awning-tricycle | 0.309 | 0.198 |
| bicycle | 0.294 | 0.132 |

## 文件说明
- `train.py`：训练脚本
- `val.py`：验证脚本
- `predict.py`：推理脚本
- `convert_and_split.py`：VisDrone 标注转换 + 数据集划分
- `ultralytics/nn/ConvModules/PConv.py`：PConv 实现
- `ultralytics/nn/SPPModules/PMHSA.py`：PMHSA 实现
- `ultralytics/cfg/models/11/conv_yaml/yolo11_PConv_PMHSA.yaml`：模型结构
