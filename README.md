# 奶龙奶蛙奶蛋识别（Nailong · Naiwa · Naidan）

一个识别网络梗「奶龙 / 奶蛙 / 奶蛋」的 CV 项目，覆盖**图像分类 → 多标签分类 → 目标检测**三种任务的完整 pipeline。

奶龙是那个被玩成抽象梗的粉龙 IP，奶蛙是它的衍生形象，奶蛋是小红书/抖音上 AI 生成的蛋形萌系角色。本项目用深度学习识别这三者，外加「无关」作为负类。

## 项目亮点

- **三任务完整链路**：分类（ResNet18）、多标签（处理同框图）、目标检测（YOLOv8）
- **数据从 55 张奶蛋扩充到 526 张**：视频抽帧 + 半自动预标注 + 人工修正的迭代闭环
- **半自动标注流程**：初版模型预标注 → 人工修框 → 迭代训练，把标注成本压到最低
- **GPU 训练**：RTX 4050，检测 mAP50 从 0.733 迭代到 0.851

## 任务与效果

| 任务 | 模型 | 指标 |
|------|------|------|
| 四分类（奶龙/奶蛙/奶蛋/无关） | ResNet18 全量微调 | test accuracy **95.8%** |
| 多标签（奶龙/奶蛙/奶蛋，处理同框） | ResNet18 + 3×sigmoid | exact-match **0.933** |
| 目标检测 | YOLOv8n | mAP50 **0.851** |

检测各类别 mAP50：

| 类别 | mAP50 | 精度 P | 召回 R |
|------|-------|--------|--------|
| 奶龙 | 0.944 | 0.97 | 0.83 |
| 奶蛋 | 0.935 | 0.92 | 0.92 |
| 奶蛙 | 0.676 | 0.75 | 0.69 |

> 奶蛙是三类里最难的：形象本身奇形怪状、类内方差大，边界跟奶龙奶蛋都沾边，所以 mAP 明显偏低。

## 技术栈

- **PyTorch** + torchvision（ResNet18）
- **ultralytics**（YOLOv8n）
- 数据工具链：ffmpeg 抽帧、pHash 去重、四分类辅助筛选同框候选

## 目录结构

```
nailongnaiwa/
├── nailong_model.py               # 分类/多标签模型定义
├── train_pretrained_classifier.py # 四分类训练
├── train_multilabel.py            # 多标签训练
├── train_yolo.py                  # YOLOv8 检测训练
├── prelabel_yolo.py               # 预标注脚本
├── models/                        # 训练好的模型权重
│   ├── yolov8n_detection.pt       # 检测（mAP50 0.851）
│   ├── four_class_resnet18_gpu.pt # 四分类
│   └── multilabel_resnet18.pt     # 多标签
├── nailong_detection/             # 检测数据配置（data.yaml）
└── tools/
    ├── pipeline.py                # 数据流水线入口（抽帧→去重→分类→训练）
    ├── prepare_four_class.py      # 划分 train/test/gen
    ├── dedup_images.py            # md5 + pHash 去重
    └── classify_frames.py         # 视频帧分类分桶
```

## 数据说明

**数据集不包含在本仓库中。** 训练图像来自公开的 meme 图与短视频抽帧：

- 奶龙：Know Your Meme 等公开 meme 图
- 奶蛙：奶龙蛙衍生形象（nai_long_frog）
- 奶蛋：小红书/抖音的 AI 生成 IP（人工收集 + 视频抽帧）
- 无关：必应图搜批量下载
- 同框：奶蛋奶蛙同框镜头抽帧

这些图涉及第三方 IP，故不重新分发。复现训练需自行按上述来源准备数据，目录组织为 `nailong_naiwa_10_demo/{nailong,naiwa,naidan,unrelated}/`。

## 快速开始

```bash
# 依赖
pip install torch torchvision ultralytics pillow imagehash

# 四分类训练
python train_pretrained_classifier.py --split-dir nailong_naiwa_splits_4class

# 多标签训练（需同框标注 manifest）
python train_multilabel.py --manifest multilabel_manifest.json

# 检测训练
python train_yolo.py

# 用训练好的权重直接推理（检测）
python -c "from ultralytics import YOLO; YOLO('models/yolov8n_detection.pt').predict('你的图片.jpg', save=True)"
```

## 致谢

本项目 fork 自 [whxoperator/nailongnaiwa](https://github.com/whxoperator/nailongnaiwa)（奶龙 vs 奶蛙二分类），在其基础上扩展了奶蛋类别、多标签分类与目标检测，并重写了数据工具链。原项目的 `RandomCornerOcclusion`（针对水印的随机角遮挡）设计保留并沿用。
