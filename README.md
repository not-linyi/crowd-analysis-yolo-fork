# 草履虫DQYのYOLO速通术

> 本项目出于学习YOLO目的，复刻自[BRT Crowd Analysis System by Hassan Raza](https://github.com/hassanrrraza/crowd-analysis-yolo)

本项目仅用于记录本人YOLO学习的过程
啊，我速通YOLO，真的假的，要上吗？

## 数据准备

### 数据集
1. 提取视频帧，用于训练YOLO模型
    ```bash
    python .\src\utils\ImageExtractor.py --video test.mp4 --output dataset/images/train
    ```
2. 使用LabelImg等目标检测标注工具，标注目标

（或者在roboflow下载数据集）

### 数据集格式要求
YOLOv8 的数据集需满足以下结构：
```
dataset/
├── images/
│   ├── train/      # 训练图像
│   └── val/        # 验证图像
└── labels/
    ├── train/      # 训练标注文件
    └── val/        # 验证标注文件
```
每个标注文件（.txt）与图像文件同名

### 配置文件（data.yaml）
创建 data.yaml 文件，定义数据集路径和类别信息：
```yaml
path: ./dataset
train: images/train
val: images/val
names: ['cardboard']  # 类别名称列表
```


## 友情链接:
- [BRT Crowd Analysis System by Hassan Raza](https://github.com/hassanrrraza/crowd-analysis-yolo)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [探索Ultralytics YOLOv8 -Ultralytics YOLO 文档](https://docs.ultralytics.com/zh/models/yolov8/)
- [YOLO代码参考](https://docs.ultralytics.com/reference/cfg/__init__/)
- [roboflow下载数据集](https://public.roboflow.com/)