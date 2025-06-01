import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


class PeopleCounter:

    def __init__(self, model_path, video_path, class_file, threshold=40):
        """
        初始化人群计数器

        参数:
            model_path (str): YOLO模型文件路径
            video_path (str): 分析视频文件路径
            class_file (str): 类别名称文件路径
            threshold (int): 人群数量阈值
        """

        self.model = YOLO(model_path)
        self.video_path = video_path

        # 阈值
        self.threshold = threshold
        self.frame_count = 0

        # 读取类别名称
        with open(class_file, "r") as my_file:
            data = my_file.read()
            self.class_list = data.split("\n")

            # 定义计数的多边形区域
        self.area1 = [(827, 155), (1261, 457), (774, 695), (610, 193)]

    def run(self, display=True, skip_frames=3):
        """
        运行人群计数器

        参数:
            display (bool): 是否显示检测结果
            skip_frames (int): 跳帧数
        """
        # 打开视频文件
        cap = cv2.VideoCapture(self.video_path)

        count = 0
        # 存储每帧处理的人数统计
        frame_counts = []

        while True:
            # 读取下一帧
            ret, frame = cap.read()

            if not ret:
                break

            count += 1

            if count % skip_frames != 0:
                continue

            # 缩放帧以适应窗口
            frame = cv2.resize(frame, (1280, 720))

            results = self.model.predict(frame)

            px = pd.DataFrame(results[0].boxes.data).astype(float)
            # 获取检测框
            detection_boxes = []

            for i in range(len(px)):
                x1, y1, x2, y2, conf, cls = px.iloc[i].values.tolist()

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                result = cv2.pointPolygonTest(np.array(self.area1, np.int32), (cx, cy), False)

                if result >= 0:
                    detection_boxes.append([x1, y1, x2, y2, conf, cls])

            current_count = len(detection_boxes)
            frame_counts.append(current_count)
            self.frame_count += 1

        cap.release()

        return frame_counts


if __name__ == "__main__":
    # 获取相对于脚本位置的路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "best.pt")
    video_path = os.path.join(base_dir, "test.mp4")
    class_file = os.path.join(base_dir, "classes.txt")

    counter = PeopleCounter(model_path, video_path, class_file, threshold=40)
    counter.run(display=True)
