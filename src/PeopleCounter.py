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

        # 颜色方案
        self.colors = {
            'normal': (0, 255, 0),  # 绿色
            'warning': (0, 255, 0),  # 浅红色
            'critical': (0, 0, 255),  # 深红色
            'text_bg': (44, 44, 44),  # 深灰色
            'accent': (255, 204, 0),  # 强调色 (金色)
            'border': (180, 180, 180),  # 边框颜色
            'polygon': (0, 140, 255),  # 多边形颜色
            'grid': (50, 50, 50),  # 网格颜色
        }

    def draw_detection_boxes(self, frame, detections):
        """
        绘制检测框

        参数:
            frame (numpy.ndarray): 输入帧
            detections (list): 检测结果列表
        """
    def draw_detections(self, detections, frame):
        """
        在图像帧上绘制检测框及其角部装饰线

        参数:
            detections (list): 检测结果列表，每个元素包含检测框坐标(x1,y1,x2,y2)、置信度和类别
            frame (np.ndarray): 待绘制的图像帧（BGR格式）

        返回值:
            None: 直接修改输入的frame参数
        """

        # 遍历所有检测结果
        for detection in detections:
            # 解析检测框参数
            x1, y1, x2, y2, conf, cls = detection

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # 设置绘制样式
            color = self.colors['normal']
            w, h = x2 - x1, y2 - y1
            thickness = 1

            # 绘制检测框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # 计算装饰线长度
            corner_length = int(min(20, w // 4, h // 4))

            # 绘制左上角装饰线
            cv2.line(frame, (x1-2, y1-2), (x1 + corner_length-2, y1-2), color, thickness)
            cv2.line(frame, (x1-2, y1-2), (x1-2, y1 + corner_length-2), color, thickness)

            # 绘制右上角装饰线
            cv2.line(frame, (x1+2 + w, y1-2), (x1+2 + w - corner_length, y1-2), color, thickness)
            cv2.line(frame, (x1+2 + w, y1-2), (x1+2 + w, y1-2 + corner_length), color, thickness)

            # 绘制左下角装饰线
            cv2.line(frame, (x1-2, y1+2 + h), (x1-2 + corner_length, y1+2 + h), color, thickness)
            cv2.line(frame, (x1-2, y1+2 + h), (x1-2, y1+2 + h - corner_length), color, thickness)

            # 绘制右下角装饰线
            cv2.line(frame, (x1+2 + w, y1+2 + h), (x1+2 + w - corner_length, y1+2 + h), color, thickness)
            cv2.line(frame, (x1+2 + w, y1+2 + h), (x1+2 + w, y1+2 + h - corner_length), color, thickness)

    def run(self, display=True, skip_frames=3):
        """
        运行人群计数器

        参数:
            display (bool): 是否显示检测结果
            skip_frames (int): 跳帧数
        """
        if display:
            cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detection Result", 1280, 720)

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

            if display:
                # 首先创建干净的帧
                display_frame = frame.copy()
                # 绘制多边形
                cv2.polylines(display_frame, [np.array(self.area1, np.int32)], True, self.colors['polygon'], 2)

                # 添加透明叠加层以突出显示计数区域
                overlay = display_frame.copy()
                cv2.fillPoly(overlay, [np.array(self.area1, np.int32)], (100, 100, 100))
                cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0, display_frame)

                # 绘制检测框
                self.draw_detections(detection_boxes, display_frame)

                # 显示
                cv2.imshow("Crowd Analysis", display_frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC键退出
                    break

        cap.release()

        if display:
            cv2.destroyAllWindows()

        return frame_counts


if __name__ == "__main__":
    # 获取相对于脚本位置的路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "best.pt")
    video_path = os.path.join(base_dir, "test.mp4")
    class_file = os.path.join(base_dir, "classes.txt")

    counter = PeopleCounter(model_path, video_path, class_file, threshold=40)
    counter.run(display=True)
