
import time
from datetime import datetime
import threading
import queue

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


class PeopleCounter:

    def __init__(self, model_path, video_path=None, class_file=None, threshold=40):
        self.frame_queue = queue.Queue(maxsize=5)  # 帧队列，限制大小以避免内存溢出
        self.stop_event = threading.Event()  # 停止事件
        self.read_thread = None  # 读取线程
        """
        初始化人群计数器

        参数:
            model_path (str): YOLO模型文件路径
            video_path (str, optional): 分析视频文件路径
            class_file (str, optional): 类别名称文件路径
            threshold (int): 人群数量阈值
        """
        self.model = YOLO(model_path)
        self.video_path = video_path

        # 阈值
        self.threshold = threshold
        self.start_time = datetime.now()
        self.frame_count = 0
        self.max_count = 0

        # 帧率计算
        self.fps_list = []
        self.fps = 0
        self.last_time = time.time()

        # 最近100帧的检测历史
        self.count_history = []
        self.max_history_length = 100

        # 绘图相关
        self.drawing = False
        self.current_point_index = -1
        self.display_frame = None

        # 读取类别名称
        if class_file:
            with open(class_file, "r") as my_file:
                data = my_file.read()
                self.class_list = data.split("\n")
        else:
            self.class_list = ['person']

        # 定义计数的多边形区域
        self.area1 = [(827, 155), (1261, 457), (774, 695), (610, 193)]

        # 颜色方案
        self.colors = {
            'normal': (0, 255, 0),  # 绿色
            'warning': (255, 165, 0),  # 橙色
            'critical': (0, 0, 255),  # 红色
            'text_bg': (44, 44, 44),  # 深灰色
            'accent': (255, 204, 0),  # 强调色 (金色)
            'border': (180, 180, 180),  # 边框颜色
            'polygon': (0, 140, 255),  # 多边形颜色
            'grid': (50, 50, 50),  # 网格颜色
        }

    def draw_detection_boxes(self, frame, detections):
        """绘制改进后的检测框"""
        for (x1, y1, w, h) in detections:
            # 改进检测框视觉效果 - 使用更细的线条
            color = self.colors['normal']
            thickness = 1  # 厚度为1

            # 使用更细的样式绘制主矩形
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, thickness)

            # 计算装饰线长度
            corner_length = int(min(w // 4, h // 4))

            if corner_length > 20:
                return

            # 绘制左上角装饰线
            cv2.line(frame, (x1 - 2, y1 - 2), (x1 + corner_length - 2, y1 - 2), color, thickness)
            cv2.line(frame, (x1 - 2, y1 - 2), (x1 - 2, y1 + corner_length - 2), color, thickness)

            # 绘制右上角装饰线
            cv2.line(frame, (x1 + 2 + w, y1 - 2), (x1 + 2 + w - corner_length, y1 - 2), color, thickness)
            cv2.line(frame, (x1 + 2 + w, y1 - 2), (x1 + 2 + w, y1 - 2 + corner_length), color, thickness)

            # 绘制左下角装饰线
            cv2.line(frame, (x1 - 2, y1 + 2 + h), (x1 - 2 + corner_length, y1 + 2 + h), color, thickness)
            cv2.line(frame, (x1 - 2, y1 + 2 + h), (x1 - 2, y1 + 2 + h - corner_length), color, thickness)

            # 绘制右下角装饰线
            cv2.line(frame, (x1 + 2 + w, y1 + 2 + h), (x1 + 2 + w - corner_length, y1 + 2 + h), color, thickness)
            cv2.line(frame, (x1 + 2 + w, y1 + 2 + h), (x1 + 2 + w, y1 + 2 + h - corner_length), color, thickness)

    def process_frame(self, frame):
        """
        处理单个视频帧

        参数:
            frame: 输入的视频帧

        返回:
            处理后的视频帧
        """
        # 计算FPS
        current_time = time.time()
        time_diff = current_time - self.last_time
        if time_diff > 0:
            self.fps = 1 / time_diff
        self.last_time = current_time
        self.fps_list.append(self.fps)
        if len(self.fps_list) > 30:  # 保留最近30帧的FPS
            self.fps_list.pop(0)
        avg_fps = sum(self.fps_list) / len(self.fps_list)
        self.fps = round(avg_fps, 1)

        # 获取检测结果
        results = self.model.predict(frame)
        # 获取检测框
        a = results[0].boxes.data
        # 将检测结果转换为DataFrame并转换为浮点型
        px = pd.DataFrame(a).astype(float)
        detection_boxes = []

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2
            w, h = x2 - x1, y2 - y1

            # 检查中心点是否在多边形区域内
            result = cv2.pointPolygonTest(np.array(self.area1, np.int32), (cx, cy), False)
            # 如果结果大于等于0，则表示中心点在多边形区域内
            if result >= 0:
                detection_boxes.append((x1, y1, w, h))

        current_count = len(detection_boxes)
        self.count_history.append(current_count)
        if len(self.count_history) > self.max_history_length:
            self.count_history.pop(0)
        self.frame_count += 1

        # 更新最大计数
        if current_count > self.max_count:
            self.max_count = current_count

        # 创建干净的帧
        display_frame = frame.copy()
        self.display_frame = display_frame

        # 绘制多边形
        cv2.polylines(display_frame, [np.array(self.area1, np.int32)], True, self.colors['polygon'], 2)

        # 添加透明叠加层以突出显示计数区域
        overlay = display_frame.copy()
        cv2.fillPoly(overlay, [np.array(self.area1, np.int32)], (100, 100, 100))
        cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0, display_frame)

        # 绘制改进的检测框
        self.draw_detection_boxes(display_frame, detection_boxes)

        return display_frame

    def _read_frames(self, skip_frames=3):
        if not self.video_path:
            raise ValueError("视频路径未设置")

        cap = cv2.VideoCapture(self.video_path)
        count = 0

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            count += 1
            if count % skip_frames != 0:
                continue

            frame = cv2.resize(frame, (1280, 720))
            try:
                self.frame_queue.put(frame, timeout=1)  # 放入队列，设置超时
            except queue.Full:
                continue  # 队列满时跳过当前帧

        cap.release()

    def run(self, skip_frames=3):
        self.stop_event.clear()
        self.read_thread = threading.Thread(target=self._read_frames, args=(skip_frames,))
        self.read_thread.start()

        # 存储每帧处理的人数统计
        frame_counts = []

        return frame_counts

    def get_stats(self):
        """
        获取当前统计数据
        
        返回:
            dict: 包含当前统计数据的字典
        """
        # 计算运行时间
        runtime = datetime.now() - self.start_time
        hours, remainder = divmod(runtime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        runtime_str = f"{hours:02}:{minutes:02}:{seconds:02}"

        # 计算平均人数
        current_count = len(self.count_history) > 0 and self.count_history[-1] or 0
        avg_count = round(sum(self.count_history) / len(self.count_history), 1) if self.count_history else 0

        return {
            "current_count": current_count,
            "max_count": self.max_count,
            "avg_count": avg_count,
            "fps": self.fps,
            "runtime": runtime_str,
            "threshold": self.threshold,
            "threshold_percentage": min(round((current_count / self.threshold) * 100), 100) if self.threshold > 0 else 0
        }

    def update_polygon(self, polygon):
        """
        更新多边形区域
        
        参数:
            polygon (list): 新的多边形顶点列表
        """
        if len(polygon) >= 3:  # 确保至少有3个点形成多边形
            self.area1 = polygon
            return True
        return False

    def get_polygon(self):
        """
        获取当前多边形区域
        
        返回:
            list: 多边形顶点列表
        """
        return self.area1

    def stop(self):
        self.stop_event.set()
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join()
