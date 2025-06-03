import os
import time

import cv2
import cvzone
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime


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

        # 帧率计算
        self.max_count = 0
        self.start_time = datetime.now()
        self.fps_list = []
        self.fps = 0
        self.last_time = time.time()

        # 最近100帧的检测历史
        self.count_history = []
        self.max_history_length = 100

        # 鼠标事件相关
        self.drawing = False  # 是否正在拖动
        self.current_point_index = -1  # 当前拖动的顶点索引
        self.display_frame = None  # 用于鼠标回调的帧

    def mouse_callback(self, event, x, y, flags, param):
        """
        鼠标回调函数，用于处理鼠标事件。

        参数:
        - event: OpenCV传递的鼠标事件类型，如点击、移动等。
        - x, y: 鼠标事件发生的坐标位置。
        - flags: 鼠标事件的标志，保留参数，未使用。
        - param: 用户定义的传递到回调函数的参数，未使用。

        返回值: 无

        本函数主要用于响应鼠标事件，以实现对特定区域顶点的拖动功能。
        """
        # 检查是否有显示帧，如果没有则直接返回，不做任何操作。
        if self.display_frame is None:
            return

        # 当检测到鼠标左键按下事件时，检查是否点击在区域1的某个顶点附近。
        if event == cv2.EVENT_LBUTTONDOWN:
            # 遍历区域1的所有顶点。
            for i, point in enumerate(self.area1):
                # 如果鼠标点击位置接近某个顶点（误差范围内），则开始记录拖动操作。
                if abs(x - point[0]) < 10 and abs(y - point[1]) < 10:
                    self.drawing = True
                    self.current_point_index = i
                    break
        # 当鼠标移动时，如果正在拖动某个顶点，则更新该顶点的位置。
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.area1[self.current_point_index] = (x, y)
        # 当检测到鼠标左键释放事件时，结束拖动操作，并重置当前顶点索引。
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_point_index = -1

    def draw_detections(self, detections, frame):
        """
        在图像帧上绘制检测框及其角装饰线

        参数:
            detections (list): 检测结果列表，每个元素包含检测框坐标(x1,y1,x2,y2)、置信度和类别
            frame (np.ndarray): 待绘制的图像帧

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

    def draw_header(self, frame):
        """绘制包含标题和基本信息的页眉"""
        # 绘制页眉背景
        header_height = 50
        cv2.rectangle(frame, (0, 0), (frame.shape[1], header_height), (30, 30, 30), -1)
        cv2.line(frame, (0, header_height), (frame.shape[1], header_height), self.colors['accent'], 2)

        # 添加标题
        title = "BRT CROWD ANALYSIS SYSTEM"
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        title_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, title, (title_x, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['accent'], 2)

    def draw_counting_bar(self, frame, current_count):
        """
        绘制人群计数进度条

        参数:
            frame (numpy.ndarray): 输入帧
            current_count (int): 当前计数值
            threshold (int): 人群数量阈值
        """
        # 更新样式
        bar_width = 200
        bar_height = 25
        padding = 20
        bar_x = padding
        bar_y = padding

        # 计算进度条应填充的程度
        percentage = current_count / self.threshold
        # 计算进度条应填充的宽度
        filled_width = min(int(percentage * bar_width), bar_width)

        # 根据百分比确定颜色
        status, color = self.get_crowd_status(current_count)

        # 绘制背景
        cv2.rectangle(frame, (bar_x - 5, bar_y - 5), (bar_x + bar_width + 5, bar_y + bar_height + 5),
                      self.colors['border'], -1)

        # 绘制空进度条
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                      (50, 50, 50), -1)

        # 绘制填充部分
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height),
                      color, -1)

        # 绘制边框
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                      self.colors['border'], 1)

        # 添加带背景的文本
        label = f'COUNT: {current_count}'
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = bar_x + 10
        text_y = bar_y + bar_height // 2 + text_size[1] // 2

        cvzone.putTextRect(frame, label, (text_x, text_y),
                           colorR=self.colors['text_bg'],
                           colorT=(255, 255, 255),
                           font=cv2.FONT_HERSHEY_SIMPLEX,
                           scale=0.7, thickness=2,
                           offset=0)

    def get_crowd_status(self, current_count):
        """根据当前人数和阈值确定人群状态"""
        percentage = (current_count / self.threshold) * 100

        if percentage < 60:
            return "NORMAL", self.colors['normal']
        elif percentage < 90:
            return "WARNING", self.colors['warning']
        else:
            return "CRITICAL", self.colors['critical']

    def draw_threshold_bar(self, frame, current_count):
        """
        绘制阈值进度条
        参数:
            frame (numpy.ndarray): 输入帧
            current_count (int): 当前计数值
        """
        bar_width = 250
        bar_height = 25
        padding = 20
        bar_x = frame.shape[1] - bar_width - padding
        bar_y = padding

        percentage = min((current_count / self.threshold), 1.0)
        percentage_display = min(int(percentage * 100), 100)
        filled_width = min(int(percentage * bar_width), bar_width)

        status, color = self.get_crowd_status(current_count)

        # 绘制阈值显示
        # 绘制背景
        cv2.rectangle(frame, (bar_x - 5, bar_y - 5), (bar_x + bar_width + 5, bar_y + bar_height + 5),
                      self.colors['border'], -1)

        # 绘制空进度条
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                      (50, 50, 50), -1)

        # 绘制填充部分
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height),
                      color, -1)

        # 绘制边框
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                      self.colors['border'], 1)

        # 添加带背景的文本
        label = f'THRESHOLD: {percentage_display}%'
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = bar_x + bar_width - text_size[0] - 10
        text_y = bar_y + bar_height // 2 + text_size[1] // 2

        cvzone.putTextRect(frame, label, (text_x, text_y),
                           colorR=self.colors['text_bg'],
                           colorT=(255, 255, 255),
                           font=cv2.FONT_HERSHEY_SIMPLEX,
                           scale=0.7, thickness=2,
                           offset=0)

    def calculate_fps(self):
        """
        计算并平滑处理每秒帧数(FPS)

        该方法通过计算当前帧与上一帧的时间间隔推导瞬时FPS，并维护最近10帧的FPS数据
        进行移动平均计算，使结果更平滑稳定。

        Returns:
            int: 经过平滑处理后的平均FPS整数值
        """
        # 获取当前时间
        current_time = time.time()
        #  计算时间间隔
        time_diff = current_time - self.last_time

        # 仅当时间间隔有效时进行FPS计算
        if time_diff > 0:
            # 计算当前瞬时FPS并加入列表
            fps = 1 / time_diff
            self.fps_list.append(fps)

            # 维护固定长度的FPS队列，移除最旧数据保持队列长度
            if len(self.fps_list) > 10:
                self.fps_list.pop(0)

            # 计算移动平均FPS值
            self.fps = sum(self.fps_list) / len(self.fps_list)

        # 更新最后记录时间戳并返回处理后的FPS值
        self.last_time = current_time
        return int(self.fps)

    def draw_statistics_panel(self, frame, current_count):
        """
        绘制统计面板，包括当前计数、最大计数、平均计数和FPS。

        参数:
            frame (numpy.ndarray): 输入帧
            current_count (int): 当前计数值
        """
        panel_width = 250
        panel_height = 180
        panel_x = 20
        panel_y = frame.shape[0] - panel_height - 40  # 上移以避免与页脚重叠

        # 更新最大计数值
        self.max_count = max(self.max_count, current_count)

        # 更新计数历史
        self.count_history.append(current_count)
        if len(self.count_history) > self.max_history_length:
            self.count_history.pop(0)

        avg_count = sum(self.count_history) / len(self.count_history) if self.count_history else 0

        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 绘制边框
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                      self.colors['accent'], 1)

        # 添加标题
        title_x = panel_x + 10
        title_y = panel_y + 30
        cv2.putText(frame, "STATISTICS", (title_x, title_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['accent'], 2)

        # 添加水平线
        cv2.line(frame, (panel_x + 10, title_y + 10), (panel_x + panel_width - 10, title_y + 10),
                 self.colors['accent'], 1)

        # 添加统计数据
        stats_y_start = title_y + 40
        stats_x = panel_x + 15
        line_height = 25  # 减小行高以更好地适应

        stats = [
            f"Current Count: {current_count}",
            f"Maximum Count: {self.max_count}",
            f"Average Count: {int(avg_count)}",
            f"FPS: {self.calculate_fps()}",
            f"Runtime: {str(datetime.now() - self.start_time).split('.')[0]}"
        ]

        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (stats_x, stats_y_start + i * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def draw_mini_graph(self, frame):
        """
        绘制迷你图表，显示当前计数历史。

        参数:
            frame (numpy.ndarray): 输入帧
        """
        if not self.count_history:
            return

        graph_width = 200
        graph_height = 100
        graph_x = frame.shape[1] - graph_width - 20
        graph_y = frame.shape[0] - graph_height - 30  # 上移以避免与页脚重叠

        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 绘制边框
        cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height),
                      self.colors['accent'], 1)

        # 添加标题 with more space below it
        title_x = graph_x + 10
        title_y = graph_y + 23  # 略微上移
        cv2.putText(frame, "COUNT HISTORY", (title_x, title_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['accent'], 1)

        # Adjust the graph area to start below the title
        graph_area_start_y = title_y + 15  # 标题后添加填充
        usable_height = graph_height - (graph_area_start_y - graph_y)

        # 绘制调整高度后的网格
        max_val = max(max(self.count_history), self.threshold)
        grid_step_y = usable_height / 4

        for i in range(1, 4):
            y_pos = int(graph_y + graph_height - i * grid_step_y)
            cv2.line(frame, (graph_x + 25, y_pos), (graph_x + graph_width, y_pos),
                     self.colors['grid'], 1)
            val = int((i / 4) * max_val)
            cv2.putText(frame, str(val), (graph_x + 5, y_pos + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # 绘制调整坐标后的图表
        points = []
        history_to_plot = self.count_history[-min(len(self.count_history), graph_width):]

        for i, count in enumerate(history_to_plot):
            x = graph_x + 25 + i * ((graph_width - 25) / len(history_to_plot))  # 调整x轴起点至标签之后
            y = graph_y + graph_height - (count / max_val) * usable_height
            points.append((int(x), int(y)))

        # 绘制折线图
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1], points[i], self.colors['accent'], 2)

    def draw_footer(self, display_frame):
        # 添加包含版权和时间戳信息的页脚
        footer_y = display_frame.shape[0] - 15  # 页脚略微提高
        # 左对齐版权信息
        copyright_text = "Note: BRT footage for educational use only. All rights reserved."
        cv2.putText(display_frame, copyright_text,
                    (20, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        # 右对齐时间戳
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_size = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        time_x = display_frame.shape[1] - time_size[0] - 20
        cv2.putText(display_frame, time_str,
                    (time_x, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def run(self, display=True, skip_frames=3):
        """
        运行人群计数器

        参数:
            display (bool): 是否显示检测结果
            skip_frames (int): 跳帧数
        """
        if display:
            cv2.namedWindow("Crowd Analysis", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Crowd Analysis", 1280, 720)
            cv2.setMouseCallback('Crowd Analysis', self.mouse_callback)

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
                self.display_frame = display_frame

                # 绘制多边形
                cv2.polylines(display_frame, [np.array(self.area1, np.int32)], True, self.colors['polygon'], 2)

                # 绘制多边形顶点
                for point in self.area1:
                    cv2.circle(display_frame, point, 5, self.colors['accent'], -1)

                # 添加透明叠加层以突出显示计数区域
                overlay = display_frame.copy()
                cv2.fillPoly(overlay, [np.array(self.area1, np.int32)], (100, 100, 100))
                cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0, display_frame)

                # 绘制检测框
                self.draw_detections(detection_boxes, display_frame)

                # 绘制页眉
                self.draw_header(display_frame)

                # 绘制进度条
                self.draw_counting_bar(display_frame, current_count)
                self.draw_threshold_bar(display_frame, current_count)
                self.draw_statistics_panel(display_frame, current_count)
                self.draw_mini_graph(display_frame)

                self.draw_footer(display_frame)

                # 显示
                cv2.imshow("Crowd Analysis", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键退出
                    break
                elif key == 32:  # 空格键暂停
                    while True:
                        key2 = cv2.waitKey(0) & 0xFF
                        if key2 == 32:  # 再次按空格继续
                            break
                        elif key2 == 27:  # 按ESC直接退出
                            cap.release()
                            cv2.destroyAllWindows()
                            return frame_counts

        cap.release()

        if display:
            cv2.destroyAllWindows()

        return frame_counts


if __name__ == "__main__":
    import argparse

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='人群计数与分析系统')
    parser.add_argument('--video', type=str, help='视频文件路径', default=None)
    parser.add_argument('--model', type=str, help='模型文件路径', default=None)
    parser.add_argument('--classes', type=str, help='类别文件路径', default=None)
    parser.add_argument('--threshold', type=int, help='人群数量阈值', default=40)
    parser.add_argument('--skip', type=int, help='跳帧数', default=3)
    args = parser.parse_args()

    # 获取相对于脚本位置的路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 使用命令行参数或默认值
    model_path = args.model if args.model else os.path.join(base_dir, "models", "best.pt")
    video_path = args.video if args.video else os.path.join(base_dir, "test.mp4")
    class_file = args.classes if args.classes else os.path.join(base_dir, "classes.txt")

    print(f"启动人群计数与分析系统...")
    print(f"模型路径: {model_path}")
    print(f"视频路径: {video_path}")
    print(f"类别文件: {class_file}")
    print(f"人群阈值: {args.threshold}")
    print(f"跳帧数: {args.skip}")

    counter = PeopleCounter(
        model_path=model_path,
        video_path=video_path,
        class_file=class_file,
        threshold=args.threshold,
    )

    # 运行
    counter.run(display=True, skip_frames=args.skip)
