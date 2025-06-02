import os
import cv2
import atexit
import time
import queue
from flask import Flask, Response, render_template, jsonify, request
from src.PeopleCounter import PeopleCounter

app = Flask(__name__, template_folder='Web')

# 获取相对于脚本位置的路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "models", "best.pt")
video_path = os.path.join(base_dir, "test.mp4")
class_file = os.path.join(base_dir, "classes.txt")

# 初始化人群计数器
counter = PeopleCounter(model_path, video_path, class_file, threshold=40)

is_paused = False  # 控制视频播放/暂停状态


def generate_frames():
    """生成视频帧"""
    global is_paused
    last_frame_bytes = None

    # 启动帧读取线程
    if counter.read_thread is None or not counter.read_thread.is_alive():
        counter.run()  # 启动读取线程

    while True:
        if not is_paused:
            try:
                frame = counter.frame_queue.get(timeout=1)  # 从队列获取帧
            except queue.Empty:
                # 队列为空时，尝试重新启动读取线程或等待
                if not counter.read_thread.is_alive():
                    counter.run()  # 重新启动读取线程
                time.sleep(0.01)  # 短暂等待
                continue

            # 处理帧
            processed_frame = counter.process_frame(frame)

            # 将帧编码为JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            last_frame_bytes = frame_bytes
        else:
            # 如果暂停，并且有上一帧，则重复发送上一帧
            if last_frame_bytes:
                frame_bytes = last_frame_bytes
            else:
                time.sleep(0.1)  # 暂停时如果没有帧，则短暂等待
                continue

        # 以流的形式返回帧
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """视频流"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_stats')
def get_stats():
    """获取统计数据"""
    try:
        stats = counter.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/get_polygon')
def get_polygon():
    """获取多边形区域"""
    try:
        polygon = counter.get_polygon()
        return jsonify({"polygon": polygon})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/update_polygon', methods=['POST'])
def update_polygon():
    """更新多边形区域"""
    try:
        data = request.json
        polygon = data.get('polygon')
        if polygon and counter.update_polygon(polygon):
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "无效的多边形数据"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    global is_paused
    is_paused = not is_paused
    return jsonify({"is_paused": is_paused})


atexit.register(counter.stop)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
