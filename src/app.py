import os
import cv2
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

# 视频捕获对象
cap = None

is_paused = False  # 控制视频播放/暂停状态


def init_video():
    """初始化视频捕获"""
    global cap
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(video_path)
    return cap


def generate_frames():
    """生成视频帧"""
    global cap, is_paused
    if cap is None or not cap.isOpened():
        cap = init_video()

    skip_frames = 3
    frame_count = 0
    last_frame_bytes = None

    while True:
        if not is_paused:
            success, frame = cap.read()
            if not success:
                # 视频结束，重新开始
                cap = init_video()
                continue

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            # 调整帧大小
            frame = cv2.resize(frame, (1280, 720))

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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
