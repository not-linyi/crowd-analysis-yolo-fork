import argparse
import os
import time

import cv2


def ExtractFrames(video_path, output_dir, max_frames=200, interval=0.01, resize=(1080, 500)):
    """
    从视频文件中提取帧

    <修改> 参数:
        video_path (str): 视频文件路径
        output_dir (str): 保存提取帧的输出目录
        max_frames (int): 最大提取帧数
        interval (float): 捕获帧之间的时间间隔（秒）
        resize (tuple): 帧调整尺寸（宽度，高度）

    <修改> 返回:
        int: 成功提取的帧数量
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件，{video_path}")
        return 0

    frame_count = 0

    while frame_count < max_frames:
        # 读取下一帧
        ret, frame = cap.read()

        if not ret:
            print(f"帧已读取完毕，已提取{frame_count}帧")
            break

        frame = cv2.resize(frame, resize)

        # 保存帧
        output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(output_path, frame)

        cv2.imshow("Extracting Frames", frame)

        # 按ESC键退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_count += 1
        time.sleep(interval)

    # 释放资源
    cv2.destroyAllWindows()
    cap.release()

    print(f"成功提取{frame_count}帧, 视频路径: {video_path}, 输出路径: {output_dir}")
    return frame_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从视频文件中提取帧")

    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    parser.add_argument("--output", type=str, required=True, help="Directory to save extracted frames")
    parser.add_argument("--max-frames", type=int, default=200, help="Maximum number of frames to extract")
    parser.add_argument("--interval", type=float, default=0.01, help="Time interval between frame captures in seconds")
    parser.add_argument("--width", type=int, default=1080, help="Width to resize frames to")
    parser.add_argument("--height", type=int, default=500, help="Height to resize frames to")

    args = parser.parse_args()

    ExtractFrames(args.video,
                  args.output,
                  args.max_frames,
                  args.interval,
                  (args.width, args.height))
