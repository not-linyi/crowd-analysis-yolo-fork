import cv2
from ultralytics import YOLO

if __name__ == "__main__":

    # 加载模型并推理
    model = YOLO('models/best.pt')
    results = model('img.png', save=True, conf=0.25)

    # 获取保存后的图像路径
    save_path = results[0].save_dir + '\\img.jpg'

    # 使用 OpenCV 加载并显示图像
    img = cv2.imread(str(save_path))
    cv2.imshow('Detection Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
