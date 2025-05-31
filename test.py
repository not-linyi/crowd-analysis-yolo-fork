
if __name__ == '__main__':
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")  # 加载预训练模型
    model.train(data="data.yaml", workers=0, epochs=100,  batch=16)
