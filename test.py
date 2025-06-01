from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("yolov8n.pt")  # 加载预训练模型
    model.train(data="data.yaml",
                workers=0,
                epochs=100,
                project='runs/test',  # 指定输出文件夹
                batch=16)


