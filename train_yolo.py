from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")
    model.train(
        data="nailong_detection/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=2,
        device=0,
        project="nailong_detection/runs",
        name="yolov8n_v2",
    )
