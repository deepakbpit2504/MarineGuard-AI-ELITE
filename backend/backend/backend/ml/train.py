from ultralytics import YOLO

def train():
    model = YOLO("yolov8n.pt")

    model.train(
        data="config/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="marine_guard_ai"
    )

if __name__ == "__main__":
    train()
