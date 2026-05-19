from ultralytics import YOLO

model = YOLO("models/best.pt")

model.export(format="onnx")
model.export(format="torchscript")

print("Model exported successfully")
