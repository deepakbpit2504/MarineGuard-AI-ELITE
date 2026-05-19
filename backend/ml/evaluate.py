from ultralytics import YOLO

model = YOLO("models/best.pt")

metrics = model.val()

print("mAP@50:", metrics.box.map50)
print("mAP@50-95:", metrics.box.map)
