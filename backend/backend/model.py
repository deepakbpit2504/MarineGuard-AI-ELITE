from ultralytics import YOLO

model = YOLO("models/best.pt")

def predict(img):
    results = model(img)

    output = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            output.append({
                "class": model.names[cls],
                "confidence": round(conf, 3)
            })

    return output
