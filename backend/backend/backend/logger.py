import json
from datetime import datetime

def log_event(data):
    event = {
        "timestamp": str(datetime.now()),
        "predictions": data
    }

    with open("logs/events.json", "a") as f:
        json.dump(event, f, indent=4)
