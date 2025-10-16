from pymongo import MongoClient
from datetime import datetime

class MongoDatabase:
    def __init__(self, uri, db_name):
        client = MongoClient(uri)
        self.db = client[db_name]
        self.faces = self.db["faces"]
        self.events = self.db["events"]

    def log_event(self, face_id, event_type, image_path):
        self.events.insert_one({
            "face_id": face_id,
            "event_type": event_type,
            "image_path": image_path,
            "timestamp": datetime.now()
        })
