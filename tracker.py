from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self, max_age=30):
        self.tracker = DeepSort(max_age=max_age)

    def update(self, detections, frame):
        deepsort_detections = []
        for (x1, y1, x2, y2, conf) in detections:
            w, h = x2 - x1, y2 - y1
            deepsort_detections.append(([x1, y1, w, h], conf, "face"))
        return self.tracker.update_tracks(deepsort_detections, frame=frame)
