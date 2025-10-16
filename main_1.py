import os
import cv2
import time
import json
import numpy as np
from numpy.linalg import norm
from deepface import DeepFace
from detector import FaceDetector
from tracker import Tracker
from db import MongoDatabase
from logger import save_cropped_face, print_log
import cv2
import logging
import torch

# Use GPU if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# === Load config ===
config_path = r"C:\Users\Abdul\OneDrive\Desktop\YOLO_Project\face_tracking_1\config.json"
with open(config_path) as f:
    config = json.load(f)

log_dir = config.get("log_dir", "logs")
os.makedirs(log_dir, exist_ok=True)

# === Logging setup ===
logging.basicConfig(
    filename=os.path.join(log_dir, "events.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logging.info(f"Using device: {device}")
 
# === Init modules ===
detector = FaceDetector(config["model_path"], config["confidence_threshold"])
tracker = Tracker(max_age=config.get("deepsort_max_age", 30))
db = MongoDatabase(config["database_uri"], config["database_name"])

# Build DeepFace model (warm up once)
DeepFace.build_model("ArcFace")

# === Tracking state ===
active_faces = {}        # visitor_id -> {"bbox": [l,t,r,b], "embedding": np.ndarray, "last_seen": ts}
track_to_visitor = {}    # deep-sort track_id -> visitor_id
exit_counters = {}       # visitor_id -> consecutive frames missed
frame_embeddings = {}
partial_counters = {}           # visitor_id -> number of consecutive partial frames


EDGE_MARGIN = 8                 # pixels from border considered partial zone
PARTIAL_EXIT_THRESHOLD = 5      # number of consecutive partial frames before confirming exit
EXIT_FRAME_THRESHOLD = 10   # frames missed before confirming exit
DETECTION_SKIP = 5  # skip every 5th frame
SIMILARITY_THRESHOLD = config.get("recognition_threshold", 0.9)  # cosine similarity threshold

# === Video setup ===
if config.get("use_live_camera", False):
    print("[INFO] Using live webcam feed...")
    cap = cv2.VideoCapture(config.get("camera_index", 0))
else:
    print(f"[INFO] Using video file: {config['input_video_path']}")
    cap = cv2.VideoCapture(config["input_video_path"])


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
OUTPUT_PATH = config.get("output_video_path", "output_tracked_out.mp4")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))


frame_idx = 0
entry_count = 0
exit_count = 0

def log_event(visitor_id, event_type, img_path=None):
    msg = f"{visitor_id} | Event: {event_type}"
    if img_path:
        msg += f" | Image: {img_path}"
    logging.info(msg)


def is_partial_face(bbox, frame_w, frame_h, margin=EDGE_MARGIN):
    x1, y1, x2, y2 = bbox
    # Too close to border → partial
    if x1 <= margin or y1 <= margin or x2 >= frame_w - margin or y2 >= frame_h - margin:
        return True
    return False


def get_embedding(face_img):
    try:
        rep = DeepFace.represent(
            face_img,
            model_name="ArcFace",
            enforce_detection=False  # we already cropped
        )[0]
        emb = np.array(rep["embedding"], dtype=np.float32)
        # 3️⃣ Normalize embedding
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    except Exception as e:
        print(f"[Warning] Could not get embedding: {e}")
        return None


def cosine_sim(a, b):
    return float(np.dot(a, b))   # since both normalized


# === Main loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    frame_ts = time.time()

    # 1) Detect and track (your Detector/Tracker return types)
    detections = []
    if frame_idx % DETECTION_SKIP != 0:
        detections = detector.detect_faces(frame)
    print(f"[Frame {frame_idx}] Faces detected: {len(detections)}")

    tracks = tracker.update(detections, frame)         # deep-sort track objects
    print(f"[Frame {frame_idx}] Active tracks: {len(tracks)}")

    # Keep list of visitor_ids seen this frame
    seen_visitors = set()

    # 2) Process each returned track (one iteration per tracked face)
    for track in tracks:
        if not track.is_confirmed():
            continue

        # DeepSort gives left,top,right,bottom
        l, t, r, b = map(int, track.to_ltrb())
        l, t = max(0, l), max(0, t)
        r, b = min(frame.shape[1], r), min(frame.shape[0], b)
        if r <= l or b <= t or (b - t < 20) or (r - l < 20):
            continue

        track_id = track.track_id
        bbox = [l, t, r, b]
        is_partial = is_partial_face(bbox, width, height)

        if is_partial:
            visitor_id = track_to_visitor.get(track.track_id)
            if visitor_id:
                partial_counters[visitor_id] = partial_counters.get(visitor_id, 0) + 1
            continue  # skip further processing for this face
        else:
            # reset partial counter if now full again
            visitor_id = track_to_visitor.get(track.track_id)
            if visitor_id:
                partial_counters[visitor_id] = 0

        # Crop & resize for embedding
        face_crop = frame[t:b, l:r]
        if face_crop.size > 0:
            face_resized = cv2.resize(face_crop, (112, 112))            
            emb = get_embedding(face_resized) 
            if emb is None:
                continue

            # 3) If this track already mapped to a visitor, reuse it
            if track_id in track_to_visitor:
                visitor_id = track_to_visitor[track_id]
            else:
                # 4) Try to match embedding to any active visitor embeddings
                visitor_id = None
                for vid, info in active_faces.items():                
                    prev_emb = info.get("embedding")
                    if prev_emb is None:
                        continue
                    sim = cosine_sim(emb, prev_emb)
                    if sim >= SIMILARITY_THRESHOLD:
                        visitor_id = vid
                        log_event(visitor_id, "recognized")
                        break

                # 5) If no match, this is a new visitor
                if visitor_id is None:
                    visitor_id = f"visitor_{int(time.time()*1000)}"
                    print("  New visitor created")
                    entry_count += 1
                    img_path = save_cropped_face(face_resized, log_dir, "entry")
                    db.log_event(visitor_id, "entry", img_path)
                    log_event(visitor_id, "entry", img_path)
                    print_log("entry", visitor_id, img_path)

                # map this deep-sort track to the chosen visitor id
                track_to_visitor[track_id] = visitor_id

            # 6) Update active_faces & bookkeeping
            active_faces[visitor_id] = {
                                "bbox": [l, t, r, b],
                                "embedding": emb,
                                "last_seen": frame_idx,
                                "last_seen_ts": frame_ts,
                                "last_seen_frame": frame.copy(),  # store the frame with face
                                "last_seen_face": face_resized.copy()
                            }

            exit_counters[visitor_id] = 0
            seen_visitors.add(visitor_id)

            # 7) Draw on frame
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"{visitor_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # 8) EXIT detection: visitors not seen this frame -> increment counter
    for vid in list(active_faces.keys()):
    #  Increment exit counter if not seen this frame
        if vid not in seen_visitors:
            exit_counters[vid] = exit_counters.get(vid, 0) + 1
        else:
            exit_counters[vid] = 0

        #  Check partial counter for this visitor
        partial_count = partial_counters.get(vid, 0)

        #  First, handle normal exit (missed frames)
        if exit_counters.get(vid, 0) >= EXIT_FRAME_THRESHOLD:
            info = active_faces.get(vid)
            if info:
                face_resized = info.get("last_seen_face")
                if face_resized is not None and face_resized.size > 0:
                    img_path = save_cropped_face(face_resized, log_dir, "exit")
                    db.log_event(vid, "exit", img_path)
                    log_event(vid, "exit", img_path)
                    print_log("exit", vid, img_path)
                    exit_count += 1

            # Cleanup
            to_remove = [tid for tid, v in track_to_visitor.items() if v == vid]
            for tid in to_remove:
                track_to_visitor.pop(tid, None)
            active_faces.pop(vid, None)
            exit_counters.pop(vid, None)
            partial_counters.pop(vid, None)
            continue  # move to next visitor

        #  Handle partial-face exit (visitor still visible but mostly partial)
        if partial_count >= PARTIAL_EXIT_THRESHOLD:
            info = active_faces.get(vid)
            if info:
                # Use last full-face frame stored before it became partial
                face_resized = info.get("last_seen_face")
                if face_resized is not None and face_resized.size > 0:
                    img_path = save_cropped_face(face_resized, log_dir, "exit_partial")
                    db.log_event(vid, "exit_partial", img_path)
                    log_event(vid, "exit_partial", img_path)
                    print_log("exit_partial", vid, img_path)
                    exit_count += 1

            # Cleanup
            to_remove = [tid for tid, v in track_to_visitor.items() if v == vid]
            for tid in to_remove:
                track_to_visitor.pop(tid, None)
            active_faces.pop(vid, None)
            exit_counters.pop(vid, None)
            partial_counters.pop(vid, None)
            continue

    cv2.imshow("Live Feed", frame)
    # 9) save frame to output video
    out.write(frame)
    # === Break conditions ===
    if config.get("use_live_camera", False):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] User quit live feed.")
            break
    else:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Stopped video playback.")
            break

# After loop: handle faces still active at video end (log exit_end)
for vid, info in list(active_faces.items()):
    l, t, r, b = map(int, info["bbox"])
    l, t = max(0, l), max(0, t)
    r, b = min(width, r), min(height, b)
    if r > l and b > t:
        try:
            face_crop = frame[t:b, l:r]
            face_resized = cv2.resize(face_crop, (112, 112))
            img_path = save_cropped_face(face_resized, log_dir, "exit_end")
            db.log_event(vid, "exit_end", img_path)
            log_event(vid, "exit_end", img_path)
            print_log("exit_end", vid, img_path)
        except cv2.error:
            pass

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"🎥 Output saved: {OUTPUT_PATH}")
print(f"Total unique entries logged: {entry_count}")
print(f"Total exits logged: {exit_count}")

logging.info(f"Processing finished. Output: {OUTPUT_PATH}")
logging.info(f"Total unique entries logged: {entry_count}")
logging.info(f"Total exits logged: {exit_count}")