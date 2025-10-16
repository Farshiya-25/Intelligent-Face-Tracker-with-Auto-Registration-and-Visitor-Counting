import os, cv2
from datetime import datetime

def save_cropped_face(face_img, log_dir, event_type):
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder = os.path.join(log_dir, f"{event_type}s", date_str)
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S%f")
    img_path = os.path.join(folder, f"{event_type}_{timestamp}.jpg")
    cv2.imwrite(img_path, face_img)
    return img_path

def print_log(event_type, face_id, img_path):
    print(f"[{event_type.upper()}] FaceID: {face_id} - Image: {img_path}")
