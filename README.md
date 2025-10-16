
# Intelligent Face Tracker with Auto Registration and Visitor Counting

This project is an AI-based face tracking and visitor counting system that detects, recognizes, and tracks individuals using live video or recorded footage. It automatically registers new visitors, logs entries/exits, and maintains event metadata for analytics.

## Features

- 🎯 **Real-time Face Detection** using YOLOv8.
- 🧠 **Automatic Registration** of new faces using DeepFace embeddings.
- 🔁 **Persistent Face Tracking** with DeepSORT tracker.
- 📊 **Visitor Counting & Logging** with entry/exit timestamps.
- 🗂️ **Structured Logs** and image snapshots for each detected face.
- ⚡ **Supports Live Camera** and pre-recorded video input.
- 🧾 **Configurable Parameters** through `config.json`.
- 🖥️ **Generates Annotated Output Video** with bounding boxes and IDs.


## Project Structure
```
face_tracking_1/
├── config.json              # Configuration file
├── db.py                    # Handles metadata and event database
├── detector.py              # YOLO-based face detection logic
├── logger.py                # Logging system for all events
├── main_1.py                # Main execution script
├── tracker.py               # DeepSORT-based tracking
├── logs/                    # Logs and cropped face entries/exits
├── outputs/                 # Processed output videos
│── requirements.txt         # Python dependencies
└── README.md
```
## Set Up Instructions

### 1️⃣ Clone the repository
```
git clone https://github.com/Farshiya-25/Intelligent-Face-Tracker-with-Auto-Registration-and-Visitor-Counting.git
cd face_tracking_1
```
### 2️⃣ Create a virtual environment
```
python -m venv venv
venv\Scripts\activate  # On Windows
```
### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```
### 4️⃣ Configure settings

Open config.json and update paths and options:
```
{
  "input_video_path": "input_video.mp4",
  "output_video_path": "outputs/output_tracked_out.mp4",
  "log_dir": "logs",
  "model_path": "yolov8n-face.pt",
  "confidence_threshold": 0.5,
  "use_live_camera": false,
  "camera_index": 0
}
```
Set "use_live_camera": true to use your webcam instead of a video file.
## Assumptions

- YOLOv8 face detector is used for robust face detection.

- DeepSORT tracker ensures consistent ID assignment across frames.

- ArcFace embedding model (via DeepFace) is used for recognition.

- All faces are resized to 112x112 pixels for uniform embedding.

- Logs and cropped faces are automatically organized under logs/.


## Architecture Diagram

         ┌──────────────────────────────┐
         │         Video Input          │
         │ (Live Camera / Pre-recorded) │
         └──────────────┬───────────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │   YOLO Face Detector │
             └──────────────────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ DeepSORT Tracker     │
             │ (Tracks IDs & Motion)│
             └──────────────────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ DeepFace Embedder    │
             │ (ArcFace Model)      │
             └──────────────────────┘
                        │
                        ▼
       ┌─────────────────────────────────────┐
       │ Database + Logger                   │
       │ - Entry & Exit logs                 │
       │ - Embeddings & Visitor Metadata     │
       └─────────────────────────────────────┘
                        │
                        ▼
              ┌──────────────────────┐
              │ Output Video & Logs  │
              └──────────────────────┘

## Demo

[![Watch Demo](https://img.shields.io/badge/🎥%20Watch-Demo-blue)](https://drive.google.com/file/d/1KJ5dFZlGU0OGvDQ3n8BCl-lAsYkqQ6yB/view?usp=sharing)



## Sample Outputs

### Logs Folder:

- /logs/entrys/2025-10-16/entry_1234.jpg

- /logs/exit_partials/2025-10-16/exit_partial_5678.jpg

### Database Entries (example):
```
{
  "_id": "ObjectId('68f0cfe4fd7c41cb5f2bf962')",
  "face_id": "visitor_1760612236042",
  "event_type": "exit_partial",
  "image_path": "logs\\exit_partials\\2025-10-16\\exit_partial_162844602900.jpg",
  "timestamp": "2025-10-16T16:28:44.604+00:00"
}

```

### Output Video:
```
outputs/output_tracked_out.mp4
```

Contains bounding boxes, IDs, and live visitor tracking overlays.
## Future Enhancements

- Add real-time dashboard for visitor analytics.

- Integrate cloud-based face recognition database.

- Improve handling of partial/occluded faces.
## Conclusion

This project demonstrates a real-time intelligent face tracking system that automatically detects, registers, and counts visitors while maintaining detailed logs and metadata.
## Credits

This project is a part of a hackathon run by https://katomaran.com
