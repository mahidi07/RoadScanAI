# RoadScanAI: Road Surface Damage Detection (Images + Videos) using YOLOv8

RoadScanAI is a Wagtail/Django web application that performs automated road-surface damage detection (e.g., potholes, cracks, rutting) on **uploaded images and dashcam videos** using a **trained YOLOv8 model**. The app supports an end-to-end workflow: **upload media → run inference → generate annotated outputs → preview results in the browser**. View the live demo: https://drive.google.com/file/d/1bR509Eg19WeDMLGIUMIxXjb3vRPRCrZe/view?usp=sharing

---

## Features

### Image Damage Detection
- Upload one or multiple road images via the web UI
- Click **Start** to run YOLO inference over all uploaded images
- Produces **annotated images** with bounding boxes + class/confidence labels
- Displays raw and result images in a **tabbed carousel** interface

### Video Damage Detection
- Upload dashcam-style videos via the web UI
- Click **Start** to run frame-by-frame YOLO inference
- Produces a fully **annotated output video** (same FPS/resolution as input)
- Displays uploaded videos and processed videos in the UI (raw vs result tabs)

---

## Tech Stack
- **Backend:** Django + Wagtail CMS
- **Detection model:** Ultralytics **YOLOv8** (`ultralytics`)
- **Media processing:** OpenCV (`cv2`)
- **Storage:** Local filesystem (Django `MEDIA_ROOT`)
- **Frontend:** Wagtail templates + Bootstrap (tabs/carousels)

---

## How It Works (Technical Overview)

### Model loading
The YOLO model is loaded **once at import time** (not per request) to avoid re-initialization overhead:

- Weights file: `weights/best.pt`
- Loaded using:
  - `yolo_model = YOLO(str(MODEL_PATH))`

---

## Media Pipeline (Folders + Logs)

The app uses a simple and reliable “folder + text log” mechanism so templates can display media without complex database models. When media is uploaded or processed, its **public URL** is written into a `.txt` file. The template reads these lists and renders carousels/tabs automatically.

### Image pipeline

**Uploads**
- Saved to: `media/uploadedPics/`
- Upload log: `media/uploadedPics/img_list.txt`  
  Each line contains a public URL such as:
  - `/media/uploadedPics/<filename>`

**Results**
- Saved to: `media/Result/`
- Results log: `media/Result/Result.txt`  
  Each line contains a public URL such as:
  - `/media/Result/<filename>`

**Start button (image inference flow)**
1. Read uploaded image URLs from `img_list.txt`
2. Convert URL → filesystem path inside `MEDIA_ROOT`
3. Run YOLO inference on the image
4. Draw bounding boxes + labels using OpenCV (`cv2.rectangle`, `cv2.putText`)
5. Save annotated image to `media/Result/`
6. Append annotated image URL to `media/Result/Result.txt`
7. UI displays:
   - Raw images from `img_list.txt`
   - Annotated images from `Result.txt`

---

### Video pipeline

**Uploads**
- Saved to: `media/uploadedVideos/`
- Upload log: `media/uploadedVideos/video_list.txt`

**Results**
- Saved to: `media/ResultVideos/`
- Results log: `media/ResultVideos/Result.txt`

**Start button (video inference flow)**
1. Read uploaded video URLs from `video_list.txt`
2. Convert URL → filesystem path inside `MEDIA_ROOT`
3. Open each input video using `cv2.VideoCapture`
4. Create an output writer using `cv2.VideoWriter` with matching FPS/width/height (e.g., `mp4v`)
5. Run YOLO inference per frame (streaming mode or sequential loop)
6. Draw bounding boxes + labels on each frame using OpenCV
7. Write annotated frames into the output MP4
8. Save annotated video to `media/ResultVideos/`
9. Append annotated video URL to `media/ResultVideos/Result.txt`
10. UI displays:
   - Uploaded videos in the Raw tab
   - Annotated MP4s in the Result tab

---

## UI Behaviour
- Upload forms support **multiple files**
- **Start** triggers inference on **all uploaded media**
- Tabs separate Raw vs Result outputs:
  - Raw Images / Result Images
  - Raw Videos / Result Videos
- Carousels allow easy preview and navigation through outputs
- Clicking an item opens it in a new tab for full-size viewing/playback

---

## Key Files / Structure
- `cam_app2/models.py`
  - Wagtail `ImagePage` / `VideoPage` with custom `serve()` handling uploads + inference
- `mysite/templates/cam_app2/image.html`
  - Image upload + Start + raw/result carousels
- `mysite/templates/cam_app2/video.html`
  - Video upload + Start + raw/result video carousels
- `weights/best.pt`
  - Trained YOLOv8 weights used by the application
- `media/`
  - Runtime uploads and results (typically not committed to Git)

---

## Notes / Considerations
- This project is intended for local development/demos (filesystem storage).
- Video inference can be compute-intensive; long dashcam clips may take time to process.
- For production deployments, consider:
  - background processing (Celery/RQ),
  - object storage (e.g., S3),
  - upload size limits,
  - progress indicators / async job status.
