from django.db import models
from django.shortcuts import render
from django.conf import settings
from django import forms

from modelcluster.fields import ParentalKey

# Wagtail admin interface components
from wagtail.admin.edit_handlers import (
    FieldPanel,
    MultiFieldPanel,
    InlinePanel,
    StreamFieldPanel,
    PageChooserPanel,
)

# Wagtail core models and fields
from wagtail.core.models import Page, Orderable
from wagtail.core.fields import RichTextField, StreamField
from wagtail.images.edit_handlers import ImageChooserPanel
from django.core.files.storage import default_storage
import shutil

from pathlib import Path

# Utilities and libraries
import sqlite3, datetime, os, uuid, glob, cv2
from PIL import Image

# Load detection model
from rfdetr import RFDETRBase

# Class labels for road damage
CLASS_NAMES = ["D00", "D10", "D20", "D40"]

# Dictionary to count types of detected damage
damage_counts = {"Longitudinal Crack": 0, "Traversal Crack": 0, "Alligator Crack": 0, "Pothole": 0}

# Load path to pre-trained model weights
BEST_WEIGHTS = os.path.join(settings.BASE_DIR, "weights", "checkpoint_best_total.pth")

# Initialize RF-DETR model once to avoid repeated loading
rf_model = RFDETRBase(
    backbone="res18",
    num_classes=4,
    checkpoints_dir=None,
    pretrain_weights=BEST_WEIGHTS,
    pretrained_encoder=None
)

# Unique identifier for image uploads
str_uuid = uuid.uuid4()

# Remove old result and upload files before processing new ones
def reset():
    files_result = glob.glob(str(Path(f'{settings.MEDIA_ROOT}/Result/*.*')), recursive=True)
    files_upload = glob.glob(str(Path(f'{settings.MEDIA_ROOT}/uploadedPics/*.*')), recursive=True)
    files = []
    if files_result:
        files.extend(files_result)
    if files_upload:
        files.extend(files_upload)
    if files:
        for f in files:
            try:
                if not f.endswith(".txt"):
                    os.remove(f)  # Delete non-text files
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))
        # Clear contents of log files
        for p in (
            Path(f'{settings.MEDIA_ROOT}/Result/Result.txt'),
            Path(f'{settings.MEDIA_ROOT}/uploadedPics/img_list.txt'),
            Path(f'{settings.MEDIA_ROOT}/Result/stats.txt'),
        ):
            with open(p, "r+") as file:
                file.truncate(0)

# Wagtail page for handling image-based detection
class ImagePage(Page):
    template = "cam_app2/image.html"
    max_count = 2  # Limit to one or two instances in CMS

    name_title    = models.CharField(max_length=100, blank=True, null=True)
    name_subtitle = RichTextField(features=["bold", "italic"], blank=True)

    content_panels = Page.content_panels + [
        MultiFieldPanel(
            [ FieldPanel("name_title"), FieldPanel("name_subtitle") ],
            heading="Page Options",
        ),
    ]

    # Prepare context dictionary with empty values
    def reset_context(self, request):
        context = super().get_context(request)
        context["my_uploaded_file_names"] = []
        context["my_result_file_names"]   = []
        context["my_staticSet_names"]     = []
        context["my_lines"]               = []
        return context
    
    def serve(self, request):
        detection_summary = {}
        result_dir = Path(settings.MEDIA_ROOT) / "Result"
        thumb_dir  = result_dir / "thumbs"

        # Ensure result directories exist and start fresh
        result_dir.mkdir(parents=True, exist_ok=True)
        if thumb_dir.exists():
            shutil.rmtree(thumb_dir)
        thumb_dir.mkdir(parents=True, exist_ok=True)

        result_log = result_dir / "Result.txt"

        # On page load (GET), reset old files
        if request.method == "GET":
            reset()

        # If files are uploaded
        if request.FILES:
            reset()
            self.reset_context(request)

            upload_dir = Path(settings.MEDIA_ROOT) / "uploadedPics"
            upload_dir.mkdir(parents=True, exist_ok=True)
            upload_log = upload_dir / "img_list.txt"
            upload_log.write_text("")  # clear old entries

            for file_obj in request.FILES.getlist("file_data"):
                uid   = uuid.uuid4().hex
                stem  = file_obj.name.rsplit(".", 1)[0]
                ext   = file_obj.name.rsplit(".", 1)[1]
                name  = f"{stem}_{uid}.{ext}"
                dest  = upload_dir / name
                default_storage.save(dest.as_posix(), file_obj)

                url = f"{settings.MEDIA_URL}uploadedPics/{name}"
                with open(upload_log, "a") as f:
                    f.write(url + "\n")

        # Run detection when 'Start Detection' is clicked
        if request.POST.get("start") is not None:
            result_log.write_text("")

            upload_log = Path(settings.MEDIA_ROOT) / "uploadedPics" / "img_list.txt"
            lines = [l.strip() for l in upload_log.read_text().splitlines() if l.strip()]

            CLASS_NAMES = ["D00","D10","D20","D40"]
            COLOR_MAP = {
                "D00": (0, 128,   0),   # green
                "D10": (0, 128,   0),
                "D20": (0, 165, 255),   # orange
                "D40": (0,   0, 255),   # red
            }

            LABEL_MAP = {
                "D00":"Longitudinal Crack",
                "D10":"Traversal Crack",
                "D20":"Alligator Crack",
                "D40":"Pothole"
            }

            for url in lines:
                rel = url[len(settings.MEDIA_URL):]
                src = Path(settings.MEDIA_ROOT) / rel
                orig = cv2.imread(src.as_posix())  # Load original image
                frame = orig.copy()
                if frame is None:
                    continue

                # Run model prediction
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                dets = rf_model.predict(pil_img, threshold=0.4)

                img_key = src.stem
                detection_summary[img_key] = []

                for (x1, y1, x2, y2), cid, conf in zip(dets.xyxy, dets.class_id, dets.confidence):
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    cls_name = CLASS_NAMES[cid - 1]
                    label_name = LABEL_MAP[cls_name]
                    color = COLOR_MAP[cls_name]
                    label_txt = f"{label_name} {conf:.2f}"

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                    cv2.putText(frame, label_txt, (x1, y1 - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    # Crop and save thumbnail image
                    crop = orig[y1:y2, x1:x2]
                    thumb_name = f"{img_key}_{cls_name}_{int(conf * 100)}.jpg"
                    thumb_path = thumb_dir / thumb_name
                    cv2.imwrite(thumb_path.as_posix(), crop)
                    thumb_url = f"{settings.MEDIA_URL}Result/thumbs/{thumb_name}"

                    # Save detection metadata
                    detection_summary[img_key].append({
                        "class": cls_name,
                        "confidence": float(conf),
                        "label": label_name,
                        "thumb_url": thumb_url,
                    })

                # Save annotated result image
                out_name = img_key + "_ann.jpg"
                out_path = result_dir / out_name
                cv2.imwrite(out_path.as_posix(), frame)

                public_url = f"{settings.MEDIA_URL}Result/{out_name}"
                with open(result_log, "a") as f:
                    f.write(public_url + "\n")

        # Prepare context and render the page
        context = super().get_context(request)
        context["detection_summary"] = detection_summary
        context["my_uploaded_file_names"] = (Path(settings.MEDIA_ROOT)/"uploadedPics"/"img_list.txt").read_text().splitlines()
        context["my_result_file_names"] = result_log.read_text().splitlines()
        return render(request, "cam_app2/image.html", context)
    
    
# Wagtail page for handling video-based damage detection
class VideoDetectPage(Page):
    template = "cam_app2/video_detect.html"
    max_count = 1  # Only one instance allowed

    name_title = models.CharField(max_length=100, blank=True, null=True)
    name_subtitle = RichTextField(features=["bold", "italic"], blank=True)

    content_panels = Page.content_panels + [
        MultiFieldPanel(
            [FieldPanel("name_title"), FieldPanel("name_subtitle")],
            heading="Page Options",
        ),
    ]

    # Prepare context with empty values
    def reset_context(self, request):
        context = super().get_context(request)
        context["my_uploaded_file_names"] = []
        context["my_result_file_names"] = []
        context["detection_summary"] = {}
        context["damage_counts"] = {"D00": 0, "D10": 0, "D20": 0, "D40": 0}
        return context

    def serve(self, request):
        detection_summary = {}
        damage_counts = {"D00": 0, "D10": 0, "D20": 0, "D40": 0}

        # Define directories for video uploads and results
        upload_dir = Path(settings.MEDIA_ROOT) / "uploadedVideosDetect"
        result_dir = Path(settings.MEDIA_ROOT) / "ResultVideosDetect"
        thumb_dir = result_dir / "thumbs"

        # Clean thumbnail directory if it exists
        if thumb_dir.exists():
            shutil.rmtree(thumb_dir)
        thumb_dir.mkdir(parents=True, exist_ok=True)

        upload_log = upload_dir / "video_list.txt"
        result_log = result_dir / "Result.txt"

        # Reset logs and delete old files on GET
        if request.method == "GET":
            for folder, log in [("uploadedVideosDetect", "video_list.txt"), ("ResultVideosDetect", "Result.txt")]:
                d = Path(settings.MEDIA_ROOT) / folder
                d.mkdir(exist_ok=True)
                for f in d.glob("*.*"):
                    if f.name != log:
                        f.unlink()
                (d / log).write_text("")

        # Handle uploaded video files
        if request.FILES:
            for folder, log in [("uploadedVideosDetect", "video_list.txt"), ("ResultVideosDetect", "Result.txt")]:
                d = Path(settings.MEDIA_ROOT) / folder
                for f in d.glob("*.*"):
                    if f.name != log:
                        f.unlink()
                (d / log).write_text("")
            self.reset_context(request)
            upload_log.write_text("")
            for fobj in request.FILES.getlist("file_data"):
                uid = uuid.uuid4().hex
                stem, ext = fobj.name.rsplit(".", 1)
                name = f"{stem}_{uid}.{ext}"
                dest = upload_dir / name
                default_storage.save(dest.as_posix(), fobj)
                url = f"{settings.MEDIA_URL}uploadedVideosDetect/{name}"
                with open(upload_log, "a") as logf:
                    logf.write(url + "\n")

        # Perform detection when "Start Detection" is clicked
        if request.POST.get("start") is not None:
            result_log.write_text("")
            lines = [l.strip() for l in upload_log.read_text().splitlines() if l.strip()]

            CLASS_NAMES = ["D00", "D10", "D20", "D40"]
            COLOR_MAP = {"D00": (0, 128, 0), "D10": (0, 128, 0), "D20": (0, 165, 255), "D40": (0, 0, 255)}
            LABEL_MAP = {
                "D00": "Longitudinal Crack",
                "D10": "Traversal Crack",
                "D20": "Alligator Crack",
                "D40": "Pothole",
            }

            # Prevent duplicate detections within short time span
            last_detected_frame = {cls: -20 for cls in CLASS_NAMES}

            for url in lines:
                rel = url[len(settings.MEDIA_URL):]
                src_path = Path(settings.MEDIA_ROOT) / rel
                cap = cv2.VideoCapture(src_path.as_posix())
                fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_name = src_path.stem + "_ann.mp4"
                out_vid = result_dir / out_name
                writer = cv2.VideoWriter(out_vid.as_posix(), fourcc, fps, (w, h))

                vid_key = src_path.stem
                detection_summary[vid_key] = []

                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    orig = frame.copy()

                    # Convert frame to PIL image for prediction
                    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    dets = rf_model.predict(pil, threshold=0.3)

                    for (x1, y1, x2, y2), cid, conf in zip(dets.xyxy, dets.class_id, dets.confidence):
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        cls = CLASS_NAMES[cid - 1]

                        # Skip detection if too close to last detection of same class
                        if frame_idx - last_detected_frame[cls] < 10:
                            continue
                        last_detected_frame[cls] = frame_idx

                        col = COLOR_MAP[cls]
                        lbl = f"{LABEL_MAP[cls]} {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), col, -1)
                        cv2.putText(frame, lbl, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                        # Save thumbnail of detected region
                        crop = orig[y1:y2, x1:x2]
                        thumb_name = f"{vid_key}_{frame_idx}_{cls}_{int(conf * 100)}.jpg"
                        thumb_path = thumb_dir / thumb_name
                        cv2.imwrite(thumb_path.as_posix(), crop)
                        thumb_url = f"{settings.MEDIA_URL}ResultVideosDetect/thumbs/{thumb_name}"

                        # Timestamp the detection
                        timestamp_sec = frame_idx / fps
                        minutes = int(timestamp_sec // 60)
                        seconds = int(timestamp_sec % 60)
                        timestamp_str = f"{minutes:02d}:{seconds:02d}"

                        detection_summary[vid_key].append({
                            "class": cls,
                            "confidence": float(conf),
                            "label": LABEL_MAP[cls],
                            "thumb_url": thumb_url,
                            "timestamp": timestamp_str,
                        })

                        damage_counts[cls] += 1  # Tally each class count

                    writer.write(frame)
                    frame_idx += 1

                cap.release()
                writer.release()

                # Re-encode video to H.264 with +faststart for web
                converted_path = result_dir / f"{src_path.stem}_converted.mp4"
                ffmpeg_cmd = f'ffmpeg -y -i "{out_vid}" -vcodec libx264 -pix_fmt yuv420p -crf 23 -preset veryfast -movflags +faststart "{converted_path}"'
                os.system(ffmpeg_cmd)

                if converted_path.exists() and converted_path.stat().st_size > 0:
                    public_url = f"{settings.MEDIA_URL}ResultVideosDetect/{converted_path.name}"
                    with open(result_log, "a") as logf:
                        logf.write(public_url + "\n")
                    try:
                        out_vid.unlink()  # Clean up the temporary .mp4
                    except:
                        pass

        # Pass data to the template for rendering
        context = super().get_context(request)
        context["damage_counts"] = damage_counts
        context["my_uploaded_file_names"] = upload_log.read_text().splitlines()
        context["my_result_file_names"] = result_log.read_text().splitlines()
        context["detection_summary"] = detection_summary
        return render(request, "cam_app2/video_detect.html", context)
