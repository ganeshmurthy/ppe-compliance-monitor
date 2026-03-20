from flask import Flask, Response, request, jsonify, Blueprint, send_from_directory
from flask_cors import CORS
import cv2
import json
import os
import time
from datetime import datetime

from multimodel import MultiModalAIDemo
from llm import LLMChat
from database import (
    get_all_configs,
    get_config_by_id,
    insert_config,
    update_config,
    replace_detection_classes,
)
from logger import get_logger

log = get_logger(__name__)


app = Flask(__name__)
api = Blueprint("api", __name__, url_prefix="/api")

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
if cors_origins.strip() == "*":
    cors_allowed_origins = "*"
else:
    cors_allowed_origins = [
        origin.strip() for origin in cors_origins.split(",") if origin.strip()
    ]

CORS(app, resources={r"/*": {"origins": cors_allowed_origins}})


# Video source is selected dynamically by the user (MP4 or RTSP from config).
demo = MultiModalAIDemo()
demo.setup_components()
log.info("MultiModalAIDemo initialized (source: user-selected from UI)")

llm_chat = LLMChat()

latest_description = "Initializing..."
latest_summary = "Processing video..."


def generate_response_frames():
    """Generate MJPEG video stream for the /api/video_feed endpoint.

    This generator runs in a loop, fetching the latest frame and detections from
    the demo (which are produced by a separate inference process). It draws
    bounding boxes and labels on each frame, encodes as JPEG, and yields
    multipart MJPEG chunks for the browser to display.

    Flow:
        - get_frame_for_display() returns the latest frame + detections (never blocks on inference)
        - Draws boxes (cyan=person, green=compliant PPE, red=non-compliant)
        - Encodes frame as JPEG and yields multipart chunk (--frame + Content-Type + Content-Length + bytes)

    The display path is decoupled from inference: frames come from a reader thread,
    detections from an inference process. If no frame is available, the loop
    continues without yielding. Duplicate frames (same frame_id) are skipped.
    """
    global latest_description, latest_summary
    log.info("Video feed: client connected, starting frame loop")
    none_count = 0
    last_none_log = 0.0
    frame_count = 0
    last_sent_frame_id = None
    try:
        while True:
            frame, detections, frame_id = demo.get_frame_for_display(
                resize_to=(1920, 1080)
            )
            if frame is None:
                none_count += 1
                if none_count == 1:
                    log.warning("Video feed: first None received (no frame to display)")
                now = time.time()
                if now - last_none_log >= 5.0:
                    log.warning(
                        "Video feed: received None %d times in last 5s (no frames to display)",
                        none_count,
                    )
                    last_none_log = now
                    none_count = 0
                continue

            # Skip sending duplicate frames (same frame_id as last sent)
            if frame_id is not None and frame_id == last_sent_frame_id:
                time.sleep(0.001)  # Avoid tight loop when waiting for new frame
                continue

            try:
                annotated_frame = frame.copy()
            except Exception as e:
                log.exception("Video feed: frame.copy() failed: %s", e)
                continue
            h_frame, w_frame = annotated_frame.shape[:2]
            # Draw bounding boxes and labels for each detection (Person, PPE items)
            line_type = cv2.LINE_AA
            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1 = max(0, min(x1, w_frame - 1))
                y1 = max(0, min(y1, h_frame - 1))
                x2 = max(0, min(x2, w_frame - 1))
                y2 = max(0, min(y2, h_frame - 1))
                if x1 >= x2 or y1 >= y2:
                    continue
                conf = detection["confidence"]
                currentClass = detection["class_name"]
                if conf > 0.5:
                    if currentClass == "Person":
                        color = (0, 255, 255)  # Cyan for person
                    elif currentClass in ["NO-Hardhat", "NO-Safety Vest", "NO-Mask"]:
                        color = (0, 0, 255)  # Red for non-compliance
                    elif currentClass in ["Hardhat", "Safety Vest", "Mask"]:
                        color = (0, 255, 0)  # Green for compliance
                    else:
                        color = (255, 255, 0)  # Yellow for other objects
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                        lineType=line_type,
                    )
                    label = f"{currentClass} {conf:.2f}"
                    if (
                        currentClass == "Person"
                        and detection.get("track_id") is not None
                    ):
                        label = f"Person #{detection['track_id']} {conf:.2f}"
                    text_size = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
                    )[0]
                    label_y1 = max(0, y1 - text_size[1] - 10)
                    label_y2 = y1
                    label_x2 = min(w_frame, x1 + text_size[0])
                    if label_x2 > x1 and label_y2 > label_y1:
                        cv2.rectangle(
                            annotated_frame,
                            (x1, label_y1),
                            (label_x2, label_y2),
                            color,
                            -1,
                            lineType=line_type,
                        )
                    text_y = max(label_y1 + text_size[1] - 2, y1 - 5)
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 0),
                        2,
                        lineType=line_type,
                    )

            # Read from shared state (updated by inference thread)
            with demo._display_lock:
                if demo._display_description:
                    latest_description = demo._display_description
                latest_summary = (
                    demo._display_summary or demo.latest_summary or latest_summary
                )

            ret, buffer = cv2.imencode(
                ".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            if not ret:
                log.warning("Video feed: cv2.imencode failed")
                continue
            frame_bytes = buffer.tobytes()
            frame_count += 1
            last_sent_frame_id = frame_id
            try:
                # Content-Length helps Chrome parse each part correctly (avoids distortion from boundary misparsing)
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
                )
                yield header + frame_bytes + b"\r\n"
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                log.warning("Video feed: client connection lost during yield: %s", e)
                break
    except GeneratorExit:
        log.debug("Video feed: client disconnected (GeneratorExit)")
    except Exception as e:
        log.exception("Video feed: exception in stream loop: %s", e)
    finally:
        log.debug("Video feed: stream ended (total frames sent: %d)", frame_count)


@api.route("/video_feed")
def video_feed():
    """Video streaming route."""
    response = Response(
        generate_response_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )
    # Disable proxy buffering to reduce periodic pauses (OpenShift/HAProxy)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@api.route("/")
def api_root():
    """Simple health response for the API root."""
    return jsonify({"status": "ok"})


@api.route("/latest_info")
def latest_info():
    """Return the latest description and summary. Reads from shared state (no inference trigger)."""
    global latest_description, latest_summary
    with demo._display_lock:
        if demo._display_description:
            latest_description = demo._display_description
        latest_summary = demo._display_summary or demo.latest_summary or latest_summary
        inference_ready = demo._results_received_count > 0
    return jsonify(
        {
            "description": latest_description,
            "summary": latest_summary,
            "inference_ready": inference_ready,
        }
    )


@api.route("/chat", methods=["POST"])
def chat():
    """Answer a question based on latest description and summary.

    Supports streaming via Server-Sent Events when ``stream=true`` is sent in
    the JSON body.  An optional ``session_id`` field enables per-session
    conversation memory (defaults to ``"default"``).
    """
    global latest_description, latest_summary
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Field 'question' is required."}), 400

    session_id = (data.get("session_id") or "default").strip()
    user_description = (data.get("description") or "").strip()

    if user_description:
        desc = user_description
    else:
        with demo._display_lock:
            desc = demo._display_description or latest_description
            # summ = demo._display_summary or demo.latest_summary or latest_summary
    context = desc.replace("Detected: ", "", 1)  # + " " + summ

    answer = llm_chat.chat(
        question=question,
        context=context,
        session_id=session_id,
    )
    return jsonify({"answer": answer})


def _parse_classes(value: str | dict) -> tuple[dict, list[tuple[int, str, bool]]]:
    """
    Parse classes from new JSON format. Returns (mapping, entries).
    Format: {"0":{"name":"Person","trackable":true},"1":{"name":"Hardhat","trackable":false}}
    mapping: {"0":"Person","1":"Hardhat"} for app_config.classes
    entries: [(model_class_index, name, trackable), ...] for detection_classes
    """
    if value is None:
        raise ValueError("Classes cannot be empty")
    obj = value if isinstance(value, dict) else json.loads(str(value).strip())
    if not isinstance(obj, dict):
        raise ValueError(
            'Classes must be an object like {"0":{"name":"Person","trackable":true}}'
        )
    if not obj:
        raise ValueError("Classes cannot be empty")
    mapping = {}
    entries = []
    for idx_str, v in obj.items():
        if not isinstance(v, dict):
            raise ValueError(
                f'Class "{idx_str}" must be an object with "name" and "trackable"'
            )
        name = v.get("name")
        if not name or not str(name).strip():
            raise ValueError(f'Class "{idx_str}" must have a non-empty "name"')
        name = str(name).strip()
        trackable = bool(v.get("trackable", False))
        try:
            model_class_index = int(idx_str)
        except ValueError:
            raise ValueError(f'Class key "{idx_str}" must be an integer (model index)')
        mapping[idx_str] = name
        entries.append((model_class_index, name, trackable))
    return mapping, entries


@api.route("/config", methods=["GET"])
def config_list():
    """List all app configs."""
    try:
        configs = get_all_configs()
        # Ensure classes is JSON-serializable (PostgreSQL JSONB may return dict)
        for c in configs:
            if isinstance(c.get("created_at"), datetime):
                c["created_at"] = c["created_at"].isoformat()
        return jsonify(configs)
    except Exception as e:
        log.exception("config_list: %s", e)
        return jsonify({"error": str(e)}), 500


@api.route("/config", methods=["POST"])
def config_create():
    """Create a new app config."""
    data = request.get_json(silent=True) or {}
    model_url = (data.get("model_url") or "").strip()
    video_source = (data.get("video_source") or "").strip()
    classes_raw = data.get("classes")
    if classes_raw is None:
        return jsonify({"error": "Field 'classes' is required"}), 400
    try:
        classes, entries = _parse_classes(classes_raw)
    except (ValueError, json.JSONDecodeError) as e:
        return jsonify({"error": str(e)}), 400
    if not model_url:
        return jsonify({"error": "Field 'model_url' is required"}), 400
    if not video_source:
        return jsonify({"error": "Field 'video_source' is required"}), 400
    try:
        config_id = insert_config(model_url, video_source)
        replace_detection_classes(config_id, entries)
        if _is_local_video_path(video_source):
            _generate_thumbnail(video_source)
        return jsonify({"id": config_id, "message": "Config created"}), 201
    except Exception as e:
        log.exception("config_create: %s", e)
        return jsonify({"error": str(e)}), 500


@api.route("/config/<int:config_id>", methods=["PUT"])
def config_update(config_id):
    """Update an existing app config."""
    data = request.get_json(silent=True) or {}
    model_url = (data.get("model_url") or "").strip()
    video_source = (data.get("video_source") or "").strip()
    classes_raw = data.get("classes")
    if classes_raw is None:
        return jsonify({"error": "Field 'classes' is required"}), 400
    try:
        classes, entries = _parse_classes(classes_raw)
    except (ValueError, json.JSONDecodeError) as e:
        return jsonify({"error": str(e)}), 400
    if not model_url:
        return jsonify({"error": "Field 'model_url' is required"}), 400
    if not video_source:
        return jsonify({"error": "Field 'video_source' is required"}), 400
    try:
        replace_detection_classes(config_id, entries)
        updated = update_config(config_id, model_url, video_source)
        if not updated:
            return jsonify({"error": "Config not found"}), 404
        if _is_local_video_path(video_source):
            _generate_thumbnail(video_source)
        return jsonify({"message": "Config updated"})
    except Exception as e:
        log.exception("config_update: %s", e)
        return jsonify({"error": str(e)}), 500


@api.route("/active_config", methods=["POST"])
def active_config_set():
    """Set the active video source from a config. Switches to that config's video (MP4 path or RTSP URL)."""
    data = request.get_json(silent=True) or {}
    config_id = data.get("config_id")
    if config_id is None:
        return jsonify({"error": "Field 'config_id' is required"}), 400
    try:
        config_id = int(config_id)
    except (ValueError, TypeError):
        return jsonify({"error": "config_id must be an integer"}), 400
    config = get_config_by_id(config_id)
    if not config:
        return jsonify({"error": "Config not found"}), 404
    video_source = (config.get("video_source") or "").strip()
    if not video_source:
        return jsonify({"error": "Config has no video source"}), 400
    try:
        demo.start_streaming(video_source, config_id=config_id)
        return jsonify({"message": "Active config set", "video_source": video_source})
    except Exception as e:
        log.exception("active_config_set: %s", e)
        return jsonify({"error": str(e)}), 500


# Config upload - use /tmp in container (writable); CONFIG_UPLOAD_DIR overrides
def _get_upload_dir():
    env_dir = os.environ.get("CONFIG_UPLOAD_DIR")
    if env_dir:
        try:
            os.makedirs(env_dir, exist_ok=True)
            return env_dir
        except (PermissionError, OSError) as e:
            log.warning("CONFIG_UPLOAD_DIR %s not writable: %s", env_dir, e)
    # /tmp is writable in containers; use it to avoid PermissionError on /data
    upload_dir = "/tmp/ppe-config-uploads"
    try:
        os.makedirs(upload_dir, exist_ok=True)
        return upload_dir
    except (PermissionError, OSError):
        return "/tmp"  # last resort


UPLOAD_DIR = _get_upload_dir()
log.info("Config upload directory: %s", UPLOAD_DIR)


def _get_thumbnail_dir():
    """Thumbnail directory. CONFIG_THUMBNAIL_DIR overrides; else derived from upload dir."""
    env_dir = os.environ.get("CONFIG_THUMBNAIL_DIR")
    if env_dir:
        try:
            os.makedirs(env_dir, exist_ok=True)
            return env_dir
        except (PermissionError, OSError) as e:
            log.warning("CONFIG_THUMBNAIL_DIR %s not writable: %s", env_dir, e)
    parent = os.path.dirname(UPLOAD_DIR.rstrip("/"))
    return os.path.join(parent, "ppe-thumbnails")


THUMBNAIL_DIR = _get_thumbnail_dir()
log.info("Thumbnail directory: %s", THUMBNAIL_DIR)
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
_THUMBNAIL_TIMESTAMP_S = 17.0


def _is_local_video_path(path):
    """True if path looks like a local file path to a video."""
    if not path or not isinstance(path, str):
        return False
    p = path.strip()
    if "://" in p:
        return False
    return any(p.lower().endswith(ext) for ext in _VIDEO_EXTENSIONS)


def _generate_thumbnail(video_path):
    """Generate a JPEG thumbnail from a video file using OpenCV. Returns path or None."""
    if not os.path.isfile(video_path):
        return None
    stem = os.path.splitext(os.path.basename(video_path))[0]
    if not stem:
        return None
    try:
        os.makedirs(THUMBNAIL_DIR, exist_ok=True)
        thumb_path = os.path.join(THUMBNAIL_DIR, stem + ".jpg")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.warning("Could not open video for thumbnail: %s", video_path)
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_pos = int(_THUMBNAIL_TIMESTAMP_S * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            if cv2.imwrite(thumb_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                log.info("Generated thumbnail: %s", thumb_path)
                return thumb_path
    except (OSError, Exception) as e:
        log.warning("Thumbnail generation failed for %s: %s", video_path, e)
    return None


@api.route("/thumbnails/<path:filename>")
def serve_thumbnail(filename):
    """Serve a thumbnail image by filename (e.g. video.jpg)."""
    if ".." in filename or "/" in filename:
        return jsonify({"error": "Invalid filename"}), 400
    if not filename.lower().endswith(".jpg"):
        return jsonify({"error": "Only .jpg thumbnails are served"}), 400
    if not os.path.isdir(THUMBNAIL_DIR):
        return jsonify({"error": "Thumbnails directory not found"}), 404
    path = os.path.join(THUMBNAIL_DIR, filename)
    if not os.path.isfile(path):
        return jsonify({"error": "Thumbnail not found"}), 404
    return send_from_directory(THUMBNAIL_DIR, filename, mimetype="image/jpeg")


@api.route("/config/upload", methods=["POST"])
def config_upload():
    """Upload a video file. Returns the path/URL for the video field."""
    if "file" not in request.files:
        return jsonify({"error": "No file in request"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No filename"}), 400
    if not f.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        return jsonify(
            {"error": "Only video files (mp4, avi, mov, mkv) are allowed"}
        ), 400
    try:
        safe_name = os.path.basename(f.filename)
        path = os.path.join(UPLOAD_DIR, safe_name)
        f.save(path)
        return jsonify({"path": path, "filename": safe_name})
    except Exception as e:
        log.exception("config_upload: %s", e)
        return jsonify({"error": str(e)}), 500


app.register_blueprint(api)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8888"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
