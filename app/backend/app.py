from flask import Flask, Response, request, jsonify, Blueprint
from flask_cors import CORS
import cv2
import time

from multimodel import MultiModalAIDemo
from llm import LLMChat
import os
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


def get_video_source():
    """
    Get video stream URL. Always streams (no file mode).

    VIDEO_STREAM_URL: RTSP/HTTP stream URL (live camera or MP4 simulation via MediaMTX).
    """
    stream_url = os.getenv("VIDEO_STREAM_URL", "").strip()
    if not stream_url:
        raise SystemExit(
            "VIDEO_STREAM_URL is required. Set it to an RTSP/HTTP stream URL "
            "(e.g. rtsp://video-stream:8554/live for local, or rtsp://camera-ip:554/stream for real camera)."
        )
    log.info(f"Using video stream: {stream_url}")
    return stream_url


video_source = get_video_source()
demo = MultiModalAIDemo(video_source)
demo.setup_components()
log.info("MultiModalAIDemo initialized and components ready")

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
    log.info("Video feed: client connected")
    none_count = 0
    last_none_log = 0.0
    frame_count = 0
    last_frame_log = 0.0
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
            # Use LINE_AA for antialiasing to reduce JPEG compression artifacts
            line_type = cv2.LINE_AA
            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Clip to frame bounds to avoid drawing artifacts
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

                    # Add text label; clip label position to avoid negative coords (causes distortion)
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
            now = time.time()
            if now - last_frame_log >= 10.0:
                log.debug("Video feed: sent %d frames (stream active)", frame_count)
                last_frame_log = now
            # Frame writer: log before/after yield to detect if browser stopped consuming
            if frame_count % 10 == 0:
                log.debug(
                    "Frame writer: about to send frame %d to browser", frame_count
                )
            try:
                # Content-Length helps Chrome parse each part correctly (avoids distortion from boundary misparsing)
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
                )
                yield header + frame_bytes + b"\r\n"
                if frame_count % 10 == 0:
                    log.debug(
                        "Frame writer: sent frame %d (browser consumed)", frame_count
                    )
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
    return jsonify({"description": latest_description, "summary": latest_summary})


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


app.register_blueprint(api)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8888"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
