import cv2
import threading
import time
from collections import defaultdict
from datetime import datetime

from database import (
    get_stream_fps,
    init_database,
    insert_observation,
    insert_person,
    set_stream_fps,
    update_person_last_seen,
)
from logger import get_logger
from runtime import Runtime

log = get_logger(__name__)


class MultiModalAIDemo:
    """Core video analysis pipeline for detection, summaries, and chat context."""

    def __init__(self, video_source):
        """Initialize the demo with a video source (file path or RTSP/HTTP URL)."""
        self.video_source = video_source
        self.is_live_stream = video_source.startswith(
            ("rtsp://", "http://", "https://")
        )
        self.cap = None
        self.runtime = None
        self.summarizer = None
        self.description_buffer = []
        self.frame_count = 0
        self.class_names = [
            "Hardhat",
            "Mask",
            "NO-Hardhat",
            "NO-Mask",
            "NO-Safety Vest",
            "Person",
            "Safety Cone",
            "Safety Vest",
            "machinery",
            "vehicle",
        ]
        self.ppe_stats = defaultdict(lambda: {"compliant": 0, "non_compliant": 0})
        self.latest_detection = defaultdict(int)
        self.latest_summary = ""

        # Object tracking and per-person PPE tracking
        self.person_history = {}  # {track_id: {"first_seen": datetime, "last_seen": datetime}}
        self.person_observations = []  # List of per-person PPE observations with timestamps
        self.latest_tracked_persons = []  # Most recent frame's tracked persons with PPE status

        # State-change tracking: only record when PPE status changes
        # {track_id: (hardhat, vest, mask)} - last known PPE state for each person
        self.person_last_state = {}

        # Throttle update_person_last_seen: only write to DB every N frames (not every frame)
        self._last_seen_update_interval = (
            30  # Update last_seen in DB every ~1 sec at 30fps
        )
        self._frames_since_last_seen_update = 0

        # Frame reader thread: consumes stream as fast as possible to avoid MediaMTX
        # "reader too slow, discarding frames" and subsequent RTP/H.264 corruption.
        self._reader_thread = None
        self._reader_stop = threading.Event()
        self._latest_frame = None
        self._latest_frame_lock = threading.Lock()
        self._reconnect_needed = False

        # Serialize capture_and_update: video_feed and latest_info both call it;
        # DeepSORT tracker is not thread-safe. Concurrent access caused IndexError.
        self._capture_lock = threading.Lock()

        # Measured or loaded FPS; used for browser throttling
        self.stream_fps = None
        # Timestamp-based FPS measurement in reader (live streams only)
        self._reader_frame_timestamps = []
        self._fps_measured = False

    def setup_components(self):
        """Load models and initialize runtime components."""
        self.cap = cv2.VideoCapture(self.video_source)
        if self.is_live_stream and not self.cap.isOpened():
            log.info("Stream not ready at startup, retrying...")
            for attempt in range(5):
                time.sleep(2)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.video_source)
                if self.cap.isOpened():
                    log.info(f"Stream connected (attempt {attempt + 1})")
                    break
        # Initialize database early so we can load/store stream FPS
        init_database()
        log.info("PostgreSQL database initialized: schema ready")

        if self.is_live_stream:
            log.info("Video is live streaming")
            if not self.cap.isOpened():
                log.info("Stream still not ready after DB init, retrying connection...")
                for attempt in range(5):
                    time.sleep(2)
                    if self.cap is not None:
                        self.cap.release()
                        self.cap = None
                    self.cap = cv2.VideoCapture(self.video_source)
                    if self.cap.isOpened():
                        log.info("Stream connected (post-init attempt %d)", attempt + 1)
                        break
            self._resolve_stream_fps()
            self._start_frame_reader()
        else:
            log.info("Video is playing from MP4")
            self._resolve_stream_fps()
        self.runtime = Runtime()
        log.info(f"Model classes: {self.runtime.CLASSES}")
        self.class_names = list(self.runtime.CLASSES.values())
        log.info(f"Using class names: {self.class_names}")

        # Initialize DeepSORT tracker for person tracking (works with OVMS detections).
        # We use DeepSORT instead of Ultralytics because: (1) OVMS already does detection;
        # using Ultralytics would duplicate inference. (2) Ultralytics tracking is bundled
        # with its detection in model.track(); it cannot accept external detections.
        from deep_sort_realtime.deepsort_tracker import DeepSort

        self.tracker = DeepSort(max_age=30, n_init=3)
        print("Object tracking enabled (DeepSORT)")

    def _resolve_stream_fps(self):
        """Get FPS: from DB if stored, else use default. Live streams measure in reader loop."""
        fps = get_stream_fps(self.video_source)
        if fps is not None and fps > 0:
            log.info("Using stored stream FPS: %.2f", fps)
            self.stream_fps = fps
            return
        # For files, try metadata first
        if not self.is_live_stream and self.cap is not None and self.cap.isOpened():
            meta_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if meta_fps and meta_fps > 0:
                self.stream_fps = float(meta_fps)
                set_stream_fps(self.video_source, self.stream_fps)
                log.info("Using file metadata FPS: %.2f", self.stream_fps)
                return
        # Live streams: use fallback; FPS measured in reader loop (non-blocking)
        self.stream_fps = 30.0
        if self.is_live_stream:
            log.info("No stored FPS; using 30.0 until measured in reader")
        else:
            log.warning("Could not get FPS; using fallback 30.0")

    def format_detection_description(
        self, detections_class_count: dict[str, int]
    ) -> str:
        """Build a short, human-readable description from detection counts."""
        description = "Detected: "
        for item in [
            "Person",
            "Hardhat",
            "Safety Vest",
            "Mask",
            "NO-Hardhat",
            "NO-Safety Vest",
            "NO-Mask",
        ]:
            if detections_class_count[item] > 0:
                description += f"{item}: {detections_class_count[item]}, "

        return description.rstrip(", ")

    def append_description(self, description):
        """Append a description to the rolling buffer with bounds."""
        self.description_buffer.append(description)
        if len(self.description_buffer) > 50:
            self.description_buffer.pop(0)

    def generate_image_description(self, frame):
        """Run detection on a frame and return a short description string."""
        detections = self.runtime.run(frame)
        detections_class_count = defaultdict(int)

        for d in detections:
            if d.class_name in ["Safety Cone", "Safety Vest", "machinery", "vehicle"]:
                continue
            detections_class_count[d.class_name] += 1

        description = self.format_detection_description(detections_class_count)
        self.append_description(description)
        return description

    def generate_summary(self, descriptions):
        """Summarize PPE compliance over a list of detection descriptions."""
        total_stats = defaultdict(int)
        frame_count = len(descriptions)

        for desc in descriptions:
            for item in [
                "Person",
                "Hardhat",
                "Safety Vest",
                "Mask",
                "NO-Hardhat",
                "NO-Safety Vest",
                "NO-Mask",
            ]:
                count = desc.count(item)
                total_stats[item] += count

        summary = "Safety Trends Summary:\n\n"
        summary += f"Total observations: {frame_count} frames\n\n"

        if total_stats["Person"] > 0:
            hardhat_compliance = (
                total_stats["Hardhat"]
                / (total_stats["Hardhat"] + total_stats["NO-Hardhat"])
                if (total_stats["Hardhat"] + total_stats["NO-Hardhat"]) > 0
                else 0
            )
            vest_compliance = (
                total_stats["Safety Vest"]
                / (total_stats["Safety Vest"] + total_stats["NO-Safety Vest"])
                if (total_stats["Safety Vest"] + total_stats["NO-Safety Vest"]) > 0
                else 0
            )
            mask_compliance = (
                total_stats["Mask"] / (total_stats["Mask"] + total_stats["NO-Mask"])
                if (total_stats["Mask"] + total_stats["NO-Mask"]) > 0
                else 0
            )

            summary += "Compliance rates:\n"
            summary += f"\n• Hardhat compliance: {hardhat_compliance:.2%} ({total_stats['Hardhat']} out of {total_stats['Hardhat'] + total_stats['NO-Hardhat']} detections)"
            summary += f"\n• Safety Vest compliance: {vest_compliance:.2%} ({total_stats['Safety Vest']} out of {total_stats['Safety Vest'] + total_stats['NO-Safety Vest']} detections)"
            summary += f"\n• Mask compliance: {mask_compliance:.2%} ({total_stats['Mask']} out of {total_stats['Mask'] + total_stats['NO-Mask']} detections)"

            overall_compliance = (
                hardhat_compliance + vest_compliance + mask_compliance
            ) / 3
            summary += f"\n\nOverall PPE compliance: {overall_compliance:.2%}\n"

            summary += "\nRecommendations:\n"
            if overall_compliance < 0.8:
                summary += f"\n• Critical: Immediate action required. {total_stats['NO-Hardhat'] + total_stats['NO-Safety Vest'] + total_stats['NO-Mask']} PPE violations detected."
                summary += "\n• Conduct an emergency safety briefing."
                summary += "\n• Increase on-site safety inspections."
            elif overall_compliance < 0.95:
                summary += f"\n• Warning: Improvement needed. {total_stats['NO-Hardhat'] + total_stats['NO-Safety Vest'] + total_stats['NO-Mask']} PPE violations detected."
                summary += "\n• Reinforce PPE policies through team meetings."
                summary += "\n• Consider additional PPE training sessions."
            else:
                summary += (
                    "\n• Good compliance observed. Maintain current safety protocols."
                )
                summary += "\n• Continue regular safety reminders and training."
        else:
            summary += "\n• No people detected in the observed period."
            summary += "\n• Check camera positioning and functionality."

        return summary

    def get_latest_detection(self):
        """Return the most recent detection counts."""
        return self.latest_detection

    def get_latest_summary(self):
        """Return the most recent summary."""
        return self.latest_summary

    def get_latest_tracked_persons(self):
        """Return the most recent tracked persons with their PPE status."""
        return self.latest_tracked_persons

    @staticmethod
    def _boxes_overlap(box1, box2):
        """Check if two bounding boxes overlap."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        if x2_1 < x1_2 or x2_2 < x1_1:
            return False
        if y2_1 < y1_2 or y2_2 < y1_1:
            return False
        return True

    def _associate_ppe_to_person(self, person_bbox, all_detections):
        """
        Determine PPE status for a specific person based on bounding box overlap.

        Returns dict with PPE status: {"hardhat": True/False/None, "vest": True/False/None, "mask": True/False/None}
        """
        status = {"hardhat": None, "vest": None, "mask": None}
        ppe_mapping = {
            "Hardhat": ("hardhat", True),
            "NO-Hardhat": ("hardhat", False),
            "Safety Vest": ("vest", True),
            "NO-Safety Vest": ("vest", False),
            "Mask": ("mask", True),
            "NO-Mask": ("mask", False),
        }

        for det in all_detections:
            class_name = det["class_name"]
            if class_name in ppe_mapping:
                ppe_bbox = det["bbox"]
                if self._boxes_overlap(person_bbox, ppe_bbox):
                    ppe_type, ppe_value = ppe_mapping[class_name]
                    if status[ppe_type] is None:
                        status[ppe_type] = ppe_value
        return status

    def _frame_reader_loop(self):
        """Read from the stream whenever a frame is available; store only the latest frame.
        Unprocessed frames are dropped (overwritten). Keeps RTSP buffer drained."""
        fail_count = 0
        cap_unopened_count = 0
        last_success_read_time = 0.0
        # DEBUG: track producer (stream reader) frame rate
        _reader_frame_count = 0
        _reader_start = time.monotonic()
        _reader_log_interval = 5.0
        while not self._reader_stop.is_set():
            cap = self.cap
            if cap is None or not cap.isOpened():
                cap_unopened_count += 1
                if cap_unopened_count >= 60:
                    self._reconnect_needed = True
                    log.info(
                        "[DISTORTION-DIAG] cap not opened for 3s, triggering reconnect"
                    )
                    cap_unopened_count = 0
                time.sleep(0.05)
                continue
            cap_unopened_count = 0
            success, frame = cap.read()
            if success and frame is not None:
                # Log long gaps between reads (possible GIL starvation -> buffer overflow)
                now = time.monotonic()
                if last_success_read_time > 0:
                    gap_ms = (now - last_success_read_time) * 1000
                    if gap_ms > 100:
                        log.info(
                            "[DISTORTION-DIAG] long gap between reads: %.0fms (possible starvation)",
                            gap_ms,
                        )
                last_success_read_time = now
                fail_count = 0
                with self._latest_frame_lock:
                    self._latest_frame = frame.copy()
                _reader_frame_count += 1
                # Timestamp-based FPS measurement (non-blocking, after first 20 frames)
                if not self._fps_measured and self.is_live_stream:
                    self._reader_frame_timestamps.append(time.perf_counter())
                    if len(self._reader_frame_timestamps) >= 20:
                        t0, t1 = (
                            self._reader_frame_timestamps[0],
                            self._reader_frame_timestamps[-1],
                        )
                        span = t1 - t0
                        if span > 0:
                            measured_fps = (len(self._reader_frame_timestamps) - 1) / span
                            if 1.0 <= measured_fps <= 120.0:
                                self.stream_fps = measured_fps
                                set_stream_fps(self.video_source, measured_fps)
                                log.info(
                                    "Measured stream FPS in reader: %.2f (from %d frames)",
                                    measured_fps,
                                    len(self._reader_frame_timestamps),
                                )
                        self._fps_measured = True
                        self._reader_frame_timestamps = []
                # No pacing: read as fast as frames arrive to avoid RTSP buffer overflow
                # and keyframe drops (which cause garbled H.264 decoding)
                # DEBUG: log producer fps every N seconds
                _elapsed = time.monotonic() - _reader_start
                if _elapsed >= _reader_log_interval:
                    _fps = _reader_frame_count / _elapsed if _elapsed > 0 else 0
                    log.info(
                        "[DEBUG frame_reader] producer fps=%.1f (read %d frames in %.1fs)",
                        _fps,
                        _reader_frame_count,
                        _elapsed,
                    )
                    _reader_frame_count = 0
                    _reader_start = time.monotonic()
            elif self._reader_stop.is_set():
                break
            else:
                fail_count += 1
                if fail_count == 1 or fail_count % 10 == 0:
                    log.info(
                        "[DISTORTION-DIAG] cap.read() failed (fail_count=%d)",
                        fail_count,
                    )
                if fail_count >= 30:
                    self._reconnect_needed = True
                    log.info(
                        "[DISTORTION-DIAG] 30 consecutive read failures, triggering reconnect"
                    )
                time.sleep(0.01)

    def _start_frame_reader(self):
        """Start background thread that reads frames at stream FPS."""
        self._reader_stop.clear()
        self._reader_frame_timestamps = []
        self._fps_measured = False
        self._reader_thread = threading.Thread(
            target=self._frame_reader_loop, daemon=True
        )
        self._reader_thread.start()

    def _stop_frame_reader(self):
        """Stop frame reader thread before reconnect/release."""
        self._reader_stop.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        with self._latest_frame_lock:
            self._latest_frame = None

    def _reconnect_stream(self):
        """Reconnect to live stream with retry and backoff."""
        log.info("[DISTORTION-DIAG] _reconnect_stream called (decoder reset)")
        self._stop_frame_reader()
        max_retries = 5
        base_delay = 2
        for attempt in range(max_retries):
            try:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                    time.sleep(0.5)  # Allow FFmpeg/OpenCV to clean up before realloc
                self.cap = cv2.VideoCapture(self.video_source)
                if self.cap.isOpened():
                    log.info(f"Reconnected to stream (attempt {attempt + 1})")
                    self._start_frame_reader()
                    return True
            except Exception as e:
                log.warning(f"Reconnect attempt {attempt + 1} failed: {e}")
            delay = base_delay * (2**attempt)
            log.debug(f"Waiting {delay}s before retry")
            time.sleep(delay)
        log.error("Failed to reconnect to stream after all retries")
        return False

    def capture_and_update(self, resize_to=None, caller=None):
        """Capture a frame, optionally resize, update detection state, and return frame data.

        caller: optional string for debug (e.g. 'video_feed', 'latest_info').
        """
        with self._capture_lock:
            return self._capture_and_update_impl(resize_to=resize_to, caller=caller)

    def _capture_and_update_impl(self, resize_to=None, caller=None):
        """Internal implementation; must be called with _capture_lock held."""
        log.debug(
            "capture_and_update: caller=%s tid=%s",
            caller or "unknown",
            threading.current_thread().name,
        )
        if self.is_live_stream and self._reader_thread is not None:
            if self._reconnect_needed:
                self._reconnect_needed = False
                log.info("[DISTORTION-DIAG] _reconnect_needed set, attempting reconnect")
                if self._reconnect_stream():
                    return self._capture_and_update_impl(
                        resize_to=resize_to, caller=caller
                    )
                return None, []
            with self._latest_frame_lock:
                frame = self._latest_frame
            if frame is None:
                return None, []
            # Heuristic: corrupt H.264 frames often have anomalous stats (all same value, etc.)
            try:
                fmean, fstd = float(frame.mean()), float(frame.std())
                if fstd < 3 or (fmean < 2 or fmean > 253):
                    log.info(
                        "[DISTORTION-DIAG] suspicious frame: mean=%.1f std=%.1f (possible corrupt)",
                        fmean,
                        fstd,
                    )
            except Exception:
                pass
            success = True
        else:
            success, frame = self.cap.read()
        if not success or frame is None:
            if self.is_live_stream:
                log.debug("Stream read failed, attempting reconnect")
                if self._reconnect_stream():
                    return self._capture_and_update_impl(
                        resize_to=resize_to, caller=caller
                    )
                return None, []
            log.debug("End of video reached, looping back to start")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return None, []

        if resize_to:
            frame = cv2.resize(frame, resize_to)

        # Run OVMS detection (via Runtime)
        runtime_detections = self.runtime.run(frame)

        # Build detections list and collect Person detections for DeepSORT
        detections = []
        counts = defaultdict(int)
        person_detections_for_tracker = []  # ([left, top, w, h], confidence, "Person")

        for d in runtime_detections:
            if d.class_name in ["Safety Cone", "Safety Vest", "machinery", "vehicle"]:
                continue
            counts[d.class_name] += 1
            x, y, w, h = d.bbox
            x1 = round(x * d.scale)
            y1 = round(y * d.scale)
            x2 = round((x + w) * d.scale)
            y2 = round((y + h) * d.scale)
            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": d.confidence,
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                }
            )
            # Only track Person detections
            if d.class_name == "Person" and d.confidence > 0.5:
                width = x2 - x1
                height = y2 - y1
                if width > 0 and height > 0:
                    person_detections_for_tracker.append(
                        ([x1, y1, width, height], d.confidence, "Person")
                    )

        # Run DeepSORT to get track IDs for persons
        tracked_person_boxes = {}  # {track_id: (x1, y1, x2, y2)}
        if person_detections_for_tracker:
            try:
                tracks = self.tracker.update_tracks(
                    person_detections_for_tracker, frame=frame
                )
            except IndexError as e:
                n_det = len(person_detections_for_tracker)
                inner = getattr(self.tracker, "tracker", self.tracker)
                n_tracks = len(getattr(inner, "tracks", []))
                log.error(
                    "tracker IndexError: caller=%s tid=%s n_detections=%s n_tracks=%s err=%s",
                    caller or "unknown",
                    threading.current_thread().name,
                    n_det,
                    n_tracks,
                    e,
                    exc_info=True,
                )
                raise
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = int(track.track_id)
                ltrb = track.to_ltrb()  # [left, top, right, bottom]
                if ltrb is not None:
                    x1, y1, x2, y2 = map(int, ltrb)
                    tracked_person_boxes[track_id] = (x1, y1, x2, y2)

        self.latest_detection = counts
        description = self.format_detection_description(counts)
        self.append_description(description)

        # Add track_id to Person detections in output (for display)
        for det in detections:
            if det["class_name"] == "Person":
                # Find matching track by bbox overlap
                for tid, pbox in tracked_person_boxes.items():
                    if self._boxes_overlap(det["bbox"], pbox):
                        det["track_id"] = tid
                        break

        # --- Object Tracking and PPE Association ---
        tracked_persons = []
        now = datetime.now()
        self._frames_since_last_seen_update += 1
        do_last_seen_db_update = (
            self._frames_since_last_seen_update >= self._last_seen_update_interval
        )

        for track_id, person_bbox in tracked_person_boxes.items():
            # Update person history (first/last seen) - in-memory every frame
            if track_id not in self.person_history:
                self.person_history[track_id] = {
                    "first_seen": now,
                    "last_seen": now,
                }
                insert_person(track_id, now, now)
            else:
                self.person_history[track_id]["last_seen"] = now
                # Throttle DB writes: only update last_seen in PostgreSQL periodically
                if do_last_seen_db_update:
                    update_person_last_seen(track_id, now)

            # Associate PPE with this person
            ppe_status = self._associate_ppe_to_person(person_bbox, detections)
            tracked_person = {
                "track_id": track_id,
                "bbox": person_bbox,
                "hardhat": ppe_status["hardhat"],
                "vest": ppe_status["vest"],
                "mask": ppe_status["mask"],
                "timestamp": now,
            }
            tracked_persons.append(tracked_person)

            # --- State-Change Recording ---
            current_state = (
                ppe_status["hardhat"],
                ppe_status["vest"],
                ppe_status["mask"],
            )
            last_state = self.person_last_state.get(track_id)

            if last_state is None or last_state != current_state:
                self.person_observations.append(
                    {
                        "track_id": track_id,
                        "timestamp": now,
                        "hardhat": ppe_status["hardhat"],
                        "vest": ppe_status["vest"],
                        "mask": ppe_status["mask"],
                        "bbox": person_bbox,
                    }
                )
                insert_observation(
                    track_id=track_id,
                    timestamp=now,
                    hardhat=ppe_status["hardhat"],
                    vest=ppe_status["vest"],
                    mask=ppe_status["mask"],
                )
                self.person_last_state[track_id] = current_state

            if len(self.person_observations) > 1000:
                self.person_observations = self.person_observations[-1000:]

        if do_last_seen_db_update:
            self._frames_since_last_seen_update = 0

        self.latest_tracked_persons = tracked_persons
        # --- End Object Tracking ---

        if self.frame_count % 50 == 0:
            self.latest_summary = self.generate_summary(self.description_buffer)

        self.frame_count += 1
        return frame, detections

    def generate_frames(self):
        """Backward-compatible wrapper to update state for one frame."""
        self.capture_and_update()
