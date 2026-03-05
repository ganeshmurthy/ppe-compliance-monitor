import cv2
import numpy as np
import threading
import time
import atexit
from collections import defaultdict
from datetime import datetime
import queue
from multiprocessing import Process, Queue, Event
from multiprocessing.shared_memory import SharedMemory

from database import (
    init_database,
    insert_person,
    update_person_last_seen,
    insert_observation,
)
from logger import get_logger
from response import process_detections
from runtime import Runtime

log = get_logger(__name__)


# --- Module-level helpers (used by both main and inference process) ---


def _boxes_overlap(box1, box2):
    """Check if two bounding boxes overlap."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    if x2_1 < x1_2 or x2_2 < x1_1:
        return False
    if y2_1 < y1_2 or y2_2 < y1_1:
        return False
    return True


def _associate_ppe_to_person(person_bbox, all_detections):
    """Determine PPE status for a person based on bounding box overlap."""
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
            if _boxes_overlap(person_bbox, ppe_bbox):
                ppe_type, ppe_value = ppe_mapping[class_name]
                if status[ppe_type] is None:
                    status[ppe_type] = ppe_value
    return status


def format_detection_description(detections_class_count: dict[str, int]) -> str:
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
        if detections_class_count.get(item, 0) > 0:
            description += f"{item}: {detections_class_count[item]}, "
    return description.rstrip(", ")


def generate_summary(descriptions: list) -> str:
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


# --- Inference process target (runs in separate process) ---


def _inference_process_target(
    config_queue: Queue,
    results_queue: Queue,
    stop_event: Event,
    frame_ready_event: Event,
):
    """
    Run in a separate process. Waits for buffer config, then runs inference loop.
    Reads frames from shared memory when frame_ready_event is set, runs OVMS+DeepSORT,
    puts (detections, description, summary) in results_queue.
    """
    shm = None
    shm_h = shm_w = 0

    try:
        # Wait for buffer config from main process (sent when first frame arrives)
        log.info("Inference process: waiting for buffer config...")
        config = config_queue.get(timeout=300)  # 5 min timeout if stream never connects
        if config is None or stop_event.is_set():
            return
        shm_name, shm_h, shm_w = config
        log.info(
            "Inference process: received config shm=%s %dx%d", shm_name, shm_h, shm_w
        )

        shm = SharedMemory(name=shm_name)

        # Initialize components (in this process)
        init_database()
        runtime = Runtime()
        from deep_sort_realtime.deepsort_tracker import DeepSort

        tracker = DeepSort(max_age=30, n_init=3)

        # State (lives in inference process only)
        description_buffer = []
        frame_count = 0
        person_history = {}
        person_last_state = {}
        person_observations = []
        last_seen_update_interval = 30
        frames_since_last_seen_update = 0
        latest_summary = ""

        log.info("Inference process: ready, entering main loop")

        while not stop_event.is_set():
            # Wait for frame (with timeout to check stop_event periodically)
            if not frame_ready_event.wait(timeout=0.5):
                continue

            # Copy frame from shared memory
            buf = np.ndarray((shm_h, shm_w, 3), dtype=np.uint8, buffer=shm.buf)
            frame = buf.copy()

            # Signal "empty" so main can put next frame
            frame_ready_event.clear()

            # Run inference
            t0 = time.perf_counter()
            runtime_detections = runtime.run(frame)
            detections, counts, person_detections_for_tracker = process_detections(
                runtime_detections
            )

            tracked_person_boxes = {}
            if person_detections_for_tracker:
                try:
                    tracks = tracker.update_tracks(
                        person_detections_for_tracker, frame=frame
                    )
                except IndexError as e:
                    log.error("Inference: tracker IndexError: %s", e, exc_info=True)
                    continue
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = int(track.track_id)
                    ltrb = track.to_ltrb()
                    if ltrb is not None:
                        x1, y1, x2, y2 = map(int, ltrb)
                        tracked_person_boxes[track_id] = (x1, y1, x2, y2)

            description = format_detection_description(counts)
            description_buffer.append(description)
            if len(description_buffer) > 50:
                description_buffer.pop(0)

            for det in detections:
                if det["class_name"] == "Person":
                    for tid, pbox in tracked_person_boxes.items():
                        if _boxes_overlap(det["bbox"], pbox):
                            det["track_id"] = tid
                            break

            now = datetime.now()
            frames_since_last_seen_update += 1
            do_last_seen_db_update = (
                frames_since_last_seen_update >= last_seen_update_interval
            )

            for track_id, person_bbox in tracked_person_boxes.items():
                if track_id not in person_history:
                    person_history[track_id] = {
                        "first_seen": now,
                        "last_seen": now,
                    }
                    insert_person(track_id, now, now)
                else:
                    person_history[track_id]["last_seen"] = now
                    if do_last_seen_db_update:
                        update_person_last_seen(track_id, now)

                ppe_status = _associate_ppe_to_person(person_bbox, detections)
                current_state = (
                    ppe_status["hardhat"],
                    ppe_status["vest"],
                    ppe_status["mask"],
                )
                last_state = person_last_state.get(track_id)

                if last_state is None or last_state != current_state:
                    person_observations.append(
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
                    person_last_state[track_id] = current_state

                if len(person_observations) > 1000:
                    person_observations = person_observations[-1000:]

            if do_last_seen_db_update:
                frames_since_last_seen_update = 0

            frame_count += 1
            if frame_count % 50 == 0:
                latest_summary = generate_summary(description_buffer)

            duration_ms = (time.perf_counter() - t0) * 1000
            if duration_ms > 100:
                log.info(
                    "Inference: frame %d took %.0fms",
                    frame_count,
                    duration_ms,
                )

            # Ensure native Python types for reliable pickling across processes
            detections_clean = []
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                item = {
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": float(d["confidence"]),
                    "class_id": int(d["class_id"]),
                    "class_name": str(d["class_name"]),
                }
                if d.get("track_id") is not None:
                    item["track_id"] = int(d["track_id"])
                detections_clean.append(item)

            # Put results for main process (block briefly to avoid dropping)
            try:
                results_queue.put(
                    (detections_clean, description, latest_summary or ""),
                    timeout=2.0,
                )
            except queue.Full:
                log.warning("Inference: results queue full, dropping result")

    except Exception as e:
        log.exception("Inference process: crashed: %s", e)
    finally:
        log.info("Inference process: exiting")
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass


class MultiModalAIDemo:
    """Core video analysis pipeline for detection, summaries, and chat context.

    Inference runs in a separate process to avoid GIL contention. Main process
    reads frames, puts them in shared memory when inference is ready, and
    consumes results for display.
    """

    def __init__(self, video_source):
        """Initialize the demo with a video source (RTSP/HTTP stream URL)."""
        self.video_source = video_source
        self.cap = None
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
        self.latest_detection = defaultdict(int)
        self.latest_summary = ""
        self.latest_tracked_persons = []

        # Frame reader thread
        self._reader_thread = None
        self._reader_stop = threading.Event()
        self._latest_frame = None
        self._latest_frame_id = None
        self._latest_frame_lock = threading.Lock()
        self._reconnect_needed = False

        # Display state (updated by results consumer from inference process)
        self._display_lock = threading.Lock()
        self._display_detections = []
        self._display_description = ""
        self._display_summary = ""
        self._results_received_count = 0

        # Multiprocessing: inference process and IPC
        self._inference_process = None
        self._config_queue = None
        self._results_queue = None
        self._stop_event = None
        self._frame_ready_event = None
        self._shm = None
        self._shm_h = self._shm_w = 0
        self._shm_initialized = False
        self._results_consumer_thread = None

    def setup_components(self):
        """Load models and initialize runtime components."""
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            log.info("Stream not ready at startup, retrying until connected...")
            attempt = 0
            while True:
                time.sleep(2)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.video_source)
                if self.cap.isOpened():
                    log.info(f"Stream connected (attempt {attempt + 1})")
                    break
                attempt += 1
                if attempt % 5 == 0 and attempt > 0:
                    log.info("Still waiting for video stream (attempt %d)...", attempt)
        log.info("Video is live streaming")
        self._start_frame_reader()

        # Initialize PostgreSQL (main process)
        init_database()
        log.info("PostgreSQL database initialized")

        # Create IPC queues and events
        self._config_queue = Queue(maxsize=1)
        self._results_queue = Queue(maxsize=1)
        self._stop_event = Event()
        self._frame_ready_event = Event()

        # Spawn inference process early (waits for config on first frame)
        self._stop_event.clear()
        self._inference_process = Process(
            target=_inference_process_target,
            args=(
                self._config_queue,
                self._results_queue,
                self._stop_event,
                self._frame_ready_event,
            ),
            daemon=True,
        )
        self._inference_process.start()
        log.info("Inference process spawned (waiting for first frame)")

        # Start results consumer thread
        self._results_consumer_thread = threading.Thread(
            target=self._results_consumer_loop,
            daemon=True,
        )
        self._results_consumer_thread.start()
        log.info("Results consumer thread started")

        # Register shutdown
        atexit.register(self._shutdown)

    def _shutdown(self):
        """Gracefully stop inference process on exit."""
        log.info("Shutting down: stopping inference process...")
        self._stop_event.set()
        if self._inference_process is not None and self._inference_process.is_alive():
            self._inference_process.join(timeout=5.0)
            if self._inference_process.is_alive():
                self._inference_process.terminate()
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass

    def _results_consumer_loop(self):
        """Consume results from inference process and update display state."""
        try:
            while True:
                try:
                    result = self._results_queue.get(timeout=0.5)
                    if result is None:
                        break
                    detections, description, summary = result
                    with self._display_lock:
                        self._display_detections = list(detections)
                        self._display_description = description
                        self._display_summary = summary or self._display_summary
                        self._results_received_count += 1
                        if self._results_received_count == 1:
                            log.info(
                                "Results consumer: first result received (%d detections)",
                                len(detections),
                            )
                except queue.Empty:
                    pass  # Timeout, keep looping
                except Exception as e:
                    log.warning("Results consumer: failed to process result: %s", e)
        except Exception as e:
            log.exception("Results consumer: %s", e)
        finally:
            log.info("Results consumer: exiting")

    def format_detection_description(
        self, detections_class_count: dict[str, int]
    ) -> str:
        """Build a short, human-readable description from detection counts."""
        return format_detection_description(detections_class_count)

    def append_description(self, description):
        """No-op in main process; inference process maintains its own buffer."""
        pass

    def generate_image_description(self, frame):
        """Legacy: not used when inference is in separate process."""
        return ""

    def generate_summary(self, descriptions):
        """Summarize PPE compliance over a list of detection descriptions."""
        return generate_summary(descriptions)

    def get_latest_detection(self):
        """Return the most recent detection counts."""
        return self.latest_detection

    def get_latest_summary(self):
        """Return the most recent summary."""
        with self._display_lock:
            return self._display_summary or self.latest_summary

    def get_latest_tracked_persons(self):
        """Return the most recent tracked persons with their PPE status."""
        return self.latest_tracked_persons

    def _frame_reader_loop(self):
        """Continuously read frames to consume stream; keeps only latest."""
        try:
            self._frame_reader_loop_impl()
        except Exception as e:
            log.exception("Frame reader: thread crashed: %s", e)
        finally:
            log.info("Frame reader: thread exiting")

    def _frame_reader_loop_impl(self):
        fail_count = 0
        read_count = 0
        while not self._reader_stop.is_set():
            cap = self.cap
            if cap is None or not cap.isOpened():
                time.sleep(0.05)
                continue
            success, frame = cap.read()
            if success and frame is not None:
                if fail_count > 0:
                    log.info(
                        "Frame reader: recovered after %d consecutive failures",
                        fail_count,
                    )
                fail_count = 0
                read_count += 1
                if read_count % 10 == 0:
                    log.info(
                        "Frame reader: received %d frames from stream",
                        read_count,
                    )
                with self._latest_frame_lock:
                    self._latest_frame = frame.copy()
                    self._latest_frame_id = read_count
            elif self._reader_stop.is_set():
                break
            else:
                fail_count += 1
                if fail_count == 1:
                    log.warning(
                        "Frame reader: first consecutive failure (cap.read returned False)"
                    )
                elif fail_count in (5, 10, 15, 20, 25):
                    log.info("Frame reader: %d consecutive failures", fail_count)
                if fail_count >= 30:
                    self._reconnect_needed = True
                    log.warning(
                        "Frame reader: 30 consecutive failures, setting _reconnect_needed"
                    )
                time.sleep(0.01)

    def _start_frame_reader(self):
        """Start background thread that consumes stream as fast as possible."""
        self._reader_stop.clear()
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
            self._latest_frame_id = None

    def _reconnect_stream(self):
        """Reconnect to live stream with retry and backoff."""
        self._stop_frame_reader()
        max_retries = 5
        base_delay = 2
        for attempt in range(max_retries):
            try:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                    time.sleep(0.5)
                self.cap = cv2.VideoCapture(self.video_source)
                if self.cap.isOpened():
                    log.info(f"Reconnected to stream (attempt {attempt + 1})")
                    self._start_frame_reader()
                    return True
            except Exception as e:
                log.warning(f"Reconnect attempt {attempt + 1} failed: {e}")
            delay = base_delay * (2**attempt)
            time.sleep(delay)
        log.error("Failed to reconnect to stream after all retries")
        return False

    def get_frame_for_display(self, resize_to=None):
        """
        Get the latest frame and detections for display. Never blocks on inference.

        When shared memory queue is empty: put frame in buffer for inference.
        When shared memory queue is not empty: do not overwrite; just return frame for display.

        Returns (frame, detections, frame_id). Frame may be None if unavailable.
        """
        frame = None
        frame_id = None
        if self._reconnect_needed:
            log.warning("get_frame_for_display: returning None (_reconnect_needed)")
            return None, [], None

        with self._latest_frame_lock:
            if self._latest_frame is not None:
                frame = self._latest_frame.copy()
                frame_id = self._latest_frame_id

        if frame is None:
            return None, [], None

        # Shared memory producer: only put frame when inference has consumed previous
        if not self._frame_ready_event.is_set():
            h, w = frame.shape[:2]
            size = h * w * 3

            if not self._shm_initialized:
                # First frame: create shared buffer and send config to inference process
                try:
                    self._shm = SharedMemory(create=True, size=size)
                    self._shm_h, self._shm_w = h, w
                    self._config_queue.put((self._shm.name, h, w))
                    self._shm_initialized = True
                    log.info(
                        "Shared buffer created %dx%d, config sent to inference", h, w
                    )
                except Exception as e:
                    log.exception("Failed to create shared memory: %s", e)
                    return frame, list(self._display_detections), frame_id

            # Only write if dimensions match (resolution unchanged)
            if h == self._shm_h and w == self._shm_w:
                buf = np.ndarray((h, w, 3), dtype=np.uint8, buffer=self._shm.buf)
                buf[:] = frame
                self._frame_ready_event.set()

        with self._display_lock:
            detections = list(self._display_detections)

        if resize_to:
            h, w = frame.shape[:2]
            tw, th = resize_to
            scale_x = tw / w
            scale_y = th / h
            frame = cv2.resize(frame, resize_to)
            scaled = []
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                scaled.append(
                    {
                        **d,
                        "bbox": (
                            round(x1 * scale_x),
                            round(y1 * scale_y),
                            round(x2 * scale_x),
                            round(y2 * scale_y),
                        ),
                    }
                )
            detections = scaled

        return frame, detections, frame_id

    def capture_and_update(self, resize_to=None, caller=None):
        """Legacy: not used when inference is in separate process."""
        return None, []

    def generate_frames(self):
        """Legacy: not used when inference is in separate process."""
        pass
