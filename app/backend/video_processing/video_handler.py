import atexit
import queue
import threading
import time
import uuid
from collections import Counter, defaultdict, deque

import cv2

from database import init_database, get_detection_classes_pipeline_maps
from logger import get_logger
from video_processing.consumer import FrameConsumer
from video_processing.inference import InferencePool
from video_processing.tracking import DB_WRITER_QUEUE_MAXSIZE, TrackerProcess

log = get_logger(__name__)


class VideoHandler:
    """Core video analysis pipeline for detection, summaries, and chat context.

    Inference runs in a pool of worker threads (each with its own Runtime) so
    multiple frames can be processed concurrently.  The heavy-lifting ops
    (gRPC I/O, cv2, numpy) release the GIL, giving true parallelism.
    """

    DESCRIPTION_BUFFER_SIZE = 50
    DESCRIPTION_VOTE_WINDOW = 50

    def __init__(self, video_source=None):
        """Initialize the demo. video_source can be None; call start_streaming() when user selects a source."""
        self.video_source = video_source
        self.cap = None
        self._is_streaming = False
        self.latest_detection = defaultdict(int)
        self.latest_summary = ""
        self.latest_description = ""
        self._display_lock = threading.Lock()

        self._frame_queue: queue.Queue = queue.Queue(maxsize=150)
        self._inference_out_queue: queue.Queue = queue.Queue(maxsize=350)
        self._stop_event = threading.Event()
        self._consumer: FrameConsumer | None = None
        self._inference_pool: InferencePool | None = None
        self._tracker: TrackerProcess | None = None
        self._active_config_id: int | None = None
        self._class_names_in_order: list[str] = []
        self._description_buffer: deque[str] = deque(
            maxlen=self.DESCRIPTION_BUFFER_SIZE
        )
        self._description_vote_buffer: deque[str] = deque(
            maxlen=self.DESCRIPTION_VOTE_WINDOW
        )
        self._epoch: int = 0

        # Client registry for broadcasting video to multiple connected clients.
        # Each client gets its own queue to receive MJPEG chunks.
        self._clients: dict[str, queue.Queue] = {}
        self._clients_lock = threading.Lock()

        # Broadcaster thread that pulls from inference queue, draws/encodes once,
        # and sends the same frame to all registered clients.
        self._broadcaster: threading.Thread | None = None

        self.init_setup()

    def init_setup(self):
        init_database()
        log.info("PostgreSQL database initialized")

        self._consumer = FrameConsumer(None, self._frame_queue, self._stop_event)
        self._consumer.start()
        log.info("FrameConsumer thread started (idle)")

        self._inference_pool = InferencePool(
            self._frame_queue, self._inference_out_queue, self._stop_event
        )
        self._inference_pool.start()
        log.info("InferencePool started (idle)")

        self._tracker = TrackerProcess()
        self._tracker.start()
        log.info("TrackerProcess started (idle)")

        self._queue_monitor = threading.Thread(
            target=self._log_queue_sizes,
            name="queue-monitor",
            daemon=True,
        )
        self._queue_monitor.start()

        atexit.register(self._shutdown)

    def register_client(self) -> tuple[str, queue.Queue]:
        """Register a new client for video streaming.

        Returns:
            tuple: (client_id, client_queue) where client_queue receives MJPEG chunks
        """
        client_id = str(uuid.uuid4())
        # Queue size of 30 frames = ~1 second buffer at 30 FPS
        # If client can't keep up, old frames are dropped to stay near real-time
        client_queue = queue.Queue(maxsize=30)

        with self._clients_lock:
            self._clients[client_id] = client_queue
            client_count = len(self._clients)

        log.info(f"Client {client_id} registered. Total clients: {client_count}")
        return client_id, client_queue

    def unregister_client(self, client_id: str) -> None:
        """Unregister a client when they disconnect.

        Args:
            client_id: The unique identifier for the client to remove
        """
        with self._clients_lock:
            if client_id in self._clients:
                del self._clients[client_id]
                client_count = len(self._clients)
                log.info(
                    f"Client {client_id} unregistered. Remaining clients: {client_count}"
                )

                # If this was the last client, stop streaming to save resources
                if client_count == 0 and self._is_streaming:
                    log.info("Last client disconnected. Stopping stream.")
                    # Schedule stop_streaming to avoid holding the lock
                    threading.Thread(target=self.stop_streaming, daemon=True).start()

    def get_active_client_count(self) -> int:
        """Get the number of currently connected clients.

        Returns:
            int: Number of active clients
        """
        with self._clients_lock:
            return len(self._clients)

    def _log_queue_sizes(self) -> None:
        while not self._stop_event.wait(timeout=2.0):
            tracker_q = self._tracker._in_queue if self._tracker else None
            tracker_size = tracker_q.qsize() if tracker_q else -1
            tracker_max = tracker_q._maxsize if tracker_q else -1
            db_depth = self._tracker._db_queue_depth.value if self._tracker else -1
            log.info(
                "queue sizes: frame_queue=%d/%d inference_out=%d/%d tracker_in=%d/%d db_writer=%d/%d",
                self._frame_queue.qsize(),
                self._frame_queue.maxsize,
                self._inference_out_queue.qsize(),
                self._inference_out_queue.maxsize,
                tracker_size,
                tracker_max,
                db_depth,
                DB_WRITER_QUEUE_MAXSIZE,
            )

    def _shutdown(self):
        log.info("Shutting down VideoHandler")
        self._stop_event.set()
        if self._tracker is not None:
            self._tracker.stop()
        if self._inference_pool is not None:
            self._inference_pool.stop()
        if self._consumer is not None:
            self._consumer.stop()

    def start_streaming(self, video_source: str, config_id: int):
        """Start video streaming with the specified source and configuration.

        Args:
            video_source: Path to video file or RTSP URL
            config_id: Database ID of the configuration to use

        Note:
            If already streaming the same config, this is a no-op.
            If streaming a different config, the old stream is stopped first.
        """
        # If already streaming this exact config, nothing to do
        if self._is_streaming and self._active_config_id == config_id:
            log.info(f"Already streaming config_id={config_id}, skipping restart")
            return

        # Clear buffers for fresh start
        self._description_buffer.clear()
        self._description_vote_buffer.clear()
        self._stop_event.clear()

        # Set video source and configuration
        self.video_source = video_source
        self._active_config_id = config_id

        # Load detection classes for this configuration
        classes, include_in_counts, trackable, name_to_id = (
            get_detection_classes_pipeline_maps(config_id)
        )
        self._class_names_in_order = [classes[i] for i in sorted(classes)]

        # Configure inference and tracking pipelines
        self._inference_pool.configure(config_id, epoch=self._epoch)
        self._tracker.configure(
            trackable_by_class_id=trackable,
            detection_class_name_to_id=name_to_id,
            epoch=self._epoch,
        )

        # Start capturing frames from the video source
        self._consumer.set_source(video_source)

        # Mark as streaming
        self._is_streaming = True

        # Start the broadcaster thread if not already running
        if self._broadcaster is None or not self._broadcaster.is_alive():
            self._broadcaster = threading.Thread(
                target=self._broadcast_loop,
                name=f"broadcaster-epoch-{self._epoch}",
                daemon=True,
            )
            self._broadcaster.start()
            log.info(f"Broadcaster thread started for epoch={self._epoch}")

        log.info(f"Streaming started: source={video_source} config_id={config_id}")

    def stop_streaming(self):
        """Stop the current video stream and clean up resources.

        This is called when:
        - The last client disconnects
        - Switching to a different video source
        - Explicitly requested via API
        """
        log.info("Stopping stream...")

        # Mark as not streaming (broadcaster thread will exit)
        self._is_streaming = False

        # Increment epoch to invalidate any in-flight frames
        self._epoch += 1

        # Stop capturing frames from video source
        if self._consumer is not None:
            self._consumer.make_idle()

        # Clear processing queues to release memory
        for q in (self._frame_queue, self._inference_out_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # Clear all client queues
        with self._clients_lock:
            for client_id, client_queue in self._clients.items():
                while not client_queue.empty():
                    try:
                        client_queue.get_nowait()
                    except queue.Empty:
                        break

        # Reset tracker state
        if self._tracker is not None:
            self._tracker.reset()

        # Clear description buffers
        self._description_buffer.clear()
        self._description_vote_buffer.clear()

        # Clear active config
        self._active_config_id = None

        # Wait for broadcaster thread to finish (it checks _is_streaming)
        if self._broadcaster is not None and self._broadcaster.is_alive():
            self._broadcaster.join(timeout=2.0)
            if self._broadcaster.is_alive():
                log.warning("Broadcaster thread did not stop cleanly")

        log.info("Streaming stopped")

    def stop_streaming_if_active_config(self, config_id: int):
        if self._active_config_id != config_id:
            return
        log.info(f"Stopping stream: active config {config_id} was deleted")
        self.stop_streaming()

    def switch_video_source(self, new_video_source: str, new_config_id: int):
        """Switch to a different video source.

        All connected clients will seamlessly switch to the new video.
        Client HTTP connections remain open - they just start receiving
        frames from the new video source.

        Args:
            new_video_source: Path to new video file or RTSP URL
            new_config_id: Database ID of the new configuration to use
        """
        # Stop the current stream (clears queues, increments epoch)
        self.stop_streaming()

        # Start streaming the new source
        # The broadcaster will start sending new frames to all clients
        self.start_streaming(new_video_source, new_config_id)

    def get_frame_and_detection_from_inference(self):
        """Block until an inference result is ready and return it."""
        while not self._stop_event.is_set():
            try:
                return self._inference_out_queue.get(timeout=0.001)
            except queue.Empty:
                continue
        return None

    def _broadcast_loop(self) -> None:
        """Broadcaster thread main loop.

        Pulls inference results, draws bounding boxes once, encodes to JPEG once,
        then broadcasts the same MJPEG chunk to all registered clients.

        This runs in a dedicated thread while _is_streaming is True.
        """
        current_epoch = self._epoch
        log.info(f"Broadcaster thread started (epoch={current_epoch})")

        try:
            while self._is_streaming:
                # Pull next inference result (blocks until available)
                result = self.get_frame_and_detection_from_inference()
                if result is None:
                    continue

                # Skip stale frames from a previous video source
                if result.epoch != current_epoch:
                    log.debug(
                        f"Broadcaster: dropping stale frame (epoch {result.epoch} != {current_epoch})"
                    )
                    continue

                # Update description and summary (used by /latest_info endpoint)
                self.latest_description = self._format_detection_description(
                    result.counts, self._class_names_in_order
                )
                self._description_vote_buffer.append(self.latest_description)
                self._description_buffer.append(self.latest_description)
                self.latest_summary = self._generate_summary(self._description_buffer)

                # Draw bounding boxes ONCE (not per client)
                frame = self.draw_detections(result.frame, result.detections)

                # Encode to JPEG ONCE (not per client)
                chunk = self.encode_mjpeg_chunk(frame)
                if chunk is None:
                    continue

                # Submit detections to tracker for database persistence
                self._tracker.submit(result.detections, epoch=current_epoch)

                # Broadcast the same chunk to all connected clients
                with self._clients_lock:
                    clients_snapshot = list(self._clients.items())

                for client_id, client_queue in clients_snapshot:
                    try:
                        # Non-blocking put - if queue is full, drop oldest frame
                        client_queue.put_nowait(chunk)
                    except queue.Full:
                        # Client is slow - drop their oldest frame and add the new one
                        # This keeps them showing recent frames instead of lagging
                        try:
                            client_queue.get_nowait()  # Remove oldest
                            client_queue.put_nowait(chunk)  # Add newest
                            log.debug(
                                f"Broadcaster: dropped frame for slow client {client_id}"
                            )
                        except queue.Empty:
                            # Race condition - queue became empty, just skip
                            pass

        except Exception as e:
            log.exception(f"Broadcaster thread error: {e}")
        finally:
            log.info(f"Broadcaster thread stopped (epoch={current_epoch})")

    def _format_detection_description(
        self,
        detections_class_count: dict[str, int],
        class_names_in_order: list[str],
    ) -> str:
        """Build a short, human-readable description from counts; order follows config/model indices."""
        description = "Detected: "
        for item in class_names_in_order:
            if detections_class_count.get(item, 0) > 0:
                description += f"{item}: {detections_class_count[item]}, "
        return description.rstrip(", ")

    def _generate_summary(self, descriptions: deque[str]) -> str:
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

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels for each detection onto *frame* (mutated in-place).

        Colors: cyan = tracked target, green = compliant PPE, red = non-compliant, yellow = other.
        Detections below VIDEO_FEED_DRAW_MIN_CONF are skipped.
        """
        VIDEO_FEED_DRAW_MIN_CONF = 0.5

        h_frame, w_frame = frame.shape[:2]
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
            if conf <= VIDEO_FEED_DRAW_MIN_CONF:
                continue
            if detection.get("track_id") is not None:
                color = (0, 255, 255)  # Cyan for tracked targets
            elif currentClass in ["NO-Hardhat", "NO-Safety Vest", "NO-Mask"]:
                color = (0, 0, 255)  # Red for non-compliance
            elif currentClass in ["Hardhat", "Safety Vest", "Mask"]:
                color = (0, 255, 0)  # Green for compliance
            else:
                color = (255, 255, 0)  # Yellow for other objects
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
                lineType=line_type,
            )
            label = f"{currentClass} {conf:.2f}"
            if detection.get("track_id") is not None:
                label = f"{currentClass} #{detection['track_id']} {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            label_y1 = max(0, y1 - text_size[1] - 10)
            label_y2 = y1
            label_x2 = min(w_frame, x1 + text_size[0])
            if label_x2 > x1 and label_y2 > label_y1:
                cv2.rectangle(
                    frame,
                    (x1, label_y1),
                    (label_x2, label_y2),
                    color,
                    -1,
                    lineType=line_type,
                )
            text_y = max(label_y1 + text_size[1] - 2, y1 - 5)
            cv2.putText(
                frame,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                2,
                lineType=line_type,
            )
        return frame

    def encode_mjpeg_chunk(self, frame, quality=95):
        """Encode *frame* as JPEG and wrap it in a multipart MJPEG chunk.

        Returns the chunk bytes ready to yield, or *None* if encoding fails.
        """
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ret:
            log.warning(
                f"Video feed: cv2.imencode failed shape={getattr(frame, 'shape', None)}"
            )
            return None
        frame_bytes = buffer.tobytes()
        header = (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
        )
        return header + frame_bytes + b"\r\n"

    def frame_generator(self):
        """Generator that yields MJPEG chunks to a single HTTP client.

        This is called once per /video_feed HTTP request. Each client gets
        frames from the broadcaster via their own queue.

        Yields:
            bytes: MJPEG-formatted frame chunks
        """
        # If no video is active, return immediately
        # Frontend will show "Select a source to start" placeholder
        if not self._is_streaming:
            log.info("frame_generator: No active video stream")
            return

        # Register this client to receive broadcasted frames
        client_id, client_queue = self.register_client()

        # Frame rate control - match the source video's frame interval
        frame_interval = self._consumer.frame_interval
        last_yield_time = time.perf_counter()

        try:
            while self._is_streaming:
                try:
                    # Pull next frame from this client's queue (with timeout)
                    # The broadcaster thread is filling this queue
                    chunk = client_queue.get(timeout=0.1)
                except queue.Empty:
                    # No frame available yet, check if still streaming and retry
                    continue

                # Maintain frame rate timing
                elapsed = time.perf_counter() - last_yield_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                last_yield_time = time.perf_counter()

                # Yield the MJPEG chunk to the HTTP response
                try:
                    yield chunk
                except (BrokenPipeError, ConnectionResetError, OSError) as e:
                    log.warning(f"Client {client_id} disconnected during yield: {e}")
                    break

        except Exception as e:
            log.exception(f"frame_generator error for client {client_id}: {e}")
        finally:
            # Always unregister the client on disconnect
            self.unregister_client(client_id)

    def get_majority_description(self) -> str:
        """Return the most common description among the last K frames."""
        if not self._description_vote_buffer:
            return ""
        counter = Counter(self._description_vote_buffer)
        return counter.most_common(1)[0][0]

    def get_latested_description(self):
        """Return the majority-vote description over the last K frames."""
        with self._display_lock:
            return self.get_majority_description()

    def get_latest_summary(self):
        """Return the most recent summary."""
        with self._display_lock:
            return self.latest_summary
