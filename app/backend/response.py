"""
Processes raw inference output into the application's detection format.

This module is backend-agnostic: it consumes raw model outputs (e.g., YOLO-format
tensors from inference servers) and produces structured detections suitable for
the PPE compliance pipeline. The output is used to draw bounding boxes around
detected objects in the video feed. It handles:

- Parsing raw tensors into Detection objects (with NMS, confidence filtering)
- Converting bounding boxes to pixel coordinates
- Filtering classes and formatting for display and tracking

Using a separate module keeps inference backend details (OVMS, Triton, etc.)
decoupled from the rest of the application. If the inference backend changes,
only this module and the runtime need to be updated.
"""

import cv2
import numpy as np
from collections import defaultdict
from pydantic import BaseModel


class Detection(BaseModel):
    """A single detection from the inference model."""

    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]  # [x, y, w, h] in model coordinates
    scale: float


# Classes to exclude from PPE pipeline (not person or PPE items)
EXCLUDED_CLASSES = frozenset(["Safety Cone", "Safety Vest", "machinery", "vehicle"])


def postprocess_image(
    outputs, scale: float, classes: dict[int, str]
) -> list[Detection]:
    """
    Parse raw inference tensor into Detection objects.

    Expects YOLO-style output format. Applies confidence threshold (0.25) and
    Non-Maximum Suppression.

    Args:
        outputs: Raw model output (numpy array or compatible).
        scale: Scale factor from preprocessing (model coords → original image).
        classes: Mapping of class_id to class_name.

    Returns:
        List of Detection objects.
    """
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(
            classes_scores
        )
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.20, 0.45, 0.5)

    detections: list[Detection] = []
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = Detection(
            class_id=class_ids[index],
            class_name=classes[class_ids[index]],
            confidence=scores[index],
            bbox=box,
            scale=scale,
        )
        detections.append(detection)

    return detections


def process_detections(
    runtime_detections: list[Detection],
) -> tuple[list[dict], defaultdict, list]:
    """
    Convert Detection objects into app format for display and tracking.

    Args:
        runtime_detections: Detections from postprocess_image or runtime.

    Returns:
        Tuple of:
        - detections: List of dicts with bbox (x1,y1,x2,y2), confidence,
          class_id, class_name.
        - counts: defaultdict of class_name -> count (excluding EXCLUDED_CLASSES).
        - person_detections_for_tracker: List of ([left, top, w, h], conf, "Person")
          for DeepSORT.
    """
    detections = []
    counts = defaultdict(int)
    person_detections_for_tracker = []

    for d in runtime_detections:
        if d.class_name in EXCLUDED_CLASSES:
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
        if d.class_name == "Person" and d.confidence > 0.5:
            width = x2 - x1
            height = y2 - y1
            if width > 0 and height > 0:
                person_detections_for_tracker.append(
                    ([x1, y1, width, height], d.confidence, "Person")
                )

    return detections, counts, person_detections_for_tracker
