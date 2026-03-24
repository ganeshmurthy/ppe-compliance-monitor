import json
import numpy as np
import cv2
import os
import time
from ovmsclient import make_grpc_client

import requests as http_requests

from logger import get_logger
from response import Detection, postprocess_image

log = get_logger(__name__)

NUMPY_DTYPE_MAP = {
    "FP32": np.float32,
    "FP16": np.float16,
    "INT32": np.int32,
    "INT64": np.int64,
    "INT8": np.int8,
    "UINT8": np.uint8,
    "BOOL": bool,
}


class Runtime:
    def __init__(
        self, classes: dict[int, str] | None = None, service_url: str | None = None
    ):
        self.service_url = service_url
        if not self.service_url or not str(self.service_url).strip():
            raise ValueError(
                "Runtime requires service_url (model_url from config). "
                "Add a config with an inferencing URL via the Config dialog."
            )
        self.input_name = os.getenv("MODEL_INPUT_NAME")
        self.model_name = os.getenv("MODEL_NAME")
        self.model_version = int(os.getenv("MODEL_VERSION"))

        runtime_type = os.getenv("RUNTIME_TYPE", "openvino").lower()
        openshift_mode = os.getenv("OPENSHIFT", "false").lower() == "true"

        if runtime_type == "kserve":
            self._session = http_requests.Session()
            self._infer_url = f"{self.service_url}/v2/models/{self.model_name}/infer"
            self.inference_fun = self.kserve_inference
        elif openshift_mode:
            grpc_url = self.service_url.replace("https://", "").replace("http://", "")
            self._grpc_client = make_grpc_client(grpc_url)
            self.inference_fun = self.remote_inference
        else:
            self._grpc_client = make_grpc_client(self.service_url)
            self.inference_fun = self.local_inference
        if not classes or len(classes) == 0:
            raise ValueError(
                "Runtime requires classes from detection_classes. "
                "Set ACTIVE_CONFIG_ID to a valid app_config id."
            )
        self.CLASSES = classes
        log.info("Runtime using %d classes from config", len(self.CLASSES))

    def preprocess_image(self, image: np.ndarray):
        """
        Preprocess the image for the model.
        """
        # Read the input image
        # image = cv2.imread("team.jpg")
        [height, width, _] = image.shape

        # Prepare a square image for inference (letterbox padding)
        length = max(height, width)
        padded = np.zeros((length, length, 3), np.uint8)
        padded[0:height, 0:width] = image
        image = padded
        scale = length / 640

        # Preprocess: normalize, resize to 640x640, swap BGR->RGB, produce NCHW blob
        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1 / 255, size=(640, 640), swapRB=True
        )
        return blob, scale

    def inference(self, image: np.ndarray) -> np.ndarray:
        """
        Inference the image for the model.
        """
        return self.inference_fun(image)

    def local_inference(self, image: np.ndarray) -> np.ndarray:
        """
        Local inference via persistent gRPC connection to OVMS.
        """
        inputs = {self.input_name: image}
        return self._grpc_client.predict(inputs, self.model_name, self.model_version)

    def remote_inference(self, image: np.ndarray) -> np.ndarray:
        """
        Remote inference via persistent gRPC connection to OVMS (binary protobuf).
        """
        inputs = {self.input_name: image}
        return self._grpc_client.predict(inputs, self.model_name, self.model_version)

    def kserve_inference(self, image: np.ndarray) -> np.ndarray:
        """
        Inference via KServe V2/Open Inference Protocol with binary tensor
        extension.  Sends/receives raw bytes instead of JSON arrays, avoiding
        the massive serialization overhead for large tensors.
        """
        input_bytes = image.astype(np.float32).tobytes()
        header_json = {
            "inputs": [
                {
                    "name": self.input_name,
                    "shape": list(image.shape),
                    "datatype": "FP32",
                    "parameters": {"binary_data_size": len(input_bytes)},
                }
            ],
            "outputs": [{"name": "output0", "parameters": {"binary_data": True}}],
        }
        header_bytes = json.dumps(header_json).encode("utf-8")

        resp = self._session.post(
            self._infer_url,
            data=header_bytes + input_bytes,
            headers={
                "Content-Type": "application/octet-stream",
                "Inference-Header-Content-Length": str(len(header_bytes)),
            },
            timeout=60.0,
        )
        resp.raise_for_status()

        header_len = int(resp.headers["Inference-Header-Content-Length"])
        resp_body = resp.content
        resp_header = json.loads(resp_body[:header_len])
        binary_buf = resp_body[header_len:]

        output = resp_header["outputs"][0]
        dtype = NUMPY_DTYPE_MAP.get(output["datatype"], np.float32)
        byte_size = output["parameters"]["binary_data_size"]
        return np.frombuffer(binary_buf[:byte_size], dtype=dtype).reshape(
            output["shape"]
        )

    def run(self, image: np.ndarray) -> list[Detection]:
        """
        Run the inference for the image.
        """
        t0 = time.perf_counter()
        blob, scale = self.preprocess_image(image)
        t1 = time.perf_counter()
        outputs = self.inference(blob)
        t2 = time.perf_counter()
        detections = postprocess_image(outputs, scale, self.CLASSES)
        t3 = time.perf_counter()

        log.debug(
            f"Inference timing — preprocess: {(t1 - t0) * 1000:.1f}ms, "
            f"inference: {(t2 - t1) * 1000:.1f}ms, "
            f"postprocess: {(t3 - t2) * 1000:.1f}ms, "
            f"total: {(t3 - t0) * 1000:.1f}ms"
        )
        return detections
