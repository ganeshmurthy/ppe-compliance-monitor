import numpy as np
import cv2
import os
import time
from ovmsclient import make_grpc_client

import tritonclient.grpc as triton_grpc

from logger import get_logger
from response import Detection, postprocess_image

log = get_logger(__name__)


class Runtime:
    def __init__(
        self,
        classes: dict[int, str] | None = None,
        service_url: str | None = None,
        model_name: str | None = None,
    ):
        self.service_url = service_url
        if not self.service_url or not str(self.service_url).strip():
            raise ValueError(
                "Runtime requires service_url (model_url from config). "
                "Add a config with an inferencing URL via the Config dialog."
            )
        _in = (os.getenv("MODEL_INPUT_NAME") or "").strip()
        self.input_name = _in or "x"
        self.model_name = (model_name or "").strip() or (
            os.getenv("MODEL_NAME") or ""
        ).strip()
        if not self.model_name:
            raise ValueError(
                "Runtime requires model_name (OVMS model id from app_config). "
                "Set it in the Configuration dialog."
            )
        _ver = (os.getenv("MODEL_VERSION") or "1").strip() or "1"
        try:
            self.model_version = int(_ver)
        except ValueError:
            self.model_version = 1
        self._model_version_str = str(self.model_version)

        runtime_type = os.getenv("RUNTIME_TYPE", "openvino").lower()
        openshift_mode = os.getenv("OPENSHIFT", "false").lower() == "true"

        if runtime_type == "kserve":
            grpc_url = self.service_url.replace("https://", "").replace("http://", "")
            self._triton_client = triton_grpc.InferenceServerClient(
                url=grpc_url,
                channel_args=[
                    ("grpc.max_send_message_length", -1),
                    ("grpc.max_receive_message_length", -1),
                    ("grpc.optimization_target", "throughput"),
                ],
            )
            self._infer_input = triton_grpc.InferInput(
                self.input_name, [1, 3, 640, 640], "FP32"
            )
            self._infer_output = triton_grpc.InferRequestedOutput("output0")
            self.inference_fun = self.kserve_inference_grpc
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

        self._pad_shape: tuple[int, int] = (0, 0)
        self._padded: np.ndarray | None = None
        self._scale: float = 1.0

    def preprocess_image(self, image: np.ndarray):
        """
        Preprocess the image for the model.
        """
        height, width = image.shape[:2]

        if (height, width) != self._pad_shape:
            self._scale = max(height, width) / 640
            new_w = int(width / self._scale)
            new_h = int(height / self._scale)
            self._resized_shape = (new_w, new_h)
            self._padded = np.zeros((640, 640, 3), np.uint8)
            self._pad_shape = (height, width)

        resized = cv2.resize(image, self._resized_shape, interpolation=cv2.INTER_LINEAR)
        self._padded[: self._resized_shape[1], : self._resized_shape[0]] = resized

        blob = cv2.dnn.blobFromImage(self._padded, scalefactor=1 / 255, swapRB=True)
        return blob, self._scale

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

    def kserve_inference_grpc(self, image: np.ndarray) -> np.ndarray:
        """
        Inference via KServe V2/Open Inference Protocol over gRPC using the
        Triton client.  Avoids HTTP serialization overhead entirely.
        """
        fp32_image = image if image.dtype == np.float32 else image.astype(np.float32)
        self._infer_input.set_data_from_numpy(fp32_image)
        result = self._triton_client.infer(
            model_name=self.model_name,
            model_version=self._model_version_str,
            inputs=[self._infer_input],
            outputs=[self._infer_output],
        )
        return result.as_numpy("output0")

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
