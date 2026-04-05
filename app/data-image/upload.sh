#!/bin/sh
set -e

echo "=== PPE Compliance Monitor Data Uploader ==="

export MC_CONFIG_DIR=/tmp/.mc

echo "Waiting for MinIO to be ready..."
until mc alias set myminio "${MINIO_ENDPOINT}" "${MINIO_ACCESS_KEY}" "${MINIO_SECRET_KEY}" 2>/dev/null; do
	echo "MinIO not ready, retrying in 2 seconds..."
	sleep 2
done
echo "MinIO connection established"

echo "Creating buckets..."
mc mb --ignore-existing myminio/models
mc mb --ignore-existing myminio/data
mc mb --ignore-existing myminio/config
echo "Buckets ready"

RUNTIME_TYPE="${RUNTIME_TYPE}"
echo "Runtime type: ${RUNTIME_TYPE}"

if [ "$RUNTIME_TYPE" = "openvino" ]; then
	echo "Checking / uploading OpenVINO model trees..."
	for d in /upload/models/*/; do
		[ -d "$d" ] || continue
		base=$(basename "$d")
		case "$base" in *-onnx) continue ;; esac
		if [ ! -f "${d}1/${base}.xml" ]; then
			continue
		fi
		if ! mc stat "myminio/models/${base}/1/${base}.xml" >/dev/null 2>&1; then
			echo "Uploading OpenVINO model: ${base}/"
			mc cp --recursive "$d" "myminio/models/${base}/"
		else
			echo "OpenVINO ${base} already present, skipping"
		fi
	done
elif [ "$RUNTIME_TYPE" = "kserve" ]; then
	echo "Checking / uploading ONNX model trees..."
	for d in /upload/models/*-onnx/; do
		[ -d "$d" ] || continue
		base=$(basename "$d")
		stem=${base%-onnx}
		onnx_path="${d}${stem}/1/model.onnx"
		if [ ! -f "$onnx_path" ]; then
			continue
		fi
		if ! mc stat "myminio/models/${base}/${stem}/1/model.onnx" >/dev/null 2>&1; then
			echo "Uploading ONNX model: ${base}/"
			mc cp --recursive "$d" "myminio/models/${base}/"
		else
			echo "ONNX ${base} already present, skipping"
		fi
	done
	# Optional Triton config (GPU / TensorRT); repo default matches ppe-onnx layout
	if [ -f /upload/triton-config/config.pbtxt ] &&
		mc stat "myminio/models/ppe-onnx/ppe/1/model.onnx" >/dev/null 2>&1; then
		echo "Uploading Triton config for ppe-onnx/ppe..."
		mc cp /upload/triton-config/config.pbtxt "myminio/models/ppe-onnx/ppe/config.pbtxt"
	fi
else
	echo "ERROR: Unknown RUNTIME_TYPE '${RUNTIME_TYPE}'. Expected 'openvino' or 'kserve'."
	exit 1
fi

echo "Uploading raw .pt files (for reference / other runtimes)..."
for f in /upload/models-pt/*.pt; do
	[ -f "$f" ] || continue
	bn=$(basename "$f")
	if ! mc stat "myminio/models/${bn}" >/dev/null 2>&1; then
		echo "Uploading ${bn}"
		mc cp "$f" "myminio/models/${bn}"
	else
		echo "${bn} already in bucket, skipping"
	fi
done

echo "Checking video file..."
if ! mc stat myminio/data/combined-video-no-gap-rooftop.mp4 >/dev/null 2>&1; then
	echo "Uploading video (combined-video-no-gap-rooftop.mp4)..."
	mc cp /upload/data/combined-video-no-gap-rooftop.mp4 myminio/data/
	echo "Video uploaded successfully"
else
	echo "Video already exists, skipping"
fi

echo "=== Data upload complete ==="

echo ""
echo "Files in MinIO:"
echo "--- models bucket ---"
mc ls myminio/models/
echo "--- data bucket ---"
mc ls myminio/data/
