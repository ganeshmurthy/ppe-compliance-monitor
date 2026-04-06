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
	echo "Checking / uploading OpenVINO model trees (ovms/<model>/1/)..."
	for d in /upload/models/ovms/*/; do
		[ -d "$d" ] || continue
		base=$(basename "$d")
		case "$base" in *-onnx) continue ;; esac
		if [ ! -f "${d}1/${base}.xml" ]; then
			continue
		fi
		if ! mc stat "myminio/models/ovms/${base}/1/${base}.xml" >/dev/null 2>&1; then
			echo "Uploading OpenVINO model: ovms/${base}/"
			mc cp --recursive "$d" "myminio/models/ovms/${base}/"
		else
			echo "OpenVINO ovms/${base} already present, skipping"
		fi
	done
	if [ -f /upload/models/ovms/config.json ]; then
		echo "Uploading OpenVINO config.json (multi-model OVMS)..."
		mc cp /upload/models/ovms/config.json myminio/models/ovms/config.json
	fi
elif [ "$RUNTIME_TYPE" = "kserve" ]; then
	echo "Checking / uploading Triton ONNX model trees (triton/<model>/1/model.onnx)..."
	for d in /upload/models/triton/*/; do
		[ -d "$d" ] || continue
		stem=$(basename "$d")
		onnx_path="${d}1/model.onnx"
		if [ ! -f "$onnx_path" ]; then
			continue
		fi
		if ! mc stat "myminio/models/triton/${stem}/1/model.onnx" >/dev/null 2>&1; then
			echo "Uploading Triton ONNX model: triton/${stem}/"
			mc cp --recursive "$d" "myminio/models/triton/${stem}/"
		else
			echo "Triton ONNX ${stem} already present, skipping"
		fi
	done
	# Optional Triton config (GPU / TensorRT); repo template targets ppe I/O shape only—do not copy to other stems.
	if [ -f /upload/triton-config/config.pbtxt ] && [ -f /upload/models/triton/ppe/1/model.onnx ]; then
		echo "Uploading Triton config for triton/ppe/..."
		mc cp /upload/triton-config/config.pbtxt myminio/models/triton/ppe/config.pbtxt
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

echo "Checking sample videos in data bucket..."
for vid in combined-video-no-gap-rooftop.mp4 bluejayclear.mp4; do
	if ! mc stat "myminio/data/${vid}" >/dev/null 2>&1; then
		echo "Uploading video (${vid})..."
		mc cp "/upload/data/${vid}" myminio/data/
		echo "Uploaded ${vid}"
	else
		echo "${vid} already in bucket, skipping"
	fi
done

echo "=== Data upload complete ==="

echo ""
echo "Files in MinIO:"
echo "--- models bucket ---"
mc ls myminio/models/
echo "--- data bucket ---"
mc ls myminio/data/
