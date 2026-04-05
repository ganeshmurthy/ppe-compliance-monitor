#!/usr/bin/env bash
# Export each /source/*.pt to OpenVINO under /models/<stem>/1/ and write OVMS config.json
set -euo pipefail

pip install --no-cache-dir 'git+https://github.com/openai/CLIP.git'

shopt -s nullglob
pts=(/source/*.pt)
if [[ ${#pts[@]} -eq 0 ]]; then
	echo "ERROR: No .pt files in /source (mount app/models)." >&2
	exit 1
fi

for pt in "${pts[@]}"; do
	stem=$(basename "$pt" .pt)
	if [[ $stem == "custome_ppe" ]]; then
		echo "skip (excluded): $stem"
		continue
	fi
	target_dir="/models/${stem}/1"
	target_xml="${target_dir}/${stem}.xml"
	if [[ -f $target_xml ]]; then
		echo "skip (exists): $stem"
		continue
	fi
	echo "exporting: $stem"
	mkdir -p "$target_dir"
	cp "$pt" "/tmp/${stem}.pt"
	yolo export "model=/tmp/${stem}.pt" format=openvino task=detect
	cp "/tmp/${stem}_openvino_model/"*.xml "$target_xml"
	cp "/tmp/${stem}_openvino_model/"*.bin "${target_dir}/${stem}.bin"
done

python3 <<'PY'
import json
import os

root = "/models"
entries = []
for name in sorted(os.listdir(root)):
    path = os.path.join(root, name)
    if not os.path.isdir(path) or name.startswith("."):
        continue
    xml = os.path.join(path, "1", f"{name}.xml")
    if os.path.isfile(xml):
        entries.append(
            {"config": {"name": name, "base_path": f"/models/{name}"}}
        )

cfg = {"model_config_list": entries}
out = os.path.join(root, "config.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print(f"Wrote {out} with {len(entries)} model(s): {[e['config']['name'] for e in entries]}")
if not entries:
    raise SystemExit("No models found under /models — export failed?")
PY

echo "yolo-model-prep: complete"
