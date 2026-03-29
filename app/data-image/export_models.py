#!/usr/bin/env python3
"""Export every *.pt under WEIGHTS_DIR to OpenVino IR and ONNX (Ultralytics).

Run after: pip install git+https://github.com/openai/CLIP.git
(in the exporter image Dockerfile).
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/weights")
OUT_OPENVINO = os.environ.get("OUT_OPENVINO", "/exported-openvino")
OUT_ONNX = os.environ.get("OUT_ONNX", "/exported-onnx")
EXCLUDE_STEMS = frozenset(
    s.strip()
    for s in os.environ.get("EXPORT_EXCLUDE_STEMS", "custome_ppe").split(",")
    if s.strip()
)


def main() -> None:
    os.makedirs(OUT_OPENVINO, exist_ok=True)
    os.makedirs(OUT_ONNX, exist_ok=True)

    pts = sorted(glob.glob(os.path.join(WEIGHTS_DIR, "*.pt")))
    if not pts:
        print(f"No .pt files in {WEIGHTS_DIR}", file=sys.stderr)
        sys.exit(1)

    for pt_path in pts:
        stem = os.path.splitext(os.path.basename(pt_path))[0]
        if stem in EXCLUDE_STEMS:
            print(f"skip (excluded): {stem}")
            continue

        d_ov = os.path.join(OUT_OPENVINO, stem, "1")
        xml_path = os.path.join(d_ov, f"{stem}.xml")
        if os.path.isfile(xml_path):
            print(f"skip (exists): {stem}")
            continue

        tmp_pt = f"/tmp/{stem}.pt"
        shutil.copy(pt_path, tmp_pt)
        subprocess.run(
            ["yolo", "export", f"model={tmp_pt}", "format=openvino", "task=detect"],
            check=True,
        )
        os.makedirs(d_ov, exist_ok=True)
        ov_dir = f"/tmp/{stem}_openvino_model"
        for pat, dest_name in (("*.xml", f"{stem}.xml"), ("*.bin", f"{stem}.bin")):
            found = glob.glob(os.path.join(ov_dir, pat))
            if not found:
                print(f"ERROR: no {pat} in {ov_dir}", file=sys.stderr)
                sys.exit(1)
            shutil.copy(found[0], os.path.join(d_ov, dest_name))

        subprocess.run(
            ["yolo", "export", f"model={tmp_pt}", "format=onnx", "task=detect"],
            check=True,
        )
        d_onx = os.path.join(OUT_ONNX, f"{stem}-onnx", stem, "1")
        os.makedirs(d_onx, exist_ok=True)
        onnx_src = f"/tmp/{stem}.onnx"
        shutil.copy(onnx_src, os.path.join(d_onx, "model.onnx"))
        print(f"exported: {stem}")

    print("export_models: done")


if __name__ == "__main__":
    main()
