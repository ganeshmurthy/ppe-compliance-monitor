#!/usr/bin/env python3
"""
Import or refresh a YOLO .pt weight for local Podman + OVMS (deploy/local/podman-compose).

Copies the weight into app/models/<stem>.pt if needed, then either:
  - first time for <stem>: run yolo-model-prep only (no cache delete), or
  - replacement: remove /models/ovms/<stem> on model_repo, then run prep again.

Usage:
  python deploy/local/import_model_weight.py bird.pt
  python deploy/local/import_model_weight.py /path/to/custom.pt

Requires: podman, podman-compose (or `podman compose` via PODMAN_COMPOSE=compose).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def deploy_local() -> Path:
    return Path(__file__).resolve().parent


def resolve_pt_path(arg: str, root: Path) -> tuple[Path, str]:
    """Return (path under app/models/<stem>.pt, stem). Copy into app/models if needed."""
    models_dir = root / "app" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    raw = Path(arg).expanduser()
    tried: list[Path] = []

    if raw.is_absolute():
        tried.append(raw)
    elif raw.parent != Path("."):
        tried.append((Path.cwd() / raw).resolve())
    else:
        name = raw.name if raw.suffix.lower() == ".pt" else f"{raw.name}.pt"
        tried.append(models_dir / name)
        tried.append((Path.cwd() / raw).resolve())

    seen: set[Path] = set()
    source: Path | None = None
    for cand in tried:
        try:
            r = cand.resolve()
        except OSError:
            continue
        if r in seen:
            continue
        seen.add(r)
        if r.is_file():
            source = r
            break

    if source is None:
        msg = ", ".join(str(t) for t in tried)
        raise FileNotFoundError(f"No file found for {arg!r}. Tried: {msg}")

    stem = source.stem
    if stem == "custome_ppe":
        print(
            "Error: custome_ppe.pt is excluded by yolo-model-prep.sh; choose another name.",
            file=sys.stderr,
        )
        sys.exit(2)

    target = models_dir / f"{stem}.pt"
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
        print(f"Copied weights → {target}")
    else:
        print(f"Using weights at {target}")

    return target, stem


def compose_cmd(compose_file: Path) -> list[str]:
    mode = (os.environ.get("PODMAN_COMPOSE") or "auto").lower()
    if mode == "compose":
        return ["podman", "compose", "-f", str(compose_file)]
    if mode == "podman-compose":
        return ["podman-compose", "-f", str(compose_file)]
    if shutil.which("podman-compose"):
        return ["podman-compose", "-f", str(compose_file)]
    return ["podman", "compose", "-f", str(compose_file)]


def run(
    cmd: list[str], *, cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, check=check, text=True)


def volume_has_export(volume: str, stem: str) -> bool:
    """True if OpenVINO IR for stem already exists on model_repo."""
    xml = f"/models/ovms/{stem}/1/{stem}.xml"
    r = subprocess.run(
        [
            "podman",
            "run",
            "--rm",
            "-v",
            f"{volume}:/models:ro",
            "docker.io/library/alpine:3.19",
            "test",
            "-f",
            xml,
        ],
        capture_output=True,
    )
    return r.returncode == 0


def clear_stem_on_volume(volume: str, stem: str) -> None:
    run(
        [
            "podman",
            "run",
            "--rm",
            "-v",
            f"{volume}:/models:z",
            "docker.io/library/alpine:3.19",
            "sh",
            "-c",
            f"rm -rf /models/ovms/{stem} && ls /models",
        ]
    )


def find_model_repo_volume(compose_file: Path, compose_base: list[str]) -> str:
    """Resolve Podman volume name backing model_repo."""
    # Prefer a running OVMS container from this compose project
    pr = subprocess.run(
        [*compose_base, "ps", "-q", "ovms"],
        cwd=compose_file.parent,
        capture_output=True,
        text=True,
    )
    cid = (pr.stdout or "").strip().splitlines()
    if cid:
        insp = subprocess.run(
            [
                "podman",
                "inspect",
                cid[-1],
                "--format",
                '{{range .Mounts}}{{if eq .Destination "/models"}}{{.Name}}{{"\\n"}}{{end}}{{end}}',
            ],
            capture_output=True,
            text=True,
        )
        names = [ln.strip() for ln in (insp.stdout or "").splitlines() if ln.strip()]
        if names:
            return names[-1]

    # Fallback: unique *model_repo* named volume
    ls = subprocess.run(
        ["podman", "volume", "ls", "-q"],
        capture_output=True,
        text=True,
    )
    candidates = [
        v.strip()
        for v in (ls.stdout or "").splitlines()
        if v.strip().endswith("_model_repo") or v.strip() == "model_repo"
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        print(
            "Error: could not find a model_repo volume. "
            "Start the stack once (podman-compose up) so the volume is created, "
            "or set up OVMS and try again.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(
        "Error: multiple model_repo volumes found; start ovms (compose up -d ovms) "
        f"so the script can pick the right mount, or prune unused volumes.\n{candidates}",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import or refresh a .pt weight for local OVMS (Podman compose)."
    )
    parser.add_argument(
        "pt",
        help="Weights file or name (e.g. bird.pt, app/models/bird.pt, /path/foo.pt)",
    )
    parser.add_argument(
        "--compose-file",
        type=Path,
        default=None,
        help="Override path to podman-compose.yaml",
    )
    args = parser.parse_args()

    root = repo_root()
    dl = deploy_local()
    compose_file = (args.compose_file or (dl / "podman-compose.yaml")).resolve()
    if not compose_file.is_file():
        print(f"Error: compose file not found: {compose_file}", file=sys.stderr)
        sys.exit(1)

    try:
        _target, stem = resolve_pt_path(args.pt, root)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    cbase = compose_cmd(compose_file)
    volume = find_model_repo_volume(compose_file, cbase)

    if volume_has_export(volume, stem):
        print(f"Existing OpenVINO export found for “{stem}” — treating as replacement.")
        print(f"Removing /models/ovms/{stem} on volume {volume} …")
        clear_stem_on_volume(volume, stem)
    else:
        print(f"No export yet for “{stem}” on volume {volume} — first-time import.")

    # Re-export all .pt weights and regenerate config.json (prep is idempotent per stem)
    print("Running yolo-model-prep …")
    run(
        [*cbase, "run", "--rm", "yolo-model-prep"],
        cwd=compose_file.parent,
    )

    # Reload OVMS with updated /models/ovms/config.json
    pr = subprocess.run(
        [*cbase, "ps", "-q", "ovms"],
        cwd=compose_file.parent,
        capture_output=True,
        text=True,
    )
    if (pr.stdout or "").strip():
        print("Restarting ovms …")
        run([*cbase, "restart", "ovms"], cwd=compose_file.parent)
    else:
        print(
            "OVMS not running — starting ovms (will wait for prep dependency if needed) …"
        )
        run([*cbase, "up", "-d", "ovms"], cwd=compose_file.parent)

    print()
    print("Done.")
    print(f"  Model stem: {stem}")
    print(f"  In the app config, set Model name to: {stem}")
    print("  OVMS gRPC host:port for backend (typical): ovms:8081")


if __name__ == "__main__":
    main()
