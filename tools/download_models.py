"""Download the pretrained MediaPipe models into assets/models/.

Usage:
    python -m tools.download_models            # the models the game uses
    python -m tools.download_models --all      # also the lite/heavy pose variants (for evaluation)

Versions are pinned ("/1/") so everybody runs the exact same weights.
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from core.config import ASSETS

BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "pose_landmarker_full.task": f"{BASE}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "face_landmarker.task": f"{BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}
EXTRA = {
    "pose_landmarker_lite.task": f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "pose_landmarker_heavy.task": f"{BASE}/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}


def download(models: dict[str, str], out_dir: Path = ASSETS / "models", force: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, url in models.items():
        path = out_dir / name
        if path.exists() and not force:
            print(f"ok       {path}")
            continue
        print(f"download {url}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
        print(f"saved    {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true", help="also download lite/heavy pose models")
    p.add_argument("--force", action="store_true", help="re-download existing files")
    args = p.parse_args()
    download({**MODELS, **(EXTRA if args.all else {})}, force=args.force)


if __name__ == "__main__":
    main()
