# Assets

| File | Used by | How to get it |
|---|---|---|
| `models/pose_landmarker_full.task` | pose/pose_estimator.py | `python -m tools.download_models` (MediaPipe, float16, version 1) |
| `models/face_landmarker.task` | face/face_tracker.py | `python -m tools.download_models` |
| `models/pose_landmarker_{lite,heavy}.task` | evaluation only | `python -m tools.download_models --all` |
| `song.wav` | scene/audio.py | generated procedurally (scene/music.py) on first start, or `python -m tools.make_default_assets` |
| `choreography.json` | gameplay/choreography.py | generated to match the song, or from a video: `python -m tools.extract_reference` |
| `effects/{sunglasses,crown,star}.png` | face/stickers.py | optional BGRA images that replace the drawn stickers |
