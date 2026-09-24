# Assets

| File | Used by | Notes |
|---|---|---|
| `song.mp3` | scene/audio.py | Use a royalty-free song (the video is shared). Missing = silent mode. |
| `choreography.json` | gameplay/choreography.py | Made with `python -m tools.extract_reference`. Missing = placeholder dance. |
| `model_dance.mp4` | tools/extract_reference.py | Video of the model dancer (one of us). |
| `models/pose_landmarker.task` | pose/pose_estimator.py | Model weights (Task 2); document the download URL + version in the main README. |
| `models/face_landmarker.task` | face/face_tracker.py | Model weights (Task 1); same. |
| `effects/*.png` | face/face_effects.py | BGRA stickers (crown, masks, ...), loaded with `cv2.IMREAD_UNCHANGED`. |
