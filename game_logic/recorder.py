"""Recording for the demo video: python main.py --record

During the game, every shown frame is saved twice: the webcam image (what the game gets, mirrored like on
the screen) and the game image (what is on the screen). They are saved in a background thread, so the game
does not get slower. When the game closes, ffmpeg makes three videos in recordings/<date_time>/:

    webcam.mp4         the webcam
    game.mp4           the game, with the music of the songs that were played
    side_by_side.mp4   webcam and game next to each other, with the music (for the peer-review video)

All three are H.264 MP4 (plays everywhere) at MAX_FPS. A frame that took longer is repeated, so the videos
play at real speed and stay in sync with the music. The song-selection previews are not in the sound.
"""
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

import config


class Recorder:
    def __init__(self, folder=None):
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
        self.folder = Path(folder or base / "recordings" / stamp)
        self.fps = config.MAX_FPS
        self.start = None               # time.perf_counter() of the first frame
        self.songs = []                 # (audio file, seconds after the start of the recording) of every song
        self.frames = queue.Queue(maxsize=50)
        self.writers = None
        self.written = 0
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def add(self, webcam_image, game_image):
        """Called by the main loop for every shown frame."""
        now = time.perf_counter()
        if self.start is None:
            self.start = now
        try:
            self.frames.put_nowait((now - self.start, webcam_image, game_image))
        except queue.Full:
            pass                        # the writer is behind: this frame is repeated instead (see _write)

    def song_started(self, audio_path):
        """Called when a song starts playing (state Playing), for the sound of the video."""
        if audio_path is not None and self.start is not None:
            self.songs.append((Path(audio_path), time.perf_counter() - self.start))

    def _write(self):
        while True:
            item = self.frames.get()
            if item is None:
                break
            t, webcam, game = item
            if self.writers is None:
                self.folder.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")         # fast to write; made smaller by ffmpeg at the end
                self.writers = [cv2.VideoWriter(str(self.folder / name), fourcc, self.fps,
                                                (image.shape[1], image.shape[0]))
                                for name, image in (("webcam.avi", webcam), ("game.avi", game))]
            # real speed: write the frame as many times as needed to reach this moment
            due = int(round(t * self.fps)) + 1
            for _ in range(max(1, due - self.written)):
                self.writers[0].write(webcam)
                self.writers[1].write(game)
                self.written += 1

    def finish(self):
        """Stop recording and make the MP4 files (takes some seconds). Returns the folder, or None."""
        self.frames.put(None)
        self.thread.join()
        if self.writers is None:
            print("recording: no frames, nothing written")
            return None
        for writer in self.writers:
            writer.release()
        print(f"recording: {self.written / self.fps:.1f} s, making the videos ...")
        if shutil.which("ffmpeg") is None:
            print(f"recording: ffmpeg not found; the raw videos are in {self.folder} (webcam.avi, game.avi)")
            return self.folder
        webcam, game = self.folder / "webcam.avi", self.folder / "game.avi"
        ok = all(self._ffmpeg(*job) for job in (
            ([webcam], None, self.folder / "webcam.mp4"),
            ([game], "game", self.folder / "game.mp4"),
            ([webcam, game], "side_by_side", self.folder / "side_by_side.mp4")))
        if ok:
            webcam.unlink()                                      # the raw files are only needed until here
            game.unlink()
        print(f"recording written to {self.folder}")
        return self.folder

    def _ffmpeg(self, videos, kind, out):
        """Convert to H.264 MP4. kind "game" / "side_by_side": add the music; side_by_side: both videos next to
        each other (each 960 x 540, so the video is 1920 x 540)."""
        command = ["ffmpeg", "-y", "-loglevel", "error"]
        for video in videos:
            command += ["-i", str(video)]
        songs = self.songs if kind else []
        for audio, _ in songs:
            command += ["-i", str(audio)]
        filters, video_out = [], "0:v"
        if kind == "side_by_side":
            filters.append("[0:v]scale=960:540[a];[1:v]scale=960:540[b];[a][b]hstack=inputs=2[v]")
            video_out = "[v]"
        audio_out = None
        if songs:
            # every song from the moment it started, at most SONG_MAX_TIME long, with the fade-out
            parts = []
            for k, (_, offset) in enumerate(songs):
                stream = f"{len(videos) + k}:a"
                trim = ""
                if config.SONG_MAX_TIME is not None:
                    fade_start = max(0.0, config.SONG_MAX_TIME - config.SONG_FADE_TIME)
                    trim = f"atrim=0:{config.SONG_MAX_TIME},afade=t=out:st={fade_start}:d={config.SONG_FADE_TIME},"
                delay = int(offset * 1000)
                filters.append(f"[{stream}]{trim}adelay={delay}|{delay}[s{k}]")
                parts.append(f"[s{k}]")
            filters.append("".join(parts) + f"amix=inputs={len(parts)}:normalize=0[music]" if len(parts) > 1
                           else f"{parts[0]}anull[music]")
            audio_out = "[music]"
        if filters:
            command += ["-filter_complex", ";".join(filters)]
        command += ["-map", video_out]
        if audio_out:
            command += ["-map", audio_out, "-c:a", "aac", "-b:a", "160k"]
        command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-r", str(self.fps), "-t", f"{self.written / self.fps:.2f}", str(out)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"recording: ffmpeg could not make {out.name}: {result.stderr.strip()[-300:]}")
            return False
        return True
