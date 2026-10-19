from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHOREOGRAPHY_DIR = ROOT / "choreography"     # every .json here (except poses.json) is a song in the song selection
POSES_PATH = ROOT / "choreography" / "poses.json"
SONG_PATH = ROOT / "choreography" / "rasputin.json"   # the song that is selected first
YOLO_MODEL_PATH = ROOT / "yolo26n-pose.pt"
MAX_FPS = 25             # the game never runs faster than this: less work and heat for the laptop
USE_OPENVINO = True      # run YOLO with OpenVINO: about 2x faster on the CPU, same model (see pose_processing.py)

HOLD_TIME = 1.0          # seconds to hold "hands up" (song selection, registration, results)
BROWSE_HOLD_TIME = 0.5   # song selection: seconds to hold one hand up to go to the previous / next song
PREVIEW_START = 0.3      # song selection: the music preview starts at this part of the song (0.3 = 30%)
LOST_TIME = 0.5          # seconds not visible before a player is "lost"
COUNTDOWN_TIME = 3.0
RESULTS_MIN_TIME = 3.0   # gestures do not work before this
RESULTS_MAX_TIME = 20.0  # back to registration after this
GRADE_GRACE = 0.3        # extra seconds to hit a pose after the move ends
SHOW_TIMELINE = True     # the coming moves slide to a hit line at the bottom (instead of the NEXT card)
RATING_TIME = 1.0        # how long "Perfect" etc. stays on screen
DEFAULT_MOVE_BEATS = 2
HEAD_ENLARGEMENT = 1.35
CROWN_LEAD = 0.10        # the crown goes to the player whose score is at least 10% higher than the other's
PLAYER_COLOR = "player_color"
SCORE_MULTIPLIER = 3 #higher easier 
SCORE_PERCENTILE = 70    # points of a move = this percentile of the frame scores: hold the pose, one lucky frame is not enough
DUET_MIN_SCORE = 50 #BOTH PLAYERS 50 points out of 100 for bonus to count
DUET_BONUS = 0.25 #%
DUET_BONUS_TIME = 1.5   # seconds the duet bonus popup stays on screen
