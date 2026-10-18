from pathlib import Path

ROOT = Path(__file__).resolve().parent
POSES_PATH = ROOT / "choreography" / "poses.json"
SONG_PATH = ROOT / "choreography" / "rasputin.json"
YOLO_MODEL_PATH = ROOT / "yolo26n-pose.pt"

HOLD_TIME = 1.5          # seconds to hold "hands up"
LOST_TIME = 0.5          # seconds not visible before a player is "lost"
COUNTDOWN_TIME = 3.0
RESULTS_MIN_TIME = 3.0   # gestures do not work before this
RESULTS_MAX_TIME = 20.0  # back to registration after this
GRADE_GRACE = 0.3        # extra seconds to hit a pose after the move ends
RATING_TIME = 1.0        # how long "Perfect" etc. stays on screen
DEFAULT_MOVE_BEATS = 2
MIN_LIMBS = 3            # fewer visible limbs: do not grade this frame
HEAD_ENLARGEMENT = 1.35
PLAYER_COLOR = "player_color"
SCORE_MULTIPLIER = 4.5
SCORE_PERCENTILE = 70    # points of a move = this percentile of the frame scores: hold the pose, one lucky frame is not enough
DUET_MIN_SCORE = 50 #BOTH PLAYERS 50 points out of 100 for bonus to count
DUET_BONUS = 0.25 #%
DUET_BONUS_TIME = 1.5   # seconds the duet bonus popup stays on screen
