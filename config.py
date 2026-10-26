from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHOREOGRAPHY_DIR = ROOT / "choreography"     # every .json here (except poses.json) is a song in the song selection
POSES_PATH = ROOT / "choreography" / "poses.json"
SONG_PATH = ROOT / "choreography" / "rasputin.json"   # the song that is selected first
YOLO_MODEL_PATH = ROOT / "yolo26n-pose.pt"
MAX_FPS = 25             # the game never runs faster than this: less work and heat for the laptop
USE_OPENVINO = True      # run YOLO with OpenVINO: about 2x faster on the CPU, same model (see pose_processing.py)

HOLD_TIME = 0.5          # seconds to hold "hands up" (song selection, registration, results); short for the 60 s demo
BROWSE_HOLD_TIME = 0.3   # song selection: seconds to hold one hand up to go to the previous / next song
PREVIEW_START = 0.3      # song selection: the music preview starts at this part of the song (0.3 = 30%)
LOST_TIME = 0.5          # seconds not visible before a player is "lost"
COUNTDOWN_TIME = 2.0     # seconds of "3, 2, 1" before the song (short for the 60 s demo)
RESULTS_MIN_TIME = 3.0   # gestures do not work before this
RESULTS_MAX_TIME = 20.0  # back to registration after this
GRADE_GRACE = 0.3        # extra seconds to hit a pose after the move ends
SHOW_TIMELINE = True     # the coming moves slide to a hit line at the bottom (instead of the NEXT card)
RATING_TIME = 1.0        # how long "Perfect" etc. stays on screen
DEFAULT_MOVE_BEATS = 2
FACE_DETECT_WIDTH = 640   # face tracking looks at a copy of the image this wide (faster, same faces)
CAMERA_SIZE = (1920, 1080)   # the webcam is read at this size (when it can); the full image is kept for small details (faces)
GAME_SIZE = (1280, 720)      # the game works on a copy at most this big: YOLO, tracking, drawing and the screen
BACKGROUND = True            # replace the room with the Just Dance stage (person segmentation, see background/)
SEGMENT_MODEL_PATH = ROOT / "yolo26n-seg.pt"   # Ultralytics downloads it when it is missing
SEGMENT_SIZE = 320           # the segmentation looks at a copy this wide: 320 is 8-11 ms faster per frame than 480
                             # (less competition with the pose model for the CPU), the edges only a little coarser
BOX_OVERLAP = 0.5            # a person belongs to a player when their boxes overlap this much (bystanders vanish)
REMOVE_BYSTANDERS = True     # remove people who are not a player (only during countdown, playing, results)
PLAYER_KEEP_TIME = 1.0       # seconds a player who is not seen for a moment keeps their place on the stage
MASK_SOFTNESS = 1.5          # pixels: how soft the edge around the players is
HEAD_ENLARGEMENT = 1.35
PLAYER_COLOR = "player_color"
SCORE_MULTIPLIER = 3 #higher easier 
SCORE_PERCENTILE = 70    # points of a move = this percentile of the frame scores: hold the pose, one lucky frame is not enough
DUET_MIN_SCORE = 50 #BOTH PLAYERS 50 points out of 100 for bonus to count
DUET_BONUS = 0.25 #%
DUET_BONUS_TIME = 1.5   # seconds the duet bonus popup stays on screen
STAR_BONUS = 100        # points for grabbing the bonus star (a Perfect move is 100)
STAR_INTERVAL = 12.0    # seconds between two stars: the first at 12 s, the next 12 s after the last one is gone ...
STAR_JITTER = 0.0       # ... plus or minus up to this much (0: always at the same time)
STAR_TIMES = [6.0, 21.0]    # fixed song times of the stars (between the animated moves of the demo); None: STAR_INTERVAL
STAR_TIME = 5.0         # seconds a star floats across the screen
STAR_END_FREE = 3.0     # no star in the last seconds of the song
STAR_SIZE = 0.05        # star radius, as part of the image height (36 px at 720p)
STAR_REACH = 0.08       # a wrist this close to the star (part of the image height, 58 px at 720p) grabs it
STAR_BURST_TIME = 0.9   # seconds the "+100" stays after a grab
# animated moves (a list of poses in the song file, see song.animated_move)
CHAIN_MIN_SCORE = 65    # a step with at least this score (Good) keeps the chain going
CHAIN_BONUS = 20        # extra points per step in a chain: the 2nd correct step in a row +20, the 3rd +40, ...
FLOW_CHECK = True       # also check with the motion signal that the player moves the right way between steps
FLOW_MIN = 0.5          # direction match (cosine, -1..1) needed to keep the chain going
MOTION_SMOOTHING = 0.5  # smoothing of the motion per player (as in Task 2)
CHAIN_TIME = 1.0        # seconds "Chain x3" stays above the player
SONG_MAX_TIME = 30.0    # every song stops after this many seconds (fits the 60 s demo video); None: full songs
RECORD_RESULTS_TIME = 10.0   # --record: the song stops this long before the video ends, the results fill the end
SONG_FADE_TIME = 1.0    # seconds the music fades out before it stops at SONG_MAX_TIME
DEBUG_IN_RESULTS = True # the debug drawings (skeletons, boxes, FPS) on the results screen (the end of the demo video)
DEBUG_SHOW_AT = None    # seconds into the song: the debug drawings turn on by themselves for a moment ...
DEBUG_SHOW_FOR = 5.0    # ... for this many seconds. DEBUG_SHOW_AT = None: not during the song
