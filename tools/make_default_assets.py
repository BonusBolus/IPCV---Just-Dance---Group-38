"""Generate the default song (assets/song.wav) and matching choreography (assets/choreography.json).

The game calls `ensure_default_assets` on start-up, so this only needs to be run by hand to
regenerate (e.g. after changing the BPM or length).

Usage:
    python -m tools.make_default_assets [--bpm 120] [--duration 64] [--force]
"""
from __future__ import annotations

import argparse
import logging

from core.config import Config, GameConfig
from gameplay.choreography import Choreography
from scene.music import generate_song

log = logging.getLogger(__name__)


def ensure_default_assets(game: GameConfig, force: bool = False) -> None:
    """Create the song/choreography if missing (only for the default paths)."""
    default = GameConfig()
    if game.song_path == default.song_path and (force or not game.song_path.exists()):
        log.info("Generating %s (%.0f s, %.0f BPM)", game.song_path.name, game.default_song_s, game.bpm)
        generate_song(game.song_path, game.default_song_s, game.bpm)
    if game.choreography_path == default.choreography_path and (force or not game.choreography_path.exists()):
        log.info("Generating %s", game.choreography_path.name)
        Choreography.synthetic(game.default_song_s, game.bpm, song=game.song_path.name).save(game.choreography_path)


def main() -> None:
    logging.basicConfig(level="INFO")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bpm", type=float, default=GameConfig.bpm)
    p.add_argument("--duration", type=float, default=GameConfig.default_song_s)
    p.add_argument("--force", action="store_true", help="overwrite existing files")
    args = p.parse_args()
    cfg = Config()
    cfg.game.bpm, cfg.game.default_song_s = args.bpm, args.duration
    ensure_default_assets(cfg.game, force=args.force)


if __name__ == "__main__":
    main()
