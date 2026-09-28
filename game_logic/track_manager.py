import json
import os

class TrackManager:
    def __init__(self, moves_path="assets/moves.json"):
        """
        Initializes the TrackManager and loads the global move library.
        """
        self.move_library = self._load_json(moves_path)
        self.current_song_name = ""
        self.audio_file_path = ""
        self.bpm = 120
        self.timeline = []

    def _load_json(self, file_path):
        """Helper method to safely load JSON files."""
        if not os.path.exists(file_path):
            print(f"Warning: File not found at {file_path}")
            return {}
        with open(file_path, "r") as f:
            return json.load(f)

    def load_song(self, song_path):
        """
        Loads a song's metadata and timeline, injecting the target angles 
        from the global move library into each timestamp cue.
        """
        song_data = self._load_json(song_path)
        
        self.current_song_name = song_data.get("song_name", "Unknown Track")
        self.audio_file_path = song_data.get("audio_file", "")
        self.bpm = song_data.get("bpm", 120)
        raw_timeline = song_data.get("timeline", [])
        
        resolved_timeline = []
        for cue in raw_timeline:
            move_id = cue.get("move_id")
            
            if move_id in self.move_library:
                move_details = self.move_library[move_id]
                
                # Build a complete cue merging timeline data with move specs
                resolved_cue = {
                    "timestamp_start": cue["timestamp_start"],
                    "timestamp_end": cue["timestamp_end"],
                    "move_id": move_id,
                    "display_name": move_details.get("display_name", move_id),
                    "target_angles": move_details.get("target_angles", {}),
                    "constraints": move_details.get("constraints", {})
                }
                resolved_timeline.append(resolved_cue)
            else:
                print(f"Warning: Move ID '{move_id}' referenced in song, but missing from moves.json!")
                
        self.timeline = resolved_timeline
        print(f"Loaded song: '{self.current_song_name}' with {len(self.timeline)} choreographed cues.")
        return song_data

    def get_active_move(self, current_time):
        """
        Finds and returns the move cue that matches the current timestamp of the song.
        Returns None if there is a transition break.
        """
        for cue in self.timeline:
            if cue["timestamp_start"] <= current_time <= cue["timestamp_end"]:
                return cue
        return None