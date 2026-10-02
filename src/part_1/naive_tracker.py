import pandas as pd

import copy
from typing import Dict, Any
from abc import ABC

from ..part_0.synthetic_data import get_synthetic_detections
from .hungarian_match import hungarian_match
from ..tracker import Tracker


class NaiveTracker(Tracker):
    """
    Class for the naive tracker logic, using Hungarian match with fixed
    threshold. Creates new tracks when there isn't a match and kill track
    after k frames without any observation. Keeps track of runned frames internally.
    If you need to restart the tracker, run .reset_tracks().
    """

    _TRACK_COLUMNS = [
        "track_id",
        "bb_left",
        "bb_top",
        "bb_width",
        "bb_height",
        "last_frame",
        "misses",
    ]
    
    
    def __init__(
        self,
        dets_df: pd.DataFrame,
        iou_threshold: float = 0.3,
        kill_track_frames: int = 15,
    ) -> None:
        """
        iou_threshold: Threshold for the Hungarian Match.
        kill_track_frames: Number of frames after which kill a track without any matches.
        dets_df (pd.DataFrame): Detections' dataframe. Must have columns bb_left, bb_top, bb_width, bb_height.
        """
        super().__init__(dets_df)
        
        self.iou_threshold = iou_threshold
        self.max_age = kill_track_frames
        
    def _tracks_to_df(self) -> pd.DataFrame:
        """Convert the internal tracks dict to a DataFrame for hungarian_match."""
        if not self.tracks:
            return pd.DataFrame(
                {col: pd.Series(dtype=dtype) for col, dtype in {
                    "track_id": "int64",
                    "bb_left": "float64",
                    "bb_top": "float64",
                    "bb_width": "float64",
                    "bb_height": "float64",
                    "last_frame": "int64",
                    "misses": "int64",
                }.items()}
            )

        rows = [{"track_id": tid, **state} for tid, state in self.tracks.items()]
        return pd.DataFrame(rows, columns=self._TRACK_COLUMNS)
    
    def update_tracks(self):
        frame_number, frame_det_df = self._get_det_df_frame()
        self._update_current_frame()
        
        # Convert tracks dict -> df only for the hungarian match
        tracks_df = self._tracks_to_df()
        # Calculate the Hungarian match
        matches, _, unmatched_dets = hungarian_match(tracks_df, frame_det_df, self.iou_threshold)

        # 3. Age unmatched tracks — recompute from the dict, NOT from hungarian_match
        matched_track_ids = {int(tid) for tid, _ in matches}
        unmatched_tracks = set(self.tracks.keys()) - matched_track_ids

        # 1. Update matched tracks
        for track_id, det_idx in matches:
            track_id = int(track_id)
            det = frame_det_df.iloc[det_idx]
            track = self.tracks[track_id]
            track['bb_left'] = det['bb_left']
            track['bb_top'] = det['bb_top']
            track['bb_width'] = det['bb_width']
            track['bb_height'] = det['bb_height']
            track['last_frame'] = frame_number
            track['misses'] = 0
            
        # 2. Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = frame_det_df.iloc[det_idx]
            self.tracks[self.next_track_id] = {
                "bb_left": det["bb_left"],
                "bb_top": det["bb_top"],
                "bb_width": det["bb_width"],
                "bb_height": det["bb_height"],
                "last_frame": frame_number,
                "misses": 0,
            }
            self.next_track_id += 1

        # Handle unmatch tracks, increment counters and deleting very old ones
        for track_id in unmatched_tracks:
            if track_id not in self.tracks:
                continue  # safety: skip if it was already removed
            self.tracks[track_id]["misses"] += 1
            if self.tracks[track_id]["misses"] >= self.max_age:
                del self.tracks[track_id]
                
                
        self.final_tracks[frame_number] = copy.deepcopy(self.tracks)

if __name__ == "__main__":
    synthetic_detector = get_synthetic_detections(0).detections # no occlusion first
    det_df = pd.DataFrame(synthetic_detector)
    naive_tracker = NaiveTracker(
        det_df,
    )
    
    print(naive_tracker.final_tracks[1])
    print(naive_tracker.final_tracks[5])