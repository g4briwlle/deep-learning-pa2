import pandas as pd

import copy
from typing import Dict, Any
from abc import ABC, abstractmethod


class Tracker(ABC):
    """
    Abstract Tracker class. Implements the main methods and __init__ functionalities
    every tracker should have for correct evaluation.
    """
    
    def _update_current_frame(self):
        self.current_frame += 1
        
        if self.current_frame > self.num_frames:
            raise Exception(f"Exceeded max number of frames. Loaded dataset only has {self.num_frames}")
            
    def _reset_frames(self):
        self.current_frame = 0
    
    def reset_tracks(self):
        """
        Reset the entire process of updating tracks for each frame.
        """
        
        self.next_track_id = 0
        self.tracks = {}
        self._reset_frames()
        self.final_tracks = {}
    
    def _get_det_df_frame(self):
        frame_number = self.current_frame + 1
        
        return frame_number, self.dets_df[self.dets_df['frame'] == frame_number]
    
    def __init__(self, dets_df: pd.DataFrame) -> None:
        self.dets_df = dets_df
                
        self.num_frames = len(self.dets_df['frame'].unique())
        self.current_frame = 0
        
        self.reset_tracks()
        
    @abstractmethod
    def update_tracks(self):
        """
        Updates the tracks_df for one frame. When running first time, just creates tracks.
        """
        
        pass
        
    def infer_tracks(self) -> Dict[str, Dict[str, Any]]:
        """
        Runs update_tracks for all frames, creating the tracks history.
        """
        
        for _ in range(self.num_frames):
            self.update_tracks()
            
        return self.final_tracks