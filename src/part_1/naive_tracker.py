import pandas as pd
import copy
from typing import Dict, Any
from abc import ABC

from ..part_0.synthetic_data import get_synthetic_detections
from .hungarian_match import hungarian_match
from ..tracker import Tracker

import sys
# Força o terminal a não quebrar caracteres em português
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


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

    def display_frame_state(self, frame_number: int) -> str:
        """Retorna uma representação em string formatada do estado das tracks em um frame específico."""
        if frame_number not in self.final_tracks or not self.final_tracks[frame_number]:
            return f"[Frame {frame_number}] Nenhuma track ativa no momento."
        
        # Converte o dicionário interno para um DataFrame apenas para visualização
        df_vis = pd.DataFrame.from_dict(self.final_tracks[frame_number], orient='index')
        df_vis.index.name = 'Track_ID'
        
        # Formatação de colunas numéricas para melhor alinhamento
        for col in ['bb_left', 'bb_top', 'bb_width', 'bb_height']:
            if col in df_vis.columns:
                df_vis[col] = df_vis[col].apply(lambda x: f"{x:.1f}")
                
        return f"ESTADO DAS TRACKS - FRAME {frame_number}:\n{df_vis.to_string()}"


if __name__ == "__main__":
    print("="*60)
    print(" INICIANDO TESTE: NAIVE TRACKER (BASELINE) ".center(60))
    print("="*60)

    # Inicializa detecções sintéticas
    print("[*] Gerando detecções sintéticas (sem oclusão)...")
    synthetic_detector = get_synthetic_detections(0).detections 
    det_df = pd.DataFrame(synthetic_detector)
    
    # Instancia o rastreador
    naive_tracker = NaiveTracker(det_df)
    print(f"[*] Rastreador inicializado. Limiar de IoU: {naive_tracker.iou_threshold}, Max Age: {naive_tracker.max_age}\n")
    
    # Simulação de frames (ajuste este loop conforme a lógica da classe Tracker pai)
    # Supondo que você precisa rodar o update_tracks para os frames existirem no dicionário
    frames_to_simulate = 5
    print("-" * 60)
    for _ in range(frames_to_simulate):
        naive_tracker.update_tracks()
    
    # Exibe os resultados formatados
    print("\n" + naive_tracker.display_frame_state(1))
    print("-" * 60)
    print("\n" + naive_tracker.display_frame_state(5))
    print("\n" + "="*60)
    print(" EXECUÇÃO CONCLUÍDA ".center(60))
    print("="*60)