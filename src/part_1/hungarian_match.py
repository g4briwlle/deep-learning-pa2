import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from typing import Tuple, List


def iou_matrix(tracks_df: pd.DataFrame, dets_df: pd.DataFrame) -> np.ndarray:
    """
    Compute IoU between every track box and every detection box.
    
    Args:
        tracks_df (pd.DataFrame): Tracks' dataframe. Must have columns bb_left, bb_top, bb_width, bb_height.
        dets_df (pd.DataFrame): Detections' dataframe. Must have columns bb_left, bb_top, bb_width, bb_height.
    
    Returns:
        np.ndarray: Returns the IoU matrix of shape (len(tracks_df), len(dets_df)).
    """
    
    if len(tracks_df) == 0 or len(dets_df) == 0:
        return np.zeros((len(tracks_df), len(dets_df)))

    cols = ['bb_left', 'bb_top', 'bb_width', 'bb_height']
    # Track boxes: [x1, y1, x2, y2]
    t = tracks_df[cols].astype(float).to_numpy()
    # Detection boxes
    d = dets_df[cols].astype(float).to_numpy()
    
    t_x1 = t[:, 0]
    t_y1 = t[:, 1]
    t_x2 = t[:, 0] + t[:, 2]
    t_y2 = t[:, 1] + t[:, 3]

    d_x1 = d[:, 0]
    d_y1 = d[:, 1]
    d_x2 = d[:, 0] + d[:, 2]
    d_y2 = d[:, 1] + d[:, 3]

    # Intersection
    inter_x1 = np.maximum(t_x1[:, None], d_x1[None, :])
    inter_y1 = np.maximum(t_y1[:, None], d_y1[None, :])
    inter_x2 = np.minimum(t_x2[:, None], d_x2[None, :])
    inter_y2 = np.minimum(t_y2[:, None], d_y2[None, :])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Union
    t_area = (t_x2 - t_x1) * (t_y2 - t_y1)
    d_area = (d_x2 - d_x1) * (d_y2 - d_y1)
    union_area = t_area[:, None] + d_area[None, :] - inter_area

    # Avoid division by zero
    union_area = np.maximum(union_area, 1e-6)
    iou = inter_area / union_area
    return iou


def hungarian_match(
    tracks_df: pd.DataFrame,
    dets_df: pd.DataFrame,
    iou_threshold: float = 0.3
) -> Tuple[
    List[Tuple[int, int]],
    List[int],
    List[int]
]:
    """
    Match tracks to detections using Hungarian algorithm on IoU.

    Args:
        tracks_df (pd.DataFrame): Active tracks. Must have columns: track_id, bb_left, bb_top, bb_width, bb_height.
        dets_df (pd.DataFrame): Current detections. Must have columns: bb_left, bb_top, bb_width, bb_height.
        iou_threshold (float): Minimum IoU to accept a match.
    
    Returns:
        List[Tuple[int, int]]: list of (track_id, det_index)
        List[int]: list of track_id
        List[int]: list of det_index
        
    """
    if len(tracks_df) == 0 or len(dets_df) == 0:
        return [], list(tracks_df['track_id']), list(range(len(dets_df)))

    ious = iou_matrix(tracks_df, dets_df).astype(np.float64)
    cost = (1 - ious).astype(np.float64)

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    matched_tracks = set()
    matched_dets = set()

    for r, c in zip(row_ind, col_ind):
        if ious[r, c] >= iou_threshold:
            track_id = tracks_df.iloc[r]['track_id']
            matches.append((track_id, c))
            matched_tracks.add(r)
            matched_dets.add(c)

    unmatched_tracks = [
        tracks_df.iloc[i]['track_id']
        for i in range(len(tracks_df))
        if i not in matched_tracks
    ]
    unmatched_dets = [
        i for i in range(len(dets_df))
        if i not in matched_dets
    ]

    return matches, unmatched_tracks, unmatched_dets