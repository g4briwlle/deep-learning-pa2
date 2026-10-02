import pandas as pd

from ..part_0.synthetic_data import get_synthetic_detections
from ..part_0.synthetic_metrics import CustomTrackerEvaluator
from .naive_tracker import NaiveTracker


def tracks_to_predictions(tracks_by_frame, valid_frames=None):
    predictions = []
    for frame, tracks in tracks_by_frame.items():
        if valid_frames is not None and frame not in valid_frames:
            continue
        for track_id, tr in tracks.items():
            predictions.append({
                "frame": int(frame),
                "id": int(track_id),
                "bb_left": float(tr["bb_left"]),
                "bb_top": float(tr["bb_top"]),
                "bb_width": float(tr["bb_width"]),
                "bb_height": float(tr["bb_height"]),
            })
    return predictions


if __name__ == "__main__":
    gt_and_dets = get_synthetic_detections(0) 
    detections = gt_and_dets.detections
    detections_df = pd.DataFrame(detections)
    
    # Creating tracks for a whole dataset
    naive_tracker = NaiveTracker(detections_df)
    
    infered_tracks = naive_tracker.infer_tracks()
    predicted_tracks = tracks_to_predictions(infered_tracks)
    
    tracker_evaluator = CustomTrackerEvaluator()
    
    ground_truth = gt_and_dets.ground_truth
    evaluation = tracker_evaluator.evaluate(ground_truth, predicted_tracks)
    print(evaluation)