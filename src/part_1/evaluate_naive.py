import pandas as pd

from ..part_0.synthetic_data import get_synthetic_detections
from ..part_0.synthetic_metrics import CustomTrackerEvaluator
from .naive_tracker import NaiveTracker


if __name__ == "__main__":
    gt_and_dets = get_synthetic_detections(0.1) 
    detections = gt_and_dets.detections
    detections_df = pd.DataFrame(detections)
    
    # Creating tracks for a whole dataset
    naive_tracker = NaiveTracker(detections_df)
    infered_tracks = naive_tracker.infer_tracks()
    
    tracker_evaluator = CustomTrackerEvaluator()
    
    ground_truth = gt_and_dets.ground_truth
    
    # print(ground_truth)
    evaluation = tracker_evaluator.evaluate_from_tracker(ground_truth, infered_tracks)
    print(evaluation)