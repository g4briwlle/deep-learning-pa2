import numpy as np
import copy
from scipy.optimize import linear_sum_assignment
from .synthetic_data_0 import *
import sys
# Força o terminal a não quebrar caracteres em português
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    

def calculate_iou(box1, box2):
    """Calculates intersection over union (IoU) between two MOT boxes"""
    x1_inter = max(box1['bb_left'], box2['bb_left'])
    y1_inter = max(box1['bb_top'], box2['bb_top'])
    x2_inter = min(box1['bb_left'] + box1['bb_width'], box2['bb_left'] + box2['bb_width'])
    y2_inter = min(box1['bb_top'] + box1['bb_height'], box2['bb_top'] + box2['bb_height'])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    area1 = box1['bb_width'] * box1['bb_height']
    area2 = box2['bb_width'] * box2['bb_height']
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area



class CustomTrackerEvaluator:
    """
    Metrics implementations for tracking (IDF1 and ID switches) for part 0
    """
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    def evaluate(self, ground_truth, predictions):
        # Organize data by frame
        frames = sorted(list(set([d['frame'] for d in ground_truth])))
        gt_by_frame = {f: [] for f in frames}
        pr_by_frame = {f: [] for f in frames}
        
        for d in ground_truth: 
            gt_by_frame[d['frame']].append(d)
        for d in predictions: 
            pr_by_frame[d['frame']].append(d)

        # exctract unique ids
        gt_ids = sorted(list(set([d['id'] for d in ground_truth])))
        pr_ids = sorted(list(set([d['id'] for d in predictions])))

        # mapping to matrix indices
        gt_id_to_idx = {id_: i for i, id_ in enumerate(gt_ids)}
        pr_id_to_idx = {id_: i for i, id_ in enumerate(pr_ids)}

        # matrix cost for idf1
        # count in how many frames gt_id and pr_id matched (iou > threshold)
        overlap_matrix = np.zeros((len(gt_ids), len(pr_ids)))

        # structure to count id switches
        # stores the last pr_id associated to a gt_id
        last_matched_pr_id = {gt_id: None for gt_id in gt_ids}
        id_switches = 0

        for f in frames:
            gt_boxes = gt_by_frame[f]
            pr_boxes = pr_by_frame[f]

            # local iou matrix of frame
            if len(gt_boxes) > 0 and len(pr_boxes) > 0:
                iou_mat = np.zeros((len(gt_boxes), len(pr_boxes)))
                for i, gt in enumerate(gt_boxes):
                    for j, pr in enumerate(pr_boxes):
                        iou_mat[i, j] = calculate_iou(gt, pr)

                # greedy association to count temporal id switch
                matched_gt_indices, matched_pr_indices = linear_sum_assignment(-iou_mat)
                
                for gt_idx, pr_idx in zip(matched_gt_indices, matched_pr_indices):
                    if iou_mat[gt_idx, pr_idx] >= self.iou_threshold:
                        gt_id = gt_boxes[gt_idx]['id']
                        pr_id = pr_boxes[pr_idx]['id']

                        # fill global idf1 matrix
                        overlap_matrix[gt_id_to_idx[gt_id], pr_id_to_idx[pr_id]] += 1

                        # checks id switch (did the predicted id change to this gt?)
                        if last_matched_pr_id[gt_id] is not None and last_matched_pr_id[gt_id] != pr_id:
                            id_switches += 1
                        
                        last_matched_pr_id[gt_id] = pr_id

        # compute idf1 based on global association
        # linear_sum_assignment finds minimal cost, so invert overlaps
        row_ind, col_ind = linear_sum_assignment(-overlap_matrix)
        
        idtp = 0 # ID True Positives
        for r, c in zip(row_ind, col_ind):
            idtp += overlap_matrix[r, c]
            
        total_gt_boxes = len(ground_truth)
        total_pr_boxes = sum(1 for d in predictions if d['id'] != -1) # ignores trash without id
        
        idfn = total_gt_boxes - idtp
        idfp = total_pr_boxes - idtp
        
        idf1 = (2 * idtp) / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) > 0 else 0.0

        return {"IDF1": idf1, "IDSW": id_switches, "IDTP": idtp, "IDFN": idfn, "IDFP": idfp}




def run_synthetic_tests():
    print("--- INITIATING SYNTHETHIC DATA METRICS ---")
    
    # Gera um Ground Truth limpo com 2 objetos para facilitar os testes
    generator = SyntheticTrackerGenerator(num_frames=20, num_objects=2, occlusion_prob_per_frame=0.0)
    gt_data = generator.generate()
    evaluator = CustomTrackerEvaluator()

    # case 1: pred = ground truth
    # idf1 = 1 qnd zero switches
    pred_case_a = copy.deepcopy(gt_data)
    res_a = evaluator.evaluate(gt_data, pred_case_a)
    print(f"\nCaso (A) Predição Perfeita:")
    print(f"Expectativa: IDF1 = 1.0 | IDSW = 0")
    print(f"Resultado : IDF1 = {res_a['IDF1']:.2f} | IDSW = {res_a['IDSW']}")

    # case 2: two entities switched starting from frame k
    # show if the code detects id's inversion between objects and the correct number of switches
    pred_case_b = copy.deepcopy(gt_data)
    frame_k = 10
    for box in pred_case_b:
        if box['frame'] >= frame_k:
            if box['id'] == 1:
                box['id'] = 2
            elif box['id'] == 2:
                box['id'] = 1
                
    res_b = evaluator.evaluate(gt_data, pred_case_b)
    print(f"\nCaso (B) Troca de Identidades no frame {frame_k}:")
    print(f"Expectativa: O rastreador inverteu 2 objetos. Esperado IDSW = 2.")
    print(f"Resultado : IDF1 = {res_b['IDF1']:.2f} | IDSW = {res_b['IDSW']}")
    print(f"-> Explicação: O IDF1 cai porque a métrica global só permite casar o GT 1 com o Pred 1 OU Pred 2, penalizando a metade do vídeo que ficou invertida.")

    # case 3: one track split in half
    # difers a switch from a tracking loss
    pred_case_c = copy.deepcopy(gt_data)
    frame_k = 10
    for box in pred_case_c:
        if box['frame'] >= frame_k and box['id'] == 1:
            box['id'] = 99 # new id, broken track
            
    res_c = evaluator.evaluate(gt_data, pred_case_c)
    print(f"\nCaso (C) Track Partida no frame {frame_k}:")
    print(f"Expectativa: O objeto 1 virou o objeto 99. Esperado IDSW = 1.")
    print(f"Resultado : IDF1 = {res_c['IDF1']:.2f} | IDSW = {res_c['IDSW']}")
    print(f"-> Explicação: Note que o IDF1 de (C) difere de (B). Na quebra (C), o objeto 2 ficou intacto, e o objeto 1 perdeu metade de seus True Positives (que viraram IDFP e IDFN para a track 99). Já na troca (B), DOIS objetos sofreram penalidade, tornando a queda de IDF1 em (B) mais severa do que em (C).")

if __name__ == "__main__":
    run_synthetic_tests()