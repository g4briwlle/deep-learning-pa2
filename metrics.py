import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def calculate_iou(box1, box2):
    """Calcula a Intersection over Union (IoU) entre duas caixas no formato [left, top, width, height]"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[0] + box1[2], box2[0] + box2[2])
    y2_inter = min(box1[1] + box1[3], box2[1] + box2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

class MOTMetrics:
    """
    Implementação própria das métricas de rastreamento para o PA2.
    Avalia IDF1, ID Switches e Fragmentações (FRAG).
    """
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    def evaluate(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
        """
        gt_df e pred_df devem conter as colunas: 
        ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height']
        """
        # Garante que os frames estão ordenados
        frames = sorted(set(gt_df['frame'].unique()) | set(pred_df['frame'].unique()))

        # Extrai IDs únicos
        gt_ids = gt_df['id'].unique()
        pred_ids = pred_df['id'].unique()

        gt_id_to_idx = {id_: i for i, id_ in enumerate(gt_ids)}
        pred_id_to_idx = {id_: i for i, id_ in enumerate(pred_ids)}

        # Matriz global de sobreposição para o cálculo do IDF1
        overlap_matrix = np.zeros((len(gt_ids), len(pred_ids)))

        # Estruturas de estado temporal para ID Switches e Fragmentações
        last_matched_pred_id = {gt_id: None for gt_id in gt_ids}
        was_tracked_last_frame = {gt_id: False for gt_id in gt_ids}
        
        id_switches = 0
        fragmentations = 0

        # Otimização: agrupamento prévio por frames
        gt_by_frame = gt_df.groupby('frame')
        pred_by_frame = pred_df.groupby('frame')

        for f in frames:
            # Obtém as caixas do frame atual (se existirem)
            gt_boxes = gt_by_frame.get_group(f).to_dict('records') if f in gt_by_frame.groups else []
            pr_boxes = pred_by_frame.get_group(f).to_dict('records') if f in pred_by_frame.groups else []

            matched_gts_this_frame = set()

            if len(gt_boxes) > 0 and len(pr_boxes) > 0:
                # Constrói matriz de IoU local do frame
                iou_mat = np.zeros((len(gt_boxes), len(pr_boxes)))
                for i, gt in enumerate(gt_boxes):
                    b1 = [gt['bb_left'], gt['bb_top'], gt['bb_width'], gt['bb_height']]
                    for j, pr in enumerate(pr_boxes):
                        b2 = [pr['bb_left'], pr['bb_top'], pr['bb_width'], pr['bb_height']]
                        iou_mat[i, j] = calculate_iou(b1, b2)

                # Associação gulosa por frame (Hungarian)
                matched_gt_indices, matched_pr_indices = linear_sum_assignment(-iou_mat)
                
                for gt_idx, pr_idx in zip(matched_gt_indices, matched_pr_indices):
                    if iou_mat[gt_idx, pr_idx] >= self.iou_threshold:
                        gt_id = gt_boxes[gt_idx]['id']
                        pr_id = pr_boxes[pr_idx]['id']
                        
                        matched_gts_this_frame.add(gt_id)

                        # 1. Alimenta matriz global do IDF1
                        overlap_matrix[gt_id_to_idx[gt_id], pred_id_to_idx[pr_id]] += 1

                        # 2. Contagem de ID Switch
                        # Se já rastreava antes e o ID previsto mudou agora = Switch
                        if last_matched_pred_id[gt_id] is not None and last_matched_pred_id[gt_id] != pr_id:
                            id_switches += 1
                        
                        # 3. Contagem de Fragmentação
                        # Se NÃO estava sendo rastreado no quadro anterior, mas JÁ TINHA sido rastreado no passado
                        if not was_tracked_last_frame[gt_id] and last_matched_pred_id[gt_id] is not None:
                            fragmentations += 1

                        # Atualiza os estados para o próximo quadro
                        last_matched_pred_id[gt_id] = pr_id

            # Atualiza o status "tracked" de todos os GTs para a checagem de fragmentação no próximo frame
            for gt_id in gt_ids:
                was_tracked_last_frame[gt_id] = (gt_id in matched_gts_this_frame)

        # Cálculo do IDF1 baseado na associação global
        row_ind, col_ind = linear_sum_assignment(-overlap_matrix)
        
        idtp = 0 # ID True Positives
        for r, c in zip(row_ind, col_ind):
            idtp += overlap_matrix[r, c]
            
        total_gt_boxes = len(gt_df)
        total_pr_boxes = len(pred_df[pred_df['id'] != -1]) # ignora predições sem identidade válida
        
        idfn = total_gt_boxes - idtp
        idfp = total_pr_boxes - idtp
        
        if (2 * idtp + idfp + idfn) > 0:
            idf1 = (2 * idtp) / (2 * idtp + idfp + idfn)
        else:
            idf1 = 0.0

        return {
            "IDF1": float(idf1),
            "IDSW": int(id_switches),
            "FRAG": int(fragmentations)
        }

# Função auxiliar para rodar a métrica diretamente ao executar o arquivo
if __name__ == "__main__":
    print("--- Teste Rápido de Métricas ---")
    # Simula um caso em que o IDSW é zero, mas ocorre uma fragmentação
    gt_mock = pd.DataFrame({
        'frame': [1, 2, 3, 4],
        'id': [1, 1, 1, 1],
        'bb_left': [10, 10, 10, 10], 'bb_top': [10, 10, 10, 10], 
        'bb_width': [50, 50, 50, 50], 'bb_height': [50, 50, 50, 50]
    })
    
    # O rastreador perde a detecção no quadro 3 (fragmentando a trajetória), mas recupera no 4 com o mesmo ID
    pred_mock = gt_mock[gt_mock['frame'] != 3].copy()
    
    evaluator = MOTMetrics(iou_threshold=0.5)
    resultados = evaluator.evaluate(gt_mock, pred_mock)
    
    print(f"Expectativa -> IDSW: 0 | FRAG: 1")
    print(f"Resultado   -> IDF1: {resultados['IDF1']:.2f} | IDSW: {resultados['IDSW']} | FRAG: {resultados['FRAG']}")