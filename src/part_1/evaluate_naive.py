import pandas as pd
import json

from ..part_0.synthetic_data import get_synthetic_detections
from ..part_0.synthetic_metrics import CustomTrackerEvaluator
from .naive_tracker import NaiveTracker

if __name__ == "__main__":
    print("============================================================")
    print("        AVALIACAO DO RASTREADOR - DADOS SINTETICOS          ")
    print("============================================================")
    
    # É útil registrar no log qual foi o nível de estresse/oclusão do teste
    occlusion_prob = 0.1
    print(f"[*] Parametros do teste: Probabilidade de Oclusao = {occlusion_prob}")
    
    print("[*] Gerando dados sinteticos (Ground Truth e Deteccoes)...")
    gt_and_dets = get_synthetic_detections(occlusion_prob) 
    detections = gt_and_dets.detections
    detections_df = pd.DataFrame(detections)
    
    print("[*] Executando o NaiveTracker em todo o dataset...")
    naive_tracker = NaiveTracker(detections_df)
    infered_tracks = naive_tracker.infer_tracks()
    
    print("[*] Calculando metricas de rastreamento...")
    tracker_evaluator = CustomTrackerEvaluator()
    ground_truth = gt_and_dets.ground_truth
    
    evaluation = tracker_evaluator.evaluate_from_tracker(ground_truth, infered_tracks)
    
    print("\n------------------------------------------------------------")
    print("                     RESULTADOS GERAIS                      ")
    print("------------------------------------------------------------")
    
    # Formata a saída para o log ficar bem espaçado e fácil de ler
    if isinstance(evaluation, dict):
        for metric_name, value in evaluation.items():
            # Formata números float para 4 casas decimais para alinhar os dados
            if isinstance(value, float):
                print(f"{metric_name:<30}: {value:.4f}")
            else:
                print(f"{metric_name:<30}: {value}")
    else:
        # Caso o seu evaluator já retorne um DataFrame ou uma string formatada
        print(evaluation)
        
    print("\n============================================================")
    print("                     EXECUCAO CONCLUIDA                     ")
    print("============================================================")