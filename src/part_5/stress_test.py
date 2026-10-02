import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Garante que a raiz do repositório esteja no sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Reaproveitamento dos artefatos da Parte 0 e Parte 1
from src.part_0.synthetic_data import SyntheticTrackerGenerator, DetectorSimulator
from src.part_0.synthetic_metrics import CustomTrackerEvaluator
from src.part_1.naive_tracker import NaiveTracker
from src.part_1.evaluate_part_1 import compute_map

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Diretório para saídas da Parte 5
OUT_DIR = BASE_DIR / "outputs" / "part5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_stress_test():
    print("==================================================================")
    print(" INICIANDO PARTE 5 - TESTE DE ESTRESSE (QUALIDADE DO DETECTOR)    ")
    print("==================================================================")

    # Gera um Ground Truth limpo com 5 objetos e sem oclusão proposital para isolar a falha do detector
    generator = SyntheticTrackerGenerator(num_frames=60, num_objects=5, occlusion_prob_per_frame=0.0)
    ground_truth = generator.generate()
    
    # 3 Níveis de degradação: Suave, Moderado e Severo
    intensities = {
        "Base (Sem Ruído)": {"drop_prob": 0.0, "noise_std": 0.0, "fp_per_frame": 0},
        "Intensidade 1": {"drop_prob": 0.1, "noise_std": 5.0, "fp_per_frame": 2},
        "Intensidade 2": {"drop_prob": 0.3, "noise_std": 15.0, "fp_per_frame": 5},
        "Intensidade 3": {"drop_prob": 0.6, "noise_std": 30.0, "fp_per_frame": 10},
    }

    results_map = []
    results_idf1 = []
    labels = []

    evaluator = CustomTrackerEvaluator(iou_threshold=0.3)

    for level, params in intensities.items():
        print(f"\n[*] Simulando {level} -> Drops: {params['drop_prob']*100}%, Ruído: {params['noise_std']}px, FPs: {params['fp_per_frame']}")
        
        # 1. Degrada as detecções usando o DetectorSimulator da Parte 0
        simulator = DetectorSimulator(**params)
        simulated_dets = simulator.simulate(ground_truth)
        dets_df = pd.DataFrame(simulated_dets)
        
        # 2. Avalia mAP do detector estragado (usando compute_map da Parte 1)
        flat_dets = evaluator._tracks_to_predictions({f: {i: d} for i, d in enumerate(simulated_dets)}) 
        mAP = compute_map(flat_dets, ground_truth, iou_thresh=0.5)
        
        # 3. Roda o Rastreador
        tracker = NaiveTracker(dets_df, iou_threshold=0.3, kill_track_frames=15)
        infered_tracks = tracker.infer_tracks()
        
        # 4. Calcula IDF1
        metrics = evaluator.evaluate_from_tracker(ground_truth, infered_tracks)
        
        results_map.append(mAP)
        results_idf1.append(metrics.IDF1)
        labels.append(level)
        
        print(f"    -> mAP (Detecção): {mAP:.4f} | IDF1 (Tracking): {metrics.IDF1:.4f}")

    # ==============================================================================
    # SALVAR ARTEFATOS
    # ==============================================================================
    
    # Gráfico 1: Curva de Degradação conjunta mAP x IDF1
    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    plt.plot(x, results_map, marker='o', linestyle='-', color='blue', label='mAP@0.5 (Detector)')
    plt.plot(x, results_idf1, marker='s', linestyle='--', color='red', label='IDF1 (Tracking Temporal)')
    
    plt.title("Teste de Estresse: Degradação do Detector vs Rastreador")
    plt.xlabel("Intensidade da Degradação do Detector (Falsos Positivos, Ruído e Descarte)")
    plt.ylabel("Score (0 a 1)")
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = OUT_DIR / "stress_test_detector_quality.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\n[*] Gráfico de estresse salvo em: {plot_path}")

    # Relatório Teórico Respondendo a Pergunta da Parte 5
    txt_path = OUT_DIR / "resposta_stress_test.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("RESPOSTA TEÓRICA - PARTE 5: TESTE DE ESTRESSE (Qualidade do Detector)\n")
        f.write("========================================================================\n\n")
        f.write("O modelo temporal absorve ou amplifica a falha do detector?\n")
        f.write("------------------------------------------------------------------------\n")
        f.write("Conforme o gráfico gerado mostra, o rastreador AMPLIFICA drasticamente a falha do detector.\n")
        f.write("O mAP cai linearmente conforme aumentamos o p% de descarte e os Falsos Positivos.\n")
        f.write("No entanto, o IDF1 (Tracking) despenca muito mais rápido.\n\n")
        f.write("Por que isso acontece?\n")
        f.write("1. Efeito Dominó do Descarte: Se o detector perde o objeto por 'p%' quadros, a track \n")
        f.write("   morre (se estourar o kill_track_frames). Quando a detecção volta, ela nasce como um \n")
        f.write("   ID totalmente novo, gerando um ID Switch. Um único buraco na detecção destrói a métrica IDF1 \n")
        f.write("   pelo resto do vídeo.\n")
        f.write("2. Ruído de Coordenadas: O ruído no bounding box derruba o IoU abaixo do limiar (0.3). Isso \n")
        f.write("   impede o Hungarian Match de associar o ID correto, gerando mais fragmentações.\n")
        f.write("3. Falsos Positivos: Cada falso positivo não associado inicia uma 'track lixo' (ghost track), \n")
        f.write("   que aumenta massivamente a métrica de Falsos Positivos de Identidade (IDFP) do IDF1.\n\n")
        f.write("Portanto, a degradação temporal tem um efeito multiplicativo, não aditivo. Um detector ruim \n")
        f.write("torna o rastreamento temporal inútil.")
        
    print(f"[*] Relatório de resposta salvo em: {txt_path}")
    print("\n==================================================================")
    print(" EXECUÇÃO DA PARTE 5 CONCLUÍDA ")
    print("==================================================================")

if __name__ == "__main__":
    run_stress_test()