import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.part_0.synthetic_data import SyntheticTrackerGenerator, DetectorSimulator
from src.part_0.synthetic_metrics import CustomTrackerEvaluator
from src.part_1.naive_tracker import NaiveTracker
from src.part_2.appearance_tracker import AppearanceTracker

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

OUT_DIR = BASE_DIR / "outputs" / "part2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_comparison():
    print("==================================================================")
    print(" INICIANDO COMPARAÇÃO LADO A LADO: BASELINE vs TRILHA B (RNN)     ")
    print("==================================================================")

    # Gera uma sequência controlada COM OCLUSÃO para evidenciar o ganho da Parte 2
    generator = SyntheticTrackerGenerator(
        num_frames=80, 
        num_objects=3, 
        occlusion_prob_per_frame=0.05, 
        occlusion_duration=10
    )
    ground_truth = generator.generate()
    
    # Simula as detecções (o detector fica congelado a partir daqui, como exigido)
    simulator = DetectorSimulator(drop_prob=0.1, noise_std=2.0, fp_per_frame=1)
    simulated_dets = simulator.simulate(ground_truth)
    dets_df = pd.DataFrame(simulated_dets)
    
    evaluator = CustomTrackerEvaluator(iou_threshold=0.3)

    # 1. RODAR O BASELINE (PA1 - Associação ingênua por IoU)
    print("\n[*] Rodando Baseline (Naive Tracker - Parte 1)...")
    baseline_tracker = NaiveTracker(dets_df, iou_threshold=0.3, kill_track_frames=15)
    baseline_tracks = baseline_tracker.infer_tracks()
    baseline_metrics = evaluator.evaluate_from_tracker(ground_truth, baseline_tracks)

    # 2. RODAR A TRILHA B (PA2 - Memória de Aparência RNN)
    print("[*] Rodando Trilha B (Appearance Tracker - Parte 2)...")
    ckpt_path = OUT_DIR / "appearance_rnn.pth"
    
    if not ckpt_path.exists():
        print(f"[ERRO] O modelo treinado não foi encontrado em {ckpt_path}.")
        print("Certifique-se de que o treinamento inicial gerou os pesos.")
        return

    # Usando seq_name genérico para a lógica interna caso teste com dados reais
    appearance_tracker = AppearanceTracker(
        dets_df=dets_df,
        seq_name="MOT17/train/MOT17-09-FRCNN", 
        checkpoint_path=ckpt_path,
        cos_threshold=0.4,
        iou_gate_threshold=0.0
    )
    appearance_tracks = appearance_tracker.infer_tracks()
    appearance_metrics = evaluator.evaluate_from_tracker(ground_truth, appearance_tracks)

    # 3. SALVAR ARTEFATOS E GRÁFICO COMPARATIVO
    print("\n==================================================================")
    print(" RESULTADOS DA COMPARAÇÃO (MESMA SEQUÊNCIA)                       ")
    print("==================================================================")
    print(f"{'Métrica':<15} | {'Baseline (PA1)':<15} | {'Trilha B (PA2)':<15}")
    print("-" * 50)
    print(f"{'IDF1':<15} | {baseline_metrics.IDF1:<15.4f} | {appearance_metrics.IDF1:<15.4f}")
    print(f"{'ID Switches':<15} | {baseline_metrics.IDSW:<15} | {appearance_metrics.IDSW:<15}")

    labels = ['IDF1 (Maior é melhor)', 'ID Switches (Menor é melhor)']
    baseline_vals = [baseline_metrics.IDF1, baseline_metrics.IDSW]
    trilha_b_vals = [appearance_metrics.IDF1, appearance_metrics.IDSW]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # Eixo primário (IDF1)
    ax1.bar(x[0] - width/2, baseline_vals[0], width, label='Baseline (PA1)', color='tab:red')
    ax1.bar(x[0] + width/2, trilha_b_vals[0], width, label='Trilha B (PA2)', color='tab:green')
    ax1.set_ylabel('Score IDF1', color='black')
    
    # Eixo secundário (IDSW - Contagem)
    ax2 = ax1.twinx()
    ax2.bar(x[1] - width/2, baseline_vals[1], width, color='tab:red')
    ax2.bar(x[1] + width/2, trilha_b_vals[1], width, color='tab:green')
    ax2.set_ylabel('Contagem (ID Switches)', color='black')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Comparação Direta: Baseline vs Trilha B (Sob Oclusão)")
    ax1.legend(loc='upper left')

    plot_path = OUT_DIR / "comparacao_PA1_vs_PA2.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[*] Gráfico salvo em: {plot_path}")

if __name__ == "__main__":
    run_comparison()