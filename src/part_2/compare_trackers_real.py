import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import zipfile
import io

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.part_0.synthetic_metrics import CustomTrackerEvaluator
from src.part_1.naive_tracker import NaiveTracker
from src.part_2.appearance_tracker import AppearanceTracker
from src.config import ZIP_PATH

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

OUT_DIR = BASE_DIR / "outputs" / "part2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_mot17_data(seq_name: str):
    """Carrega as detecções e o ground truth reais diretamente do ZIP."""
    gt_path = f"{seq_name}/gt/gt.txt"
    det_path = f"{seq_name}/det/det.txt"
    
    cols = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'visibility']
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        # Ground Truth
        with z.open(gt_path) as f:
            gt_df = pd.read_csv(f, names=cols)
            # Filtra apenas pedestres válidos
            gt_df = gt_df[gt_df['class'] == 1].copy()
            ground_truth = gt_df.to_dict('records')
            
        # Detections
        with z.open(det_path) as f:
            det_df = pd.read_csv(f, names=cols)
            # Nas detecções públicas, a classe/id geralmente vêm como -1
            det_df = det_df[det_df['conf'] > 0.5].copy() # Filtro de confiança leve
            
    return ground_truth, det_df

def run_real_comparison():
    print("==================================================================")
    print(" COMPARAÇÃO LADO A LADO: BASELINE vs TRILHA B (DADOS REAIS MOT17) ")
    print("==================================================================")

    seq_name = "MOT17/train/MOT17-11-FRCNN"
    
    print(f"[*] Carregando dados reais da sequência {seq_name}...")
    ground_truth, dets_df = load_mot17_data(seq_name)
    
    # Para acelerar o teste, vamos usar apenas os primeiros 500 frames
    max_frames = 200
    ground_truth = [gt for gt in ground_truth if gt['frame'] <= max_frames]
    dets_df = dets_df[dets_df['frame'] <= max_frames].copy()
    
    evaluator = CustomTrackerEvaluator(iou_threshold=0.3)

    # 1. RODAR O BASELINE (PA1 - Associação ingênua por IoU)
    print("\n[*] Rodando Baseline (Naive Tracker - Parte 1)...")
    baseline_tracker = NaiveTracker(dets_df, iou_threshold=0.3, kill_track_frames=15)
    baseline_tracks = baseline_tracker.infer_tracks()
    
    # Filtramos apenas os frames válidos processados para avaliação justa
    valid_frames = list(range(1, max_frames + 1))
    baseline_metrics = evaluator.evaluate_from_tracker(ground_truth, baseline_tracks, valid_frames)

    # 2. RODAR A TRILHA B (PA2 - Memória de Aparência RNN)
    print("[*] Rodando Trilha B (Appearance Tracker - Parte 2)...")
    ckpt_path = OUT_DIR / "appearance_rnn.pth"
    
    appearance_tracker = AppearanceTracker(
        dets_df=dets_df,
        seq_name=seq_name, 
        checkpoint_path=ckpt_path,
        cos_threshold=0.15, 
        iou_gate_threshold=0.0
    )
    appearance_tracks = appearance_tracker.infer_tracks()
    appearance_metrics = evaluator.evaluate_from_tracker(ground_truth, appearance_tracks, valid_frames)

    # 3. SALVAR ARTEFATOS E GRÁFICO COMPARATIVO
    print("\n==================================================================")
    print(" RESULTADOS DA COMPARAÇÃO (MOT17-09 - REAIS)                      ")
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
    
    ax1.bar(x[0] - width/2, baseline_vals[0], width, label='Baseline (PA1)', color='tab:red')
    ax1.bar(x[0] + width/2, trilha_b_vals[0], width, label='Trilha B (PA2)', color='tab:green')
    ax1.set_ylabel('Score IDF1', color='black')
    
    ax2 = ax1.twinx()
    ax2.bar(x[1] - width/2, baseline_vals[1], width, color='tab:red')
    ax2.bar(x[1] + width/2, trilha_b_vals[1], width, color='tab:green')
    ax2.set_ylabel('Contagem (ID Switches)', color='black')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Comparação Direta: Baseline vs Trilha B (Dados Reais MOT17-09)")
    ax1.legend(loc='upper left')

    plot_path = OUT_DIR / "3_comparacao_PA1_vs_PA2.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[*] Gráfico salvo em: {plot_path}")

if __name__ == "__main__":
    run_real_comparison()