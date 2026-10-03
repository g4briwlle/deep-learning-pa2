"""CLI de avaliação comparativa — Parte 2 (Trilha B) vs baseline da Parte 1.

Roda o NaiveTracker (PA1) e o AppearanceTracker (PA2) na mesma sequência do
MOT17 e reporta IDF1 e ID Switches lado a lado. Não altera o script original
src/part_2/compare_pa1_pa2.py; apenas expõe os mesmos parâmetros como CLI.
"""
import argparse
import sys
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.part_0.synthetic_metrics import CustomTrackerEvaluator
from src.part_1.naive_tracker import NaiveTracker
from src.part_2.appearance_tracker import AppearanceTracker
from src.config import ZIP_PATH


def parse_args():
    p = argparse.ArgumentParser(
        description="Compara Baseline (PA1) vs Trilha B (PA2) em uma sequência "
                    "real do MOT17.")
    p.add_argument("--seq-name", type=str,
                   default="MOT17/train/MOT17-11-FRCNN",
                   help="Caminho interno da sequência dentro do ZIP "
                        "(default: MOT17/train/MOT17-11-FRCNN).")
    p.add_argument("--max-frames", type=int, default=200,
                   help="Avalia apenas os primeiros N frames (default: 200).")
    p.add_argument("--iou-threshold", type=float, default=0.3,
                   help="IoU mínimo do avaliador e do baseline (default: 0.3).")
    p.add_argument("--kill-track-frames", type=int, default=15,
                   help="Quadros sem observação antes de matar a track do "
                        "baseline (default: 15).")
    p.add_argument("--cos-threshold", type=float, default=0.15,
                   help="Limiar de similaridade de cosseno da Trilha B "
                        "(default: 0.15).")
    p.add_argument("--iou-gate-threshold", type=float, default=0.0,
                   help="Portão geométrico de IoU da Trilha B (default: 0.0).")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Caminho do checkpoint da Trilha B "
                        "(default: outputs/part2/appearance_rnn.pth).")
    p.add_argument("--output-dir", type=str, default="outputs/part2",
                   help="Diretório de saída (default: outputs/part2).")
    p.add_argument("--plot-name", type=str,
                   default="3_comparacao_PA1_vs_PA2.png",
                   help="Nome do gráfico salvo (default: 3_comparacao_PA1_vs_PA2.png).")
    return p.parse_args()


def load_mot17_data(seq_name: str):
    """Carrega GT e detecções reais direto do ZIP (idêntico ao script original)."""
    gt_path = f"{seq_name}/gt/gt.txt"
    det_path = f"{seq_name}/det/det.txt"
    cols = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height",
            "conf", "class", "visibility"]

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        with z.open(gt_path) as f:
            gt_df = pd.read_csv(f, names=cols)
            gt_df = gt_df[gt_df["class"] == 1].copy()
            ground_truth = gt_df.to_dict("records")

        with z.open(det_path) as f:
            det_df = pd.read_csv(f, names=cols)
            det_df = det_df[det_df["conf"] > 0.5].copy()

    return ground_truth, det_df


def run(args):
    out_dir = BASE_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = (BASE_DIR / args.ckpt) if args.ckpt else (out_dir / "appearance_rnn.pth")

    print("==================================================================")
    print(" COMPARAÇÃO LADO A LADO: BASELINE vs TRILHA B (DADOS REAIS MOT17) ")
    print("==================================================================")
    print(f"[*] Sequência : {args.seq_name}")
    print(f"[*] Frames    : 1..{args.max_frames}")
    print(f"[*] Checkpoint: {ckpt_path}")

    print(f"\n[*] Carregando dados reais da sequência {args.seq_name}...")
    ground_truth, dets_df = load_mot17_data(args.seq_name)

    ground_truth = [gt for gt in ground_truth if gt["frame"] <= args.max_frames]
    dets_df = dets_df[dets_df["frame"] <= args.max_frames].copy()

    evaluator = CustomTrackerEvaluator(iou_threshold=args.iou_threshold)

    # 1) Baseline
    print("\n[*] Rodando Baseline (Naive Tracker - Parte 1)...")
    baseline_tracker = NaiveTracker(dets_df,
                                    iou_threshold=args.iou_threshold,
                                    kill_track_frames=args.kill_track_frames)
    baseline_tracks = baseline_tracker.infer_tracks()
    valid_frames = list(range(1, args.max_frames + 1))
    baseline_metrics = evaluator.evaluate_from_tracker(
        ground_truth, baseline_tracks, valid_frames)

    # 2) Trilha B
    print("[*] Rodando Trilha B (Appearance Tracker - Parte 2)...")
    appearance_tracker = AppearanceTracker(
        dets_df=dets_df,
        seq_name=args.seq_name,
        checkpoint_path=ckpt_path,
        cos_threshold=args.cos_threshold,
        iou_gate_threshold=args.iou_gate_threshold,
    )
    appearance_tracks = appearance_tracker.infer_tracks()
    appearance_metrics = evaluator.evaluate_from_tracker(
        ground_truth, appearance_tracks, valid_frames)

    # 3) Tabela
    print("\n==================================================================")
    seq_short = args.seq_name.rstrip("/").split("/")[-1]
    print(f" RESULTADOS DA COMPARAÇÃO ({seq_short})")
    print("==================================================================")
    print(f"{'Métrica':<15} | {'Baseline (PA1)':<15} | {'Trilha B (PA2)':<15}")
    print("-" * 50)
    print(f"{'IDF1':<15} | {baseline_metrics.IDF1:<15.4f} | {appearance_metrics.IDF1:<15.4f}")
    print(f"{'ID Switches':<15} | {baseline_metrics.IDSW:<15} | {appearance_metrics.IDSW:<15}")

    # 4) Gráfico
    labels = ["IDF1 (Maior é melhor)", "ID Switches (Menor é melhor)"]
    baseline_vals = [baseline_metrics.IDF1, baseline_metrics.IDSW]
    trilha_b_vals = [appearance_metrics.IDF1, appearance_metrics.IDSW]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.bar(x[0] - width/2, baseline_vals[0], width, label="Baseline (PA1)", color="tab:red")
    ax1.bar(x[0] + width/2, trilha_b_vals[0], width, label="Trilha B (PA2)", color="tab:green")
    ax1.set_ylabel("Score IDF1", color="black")

    ax2 = ax1.twinx()
    ax2.bar(x[1] - width/2, baseline_vals[1], width, color="tab:red")
    ax2.bar(x[1] + width/2, trilha_b_vals[1], width, color="tab:green")
    ax2.set_ylabel("Contagem (ID Switches)", color="black")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title(f"Comparação Direta: Baseline vs Trilha B ({seq_short})")
    ax1.legend(loc="upper left")

    plot_path = out_dir / args.plot_name
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[*] Gráfico salvo em: {plot_path}")


if __name__ == "__main__":
    run(parse_args())