"""
evaluate_part1.py — Quantificação do fracasso do baseline (Parte 1, passo 5).

Uso:
    python evaluate_part1.py --data-root /caminho/MOT17 --detector FRCNN \
                             --out-dir results/part1

Estrutura esperada do MOT17 baixado:
    <data-root>/MOT17-02-FRCNN/gt/gt.txt
    <data-root>/MOT17-02-FRCNN/det/det.txt
    <data-root>/MOT17-13-FRCNN/...

O tracker ingênuo é importado de `naive_tracker.py` (função `run_naive_tracker`),
que recebe dets_by_frame e devolve tracks_by_frame no formato descrito no README.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..part_0.synthetic_metrics import calculate_iou

# ===========================================================================
# 3. mAP@0.5 para a detecção (classe única: pessoa)
# ===========================================================================

def compute_map(predictions: list, ground_truth: list, iou_thresh=0.5) -> float:
    """
    AP@0.5 calculado sobre a sequência inteira (padrão COCO/VOC adaptado a 1 classe).
    Agora aceita listas planas (flat lists) nas predições e no ground truth.
    """
    flat = []
    for d in predictions:
        # Passa o dicionário 'd' inteiro em vez de usar _box(d)
        flat.append((d.get("score", 1.0), d["frame"], d))
    
    if not flat:
        return 0.0
    flat.sort(key=lambda x: -x[0])

    gt_index = {}
    total_gt = len(ground_truth)
    if total_gt == 0:
        return 0.0
        
    for g in ground_truth:
        fr = g["frame"]
        if fr not in gt_index:
            gt_index[fr] = []
        # Passa o dicionário 'g' inteiro em vez de usar _box(g)
        gt_index[fr].append({"box": g, "matched": False})

    tp = np.zeros(len(flat))
    fp = np.zeros(len(flat))
    for i, (_, fr, box) in enumerate(flat):
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt_index.get(fr, [])):
            if g["matched"]:
                continue
            # Agora 'box' e 'g["box"]' são dicionários compatíveis com calculate_iou
            v = calculate_iou(box, g["box"])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_thresh:
            tp[i] = 1
            gt_index[fr][best_j]["matched"] = True
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / total_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recall >= t
        ap += (precision[mask].max() if mask.any() else 0.0) / 11.0
    return float(ap)

# ===========================================================================
# 4. Eixo de dificuldade: duração de oclusão
# ===========================================================================

def occlusion_durations(ground_truth: list) -> List[int]:
    """Lista de gaps (em quadros) entre aparições consecutivas de cada ID do GT."""
    per_id: Dict[int, List[int]] = {}
    for d in ground_truth:
        per_id.setdefault(d["id"], []).append(d["frame"])
        
    gaps = []
    for frames in per_id.values():
        frames.sort()
        for i in range(1, len(frames)):
            g = frames[i] - frames[i - 1] - 1
            if g > 0:
                gaps.append(g)
    return gaps



# ===========================================================================
# 5. Runner: tabela + gráfico
# ===========================================================================

@dataclass
class SeqResult:
    name: str          # <- ADICIONADO PARA CORRIGIR O ACESSO NO PLOT
    mAP: float
    IDF1: float
    IDSW: int
    num_gt_ids: int
    num_pred_ids: int
    id_ratio: float
    idsw_per_gt: float
    occlusion_median: float
    occlusion_mean: float
    occlusion_max: int


# def run_sequence(name, tracks, gt, dets, min_score=0.0) -> SeqResult:
#     m = compute_identity_metrics(tracks, gt, iou_thresh=0.5)
#     mAP = compute_map(dets, gt, iou_thresh=0.5)

#     gaps = occlusion_durations(gt)
#     if gaps:
#         occ_med = float(np.median(gaps))
#         occ_mean = float(np.mean(gaps))
#         occ_max = int(np.max(gaps))
#     else:
#         occ_med = occ_mean = 0.0
#         occ_max = 0

#     return SeqResult(
#         mAP=mAP, IDF1=m.IDF1, IDSW=m.IDSW,
#         num_gt_ids=m.num_gt_ids, num_pred_ids=m.num_pred_ids,
#         id_ratio=m.id_ratio, idsw_per_gt=m.idsw_per_gt,
#         occlusion_median=occ_med, occlusion_mean=occ_mean, occlusion_max=occ_max,
#     )


def print_table(results: List[SeqResult]) -> None:
    header = f"{'seq':<22}{'mAP':>7}{'IDF1':>7}{'IDSW':>6}{'FRAG':>6}" \
             f"{'pred/gt':>9}{'sw/gt':>7}{'occ_med':>9}{'occ_max':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.name:<22}{r.mAP:>7.3f}{r.IDF1:>7.3f}{r.IDSW:>6d}{r.FRAG:>6d}"
              f"{r.id_ratio:>9.2f}{r.idsw_per_gt:>7.2f}"
              f"{r.occlusion_median:>9.1f}{r.occlusion_max:>9d}")


def save_csv(results: List[SeqResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "seq", "mAP", "IDF1", "IDSW", "FRAG",
            "num_gt_ids", "num_pred_ids", "id_ratio", "idsw_per_gt",
            "occ_median", "occ_mean", "occ_max",
        ])
        for r in results:
            w.writerow([
                r.name, f"{r.mAP:.4f}", f"{r.IDF1:.4f}", r.IDSW, r.FRAG,
                r.num_gt_ids, r.num_pred_ids,
                f"{r.id_ratio:.4f}", f"{r.idsw_per_gt:.4f}",
                f"{r.occlusion_median:.2f}", f"{r.occlusion_mean:.2f}", r.occlusion_max,
            ])


def plot_descolamento(results: List[SeqResult], out_path: str) -> None:
    """
    Gráfico obrigatório da Parte 1.5.
    Painel superior: mAP e IDF1.
    Painel inferior: id_ratio (IDs previstas / reais) e IDSW por ID real.
    Sequências ordenadas por duração mediana de oclusão (crescente).
    """
    results = sorted(results, key=lambda r: r.occlusion_median)
    x = np.arange(len(results))
    labels = [r.name.replace("MOT17-", "") for r in results]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(8, 1.2 * len(results)), 8), sharex=True
    )

    # --- Painel superior ----------------------------------------------------
    ax1.plot(x, [r.mAP for r in results], marker="o", label="mAP@0.5 (detecção)")
    ax1.plot(x, [r.IDF1 for r in results], marker="s", label="IDF1 (identidade)")
    ax1.set_ylabel("Score")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    ax1.set_title("Descolamento detecção × identidade "
                  "(ordenado por duração mediana de oclusão, crescente)")

    # --- Painel inferior ----------------------------------------------------
    ax2.plot(x, [r.id_ratio for r in results], marker="o",
             color="tab:red", label="IDs previstas / IDs reais")
    ax2.plot(x, [r.idsw_per_gt for r in results], marker="s",
             color="tab:purple", label="ID switches / ID real")
    ax2.set_ylabel("Razão / contagem por ID real")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    # Anotação com a mediana de oclusão no eixo x
    for xi, r in zip(x, results):
        ax2.annotate(f"{r.occlusion_median:.0f}q",
                     (xi, 0), xytext=(0, -22),
                     textcoords="offset points",
                     ha="center", fontsize=8, color="gray")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_xlabel("Sequência (mediana da oclusão, em quadros, anotada abaixo)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# ===========================================================================
# 6. Main
# ===========================================================================

def discover_sequences(root: str, detector: str) -> List[str]:
    seqs = []
    for name in sorted(os.listdir(root)):
        if name.startswith("MOT17-") and name.endswith(detector):
            if os.path.isdir(os.path.join(root, name)):
                seqs.append(name)
    return seqs

def _box(d: dict) -> List[float]:
    return [d["bb_left"], d["bb_top"], d["bb_width"], d["bb_height"]]

def _per_frame_matching(tracks_by_frame, gt_by_frame, iou_thresh):
    """
    Faz o casamento por quadro (Hungarian por IoU) e devolve:
        - match_count[g_idx, p_idx]: quantas vezes g e p foram casados
        - gt_assignment[gid]: {frame: pred_id}  (para contar IDSW/FRAG)
        - gt_ids, pred_ids (listas ordenadas), índices
    """
    gt_ids = sorted({g for fr in gt_by_frame.values() for g in fr})
    pred_ids = sorted({p for fr in tracks_by_frame.values() for p in fr})
    g_idx = {g: i for i, g in enumerate(gt_ids)}
    p_idx = {p: i for i, p in enumerate(pred_ids)}

    match_count = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
    gt_assignment: Dict[int, Dict[int, int]] = {g: {} for g in gt_ids}

    all_frames = sorted(set(gt_by_frame) | set(tracks_by_frame))
    for f in all_frames:
        gts = gt_by_frame.get(f, {})
        preds = tracks_by_frame.get(f, {})
        if not gts or not preds:
            continue
        g_list = sorted(gts.keys())
        p_list = sorted(preds.keys())
        M = np.zeros((len(g_list), len(p_list)))
        for i, gid in enumerate(g_list):
            gb = _box(gts[gid])
            for j, pid in enumerate(p_list):
                M[i, j] = calculate_iou(gb, _box(preds[pid]))
        r, c = linear_sum_assignment(-M)
        for ri, ci in zip(r, c):
            if M[ri, ci] >= iou_thresh:
                gid, pid = g_list[ri], p_list[ci]
                match_count[g_idx[gid], p_idx[pid]] += 1
                gt_assignment[gid][f] = pid

    return gt_ids, pred_ids, match_count, gt_assignment

@dataclass
class IdentityMetrics2:
    mAP: float = 0.0
    IDF1: float = 0.0
    IDTP: int = 0
    IDFP: int = 0
    IDFN: int = 0
    IDSW: int = 0
    FRAG: int = 0
    num_gt_ids: int = 0
    num_pred_ids: int = 0
    
    @property
    def id_ratio(self) -> float:
        return self.num_pred_ids / self.num_gt_ids if self.num_gt_ids else float("nan")

    @property
    def idsw_per_gt(self) -> float:
        return self.IDSW / self.num_gt_ids if self.num_gt_ids else float("nan")

def main():
    import os
    import pandas as pd
    import numpy as np
    
    from .naive_tracker import NaiveTracker
    from ..part_0.synthetic_metrics import CustomTrackerEvaluator
    from ..part_0.synthetic_data import get_synthetic_detections

    print("============================================================")
    print("         AVALIACAO DO RASTREADOR - PARTE 1 (BASELINE)       ")
    print("============================================================")

    resultados = []
    # Variamos a dificuldade (duração da oclusão) para preencher o eixo X do gráfico
    niveis_dificuldade = [0.0, 0.1, 0.2, 0.3, 0.4] 

    for prob_oclusao in niveis_dificuldade:
        nome_seq = f"Oclusao_{prob_oclusao:.1f}"
        print(f"\n[*] Processando sequencia simulada: {nome_seq}")
        
        gt_and_dets = get_synthetic_detections(prob_oclusao)
        ground_truth = gt_and_dets.ground_truth
        detections = gt_and_dets.detections
        
        naive_tracker = NaiveTracker(pd.DataFrame(detections))
        infered_tracks = naive_tracker.infer_tracks()

        evaluator = CustomTrackerEvaluator(0.5)
        identity_metrics = evaluator.evaluate_from_tracker(ground_truth, infered_tracks)
        flat_preds = evaluator._tracks_to_predictions(infered_tracks)

        mAP = compute_map(flat_preds, ground_truth, iou_thresh=0.5)

        gaps = occlusion_durations(ground_truth)
        if gaps:
            occ_med = float(np.median(gaps))
            occ_mean = float(np.mean(gaps))
            occ_max = int(np.max(gaps))
        else:
            occ_med = occ_mean = 0.0
            occ_max = 0

        num_gt_ids = len(set(d['id'] for d in ground_truth))
        num_pred_ids = len(set(d['id'] for d in flat_preds))
        id_ratio = num_pred_ids / num_gt_ids if num_gt_ids > 0 else float("nan")
        idsw_per_gt = identity_metrics.IDSW / num_gt_ids if num_gt_ids > 0 else float("nan")

        result = SeqResult(
            name=nome_seq, # Passamos o identificador da sequência
            mAP=mAP,
            IDF1=identity_metrics.IDF1,
            IDSW=identity_metrics.IDSW,
            num_gt_ids=num_gt_ids,
            num_pred_ids=num_pred_ids,
            id_ratio=id_ratio,
            idsw_per_gt=idsw_per_gt,
            occlusion_median=occ_med,
            occlusion_mean=occ_mean,
            occlusion_max=occ_max,
        )
        resultados.append(result)

    print("\n------------------------------------------------------------")
    print("                     RESULTADOS GERAIS                      ")
    print("------------------------------------------------------------")
    
    # Imprime no terminal de forma clara para o arquivo txt do 'tee'
    print(f"{'Sequencia':<15} | {'mAP':<6} | {'IDF1':<6} | {'IDSW':<5} | {'Pred/GT Ratio':<15} | {'Med Oclusao'}")
    print("-" * 75)
    for r in resultados:
        print(f"{r.name:<15} | {r.mAP:<6.4f} | {r.IDF1:<6.4f} | {r.IDSW:<5} | {r.id_ratio:<15.2f} | {r.occlusion_median:<5.1f}")
         
    # 3. Chamar a função do gráfico que já existe no seu arquivo original
    # Garantimos que a pasta de destino existe
    os.makedirs("outputs/part1", exist_ok=True)
    caminho_plot = "outputs/part1/descolamento_parte1.png"
    
    print("\n[*] Gerando e salvando grafico obrigatorio...")
    plot_descolamento(resultados, caminho_plot)
    print(f"[*] Grafico salvo com sucesso em: {caminho_plot}")
    
    print("\n============================================================")
    print("                     EXECUCAO CONCLUIDA                     ")
    print("============================================================")

if __name__ == "__main__":
    main()