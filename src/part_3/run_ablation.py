"""Parte 3 — Eixo 1: ablação da célula recorrente (RNN simples / GRU / LSTM)
em diferentes comprimentos de BPTT truncado (T ∈ {4, 8, 16, 32}), com 3 seeds.
"""
import os
import sys
import json
import random
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# -- imports do repositório (o cwd precisa ser a raiz; ver comando no final) --
from src.config import DATA_DIR, CACHE_DIR, ZIP_PATH
from src.mot_reader import MOTSequenceDataset
from src.part_2.track_b_model import AppearanceRNN
from src.part_3.variants import (
    AppearanceRNN_Simple, AppearanceRNN_LSTM, get_hidden,
)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

torch.set_num_threads(os.cpu_count())

# ------------------------------------------------------------------ #
# Configuração da ablação
# ------------------------------------------------------------------ #
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "outputs" / "part3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARCHS = {
    "gru":    AppearanceRNN,         # baseline já treinado na Parte 2
    "simple": AppearanceRNN_Simple,
    "lstm":   AppearanceRNN_LSTM,
}
BPTT_LENGTHS = [4, 8, 16, 32]
SEEDS = [0, 1, 2]
SHORT_EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3
EMB_DIM = 128
NUM_WORKERS = 4
VIDEO_INDEX = 0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(bptt_len: int, shuffle: bool = True):
    """MOTSequenceDataset aceita seq_length; get_dataloader não aceita.
    Por isso instanciamos o Dataset direto. O cache .dat NÃO depende de T
    (só a lista de sequências em RAM), então o primeiro run constrói e os
    demais reaproveitam."""
    ds = MOTSequenceDataset(
        train=True,
        video_index=VIDEO_INDEX,
        seq_length=bptt_len,
    )
    print(f"  [dataset] seq_length={bptt_len}, {len(ds)} sequences", flush=True)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS)


def train_short(model_cls, bptt_len: int, seed: int, device):
    """Treino curto para uma config (arch, T, seed). Devolve (model, history)."""
    set_seed(seed)
    model = model_cls(emb_dim=EMB_DIM).to(device)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=LR)
    triplet = nn.TripletMarginLoss(margin=1.0, p=2)

    loader = make_loader(bptt_len=bptt_len, shuffle=True)

    history = []
    for epoch in range(SHORT_EPOCHS):
        ep_loss, n = 0.0, 0
        for image_sequences, _person_ids in loader:
            # image_sequences: (B, T, C, H, W). O dataset garante T == bptt_len.
            image_sequences = image_sequences.to(device)
            B, T, C, H, W = image_sequences.shape

            optimizer.zero_grad()
            state = None
            batch_loss = 0.0
            for t in range(T - 1):
                emb_t = model.forward_cnn(image_sequences[:, t])
                state = model.forward_rnn(emb_t, state)
                emb_next = model.forward_cnn(image_sequences[:, t + 1])

                anchor = get_hidden(state)                # LSTM-safe
                positive = emb_next
                negative = torch.roll(emb_next, shifts=1, dims=0)

                batch_loss = batch_loss + triplet(anchor, positive, negative)

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            optimizer.step()

            ep_loss += batch_loss.item()
            n += 1

        avg = ep_loss / max(n, 1)
        history.append(avg)
        print(f"  [{model_cls.__name__} | T={bptt_len} | seed={seed}] "
              f"epoch {epoch+1}/{SHORT_EPOCHS} loss={avg:.4f}", flush=True)

    return model, history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    print(f"[*] Combinações a rodar: "
          f"{len(ARCHS)} archs × {len(BPTT_LENGTHS)} T × {len(SEEDS)} seeds "
          f"= {len(ARCHS)*len(BPTT_LENGTHS)*len(SEEDS)} treinos\n", flush=True)

    results = []
    for (arch_name, bptt_len, seed) in product(ARCHS.keys(), BPTT_LENGTHS, SEEDS):
        print(f"=== {arch_name} | T={bptt_len} | seed={seed} ===", flush=True)
        model_cls = ARCHS[arch_name]
        model, history = train_short(model_cls, bptt_len, seed, device)

        ckpt = OUT_DIR / f"ckpt_{arch_name}_T{bptt_len}_seed{seed}.pth"
        torch.save(model.state_dict(), ckpt)
        print(f"  [ckpt] {ckpt.name}", flush=True)

        results.append({
            "arch": arch_name,
            "T": bptt_len,
            "seed": seed,
            "final_loss": history[-1],
            "history": json.dumps(history),
            "ckpt": str(ckpt.relative_to(BASE_DIR)),
        })

    df = pd.DataFrame(results)
    csv_path = OUT_DIR / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[*] Tabela de resultados: {csv_path}")

    # -------- Plot: loss final média ± std por (arch, T) --------
    fig, ax = plt.subplots(figsize=(8, 5))
    for arch_name in ARCHS:
        sub = df[df["arch"] == arch_name]
        grp = sub.groupby("T")["final_loss"].agg(["mean", "std"]).sort_index()
        ax.errorbar(grp.index, grp["mean"], yerr=grp["std"].fillna(0),
                    marker="o", capsize=4, label=arch_name)
    ax.set_xscale("log", base=2)
    ax.set_xticks(BPTT_LENGTHS, labels=[str(t) for t in BPTT_LENGTHS])
    ax.set_xlabel("Comprimento de BPTT truncado (T)")
    ax.set_ylabel("Loss final (média ± std, 3 seeds)")
    ax.set_title("Parte 3 — Eixo 1: célula recorrente × BPTT")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plot_path = OUT_DIR / "ablation_cell_bptt.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[*] Plot: {plot_path}")


if __name__ == "__main__":
    main()