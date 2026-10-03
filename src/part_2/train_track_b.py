"""CLI de treino do Track B (Appearance RNN) — Parte 2.

Não substitui src/part_2/track_b_model.py; apenas expõe o mesmo treino
como um script parametrizável, para virar um único comando no README.
"""
import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

torch.set_num_threads(os.cpu_count())

from src.mot_reader import get_dataloader
from src.part_2.track_b_model import AppearanceRNN


def parse_args():
    p = argparse.ArgumentParser(
        description="Treina o modelo Track B (Appearance RNN) da Parte 2.")
    p.add_argument("--epochs", type=int, default=22,
                   help="Número de épocas (default: 22, igual ao script original).")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size (default: 32).")
    p.add_argument("--video-index", type=int, default=0,
                   help="Índice do vídeo de treino, 0..6 (default: 0).")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate do Adam (default: 1e-3).")
    p.add_argument("--emb-dim", type=int, default=128,
                   help="Dimensão do embedding / estado oculto (default: 128).")
    p.add_argument("--output-dir", type=str, default="outputs/part2",
                   help="Diretório de saída (default: outputs/part2).")
    p.add_argument("--ckpt-name", type=str, default="appearance_rnn.pth",
                   help="Nome do checkpoint (default: appearance_rnn.pth).")
    return p.parse_args()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    print(f"[*] Config: epochs={args.epochs} batch={args.batch_size} "
          f"video_index={args.video_index} lr={args.lr} emb_dim={args.emb_dim}")

    out_dir = BASE_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AppearanceRNN(emb_dim=args.emb_dim).to(device)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

    train_loader = get_dataloader(
        train=True, video_index=args.video_index,
        batch_size=args.batch_size, shuffle=True,
    )

    loss_history = []
    for epoch in range(args.epochs):
        epoch_loss, batch_count = 0.0, 0
        for image_sequences, _person_ids in train_loader:
            image_sequences = image_sequences.to(device)
            B, T, C, H, W = image_sequences.shape

            optimizer.zero_grad()
            batch_loss = 0.0
            h_t = None
            for t in range(T - 1):
                emb_t = model.forward_cnn(image_sequences[:, t])
                h_t = model.forward_rnn(emb_t, h_t)
                emb_next = model.forward_cnn(image_sequences[:, t + 1])

                anchor = h_t
                positive = emb_next
                negative = torch.roll(emb_next, shifts=1, dims=0)
                batch_loss = batch_loss + triplet_loss_fn(anchor, positive, negative)

            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_count}, "
                      f"Loss: {batch_loss.item():.4f}", flush=True)

        avg = epoch_loss / max(batch_count, 1)
        loss_history.append(avg)
        print(f"--- Fim da Epoch {epoch+1}/{args.epochs} | "
              f"Loss Média: {avg:.4f} ---", flush=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, args.epochs + 1), loss_history, marker="o", color="purple")
    plt.title("Treinamento Triplet Loss - Memória de Aparência (MOT17)")
    plt.xlabel("Época")
    plt.ylabel("Loss Média")
    plt.grid(True)
    plt.savefig(out_dir / "trilhaB_loss_MOT17.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[*] Gráfico salvo em: {out_dir / 'trilhaB_loss_MOT17.png'}")

    ckpt_path = out_dir / args.ckpt_name
    torch.save(model.state_dict(), ckpt_path)
    print(f"[*] Checkpoint salvo em: {ckpt_path}")


if __name__ == "__main__":
    train(parse_args())