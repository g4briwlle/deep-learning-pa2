import os
import sys
from pathlib import Path

# Garante que a raiz do repositório esteja no sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights

# Imports consistentes com a estrutura de pacotes do repositório
from src.config import DATA_DIR, CACHE_DIR
from src.mot_reader import get_dataloader

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Diretório para saídas executáveis da Parte 2
OUT_DIR = BASE_DIR / "outputs" / "part2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 1. ARQUITETURA DO MODELO (Trilha B - Enunciado: Encoder pequeno + Agregador)
# ==============================================================================
class AppearanceRNN(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        # Carrega a ResNet18 pré-treinada
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # 🚀 CONGELA O BACKBONE PARA VOAR NA CPU:
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Apenas a camada final de projeção (fc) será treinada
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, emb_dim)
        
        # A GRU continua treinando normalmente
        self.rnn = nn.GRUCell(input_size=emb_dim, hidden_size=emb_dim)

    def forward_cnn(self, x):
        """Passa o recorte (crop) de 128x64 pela CNN para extrair a aparência daquele instante."""
        return self.encoder(x)

    def forward_rnn(self, curr_emb, prev_hidden):
        """Atualiza a memória de aparência com a nova observação."""
        if prev_hidden is None:
            prev_hidden = torch.zeros_like(curr_emb)
        return self.rnn(curr_emb, prev_hidden)

# ==============================================================================
# 2. TREINAMENTO REAL NOS DADOS MOT17   
# ==============================================================================
def train_model():
    print("Iniciando Treinamento da Trilha B com dados MOT17 reais...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AppearanceRNN(emb_dim=128).to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3) # Pode usar lr um pouco maior (1e-3)
    
    # Enunciado exige Perda contrastiva ou triplet sobre as identidades do ground truth
    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
    
    # Usa o dataloader do arquivo 'mot_reader.py' já feito no trabalho
    # Vídeo índice 3, seq_length controla a janela temporal
    train_loader = get_dataloader(train=True, video_index=0, batch_size=32, shuffle=True)
    
    loss_history = []
    epochs = 10
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        # image_sequences: (Batch, Time, C, H, W) -> do mot_reader.py
        for image_sequences, person_ids in train_loader:
            image_sequences = image_sequences.to(device)
            B, T, C, H, W = image_sequences.shape
            
            optimizer.zero_grad()
            batch_loss = 0.0
            
            h_t = None
            
            # Backpropagation Through Time (BPTT) na janela de T quadros
            for t in range(T - 1):
                curr_frame = image_sequences[:, t, :, :, :]
                next_frame = image_sequences[:, t+1, :, :, :]
                
                # Extrai aparência atual e atualiza agregador
                emb_t = model.forward_cnn(curr_frame)
                h_t = model.forward_rnn(emb_t, h_t)
                
                # Extrai aparência do frame seguinte (Ground Truth da mesma pessoa)
                emb_next = model.forward_cnn(next_frame)
                
                # Triplet Loss:
                # Anchor: Estado acumulado até t (h_t)
                # Positive: Observação da MESMA pessoa em t+1 (emb_next)
                # Negative: Observação de OUTRA pessoa no mesmo batch (deslocando o tensor em 1)
                anchor = h_t
                positive = emb_next
                # Usar um shift de 1 garante que pegamos IDs diferentes contanto que batch_size > 1
                negative = torch.roll(emb_next, shifts=1, dims=0) 
                
                loss = triplet_loss_fn(anchor, positive, negative)
                batch_loss += loss
            
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            batch_count += 1
            
            if batch_count % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {batch_loss.item():.4f}", flush=True)
                
        avg_loss = epoch_loss / batch_count
        loss_history.append(avg_loss)
        print(f"--- Fim da Epoch {epoch+1} | Loss Média: {avg_loss:.4f} ---")

    # Salva o gráfico da função de perda (Artefato para apresentação)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker='o', color='purple')
    plt.title("Treinamento Triplet Loss - Memória de Aparência (MOT17)")
    plt.xlabel("Época")
    plt.ylabel("Loss Média")
    plt.grid(True)
    plot_path = OUT_DIR / "trilhaB_loss_MOT17.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[*] Gráfico de Loss salvo em: {plot_path}")
    
    # Salva os pesos (requisito do checkpoint nos Entregáveis)
    ckpt_path = OUT_DIR / "appearance_rnn.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[*] Checkpoint salvo em: {ckpt_path}")
    
    return model

if __name__ == "__main__":
    model = train_model()
    print("Processo da Parte 2 (Trilha B) finalizado com sucesso. Confira a pasta outputs/part2/")