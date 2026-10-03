import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Garante que a raiz do repositório esteja no sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.part_2.track_b_model import AppearanceRNN
from src.mot_reader import get_dataloader

# Diretório para saídas da Parte 4
OUT_DIR = BASE_DIR / "outputs" / "part4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def measure_gradient_horizon(model_path: Path, seq_length: int = 32):
    """
    Mede a norma do gradiente (horizonte analítico) ao longo de seq_length quadros.
    Isso responde à pergunta: "Até quantos quadros no passado o gradiente sobrevive?"
    """
    print(f"==================================================================")
    print(f" INICIANDO PARTE 4 - HORIZONTE DE MEMÓRIA ANALÍTICO (k={seq_length}) ")
    print(f"==================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Carrega o modelo treinado
    model = AppearanceRNN(emb_dim=128).to(device)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"[*] Modelo carregado de {model_path}")
    else:
        print("[!] Checkpoint não encontrado. Usando modelo não treinado para demonstração de arquitetura.")

    # O modelo precisa estar em modo de treino para calcular gradientes
    model.train()
    
    # 2. Carrega um batch de dados reais
    # seq_length define a janela máxima do horizonte. No mot_reader, precisamos 
    # instanciar o dataset com esse T. Como o DataLoader retorna (B, T, C, H, W)...
    loader = get_dataloader(train=True, video_index=0, batch_size=2, shuffle=False)
    
    # Pega apenas o primeiro batch
    iterator = iter(loader)
    image_sequences, _ = next(iterator)
    
    # Garante que temos quadros suficientes no tensor lido. Se o loader padrão 
    # carrega T menor que seq_length, limitamos seq_length ao máximo disponível.
    B, T, C, H, W = image_sequences.shape
    actual_seq_length = min(seq_length, T)
    
    image_sequences = image_sequences.to(device)
    
    # Lista para armazenar as referências aos tensores de estado oculto (h_t)
    hidden_states = []
    
    h_t = None
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # Apenas para o dummy backward
    optimizer.zero_grad()

    print(f"[*] Fazendo forward de {actual_seq_length} quadros para extrair gradientes...")
    
    for t in range(actual_seq_length):
        curr_frame = image_sequences[:, t, :, :, :]
        
        # Extrai aparência
        emb_t = model.forward_cnn(curr_frame)
        
        # Atualiza a GRU
        h_t = model.forward_rnn(emb_t, h_t)
        
        # Retém o gradiente do tensor folha para podermos ler depois do backward
        h_t.retain_grad()
        hidden_states.append(h_t)

    # 3. Força um gradiente no último estado oculto (t = final)
    # Como não temos uma "loss" real de rastreamento de um objeto até o fim aqui,
    # simulamos um erro no último frame fazendo o backward na média do último h_t.
    dummy_loss = hidden_states[-1].mean()
    dummy_loss.backward()

    # 4. Coleta a norma dos gradientes retrocedendo no tempo (de t=final até t=0)
    # O gradiente de hidden_states[-1] é o mais forte. O de hidden_states[0] é o mais fraco (que sofreu decaimento).
    grad_norms = []
    
    # k representa os passos para trás (k=0 é o estado atual/final, k=31 é o estado há 31 quadros)
    for k, h in enumerate(reversed(hidden_states)):
        if h.grad is not None:
            norm = h.grad.norm().item()
            grad_norms.append(norm)
        else:
            grad_norms.append(0.0)

    # 5. Salvar Artefatos
    _save_artifacts(grad_norms, actual_seq_length)

def _save_artifacts(grad_norms: list, seq_length: int):
    # Eixo X é 'k' (passos no passado em relação a t)
    k_steps = np.arange(len(grad_norms))
    
    # Normaliza pelo valor máximo (k=0) para facilitar visualização de queda percentual
    max_norm = max(grad_norms) if max(grad_norms) > 0 else 1
    grad_norms_normalized = [g / max_norm for g in grad_norms]

    plt.figure(figsize=(10, 6))
    plt.plot(k_steps, grad_norms_normalized, marker='o', linestyle='-', color='purple', linewidth=2)
    plt.title("Parte 4: Horizonte Analítico de Memória (Decaimento do Gradiente)")
    plt.xlabel("k (quadros no passado)")
    plt.ylabel(r"Norma Normalizada do Gradiente $||\partial L_t / \partial h_{t-k}||$")
    plt.grid(True, alpha=0.3)
    plt.yscale('log') # Escala logarítmica é ideal para ver o 'vanishing gradient'
    
    plot_path = OUT_DIR / "memory_horizon_gradient.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] Gráfico do horizonte salvo em: {plot_path}")

    # Salva os valores brutos para constar na apresentação ou log
    txt_path = OUT_DIR / "memory_horizon_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("RELATÓRIO - PARTE 4: HORIZONTE ANALÍTICO (GRADIENTE QUE SOME)\n")
        f.write("==============================================================\n")
        f.write(f"Janela de tempo avaliada: {seq_length} quadros.\n\n")
        f.write("k (Passos p/ trás) | Norma Bruta | Norma Retida (%)\n")
        f.write("-" * 55 + "\n")
        for k, (bruta, norm) in enumerate(zip(grad_norms, grad_norms_normalized)):
            f.write(f"{k:<18} | {bruta:^11.6f} | {norm*100:>13.2f}%\n")
        
        f.write("\nDIAGNÓSTICO:\n")
        f.write("Observando a queda exponencial (ou resistência) da norma na escala logarítmica,\n")
        f.write("podemos definir o 'horizonte efetivo' como o valor k onde a norma cai abaixo de 5% do sinal original.\n")
        f.write("Buracos de detecção (oclusões) maiores que esse horizonte k não receberão sinal de supervisão efetivo.\n")
        
    print(f"[*] Relatório de diagnóstico salvo em: {txt_path}")
    print("==================================================================")

if __name__ == "__main__":
    checkpoint_file = BASE_DIR / "outputs" / "part2" / "appearance_rnn.pth"
    # Você pode precisar ajustar o DataLoader localmente para puxar T=32
    measure_gradient_horizon(model_path=checkpoint_file, seq_length=16)