"""Variantes da célula recorrente — Parte 3, Eixo 1.

Não altera nada da Parte 2: a AppearanceRNN original (GRU) continua sendo
importada de src.part_2.track_b_model. Aqui ficam apenas as duas variantes
novas, com a mesma interface forward_cnn / forward_rnn.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class _Encoder(nn.Module):
    """Encoder ResNet18 congelado + projeção treinável (mesmo setup da Parte 2)."""
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, emb_dim)

    def forward_cnn(self, x):
        return self.encoder(x)


class AppearanceRNN_Simple(_Encoder):
    """Variante com nn.RNNCell — é a que deve quebrar por gradiente que some."""
    def __init__(self, emb_dim: int = 128):
        super().__init__(emb_dim)
        self.rnn = nn.RNNCell(input_size=emb_dim, hidden_size=emb_dim)

    def forward_rnn(self, curr_emb, prev_hidden):
        if prev_hidden is None:
            prev_hidden = torch.zeros_like(curr_emb)
        return self.rnn(curr_emb, prev_hidden)


class AppearanceRNN_LSTM(_Encoder):
    """Variante com nn.LSTMCell. O estado é uma tupla (h, c)."""
    def __init__(self, emb_dim: int = 128):
        super().__init__(emb_dim)
        self.rnn = nn.LSTMCell(input_size=emb_dim, hidden_size=emb_dim)

    def forward_rnn(self, curr_emb, prev_state):
        if prev_state is None:
            h = torch.zeros_like(curr_emb)
            c = torch.zeros_like(curr_emb)
        else:
            h, c = prev_state
        h_new, c_new = self.rnn(curr_emb, (h, c))
        return (h_new, c_new)


def get_hidden(state):
    """Uniformiza o acesso ao estado oculto: LSTM devolve (h, c)."""
    return state[0] if isinstance(state, tuple) else state