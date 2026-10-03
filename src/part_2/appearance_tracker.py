import sys
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import zipfile
import io
from scipy.optimize import linear_sum_assignment
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.tracker import Tracker
from src.config import ZIP_PATH
from src.part_2.track_b_model import AppearanceRNN
from src.part_1.hungarian_match import iou_matrix

class AppearanceTracker(Tracker):
    """
    Rastreador da Trilha B: Usa memória temporal de aparência (GRU)
    com matching por Cosseno e portão geométrico de IoU.
    """
    def __init__(
        self,
        dets_df: pd.DataFrame,
        seq_name: str,
        checkpoint_path: Path,
        cos_threshold: float = 0.3,
        iou_gate_threshold: float = 0.0,
        kill_track_frames: int = 15,
        device: str = "cpu"
    ):
        super().__init__(dets_df)
        self.seq_name = seq_name
        self.cos_threshold = cos_threshold
        self.iou_gate_threshold = iou_gate_threshold
        self.max_age = kill_track_frames
        self.device = torch.device(device)
        
        # Carrega o modelo com os pesos treinados
        self.model = AppearanceRNN(emb_dim=128).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _extract_crops(self, frame_number: int, frame_det_df: pd.DataFrame) -> torch.Tensor:
        """Lê o quadro dentro do ZIP do MOT17 e extrai os embeddings de cada detecção."""
        img_name = f"{int(frame_number):06d}.jpg"
        img_internal_path = f"{self.seq_name}/img1/{img_name}"

        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            with z.open(img_internal_path) as f:
                img_bytes = f.read()
                full_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        crops = []
        for _, row in frame_det_df.iterrows():
            l = max(0, int(row['bb_left']))
            t = max(0, int(row['bb_top']))
            r = max(l + 1, int(l + row['bb_width']))
            b = max(t + 1, int(t + row['bb_height']))
            crop = full_img.crop((l, t, r, b))
            crops.append(self.transform(crop))

        if not crops:
            return torch.empty((0, 128), device=self.device)

        batch_crops = torch.stack(crops).to(self.device)
        with torch.no_grad():
            embs = self.model.forward_cnn(batch_crops)
        return embs

    def _tracks_to_df(self) -> pd.DataFrame:
        if not self.tracks:
            return pd.DataFrame(columns=["track_id", "bb_left", "bb_top", "bb_width", "bb_height"])
        rows = [{"track_id": tid, **state} for tid, state in self.tracks.items()]
        return pd.DataFrame(rows)

    def update_tracks(self):
        frame_number, frame_det_df = self._get_det_df_frame()
        self._update_current_frame()

        if len(frame_det_df) == 0:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["misses"] += 1
                if self.tracks[tid]["misses"] >= self.max_age:
                    del self.tracks[tid]
            self.final_tracks[frame_number] = copy.deepcopy(self.tracks)
            return

        # 1. Extrair os embeddings das detecções atuais
        det_embs = self._extract_crops(frame_number, frame_det_df)
        tracks_df = self._tracks_to_df()

        if len(tracks_df) == 0:
            # Inicializa todas as detecções como novas tracks
            for idx, (_, row) in enumerate(frame_det_df.iterrows()):
                h_0 = self.model.forward_rnn(det_embs[idx].unsqueeze(0), None)
                self.tracks[self.next_track_id] = {
                    "bb_left": row["bb_left"], "bb_top": row["bb_top"],
                    "bb_width": row["bb_width"], "bb_height": row["bb_height"],
                    "last_frame": frame_number, "misses": 0,
                    "hidden": h_0.squeeze(0).detach()
                }
                self.next_track_id += 1
            self.final_tracks[frame_number] = copy.deepcopy(self.tracks)
            return

        # 2. Similaridade de Cosseno entre estados ocultos das tracks e novos embeddings
        track_ids = list(self.tracks.keys())
        track_hiddens = torch.stack([self.tracks[tid]["hidden"] for tid in track_ids])

        th_norm = F.normalize(track_hiddens, p=2, dim=1)
        de_norm = F.normalize(det_embs, p=2, dim=1)
        cos_sim = torch.mm(th_norm, de_norm.t()).cpu().numpy()

        cost_matrix = 1.0 - cos_sim

        # 3. Portão Geométrico Estrito de IoU
        ious = iou_matrix(tracks_df, frame_det_df)
        
        # Calcula a distância em pixels entre os centros
        tracks_centers = tracks_df[['bb_left', 'bb_top']].values + tracks_df[['bb_width', 'bb_height']].values / 2
        dets_centers = frame_det_df[['bb_left', 'bb_top']].values + frame_det_df[['bb_width', 'bb_height']].values / 2
        dist_matrix = np.linalg.norm(tracks_centers[:, None, :] - dets_centers[None, :, :], axis=2)

        # Distância máxima permitida (ex: 50 pixels por frame ausente)
        max_dist = np.array([[50 * (self.tracks[tid]["misses"] + 1) for tid in track_ids]]).T
        
        # Bloqueia a associação se estiver muito longe (mesmo com aparência parecida)
        cost_matrix[dist_matrix > max_dist] = 1e5

        # 4. Hungarian Match
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks, matched_dets = set(), set()
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cos_sim[r, c] >= self.cos_threshold and cost_matrix[r, c] < 1e4:
                matches.append((track_ids[r], c))
                matched_tracks.add(track_ids[r])
                matched_dets.add(c)

        # 5. Atualizar tracks casadas com novo embedding na recorrência
        for tid, det_idx in matches:
            det = frame_det_df.iloc[det_idx]
            curr_emb = det_embs[det_idx].unsqueeze(0)
            prev_h = self.tracks[tid]["hidden"].unsqueeze(0)
            
            with torch.no_grad():
                new_h = self.model.forward_rnn(curr_emb, prev_h)

            self.tracks[tid].update({
                "bb_left": det["bb_left"], "bb_top": det["bb_top"],
                "bb_width": det["bb_width"], "bb_height": det["bb_height"],
                "last_frame": frame_number, "misses": 0,
                "hidden": new_h.squeeze(0).detach()
            })

        # 6. Criar novas tracks para detecções não casadas
        unmatched_dets = [j for j in range(len(frame_det_df)) if j not in matched_dets]
        for det_idx in unmatched_dets:
            det = frame_det_df.iloc[det_idx]
            h_0 = self.model.forward_rnn(det_embs[det_idx].unsqueeze(0), None)
            self.tracks[self.next_track_id] = {
                "bb_left": det["bb_left"], "bb_top": det["bb_top"],
                "bb_width": det["bb_width"], "bb_height": det["bb_height"],
                "last_frame": frame_number, "misses": 0,
                "hidden": h_0.squeeze(0).detach()
            }
            self.next_track_id += 1

        # 7. Incrementar misses e remover mortas
        unmatched_tracks = set(self.tracks.keys()) - matched_tracks
        for tid in unmatched_tracks:
            self.tracks[tid]["misses"] += 1
            if self.tracks[tid]["misses"] >= self.max_age:
                del self.tracks[tid]

        self.final_tracks[frame_number] = copy.deepcopy(self.tracks)