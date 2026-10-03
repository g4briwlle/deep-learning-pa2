import argparse
import os
import cv2
import pandas as pd
import numpy as np
import torch
import zipfile
import io
from pathlib import Path
from PIL import Image

# Importa o rastreador da Trilha B criado na Parte 2
from src.part_2.appearance_tracker import AppearanceTracker
from src.config import ZIP_PATH

def get_color(track_id: int):
    """Gera uma cor RGB consistente e bem distinguível baseada no track_id."""
    np.random.seed(track_id * 100)
    color = np.random.randint(100, 255, size=3).tolist()
    return tuple(color)

def find_sequence_prefix(seq_name: str) -> str:
    """Busca o caminho raiz correto da sequência dentro do ZIP (ex: MOT17/test/MOT17-01-FRCNN)"""
    if not Path(ZIP_PATH).exists():
        raise FileNotFoundError(f"Arquivo ZIP não encontrado: {ZIP_PATH}")
        
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        all_files = z.namelist()
        
    # Procura algum arquivo det.txt que contenha o nome da sequência
    for f in all_files:
        if seq_name in f and f.endswith("det/det.txt"):
            # Retorna tudo antes de "/det/det.txt"
            return f.replace("/det/det.txt", "")
            
    raise ValueError(f"Sequência '{seq_name}' não encontrada no arquivo {ZIP_PATH}")

def load_public_detections_from_zip(seq_prefix: str) -> pd.DataFrame:
    """Carrega o arquivo det.txt de uma sequência do MOT17 DIRETAMENTE DO ZIP com filtro permissivo."""
    det_internal_path = f"{seq_prefix}/det/det.txt"
    cols = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'visibility']
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        with z.open(det_internal_path) as f:
            df = pd.read_csv(f, names=cols)
            
    print(f"    -> [DEBUG] Caixas brutas lidas do det.txt: {len(df)}")
        
    # Filtro super permissivo: ignora a coluna 'class' completamente (que costuma vir vazia/errada no det.txt)
    # E aceita valores de confiança maiores que um limiar bem baixo (alguns detectores como DPM usam conf negativa)
    df = df[df['conf'] > -10.0].copy()
    
    print(f"    -> [DEBUG] Caixas repassadas ao rastreador: {len(df)}")
    
    return df

def get_image_list_from_zip(seq_prefix: str) -> list[str]:
    """Retorna uma lista ordenada dos caminhos internos das imagens de uma sequência no ZIP."""
    img_dir_prefix = f"{seq_prefix}/img1/"
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        all_files = z.namelist()
        
    img_files = [f for f in all_files if f.startswith(img_dir_prefix) and f.endswith('.jpg')]
    return sorted(img_files)

def read_image_from_zip(internal_path: str) -> np.ndarray:
    """Lê uma imagem do ZIP e retorna como array BGR do OpenCV."""
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        with z.open(internal_path) as f:
            img_bytes = f.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img_cv2

def main(seq_name: str, output_path: str, checkpoint_path: str):
    print(f"[*] Buscando sequência '{seq_name}' no arquivo {ZIP_PATH}...")
    seq_prefix = find_sequence_prefix(seq_name)
    
    print(f"[*] Carregando detecções do caminho: {seq_prefix}/det/det.txt")
    dets_df = load_public_detections_from_zip(seq_prefix)
    
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Pesos do modelo não encontrados em {ckpt_file}.")
        
    print("[*] Instanciando o modelo temporal AppearanceTracker (Trilha B)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracker = AppearanceTracker(
        dets_df=dets_df,
        seq_name=seq_prefix,
        checkpoint_path=ckpt_file,
        cos_threshold=0.2,
        iou_gate_threshold=0.0,
        device=device
    )
    
    print("[*] Executando associação temporal com RNN (isso pode levar alguns minutos)...")
    tracks_by_frame = tracker.infer_tracks()
    
    unique_ids = set()
    for frame, tracks in tracks_by_frame.items():
        unique_ids.update(tracks.keys())
        
    print(f"[*] Contagem final: {len(unique_ids)} objetos únicos identificados na sequência.")
    
    print("[*] Gerando vídeo de saída com identidades coloridas...")
    image_files_internal = get_image_list_from_zip(seq_prefix)
    
    if not image_files_internal:
        raise ValueError(f"Nenhuma imagem encontrada na pasta img1 de {seq_prefix} dentro do ZIP")
        
    frame_example = read_image_from_zip(image_files_internal[0])
    height, width, _ = frame_example.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    
    for idx, internal_img_path in enumerate(image_files_internal):
        frame_number = idx + 1
        frame = read_image_from_zip(internal_img_path)
        
        current_tracks = tracks_by_frame.get(frame_number, {})
        
        for track_id, bbox in current_tracks.items():
            left, top = int(bbox['bb_left']), int(bbox['bb_top'])
            w, h = int(bbox['bb_width']), int(bbox['bb_height'])
            right, bottom = left + w, top + h
            
            color = get_color(track_id)
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
            
            label = f"ID: {track_id}"
            (t_w, t_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (left, top - t_h - 5), (left + t_w, top), color, -1)
            cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
        out_video.write(frame)
        
        if frame_number % 50 == 0:
            print(f"    - Renderizados {frame_number}/{len(image_files_internal)} frames...")

    out_video.release()
    print(f"[*] Vídeo salvo com sucesso em: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia do Rastreador Temporal (PA2 - Trilha B)")
    parser.add_argument("--seq_name", type=str, required=True, help="Nome da sequencia (ex: MOT17-01-FRCNN)")
    parser.add_argument("--output", type=str, default="inferencia_out.mp4", help="Caminho do video de saida")
    parser.add_argument("--checkpoint", type=str, default="outputs/part2/appearance_rnn.pth", help="Caminho para os pesos treinados (.pth)")
    
    args = parser.parse_args()
    main(args.seq_name, args.output, args.checkpoint)