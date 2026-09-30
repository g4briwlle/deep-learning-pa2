import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import io
from PIL import Image
import zipfile
import sys
# Força o terminal a não quebrar caracteres em português
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

class MOTSequenceDataset(Dataset):
    def __init__(self, zip_path, seq_name, seq_length=8, transform=None):
        """
        zip_path: Caminho para o arquivo ZIP principal (ex: 'MOT17.zip').
        seq_name: Caminho interno da sequência no ZIP (ex: 'MOT17/train/MOT17-04-FRCNN').
        seq_length: Time window (T) the RNN will process at once.
        transform: Torchvision transforms (e.g., resize and tensor conversion).
        """
        self.zip_path = zip_path
        self.seq_name = seq_name
        self.seq_length = seq_length
        
        # Convolutional models require fixed-size images. 128x64 is standard for pedestrians.
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((128, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        # Ler o ground truth (gt.txt) diretamente de dentro do ZIP
        gt_internal_path = f"{seq_name}/gt/gt.txt"
        cols = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'visibility']
        
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            with z.open(gt_internal_path) as f:
                df = pd.read_csv(f, names=cols)

        # Filter data: In MOT17, class 1 represents 'pedestrian'
        df = df[df['class'] == 1]
        
        # Group by identity (ID) to create temporal sequences of the same person
        self.sequences = []
        
        for person_id, group in df.groupby('id'):
            # Sort by frame to ensure correct temporal order
            group = group.sort_values('frame')
            
            # Split the frames into chunks of size 'seq_length'
            for i in range(0, len(group) - seq_length + 1, seq_length):
                seq_chunk = group.iloc[i : i + seq_length]
                self.sequences.append(seq_chunk)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieves a sequence, crops the bounding boxes, and returns a Tensor
        with shape (Time, Channels, Height, Width).
        """
        seq_chunk = self.sequences[idx]
        crops = []
        
        # Get the person ID for this sequence chunk
        person_id = seq_chunk.iloc[0]['id']

        # Abre o zip apenas uma vez por sequência para extrair os frames necessários
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            for _, row in seq_chunk.iterrows():
                img_name = f"{int(row['frame']):06d}.jpg"
                img_internal_path = f"{self.seq_name}/img1/{img_name}"
                
                # Extrai a imagem do zip para a memória
                with z.open(img_internal_path) as f:
                    img_bytes = f.read()
                    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                
                # Calculate bounding box
                left = max(0, int(row['bb_left']))
                top = max(0, int(row['bb_top']))
                right = int(left + row['bb_width'])
                bottom = int(top + row['bb_height'])
                
                # Crop and transform
                crop = image.crop((left, top, right, bottom))
                crop_tensor = self.transform(crop)
                crops.append(crop_tensor)

        # Stack into shape (seq_length, C, H, W)
        sequence_tensor = torch.stack(crops)
        
        return sequence_tensor, torch.tensor(person_id, dtype=torch.long)


######### testing

# Aponte para o arquivo ZIP principal
caminho_zip = "MOT17.zip"
# Aponte para a sequência desejada dentro do ZIP
nome_sequencia = "MOT17/train/MOT17-04-FRCNN"

# Instancie o Dataset modificado
my_dataset = MOTSequenceDataset(zip_path=caminho_zip, seq_name=nome_sequencia, seq_length=8)
print(f"Total sequences extracted: {len(my_dataset)}")

# Crie o DataLoader
my_dataloader = DataLoader(my_dataset, batch_size=4, shuffle=True)

# Loop through one batch to verify shapes
for batch_idx, (image_sequences, person_ids) in enumerate(my_dataloader):
    # image_sequences shape: (Batch, Time, Channels, Height, Width)
    print(f"Batch {batch_idx}:")
    print(f" - Image tensor shape: {image_sequences.shape}")
    print(f" - Person IDs in this batch: {person_ids}")
    break