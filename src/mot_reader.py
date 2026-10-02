"""Module to read data from the MOT zip file using a memory-mapped cache"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import os
import io
import zipfile
import sys
from pathlib import Path

from .config import DATA_DIR, CACHE_DIR, ZIP_PATH

# Force terminal encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

TRAINING_VIDEOS_NUMBERS_LIST = [
    2,
    4,
    5,
    9,
    10,
    11,
    13,
]
TEST_VIDEOS_NUMBER_LIST = list(set(range(1, 15)) - set(TRAINING_VIDEOS_NUMBERS_LIST))

class MOTSequenceDataset(Dataset):
    def __init__(
        self,
        train: bool,
        video_index: int,
        seq_length: int = 8,
        transform: torch.nn.Module | None = None,
        zip_path: Path = ZIP_PATH,
        cache_dir: Path = CACHE_DIR,
    ):
        """
        train (bool): If True, loads training data. Otherwise, test data.
        video_index (int): Index of video to use. Must be integer between 0 and 6, included.
        seq_length (int): Time window (T) the RCNN will process at once.
        transform: Torchvision transforms (applied dynamically on __getitem__).
        zip_path (Path): Path to the main ZIP file (e.g., 'MOT17.zip'). Default is DATA_DIR / 'MOT17.zip'.
        cache_dir (Path): Directory where the memory-mapped file and indexing CSV will be saved. Default is CACHE_DIR.
        """
        if video_index < 0 or video_index > 6:
            raise ValueError(f"`video_index` must be between 0 and 6, included")
        
        self.zip_path = zip_path
        self.seq_length = seq_length
        self.cache_dir = cache_dir
        
        # Logic for getting correct video name
        if train:
            video_number = TRAINING_VIDEOS_NUMBERS_LIST[video_index]
        else:
            video_number = TEST_VIDEOS_NUMBER_LIST[video_index]
        if video_number < 10:
            video_number_str = f'0{video_number}'
        else:
            video_number_str = str(video_number)
            
        # Alwyas getting FRCNN
        self.seq_name = f'MOT17/{'train' if train else 'test'}/MOT17-{video_number_str}-FRCNN'
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Unique names for the cache files based on the sequence
        safe_seq_name = self.seq_name.replace("/", "_")
        self.mmap_path = os.path.join(self.cache_dir, f"{safe_seq_name}_crops.dat")
        self.df_cache_path = os.path.join(self.cache_dir, f"{safe_seq_name}_metadata.csv")

        # Transform logic: ToTensor automatically converts numpy arrays [0, 255] to [0.0, 1.0]
        # Resize is intentionally omitted here because it will be done ONCE during caching
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        # If cache doesn't exist, we build it. This runs only ONCE.
        if not os.path.exists(self.mmap_path) or not os.path.exists(self.df_cache_path):
            self._build_cache()

        # Load the cached metadata
        self.df = pd.read_csv(self.df_cache_path)
        self.total_crops = len(self.df)

        # Open the memory-mapped file in READ-ONLY mode ('r')
        # Lazy initialization: deixa como None aqui para o Windows conseguir serializar o dataset
        self.mmap_data = None

        self._prepare_sequences()

    def _build_cache(self):
        """
        Extracts images from the ZIP, crops bounding boxes, resizes them,
        and saves raw pixel data sequentially into a binary memmap file.
        """
        print(f"Building cache for {self.seq_name}... This will take a few minutes but runs only once.")
        
        gt_internal_path = f"{self.seq_name}/gt/gt.txt"
        cols = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'visibility']
        
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            with z.open(gt_internal_path) as f:
                df = pd.read_csv(f, names=cols)

        # Filter for pedestrians (class == 1) and valid visibility if desired
        df = df[df['class'] == 1].copy()
        
        # Sort by frame and ID to establish a predictable processing order
        df = df.sort_values(by=['frame', 'id']).reset_index(drop=True)
        total_crops = len(df)

        # Create a writable memory map file
        mmap_writer = np.memmap(
            self.mmap_path,
            dtype=np.uint8,
            mode='w+', # Create or overwrite
            shape=(total_crops, 128, 64, 3)
        )

        resize_op = transforms.Resize((128, 64))

        # We group by 'frame' to drastically speed up processing.
        # This prevents opening the same frame from the ZIP multiple times.
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            for frame_idx, (frame, group) in enumerate(df.groupby('frame')):
                img_name = f"{int(frame):06d}.jpg" # type: ignore
                img_internal_path = f"{self.seq_name}/img1/{img_name}"
                
                try:
                    with z.open(img_internal_path) as f:
                        img_bytes = f.read()
                        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                except KeyError:
                    print(f"Warning: Could not find image {img_internal_path} in ZIP. Skipping frame.")
                    continue

                # Crop and store every pedestrian in this frame
                for original_idx, row in group.iterrows():
                    left = max(0, int(row['bb_left']))
                    top = max(0, int(row['bb_top']))
                    right = int(left + row['bb_width'])
                    bottom = int(top + row['bb_height'])
                    
                    # Prevent degenerate bounding boxes
                    right = max(left + 1, right)
                    bottom = max(top + 1, bottom)
                    
                    crop = image.crop((left, top, right, bottom))
                    crop = resize_op(crop)
                    
                    # Write the numpy array directly to the disk-backed memmap
                    mmap_writer[original_idx] = np.array(crop, dtype=np.uint8) # type: ignore

        # Flush changes to disk and close the writer
        mmap_writer.flush()
        del mmap_writer 

        # Save an index mapping directly in the DataFrame
        df['mmap_idx'] = df.index
        df.to_csv(self.df_cache_path, index=False)
        print("Cache built successfully!")

    def _prepare_sequences(self):
        """
        Groups the pre-processed rows into sequences of length 'seq_length'.
        Stores only the memory-map pointers (indices) to keep RAM usage low.
        """
        self.sequences = []
        for person_id, group in self.df.groupby('id'):
            group = group.sort_values('frame')
            
            for i in range(0, len(group) - self.seq_length + 1, self.seq_length):
                seq_chunk = group.iloc[i : i + self.seq_length]
                self.sequences.append({
                    'mmap_indices': seq_chunk['mmap_idx'].values,
                    'person_id': person_id
                })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Fetches the sequence directly from the disk-backed memmap array.
        """
        # Abre o memmap apenas quando o worker realmente for ler o primeiro dado
        if self.mmap_data is None:
            self.mmap_data = np.memmap(
                self.mmap_path,
                dtype=np.uint8,
                mode='r',
                shape=(self.total_crops, 128, 64, 3)
            )
        seq_info = self.sequences[idx]
        indices = seq_info['mmap_indices']
        person_id = seq_info['person_id']

        crops = []
        for mmap_idx in indices:
            # Slicing the memmap and copying to memory to make it writable
            raw_crop_np = self.mmap_data[mmap_idx].copy()
            
            # Apply ToTensor and Normalize
            crop_tensor = self.transform(raw_crop_np)
            crops.append(crop_tensor)

        # Stack into shape (seq_length, C, H, W)
        sequence_tensor = torch.stack(crops)
        
        return sequence_tensor, torch.tensor(person_id, dtype=torch.long)

def get_dataloader(
    train: bool,
    video_index: int,
    batch_size: int = 4,
    shuffle : bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Creates the dataloader for given video.
    
    Args:
        train (bool): If True, creates data loader for a training video
        video_index (int): Index (0 to 6) of video to load.
    
    Returns:
        DataLoader: The video dataloader.
    """
    # First run will take some time to build the .dat and .csv cache files.
    # Subsequent runs will load instantly.
    dataset = MOTSequenceDataset(
        train=train,
        video_index=video_index
    )
    print(f"Total sequences extracted: {len(dataset)}. Corresponds to video with approximately {len(dataset) * batch_size / 60} seconds.")

    # You can now safely increase num_workers without I/O blocking
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

######### testing
if __name__ == "__main__":
    video_index = 1 # second video

    my_dataloader = get_dataloader(True, video_index)

    for batch_idx, (image_sequences, person_ids) in enumerate(my_dataloader):
        print(f"Batch {batch_idx}:")
        print(f" - Image tensor shape: {image_sequences.shape}", '# (Batch, Time, Channels, Height, Width) ')
        print(f" - Person IDs in this batch: {person_ids}")
        break