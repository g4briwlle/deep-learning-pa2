import torch
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, to_tensor
from pathlib import Path
import zipfile
import io
from PIL import Image

# Import the ZIP_PATH from your config
from src.mot_reader import ZIP_PATH

class TorchDetector:

    def __init__(self) -> None:
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        # Threshold de 0.75 para filtrar as detecções falsas de fundo
        self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights, box_score_thresh=0.75)
        self.model.eval()

    def run_inference(self, image_tensor: torch.Tensor):
        # image_tensor has shape [C, H, W] in [0.0, 1.0] range
        self.img = image_tensor

        # Re-apply the official torchvision transforms for the detector
        preprocess = self.weights.transforms()
        batch = [preprocess(self.img)]

        with torch.no_grad():
            self.prediction = self.model(batch)[0]

    def plot_predictions(self, save_path: Path = None):
        person_class_id = 1
        person_indices = self.prediction["labels"] == person_class_id

        person_boxes = self.prediction["boxes"][person_indices]
        person_scores = self.prediction["scores"][person_indices]

        labels = [f"person {score:.2f}" for score in person_scores]
        
        # Convert from float [0.0, 1.0] to uint8 [0, 255] for drawing
        img_uint8 = (self.img * 255).clamp(0, 255).to(torch.uint8)

        box = draw_bounding_boxes(
            img_uint8,
            boxes=person_boxes,
            labels=labels,
            colors="red",
            width=3, # Thicker line for high-res images
            font_size=12
        )

        im = to_pil_image(box.detach())
        
        plt.figure(figsize=(16, 9)) # Widescreen ratio for MOT frames
        plt.imshow(im)
        plt.axis("off")
        
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot successfully saved to: {save_path}")
            
        plt.show()
        
if __name__ == "__main__":
    # Para testar o detector da Parte 1, precisamos do quadro completo (Full Scene),
    # e não do recorte de 128x64 da Parte 2.
    seq_name = "MOT17/train/MOT17-02-FRCNN"
    img_internal_path = f"{seq_name}/img1/000001.jpg"
    
    print(f"Extraindo quadro original do ZIP: {img_internal_path}")
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        with z.open(img_internal_path) as f:
            img_bytes = f.read()
            raw_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
    # Converte PIL Image para Tensor PyTorch
    full_frame_tensor = to_tensor(raw_image)
    
    torch_detector = TorchDetector()
    torch_detector.run_inference(full_frame_tensor)
    
    # Define o arquivo de saída
    out_path = Path("outputs/part1/torch_detector_full_frame.png")
    torch_detector.plot_predictions(save_path=out_path)