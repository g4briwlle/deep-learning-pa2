import torch
from torchvision.io.image import decode_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

from pathlib import Path
from ..mot_reader import get_dataloader


class TorchDetector:

    def __init__(self) -> None:
        # Loading the model with the best available pre-trained weights
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights, box_score_thresh=0.9)
        self.model.eval()

    def run_inferece(self, image: torch.Tensor):
        self.img = image

        # Initializing the inference transforms and apply them
        preprocess = self.weights.transforms()
        batch = [preprocess(self.img)]

        # Making the prediction
        with torch.no_grad():
            self.prediction = self.model(batch)[0]

    def plot_predictions(self):
        # Filtering for the "person" class
        # The COCO dataset uses label 1 for "person"
        person_class_id = 1
        person_indices = self.prediction["labels"] == person_class_id

        person_boxes = self.prediction["boxes"][person_indices]
        person_scores = self.prediction["scores"][person_indices]

        # 6. Visualize the results
        # Create a list of labels for the detected persons
        labels = ["person" for _ in person_boxes]

        # Draw the bounding boxes on the image
        box = draw_bounding_boxes(
            self.img,
            boxes=person_boxes,
            labels=labels,
            colors="red",
            width=4,
            font_size=30
        )

        # Convert the tensor to a PIL image and display it
        im = to_pil_image(box.detach())
        im.show()
        
if __name__ == "__main__":
    # Testing with first frame of first video
    dataloader = get_dataloader(True, 0)
    
    image_sequence, person_ids = next(dataloader._get_iterator())
    first_image = image_sequence[0]
    
    torch_detector = TorchDetector()
    
    torch_detector.run_inferece(first_image)
    torch_detector.plot_predictions()