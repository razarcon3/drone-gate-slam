import time
import torch
import cv2
import numpy as np
from torchvision import transforms
from dataclasses import dataclass
from typing import List
import pathlib

@dataclass
class CornerPredictions:
    # box format is [x1, y1, x2, y2]
    boxes: List
    scores: List

class CornerDetector():
    MODELS = {"FasterRCNN", "YoloV11"}
    
    def __init__(self, device, model="FasterRCNN"):
        if model not in CornerDetector.MODELS:
            raise ValueError(f"Invalid model selected. Allowed model types are: {', '.join(CornerDetector.MODELS)}")
        
        self.device = device
        
        try:
            current_file_path = pathlib.Path(__file__).resolve()
            model_path = current_file_path.parent.parent / "resource/frcnn_full_model.pth"
            if model == "FasterRCNN":
                self.model = torch.load(model_path, weights_only=False)
        except Exception as e:
            print(f"Could not load model due to {e}")
        
        
        self.model.eval()
        self.model.to(self.device)
        
        self.current_predictions: CornerPredictions = None
        self.current_img_rgb: np.ndarray = None
    
    @torch.no_grad()
    def createPredictions(self, image_rgb: np.ndarray):
        img_tensor = transforms.ToTensor()(image_rgb).to(self.device)
        
        start = time.time()
        predictions = self.model([img_tensor])
        end = time.time()
        
        print(f"Inference Time: {(end - start) * 1000} ms")
        
        # Move results to CPU and convert to lists for visualization
        predictions = [{k: v.cpu().numpy() for k, v in p.items()} for p in predictions]
        
        pred_boxes = predictions[0]['boxes'].tolist()
        pred_scores = predictions[0]['scores'].tolist()
        
        # apply non-maximum suppression to avoid several boxes for one gate
        selected_indices = cv2.dnn.NMSBoxes(pred_boxes, pred_scores, score_threshold=0.8, nms_threshold=0.7)
        filtered_pred_boxes = []
        filtered_pred_scores = []

        if len(selected_indices) > 0:  # check that selected indices is not empty
            for idx in selected_indices:
                filtered_pred_boxes.append(pred_boxes[idx])
                filtered_pred_scores.append(pred_scores[idx])
        
        filtered_pred_boxes = [[int(coord) for coord in box] for box in filtered_pred_boxes]
        
        self.current_img_rgb = image_rgb
        self.current_predictions = CornerPredictions(filtered_pred_boxes, filtered_pred_scores)
        return CornerPredictions(filtered_pred_boxes, filtered_pred_scores)

    def getCurrentVisualization(self) -> np.ndarray:
        img_bgr = cv2.cvtColor(self.current_img_rgb, cv2.COLOR_RGB2BGR)
        
        if self.current_predictions.boxes is not None and len(self.current_predictions.boxes) != 0:
            for i, box in enumerate(self.current_predictions.boxes):
                x1, y1, x2, y2 = [round(coord) for coord in box]
                
                cv2.circle(img_bgr, ((x1 + x2) // 2, (y1 + y2) // 2), 3, (0, 0, 255))
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red
                if self.current_predictions.scores is not None and len(self.current_predictions.scores) != 0:
                    cv2.putText(img_bgr, f"{self.current_predictions.scores[i]:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Red text
        return img_bgr