import numpy as np
from CornerDetector import CornerPrediction, CornerDetector
import torch

class FrontEnd:
    def __init__(self):

    
        self.curr_keyframe_id = 0
        
        # cotracker testing counter
        self.tracking = False
        self.curr_track_queries = []
        
        self.prev_rgb_image = None
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        print(f"FrontEnd using device {self.device}")
        
        # Load tracking and detection models
        self.cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(self.device)
        self.corner_detector = CornerDetector(device=self.device)

    def extract_bbox_centroid(bbox):
        x1, y1, x2, y2 = [round(coord) for coord in bbox]

        return ((x1 + x2) // 2 , ((y1 + y2) // 2))
    
    def process_image(self, curr_rgb_image: np.ndarray):
        corner_prediction: CornerPrediction = self.corner_detector.createPrediction(curr_rgb_image)
        
        # initiate tracking
        if self.tracking == False:
            for bbox in corner_prediction.boxes:
                centroid = FrontEnd.extract_bbox_centroid(bbox)
                self.curr_track_queries.append([0., float(centroid[0]), float(centroid[1])])
            self.tracking = True
            
        # run cotracking
        if self.prev_rgb_image is not None and len(self.curr_track_queries) != 0:
            queries = torch.tensor(self.curr_track_queries)
            if self.device.type == "cuda":
                queries = queries.cuda()
            
            stacked_frames = np.stack([self.prev_rgb_image, curr_rgb_image])
            video_chunk = (
                torch.tensor(
                    stacked_frames, device=self.device # Shape becomes (2, H, W, C)
                )
                .float()                                  # Shape (2, H, W, C), dtype float32
                .permute(0, 3, 1, 2)                      # Shape becomes (2, C, H, W)
                [None]                                    # Shape becomes (1, 2, C, H, W)
            )

            pred_tracks, pred_visibility = self.cotracker(video_chunk ,queries=queries[None])
            print(pred_tracks.shape)
            print(pred_tracks)
        
        
        self.prev_rgb_image = curr_rgb_image