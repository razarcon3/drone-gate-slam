import numpy as np
from CornerDetector import CornerPrediction, CornerDetector
import torch
from dataclasses import dataclass
from typing import Dict, List
import copy
from k_means_constrained import KMeansConstrained
import time

@dataclass
class CornerObservation:
    corner_id: int # corner id
    gate_id: int # gate id
    point_2d: List # (x,y) point this landmark was seen

class FrontEnd:
    def __init__(self):
        self.curr_keyframe_id: int = 0
        
        self.curr_corners_tracked: List[CornerObservation] = [] # list containing corner landmarks being tracked        
        self.prev_rgb_image: np.ndarray = None
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        print(f"FrontEnd using device {self.device}")
        
        # Load tracking and detection models
        self.cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(self.device)
        self.corner_detector: CornerDetector = CornerDetector(device=self.device)

    def extract_bbox_centroid(bbox):
        x1, y1, x2, y2 = [round(coord) for coord in bbox]

        return ((x1 + x2) // 2 , ((y1 + y2) // 2))
    
    def _prune_corner_prediction(self, corner_prediction: CornerPrediction):
        if (corner_prediction.boxes):
            return
        # Convert points to NumPy array for easier comparison if not already
        points_arr = np.asarray([corner_observation.point_2d for corner_observation in self.curr_corners_tracked])

        # True/False array indicating which bboxes in the corner_prediction do not contain any tracked points
        # True if bbox doesn't contain any tracked points
        # False if bbox does contain a tracked point
        results = []
        for bbox in corner_prediction.boxes:
            xmin, ymin, xmax, ymax = bbox
            contains_no_tracked_points = True

            # Check each point efficiently
            for point in points_arr:
                px, py = point
                # Check if the point's coordinates are within the bbox boundaries (inclusive)
                if xmin <= px <= xmax and ymin <= py <= ymax:
                    contains_no_tracked_points = False
                    break # Found a point in this bbox, no need to check other points

            results.append(contains_no_tracked_points)

        return results
        
    def _update_curr_corners_tracked(self, curr_rgb_image: np.ndarray):
        queries = torch.tensor([[0. , float(corner.point_2d[0]), float(corner.point_2d[1])] for corner in self.curr_corners_tracked])
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
    
        # update visible corners with new track points
        visibility_mask = pred_visibility[0][-1][:].tolist()
        pred_tracks = pred_tracks[0][-1][:].tolist()
        
        updated_curr_corners_tracked: List[CornerObservation] = []
        for i in range(len(visibility_mask)):
            if visibility_mask[i]:
                updated_corner = copy.deepcopy(self.curr_corners_tracked[i])
                updated_curr_corners_tracked.append(updated_corner)
                updated_corner.keyframe_id = self.curr_keyframe_id
                updated_corner.point_2d = pred_tracks[i]
                
        self.curr_corners_tracked = updated_curr_corners_tracked
    
    def process_image(self, curr_rgb_image: np.ndarray, curr_depth_image: np.ndarray):            
        # run cotracking if possible
        if self.prev_rgb_image is not None and len(self.curr_corners_tracked) != 0:
            self._update_curr_corners_tracked(curr_rgb_image)
        
        # get current corner prediction from the image
        corner_prediction: CornerPrediction = self.corner_detector.createPrediction(curr_rgb_image)
        
        # add new corner observations from the predictions if new gate(s) seen
        if len(corner_prediction.boxes) >= 4:
            # prune detections (remove bboxes that contain current tracks)
            self._prune_corner_prediction(corner_prediction)
            
            if len(corner_prediction.boxes) >= 4:
                return
                # with remaining detections check for gates (4 bboxes) and add their points to the current corners being tracked

        
        # construct and return keyframe info with current tracked points 
        self.prev_rgb_image = curr_rgb_image


    def getKeyFrame(self):
        pass