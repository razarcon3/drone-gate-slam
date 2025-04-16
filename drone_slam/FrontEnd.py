import numpy as np
from CornerDetector import CornerPredictions, CornerDetector
import torch
from dataclasses import dataclass
from typing import Dict, List
import copy
import time
from boxmot import ByteTrack
import cv2
import itertools # For efficient pair generation
import math

@dataclass
class CornerObservation:
    corner_id: int # corner id
    gate_id: int # gate id
    point_2d: np.ndarray # (x,y) point this landmark was seen
    depth: int
    tracking_id: int

def line_intersection(line1, line2):
    """
    Finds the intersection point of two lines given in the form (x1, y1, x2, y2).
    Returns (x, y) intersection point or None if lines are parallel or coincident.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Calculate determinant
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Check if lines are parallel (denominator is zero or very close to it)
    if abs(denominator) < 1e-6: # Use a small epsilon for floating point comparison
        return None

    # Calculate intersection point using Cramer's rule or similar method
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

    # Calculate intersection coordinates
    intersect_x = x1 + t * (x2 - x1)
    intersect_y = y1 + t * (y2 - y1)

    return (intersect_x, intersect_y)


def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=2):
    """
    Draws a bounding box on the given image.

    Args:
        image (numpy.ndarray): The image (in BGR format) to draw on.
                               This image will be modified in-place.
        bbox (tuple): A tuple containing the bounding box coordinates
                      in the format (x1, y1, x2, y2), where (x1, y1)
                      is the top-left corner and (x2, y2) is the
                      bottom-right corner.
        color (tuple): The color of the bounding box in BGR format.
                       Default is green (0, 255, 0).
        thickness (int): The thickness of the bounding box lines.
                         Default is 2. Use -1 or cv2.FILLED to fill.

    Returns:
        numpy.ndarray: The image with the bounding box drawn on it.
                       (Note: The input image object is modified directly).
    """
    # Extract coordinates, ensuring they are integers
    x1, y1, x2, y2 = map(int, bbox)

    # Define the top-left and bottom-right points
    pt1 = (x1, y1)
    pt2 = (x2, y2)

    # Draw the rectangle on the image
    # cv2.rectangle modifies the image in-place
    cv2.rectangle(image, pt1, pt2, color, thickness)


def find_target_points_in_bboxes(img, bboxes, rgb_img):
    """
    Finds the intersection of Hough lines closest to the center within each
    specified bounding box.

    Args:
        image_path (str): Path to the input image file.
        bboxes (list): A list of bounding box tuples, where each tuple is in
                       (x1, y1, x2, y2) format (top-left, bottom-right corner).

    Returns:
        list: A list of closest intersection points found. Each element
              corresponds to a bbox in the input list. Contains (x, y) tuples
              or None if no intersection was found for that bbox.
        np.ndarray: The original image with all processed bounding boxes and
                    found intersection points drawn on it. Returns None for the
                    image if image loading fails.
    """
    # output_img = img.copy() # Create a single copy for drawing all results
    output_img = rgb_img.copy()
    
    img_h, img_w = img.shape[:2]
    all_closest_points = [] # To store results for each bbox
    target_points = []
    
    # --- Iterate through each bounding box ---
    for i, bbox in enumerate(bboxes):
        # print(f"\n--- Processing Bbox {i+1}: {bbox} ---")
        closest_intersection_for_this_bbox = None
        min_distance_for_this_bbox = float('inf')
        # *** NEW: Set to store unique lines involved in ANY intersection in this bbox ***
        intersecting_lines_to_draw_this_bbox = set()

        # --- 2. Define Bbox (x1, y1, x2, y2) and Extract ROI ---
        # (Validation and ROI extraction logic remains the same as previous version)
        try:
            x1, y1, x2, y2 = map(int, bbox)
            bbox_h = y2 - y1
            bbox_w = x2 - x1
            
            x1 -= int(bbox_w * 0.1)
            x2 += int(bbox_w * 0.1)
            
            y1 -= int(bbox_h * 0.1)
            y2 += int(bbox_h * 0.1)
            
        except (ValueError, TypeError):
            print(f"Error: Invalid bbox format for {bbox}. Skipping.")
            all_closest_points.append(None)
            continue
        
        # remove any negative bboxes
        if any(v < 0 for v in (x1, y1, x2, y2)):
            all_closest_points.append(None)
            continue
        
        if x1 >= x2 or y1 >= y2:
            print(f"Error: Invalid bounding box coordinates (x1>=x2 or y1>=y2) in {bbox}. Skipping.")
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            all_closest_points.append(None)
            continue
        roi_offset_x, roi_offset_y = x1, y1
        x1_clamp = max(0, x1); y1_clamp = max(0, y1)
        x2_clamp = min(img_w, x2); y2_clamp = min(img_h, y2)
        if x1_clamp >= x2_clamp or y1_clamp >= y2_clamp:
             print(f"Error: Bounding box {bbox} is entirely outside image dimensions. Skipping.")
             cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
             all_closest_points.append(None)
             continue
        elif x1 != x1_clamp or y1 != y1_clamp or x2 != x2_clamp or y2 != y2_clamp:
             print(f"Warning: Bbox {bbox} partially outside image. Clamping ROI to ({x1_clamp},{y1_clamp},{x2_clamp},{y2_clamp}).")
             roi_offset_x, roi_offset_y = x1_clamp, y1_clamp
        roi = img[y1_clamp:y2_clamp, x1_clamp:x2_clamp]
        if roi.size == 0:
            print("Error: ROI extracted is empty. Skipping.")
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            all_closest_points.append(None)
            continue

        # Draw the (original) valid bbox outline
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green valid bbox

        # --- 3. Preprocessing within ROI ---
        # (Preprocessing logic remains the same)
        # gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # blurred_roi = cv2.GaussianBlur(gray_roi, (3, 3), sigmaX=1, sigmaY=1)
        median_blurred_roi = cv2.medianBlur(roi, 3) 
        gaussian_blurred_roi = cv2.GaussianBlur(median_blurred_roi, (3, 3), sigmaX=1, sigmaY=1)
        
        # threshold the image
        min_val = np.min(gaussian_blurred_roi)
        threshold = 1
        lower_bound = max(0, min_val - threshold) 
        upper_bound = min_val + threshold
        thresholded_roi = np.where((gaussian_blurred_roi >= lower_bound) & (gaussian_blurred_roi <= upper_bound), gaussian_blurred_roi, 0)
        thresholded_roi = np.where(thresholded_roi == 0, thresholded_roi, 255).astype(np.uint8)
        
        # apply canny
        edges_roi = cv2.Canny(thresholded_roi, 50, 150, apertureSize=3)
        
        # dilate canny
        dilation_kernel_size = (2, 2)
        dilation_iterations = 1 # How many times to apply dilation

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_kernel_size)
        # Visualization
        # MIN_DEPTH_METERS = 0
        # MAX_DEPTH_METERS = 100
        
        # gaussian_blurred_roi_vis = 255 - np.interp(gaussian_blurred_roi, (MIN_DEPTH_METERS, MAX_DEPTH_METERS), (0, 255)).astype(np.uint8)

        # cv2.imshow("thresholded_edges", thresholded_roi)
        # cv2.imshow("before thresholding", gaussian_blurred_roi_vis)
        # cv2.imshow("edges image", edges_roi)
        # cv2.waitKey(0)
        
        # --- 4. Hough Line Transform within ROI ---
        # (HoughLinesP call remains the same)
        height, width = y2 - y1, x2 - x1
        lines = cv2.HoughLinesP(edges_roi, rho=1, theta=np.pi / 180, threshold=10, minLineLength=min(int(height*0.1), int(width*0.1)), maxLineGap=20)

        if lines is None:
            print(f"No lines detected within bbox {i+1}.")
            all_closest_points.append(None)
            continue

        # --- 5. Find Intersections and Closest Point (within this bbox) ---
        bbox_center_x = (x1 + x2) / 2.0
        bbox_center_y = (y1 + y2) / 2.0
        intersections_found_in_bbox = False # Flag to track if any intersection occurred
        intersection_pts = []
        
        for line1_data, line2_data in itertools.combinations(lines, 2):
            x1a_roi, y1a_roi, x2a_roi, y2a_roi = line1_data[0]
            x1b_roi, y1b_roi, x2b_roi, y2b_roi = line2_data[0]

            line1_global = (x1a_roi + roi_offset_x, y1a_roi + roi_offset_y, x2a_roi + roi_offset_x, y2a_roi + roi_offset_y)
            line2_global = (x1b_roi + roi_offset_x, y1b_roi + roi_offset_y, x2b_roi + roi_offset_x, y2b_roi + roi_offset_y)

            intersection_pt = line_intersection(line1_global, line2_global)

            if intersection_pt:
                ix, iy = intersection_pt
                epsilon = 1e-6
                # Check if intersection point is strictly inside the *original* bbox
                if (x1 + epsilon <= ix < x2 - epsilon and
                    y1 + epsilon <= iy < y2 - epsilon):

                    intersections_found_in_bbox = True # Mark that at least one intersection was found

                    # *** Add BOTH lines to the set for drawing ***
                    # Convert to tuple of ints for hashing/set storage
                    line1_tuple = tuple(map(int, line1_global))
                    line2_tuple = tuple(map(int, line2_global))
                    intersecting_lines_to_draw_this_bbox.add(line1_tuple)
                    intersecting_lines_to_draw_this_bbox.add(line2_tuple)

                    # --- Still track the closest intersection point ---
                    intersection_pts.append([ix, iy])
                    distance = math.hypot(ix - bbox_center_x, iy - bbox_center_y)
                    if distance < min_distance_for_this_bbox:
                        min_distance_for_this_bbox = distance
                        closest_intersection_for_this_bbox = np.array([ix, iy]).astype(np.uint32)

        # --- Store result (closest point) for this bbox ---
        all_closest_points.append(closest_intersection_for_this_bbox)

        # --- Draw results for this bbox ---
        # *** Draw ALL unique lines involved in ANY intersection within this bbox ***
        if intersecting_lines_to_draw_this_bbox:
        #     print(f"Bbox {i+1}: Found {len(intersecting_lines_to_draw_this_bbox)} unique line segments involved in intersections.")
            line_color = (255, 0, 0) # Blue for intersecting lines (BGR)
            for line_segment in intersecting_lines_to_draw_this_bbox:
                lx1, ly1, lx2, ly2 = line_segment
                cv2.line(output_img, (lx1, ly1), (lx2, ly2), line_color, 1)
        # # else: # This case can happen if lines were detected but none intersected within the box
        # #    print(f"Bbox {i+1}: Lines detected but no intersections found within the box.")

        
        
        # if len(intersection_pts) == 4:
        #     target_pt = np.mean(np.array(intersection_pts), axis=0).astype(np.uint32)
        #     target_points.append(target_pt)
            
        #     cv2.circle(output_img, (target_pt[0], target_pt[1]), 2, (0, 255, 0), -1) # Red filled circle for closest point
        #     for pt in intersection_pts:
        #         cv2.circle(output_img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1) # Red filled circle for closest point
        # else:
        #     target_points.append(None)
            
        #     # Visualization

        # # *** Draw the closest intersection point (if any intersection occurred) ***
        if closest_intersection_for_this_bbox is not None:
            # We already know an intersection happened if this is not None
            # print(f"Closest intersection for bbox {i+1} found at: {closest_intersection_for_this_bbox}")
            ix_int, iy_int = map(int, map(round, closest_intersection_for_this_bbox))
            cv2.circle(output_img, (int(bbox_center_x), int(bbox_center_y)), 2, (255, 0, 0), -1)
            cv2.circle(output_img, (ix_int, iy_int), 2, (0, 0, 255), -1) # Red filled circle for closest point
        # elif intersections_found_in_bbox:
        #      # This case should be rare: intersections occurred, but calculation failed for closest?
        #      print(f"Warning: Intersections found in bbox {i+1}, but failed to determine closest point.")
        # elif not intersecting_lines_to_draw_this_bbox and lines is not None:
        #      print(f"Bbox {i+1}: Lines detected but no intersections found strictly within the box.")
        # If lines was None initially, that case was handled earlier.

        # --- End of loop for one bbox ---

        # find weighted centroid of lines
    return all_closest_points, output_img

def put_text_above_point(image, text, point, font_scale=0.4, color=(255, 255, 255), thickness=1, vertical_offset=10):
    """
    Draws text slightly above a specified point, horizontally centered.

    Args:
        image: The OpenCV image (numpy array) to draw on.
        text: The string to display.
        point: A tuple (x, y) representing the coordinates of the point.
        font_scale: Font size multiplier (e.g., 0.4 for small text).
        color: Text color in BGR format (e.g., (0, 255, 0) for green).
        thickness: Text line thickness.
        vertical_offset: How many pixels above the point's y-coordinate the
                         bottom of the text should start.
    """
    px, py = point
    fontFace = cv2.FONT_HERSHEY_SIMPLEX  # Choose a font
    lineType = cv2.LINE_AA             # Anti-aliased line type

    # 1. Get the size of the text box
    (text_width, text_height), baseline = cv2.getTextSize(text, fontFace, font_scale, thickness)

    # 2. Calculate the bottom-left corner (org) of the text
    # Center the text horizontally over the point (px)
    org_x = px - text_width // 2
    # Place the text baseline 'vertical_offset' pixels above the point (py)
    org_y = py - vertical_offset

    # Ensure coordinates are integers (required by putText)
    org = (int(org_x), int(org_y))

    # 3. Put the text on the image
    cv2.putText(image, text, org, fontFace, font_scale, color, thickness, lineType)

def convert_corner_prediction_to_detarray(corner_pred: CornerPredictions) -> np.ndarray:
    detections = []
    for bbox, score in zip(corner_pred.boxes, corner_pred.scores):
        detection = []
        detection += bbox
        detection.append(score)
        detection.append(0) # for the class 0
        detections.append(detection)
    return np.array(detections)
class FrontEnd:
    def __init__(self):
        self.curr_keyframe_id: int = 0
        
        self.curr_corners_tracked: List[CornerObservation] = [] # list containing corner landmarks being tracked        
        self.prev_rgb_image: np.ndarray = None
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"FrontEnd using device {self.device}")
        
        self.gate_idx = 0
        
        fx = 472.0
        fy = 472.0
        cx = 472.0
        cy = 240.0
        self.intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,   1]
        ],dtype=np.float32)
        
        # Load tracking and detection models
        self.corner_detector: CornerDetector = CornerDetector(device=self.device)
        self._initialize_tracker()

    def backproject(self, point: np.ndarray, depth) -> np.ndarray:
        homogenous_point = np.array([[point[0]], [point[1]], [1]], dtype=np.float32)
        direction = np.linalg.inv(self.intrinsic) @ homogenous_point
        
        return ((direction / np.linalg.norm(direction)) * depth).squeeze()
        
    
    
    def _initialize_tracker(self):
        self.tracker: ByteTrack = ByteTrack(
            min_conf=0.7,
            track_thresh=0.7,
            track_buffer=5,
            frame_rate=20,
            per_class=False
        )
    
    def _get_lost_tracks(self):
        outputs = []
        for t in self.tracker.lost_stracks:
            output = []
            output.extend(t.xyxy)
            output.append(t.id)
            output.append(t.conf)
            output.append(t.cls)
            output.append(t.det_ind)
            outputs.append(output)
        outputs = np.asarray(outputs)
        return outputs
    
    def extract_bbox_centroid(bbox):
        x1, y1, x2, y2 = [round(coord) for coord in bbox]

        return ((x1 + x2) // 2 , ((y1 + y2) // 2))
    
    def process_image(self, curr_rgb_image: np.ndarray, curr_depth_image: np.ndarray):     
        
        start_time = time.time()       
        # run cotracking if possible
        # if self.prev_rgb_image is not None and len(self.curr_corners_tracked) != 0:
        #     self._update_curr_corners_tracked(curr_rgb_image)
        visualization_img = np.copy(curr_rgb_image)
        # get current corner prediction from the image
        corner_predictions: CornerPredictions = self.corner_detector.createPredictions(curr_rgb_image)
        # must add a new gate/set of corners to track
        if len(self.curr_corners_tracked) == 0 and len(corner_predictions.boxes) >= 4:
            # extract refined 2D points from corners and backproject to 3D                
            # detection_img = self.corner_detector.getCurrentVisualization()
            # cv2.imshow("detection_img", detection_img)
            target_points_2d, hough_viz_img = find_target_points_in_bboxes(curr_depth_image, corner_predictions.boxes, curr_rgb_image)
            
            # prune predictions based on whether they have a target point or not
            has_target_point = [False if tp is None else True for tp in target_points_2d]
            corner_predictions.boxes = [corner_predictions.boxes[i] for i in range(len(corner_predictions.boxes)) if has_target_point[i] is not None]
            corner_predictions.scores = [corner_predictions.scores[i] for i in range(len(corner_predictions.scores)) if has_target_point[i] is not None]
            
            # filter target points and create 3D points
            target_points_2d = [tp for tp in target_points_2d if tp is not None]
            depths = np.array([curr_depth_image[tuple(pt[::-1])] for pt in target_points_2d])
            target_points_3d = np.array([self.backproject(target_point_2d, depth) for target_point_2d, depth in zip(target_points_2d, depths)])
                        
            if len(target_points_2d) >= 4:
                # take the closest 4 points if we have more detections
                if len(target_points_2d) > 4:                        
                    closest_indices = np.argsort(depths)[:4]
                    corner_predictions.boxes = corner_predictions.boxes[closest_indices]
                    corner_predictions.scores = corner_predictions[closest_indices]
                    depths = depths[closest_indices]
                    target_points_3d = target_points_3d[closest_indices]
                    target_points_2d = target_points_2d[closest_indices]
                    
                target_points_2d = np.array(target_points_2d)
                # geometric verification and registration
                point0 = target_points_3d[0]
                
                distances = np.linalg.norm(target_points_3d[-3:] - point0, axis=1)
                tolerance = 0.1
                measured_edge_dist = np.min(distances)
                measured_hypot_dist = np.max(distances)
                
                adjacent_corner_indices = np.array((distances >= measured_edge_dist - tolerance) & (distances <= measured_edge_dist + tolerance))
                adjacent_corner_dists = distances[adjacent_corner_indices]
                
                if adjacent_corner_dists.shape[0] == 2:
                    measured_edge_dist = np.mean(adjacent_corner_dists, axis=0)
                    hypot_check = (measured_hypot_dist >= measured_edge_dist * np.sqrt(2) - tolerance) & (measured_hypot_dist <= measured_edge_dist * np.sqrt(2) + tolerance)
                    
                    if hypot_check:
                        # reorder the corner predictions and depths as necessary for registering new gate
                        adjacent_corner_boxes = (np.array(corner_predictions.boxes[-3:])[adjacent_corner_indices]).tolist()
                        adjacent_corner_scores = (np.array(corner_predictions.scores[-3:])[adjacent_corner_indices]).tolist()
                        adjacent_corner_depths = depths[-3:][adjacent_corner_indices]
                        
                        hypotenuse_box = np.array(corner_predictions.boxes[-3:])[~adjacent_corner_indices].squeeze().tolist()
                        hypotenuse_score = np.array(corner_predictions.scores[-3:])[~adjacent_corner_indices].squeeze()
                        hypotenuse_depth = depths[-3:][~adjacent_corner_indices].squeeze()
                        
                        
                        corner_predictions.boxes[-3:] = [adjacent_corner_boxes[0], hypotenuse_box, adjacent_corner_boxes[1]]
                        corner_predictions.scores[-3:] = [adjacent_corner_scores[0], hypotenuse_score, adjacent_corner_scores[1]]
                        depths[-3:] = [adjacent_corner_depths[0], hypotenuse_depth, adjacent_corner_depths[1]]
                        
                        
                        
                        # create a new gate and add it's corners
                        verified_corners_2d = target_points_2d[-3:][adjacent_corner_indices]
                        hypot_corner_2d = target_points_2d[-3:][~adjacent_corner_indices].squeeze()
                        
                        detection_array = convert_corner_prediction_to_detarray(corner_predictions)
                        
                        self._initialize_tracker()
                        tracking_results = self.tracker.update(detection_array, curr_rgb_image)
                        track_ids = tracking_results[:,4]
                        
                        self.curr_corners_tracked.append(CornerObservation(0, self.gate_idx,    target_points_2d[0],                depths[0], track_ids[0]))
                        self.curr_corners_tracked.append(CornerObservation(1, self.gate_idx, verified_corners_2d[0],                depths[1], track_ids[1]))
                        self.curr_corners_tracked.append(CornerObservation(2, self.gate_idx,        hypot_corner_2d,                depths[2], track_ids[2]))
                        self.curr_corners_tracked.append(CornerObservation(3, self.gate_idx, verified_corners_2d[1],                depths[3], track_ids[3]))
                        
                        
                        self.gate_idx += 1
                    
        
                # print(f"Hough Processing Time: {(end_time - start_time) * 1000}")
                # cv2.imshow("hough_viz_img", hough_viz_img)
                # cv2.waitKey(1)
                # with remaining detections check for gates (4 bboxes) and add their points to the current corners being tracked
        elif len(self.curr_corners_tracked) > 0:
            # track the current corners
            detection_array = convert_corner_prediction_to_detarray(corner_predictions)
            tracking_results = self.tracker.update(detection_array, curr_rgb_image)
            lost_tracks = self._get_lost_tracks().tolist()
            
            tracking_results = np.array(tracking_results.tolist() + lost_tracks)
                        
            new_curr_corners_tracked = []
            for tracked_corner in self.curr_corners_tracked:
                for tracking_result in tracking_results:
                    if tracking_result[4] == tracked_corner.tracking_id:
                        resulting_bbox = tracking_result[:4]
                        target_points_2d, hough_viz_img = find_target_points_in_bboxes(curr_depth_image, [resulting_bbox.astype(np.int32)], visualization_img)
                        visualization_img = hough_viz_img
                        
                        potential_target_point = target_points_2d[0]
                        if potential_target_point is not None:
                            tracked_corner.depth = curr_depth_image[tuple(potential_target_point[::-1])]
                            tracked_corner.point_2d = potential_target_point
                        else:
                            tracked_corner.depth = None
                            tracked_corner.point_2d = None
                            
                        # find a target point
                        new_curr_corners_tracked.append(tracked_corner)
            
            self.curr_corners_tracked = new_curr_corners_tracked
            
        # cv2.imshow("visualization image", visualization_img)
            # cv2.waitKey(0)
        end_time = time.time()
        
        print(f"Total Computation Time : {(end_time - start_time)*1000}ms\n")
        visualization_img2 = np.copy(curr_rgb_image)
        self.tracker.plot_results(visualization_img2, show_trajectories=False)

        detection_img = self.corner_detector.getCurrentVisualization()
        # cv2.imshow("detection_img", detection_img)

        for corner_obs in self.curr_corners_tracked:
            if corner_obs.depth is not None:
                put_text_above_point(visualization_img2, text=str(corner_obs.gate_id), point=corner_obs.point_2d.astype(np.uint32))
                cv2.circle(visualization_img2, corner_obs.point_2d.astype(np.uint32), 4, (255,0,0), -1)

        # cv2.imshow("vis image2", visualization_img2)
        # cv2.waitKey(1)
        # construct and return keyframe info with current tracked points 
        self.prev_rgb_image = curr_rgb_image


    def getCornerObservations(self):
        return self.curr_corners_tracked