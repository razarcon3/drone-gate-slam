import numpy as np



class FrontEnd:
    MODES = {"Online", "Offline"}
    
    def __init__(self, mode="Online"):
        # Configure the mode (Offline or Online)
        if mode not in self.ALLOWED_OPTIONS:
            raise ValueError(f"Invalid mode requested: {mode}. Must be one of: {self.MODES}")
        self.mode = mode
        
        if mode == "Offline":
            self.prev_GT_pose = None # used for generating fake relative odometry from "VIO"
    
        self.keyframe_id = 0
    
    # Used for initializing the ground truth pose for fake odom measurements
    def initialize_GT_pose(self, init_GT_pose):
        self.prev_GT_pose = init_GT_pose
    
    def process_measurements(self, image: np.ndarray, ):
        
        if self.mode == "Offline":
            # Every incoming image initiates a new keyframe
            