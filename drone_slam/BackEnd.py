from FrontEnd import CornerObservation
import gtsam
import numpy as np
from typing import List

CORNERS_PER_GATE = 4

def get_corner_key(gate_idx: int, corner_idx: int) -> gtsam.Symbol:
        global_corner_idx = gate_idx * CORNERS_PER_GATE + corner_idx
        return gtsam.symbol('c', global_corner_idx)

class BackEnd:
    def __init__(self):
        self.curr_pose = gtsam.Pose3()
        
        self.initial_estimate = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()
        
        self.landmark_name_to_key = {}
        
        self.pose_index = 0
        self.prev_pose_key = gtsam.symbol('x', 0)
        
        # Noise Models
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]))
        self.landmark_noise = gtsam.noiseModel.Isotropic.Sigma(dim=2, sigma=2.0)
        
        fx = 472.0
        fy = 472.0
        s  = 0
        cx = 472.0
        cy = 240.0
        self.calibration = gtsam.Cal3_S2(fx, fy, s, cx, cy)
        
    def process_key_frame(self, new_vio_pose: gtsam.Pose3, corner_observations: List[CornerObservation]):    
        curr_pose_key = gtsam.symbol('x', self.pose_index)
        if self.pose_index == 0:
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas([1e-6] * 6)
            self.graph.add(gtsam.PriorFactorPose3(curr_pose_key, self.curr_pose, prior_noise))
        else:
            relative_pose = self.curr_pose.between(new_vio_pose)
            self.graph.add(gtsam.BetweenFactorPose3(self.prev_pose_key, curr_pose_key, relative_pose, self.odometry_noise))
            
        self.pose_index += 1
        self.curr_pose = new_vio_pose        
        self.initial_estimate.insert(curr_pose_key, new_vio_pose)
        self.prev_pose_key = curr_pose_key


        
        # convert gtsam to ros (R @ gtsam_coord = ros_coord)
        gtsam_ros_rot = gtsam.Rot3(np.array([
            [0,0,1],
            [-1,0,0],
            [0,-1,0]
        ]))
        
        T_gtsam_ros = gtsam.Pose3(gtsam_ros_rot, np.array([0,0,0]))
        T_gtsam_world = self.curr_pose.compose(T_gtsam_ros)
        
        est_camera = gtsam.PinholeCameraCal3_S2(T_gtsam_world, self.calibration)
        # add landmarks
        for co in corner_observations:
            if co.point_2d is None:
                continue
            co_key = get_corner_key(co.gate_id, co.corner_id)
            
            if co.gate_id not in self.landmark_name_to_key.keys():
                self.landmark_name_to_key[co.gate_id] = {}
            if co.corner_id not in self.landmark_name_to_key[co.gate_id].keys():
                self.landmark_name_to_key[co.gate_id][co.corner_id] = co_key
                co_3d_guess = est_camera.backproject(co.point_2d, co.depth)
                self.initial_estimate.insert(co_key, co_3d_guess)
            
            self.graph.add(
                gtsam.GenericProjectionFactorCal3_S2(
                    measured=np.expand_dims(co.point_2d, axis=1),
                    noiseModel=self.landmark_noise,
                    poseKey=curr_pose_key,
                    pointKey=co_key,
                    k=self.calibration,
                    body_P_sensor=T_gtsam_ros
                )
            )

    def solveDogleg(self):
        params = gtsam.DoglegParams()
        params.setMaxIterations(10000)
        optimizer = gtsam.DoglegOptimizer(self.graph, self.initial_estimate, params)

        return optimizer.optimize()