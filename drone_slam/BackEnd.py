from FrontEnd import CornerObservation
import gtsam
import numpy as np
from typing import List
from dataclasses import dataclass, field

CORNERS_PER_GATE = 4

@dataclass
class LandmarkTriangulation:
    gtsam_key: gtsam.symbol
    cameras: List = field(default_factory=list)
    points_2d: List = field(default_factory=list)


def get_corner_key(gate_idx: int, corner_idx: int) -> gtsam.Symbol:
        global_corner_idx = gate_idx * CORNERS_PER_GATE + corner_idx
        return gtsam.symbol('c', global_corner_idx)

class BackEnd:
    def __init__(self):
        self.curr_pose = gtsam.Pose3()
        
        self.initial_estimate = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()
        
        # Noise Models
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
        self.landmark_projection_noise = gtsam.noiseModel.Isotropic.Sigma(dim=2, sigma=5.0)
        self.landmark_ranging_noise = gtsam.noiseModel.Isotropic.Sigma(1, 0.1)
        self.gate_edge_range_noise = gtsam.noiseModel.Isotropic.Sigma(1, 1e-6)
        self.fixed_prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas([1e-9] * 6)
        
        # Variable name to key mappers
        self.landmark_name_to_key = {}
        self.known_object_name_to_key = {}
        
        self.pose_index = 0
        self.prev_pose_key = gtsam.symbol('x', 0)
        
        self.known_object_positions = {
            "Cylinder7": [35.7, -0.7, 1.99]
        }
        
        # Add priors for any known ground truth positions:
        for key_id, gt_obj_name in enumerate(self.known_object_positions.keys()):
            gt_obj_key = gtsam.symbol("g", key_id)
            gt_obj_trans = self.known_object_positions[gt_obj_name]
            gt_obj_pose = gtsam.Pose3(
                gtsam.Rot3(np.identity(3)),
                np.array(gt_obj_trans)
            )
            
            self.known_object_name_to_key[gt_obj_name] = gt_obj_key
            
            # Insert initial estimate and add a constraint to the first state
            self.initial_estimate.insert(gt_obj_key, gt_obj_pose)
            self.graph.add(gtsam.BetweenFactorPose3(gtsam.symbol("x", 0), gt_obj_key, gt_obj_pose, self.fixed_prior_pose_noise))
        
        fx = 427.0
        fy = 427.0
        s  = 0
        cx = 427.0
        cy = 240.0
        self.calibration = gtsam.Cal3_S2(fx, fy, s, cx, cy)
        
    def process_key_frame(self, new_vio_pose: gtsam.Pose3, corner_observations: List[CornerObservation], known_object_detection = []):    
        curr_pose_key = gtsam.symbol('x', self.pose_index)
        if self.pose_index == 0:
            self.initial_estimate.insert(curr_pose_key, self.curr_pose)
            self.graph.add(gtsam.PriorFactorPose3(curr_pose_key, self.curr_pose, self.fixed_prior_pose_noise))
            self.pose_index += 1
            return
        else:
            # Add motion constraint from VIO odometry measurement
            relative_pose = self.curr_pose.between(new_vio_pose)
            self.graph.add(gtsam.BetweenFactorPose3(self.prev_pose_key, curr_pose_key, relative_pose, self.odometry_noise))
            
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
        
        
        # add any detected known landmarks:
        for kod in known_object_detection:
            kod_key = self.known_object_name_to_key[kod["obj_name"]]
            self.graph.add(
                gtsam.GenericProjectionFactorCal3_S2(
                    measured=np.expand_dims(kod["point_2d"], axis=1),
                    noiseModel=self.landmark_projection_noise,
                    poseKey=curr_pose_key,
                    pointKey=kod_key,
                    k=self.calibration,
                    body_P_sensor=T_gtsam_ros
                )
            )
            self.graph.add(gtsam.RangeFactorPose3(curr_pose_key, kod_key, kod["ranging_distance"], self.fixed_prior_pose_noise))
            
        
        # add detected landmarks
        for co in corner_observations:
            if co.point_2d is None:
                continue
            co_key = get_corner_key(co.gate_id, co.corner_id)
            
            if co.gate_id not in self.landmark_name_to_key.keys():
                self.landmark_name_to_key[co.gate_id] = {}
            if co.corner_id not in self.landmark_name_to_key[co.gate_id].keys():
                self.landmark_name_to_key[co.gate_id][co.corner_id] = LandmarkTriangulation(co_key)
                # initialize on first observation
                # co_3d_guess = est_camera.backproject(co.point_2d, co.depth)
                # self.initial_estimate.insert(co_key, co_3d_guess)
            
            # add information to triangulate + initialize later
            self.landmark_name_to_key[co.gate_id][co.corner_id].cameras.append(est_camera)
            self.landmark_name_to_key[co.gate_id][co.corner_id].points_2d.append(co.point_2d)
            
            self.graph.add(
                gtsam.GenericProjectionFactorCal3_S2(
                    measured=np.expand_dims(co.point_2d, axis=1),
                    noiseModel=self.landmark_projection_noise,
                    poseKey=curr_pose_key,
                    pointKey=co_key,
                    k=self.calibration,
                    body_P_sensor=T_gtsam_ros
                )
            )
            
            self.graph.add(
                gtsam.RangeFactor3D(
                    curr_pose_key,
                    co_key,
                    co.depth,
                    self.landmark_ranging_noise
                )
            )
            
        self.pose_index += 1

    def solveDogleg(self):
        # triangulate the corners for initialization
        for gate_dict in self.landmark_name_to_key.values():
            for landmark_triangulation in gate_dict.values():
                corner_estimate_3d = gtsam.triangulatePoint3(gtsam.CameraSetCal3_S2(landmark_triangulation.cameras), gtsam.Point2Vector(landmark_triangulation.points_2d), rank_tol=1e-9, optimize=True)
                self.initial_estimate.insert(landmark_triangulation.gtsam_key, corner_estimate_3d)

        # add between factors for the corners of each gate
        for gate_dict in self.landmark_name_to_key.values():
            for i in range(4):
                self.graph.add(gtsam.RangeFactor3(gate_dict[i].gtsam_key, gate_dict[(i+1)%4].gtsam_key, 2.4, self.gate_edge_range_noise))
        
        
        params = gtsam.DoglegParams()
        params.setMaxIterations(10000)
        optimizer = gtsam.DoglegOptimizer(self.graph, self.initial_estimate, params)
        results = optimizer.optimize()
        
        return results