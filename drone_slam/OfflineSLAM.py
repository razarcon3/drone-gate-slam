#!/usr/bin/env python3

import argparse
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String
import rosbag2_py
import time
import cv2
from cv_bridge import CvBridge, CvBridgeError  # Import CvBridge
from sensor_msgs.msg import Image  # Import the Image message type
from nav_msgs.msg import Odometry
from CornerDetector import CornerDetector
from FrontEnd import FrontEnd
from BackEnd import BackEnd
from VisualizationUtils import plot_graph_values

import gtsam
import torch
import numpy as np

RGB_IMAGE_TOPIC = "/airsim_node/Drone1/front_center/rgb"
DEPTH_IMAGE_TOPIC = "/airsim_node/Drone1/front_center/depth"
DEPTH_VIS_TOPIC = "/airsim_node/Drone1/front_center/depthvis"
ODOM_LOCAL_TOPIC = "/airsim_node/Drone1/odom_local"

KEYFRAME_POSE_TOPIC = "/keyframe_pose"

def odometry_to_gtsam_pose3(odometry_msg: Odometry) -> gtsam.Pose3:
    """
    Converts a ROS 2 nav_msgs/msg/Odometry message to a gtsam.Pose3 object.

    Args:
        odometry_msg: The input nav_msgs.msg.Odometry message.

    Returns:
        A gtsam.Pose3 object representing the pose from the Odometry message.
    """
    # Extract position
    position = odometry_msg.pose.pose.position
    translation = gtsam.Point3(position.x, position.y, position.z)
    # Or using numpy:
    # translation_np = np.array([position.x, position.y, position.z])


    # Extract orientation (quaternion)
    orientation = odometry_msg.pose.pose.orientation
    # IMPORTANT: GTSAM's Quaternion constructor expects (w, x, y, z)
    rotation = gtsam.Rot3.Quaternion(orientation.w,
                                     orientation.x,
                                     orientation.y,
                                     orientation.z)

    # Create gtsam.Pose3
    # gtsam_pose = gtsam.Pose3(rotation, translation_np) # If using numpy array
    gtsam_pose = gtsam.Pose3(rotation, translation) # If using gtsam.Point3

    return gtsam_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", help="input bag path (folder or filepath) to read from"
    )

    args = parser.parse_args()
    
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=args.input, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )

    topic_types = reader.get_all_topics_and_types()

    def typename(topic_name):
        for topic_type in topic_types:
            if topic_type.name == topic_name:
                return topic_type.type
        raise ValueError(f"topic {topic_name} not in bag")

    # Initialize CvBridge
    bridge = CvBridge()

    front_end = FrontEnd()
    
    num_rgb = 0
    num_depth = 0
    num_keyframe_poses = 0
    
    depth_images = []
    rgb_images = []
    key_frame_poses_gtsam = []
    gt_positions = []
    key_frame_positions = []
    
    curr_rgb = None
    curr_depth = None
    
    processed = False
    i = 0
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        
        
        if topic == RGB_IMAGE_TOPIC:  # Only process the specified image topic
            i += 1
            msg_type = get_message(typename(topic))
            msg = deserialize_message(data, msg_type)

            try:
                # Convert ROS Image message to OpenCV image
                curr_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")  # Or "passthrough" for no encoding change
                # "bgr8": color image with blue-green-red color order
                # "mono8" or "gray": Grayscale image
                # "passthrough": Keeps the original encoding.  Important if you have, e.g., 16-bit images.
                num_rgb += 1
                processed = True
                rgb_images.append(curr_rgb)
            except CvBridgeError as e:
                print(e)
                continue  # Skip to the next message if conversion fails
            #corner_detector.createPrediction(cv_image)
            #visualization_img = corner_detector.getCurrentVisualization()
            
            # Now you have the image in OpenCV format (cv_image)
            # You can display it, process it, save it, etc.
            print(f"Received RGB image at timestamp: {timestamp * 1e-9}")
        
        if topic == KEYFRAME_POSE_TOPIC:
            msg_type = get_message(typename(topic))
            msg = deserialize_message(data, msg_type)
            print(topic, timestamp * 1e-9)
            num_keyframe_poses += 1
            key_frame_pose_gtsam = odometry_to_gtsam_pose3(msg)
            
            # apply correction factor to y
            rot = key_frame_pose_gtsam.rotation()
            translation = key_frame_pose_gtsam.translation()
            translation[1] = translation[1]
            key_frame_pose_gtsam = gtsam.Pose3(rot, translation)
            
            key_frame_poses_gtsam.append(key_frame_pose_gtsam)
            
            key_frame_positions.append(key_frame_pose_gtsam.translation())
        
        if topic == ODOM_LOCAL_TOPIC:
            msg_type = get_message(typename(topic))
            msg = deserialize_message(data, msg_type)
            gt_translation = odometry_to_gtsam_pose3(msg).translation()
            gt_positions.append(gt_translation)
            
        
        if topic == DEPTH_IMAGE_TOPIC:
            num_depth += 1
            processed = True
            
            msg_type = get_message(typename(topic))
            msg = deserialize_message(data, msg_type)
            
            # Convert ROS Image message to OpenCV image
            curr_depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")  # Or "passthrough" for no encoding change
            print(f"Received Depth Image at timestamp: {timestamp * 1e-9}")
            depth_images.append(curr_depth)
            
        # if num_depth == num_rgb and processed == True and i >= 200:
            # front_end.process_image(curr_rgb, curr_depth)
            # processed = False
    print(f"Number of Depth: {num_depth}, Number of RGB: {num_rgb}, Number of Keyframe Poses: {num_keyframe_poses}")
    del reader

    
    back_end = BackEnd()
    
    keyframe_idx = 0
    i = 0
    for rgb_img, depth_img in zip(rgb_images, depth_images):
        front_end.process_image(rgb_img, depth_img)
        if i == 2:
            # run backend
            curr_keyframe_pose = key_frame_poses_gtsam[keyframe_idx]
            back_end.process_key_frame(curr_keyframe_pose, front_end.getCornerObservations())
            keyframe_idx += 1
            
            i = 0
        i +=1

    results = back_end.solveDogleg()
    
    plot_graph_values(results, keyframe_idx, front_end.gate_idx, gt_positions, key_frame_positions, None)
    

if __name__ == "__main__":
    main()