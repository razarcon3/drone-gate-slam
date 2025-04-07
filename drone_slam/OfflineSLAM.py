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
from CornerDetector import CornerDetector
from FrontEnd import FrontEnd

import torch
import numpy as np

RGB_IMAGE_TOPIC = "/airsim_node/Drone1/front_center/rgb"
KEYFRAME_POSE_TOPIC = "/keyframe_pose"


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

    corner_detector = CornerDetector(device = torch.device("cuda"))
    front_end = FrontEnd()
    
    images = []
    num_images = 0
    offset_counter = 0
    
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        
        
        if topic == RGB_IMAGE_TOPIC:  # Only process the specified image topic
            msg_type = get_message(typename(topic))
            msg = deserialize_message(data, msg_type)

            try:
                # Convert ROS Image message to OpenCV image
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")  # Or "passthrough" for no encoding change
                # "bgr8": color image with blue-green-red color order
                # "mono8" or "gray": Grayscale image
                # "passthrough": Keeps the original encoding.  Important if you have, e.g., 16-bit images.

            except CvBridgeError as e:
                print(e)
                continue  # Skip to the next message if conversion fails
            
            front_end.process_image(cv_image)
            #corner_detector.createPrediction(cv_image)
            #visualization_img = corner_detector.getCurrentVisualization()
            
            # Now you have the image in OpenCV format (cv_image)
            # You can display it, process it, save it, etc.
            print(f"Received RGB image at timestamp: {timestamp}")
            #cv2.imshow("Image window", visualization_img)
            #cv2.waitKey(0)
            
            # if offset_counter >= 120:
            #     if num_images < 50:
            #         images.append(cv_image)
            #         num_images += 1
            # offset_counter += 1
            
        
        # if topic == 
        
        if topic == KEYFRAME_POSE_TOPIC:
            msg_type = get_message(typename(topic))
            msg = deserialize_message(data, msg_type)
            print(topic, timestamp)
    
    
    # queries = torch.tensor([
    #     [0., 177., 66.],  # point tracked from the first frame
    # ]).cuda()
    
    # video_chunk = (
    #             torch.tensor(
    #                 np.stack(images), device=torch.device("cuda") # Shape becomes (2, H, W, C)
    #             )
    #             .float()                                  # Shape (2, H, W, C), dtype float32
    #             .permute(0, 3, 1, 2)                      # Shape becomes (2, C, H, W)
    #             [None]                                    # Shape becomes (1, 2, C, H, W)
    #         )
    
    # cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(torch.device("cuda"))
    # pred_tracks, pred_visibility = cotracker(video_chunk ,queries=queries[None])
    
    
    del reader


if __name__ == "__main__":
    main()