import argparse
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String
import rosbag2_py
import time
import cv2
from cv_bridge import CvBridge, CvBridgeError  # Import CvBridge
from sensor_msgs.msg import Image  # Import the Image message type

IMAGE_TOPIC = "/airsim_node/Drone1/front_center_Scene/image"


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

    latest_depth_image, latest_GT_odom_local = None, None


    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic == IMAGE_TOPIC:  # Only process the specified image topic
            msg_type = get_message(typename(topic))
            msg = deserialize_message(data, msg_type)

            try:
                # Convert ROS Image message to OpenCV image
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")  # Or "passthrough" for no encoding change
                # "bgr8": color image with blue-green-red color order
                # "mono8" or "gray": Grayscale image
                # "passthrough": Keeps the original encoding.  Important if you have, e.g., 16-bit images.

            except CvBridgeError as e:
                print(e)
                continue  # Skip to the next message if conversion fails

            # Now you have the image in OpenCV format (cv_image)
            # You can display it, process it, save it, etc.
            print(f"Received image at timestamp: {timestamp}")
            cv2.imshow("Image window", cv_image)
            cv2.waitKey(3)
        
        
        msg_type = get_message(typename(topic))
        msg = deserialize_message(data, msg_type)
        print(topic)
        
    del reader


if __name__ == "__main__":
    main()