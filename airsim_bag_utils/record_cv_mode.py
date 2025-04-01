import argparse
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
# from std_msgs.msg import String # No longer needed unless used elsewhere
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Pose # Import Pose
from sensor_msgs.msg import Image
from std_msgs.msg import Header

import rosbag2_py
import time
import cv2 # Not used in this snippet, can remove if not needed elsewhere
import cosysairsim as airsim
import time # Imported twice, remove one
import numpy as np # Keep for potential future use or other parts
# from scipy.spatial.transform import Rotation as ScipyRotation # No longer needed for this core conversion
from cv_bridge import CvBridge, CvBridgeError  # Not used here
# from sensor_msgs.msg import Image  # Not used here

# Topic to read from
ODOM_LOCAL_TOPIC = "/airsim_node/Drone1/odom_local"

# topics to write
FC_IMG_RGB_TOPIC = "/airsim_node/Drone1/front_center/rgb"
FC_IMG_MONO_TOPIC = "/airsim_node/Drone1/front_center/mono"
FC_IMG_DEPTH_TOPIC = "/airsim_node/Drone1/front_center/depth"
FC_IMG_DEPTHVIS_TOPIC = "/airsim_node/Drone1/front_center/depthvis"
FR_IMG_MONO_TOPIC = "/airsim_node/Drone1/front_right/mono"
ODOM_LOCAL_PATH_TOPIC = "/airsim_node/Drone1/odom_local_path"

IMAGE_CAPTURE_FREQ = 24 # in HZ how many times image is captured per second
VISUALIZE = False

# Constants for visualization
MIN_DEPTH_METERS = 0
MAX_DEPTH_METERS = 100

def convert_ros2_to_airsim_pose(ros2_pose: Pose) -> airsim.Pose:
    """
    Converts a ROS2 geometry_msgs.msg.Pose (ENU) to an AirSim Pose (NED).

    Args:
        ros2_pose: The ROS2 Pose message.

    Returns:
        An airsim.Pose object representing the same pose in AirSim's NED frame.
    """
    # Position Conversion (ROS ENU to AirSim NED)
    airsim_pos = airsim.Vector3r(
        x_val=ros2_pose.position.x,
        y_val=-ros2_pose.position.y, # Negate Y
        z_val=-ros2_pose.position.z  # Negate Z
    )

    # Orientation Conversion (ROS Quaternion xyzw to AirSim Quaternion wxyz)
    # ROS quat: [ros2_pose.orientation.x, ros2_pose.orientation.y, ros2_pose.orientation.z, ros2_pose.orientation.w]
    # AirSim quat: [w, x, y, z]
    airsim_ori = airsim.Quaternionr(
        w_val=ros2_pose.orientation.w,
        x_val=ros2_pose.orientation.x,
        y_val=-ros2_pose.orientation.y, # Negate Y component
        z_val=-ros2_pose.orientation.z  # Negate Z component
    )

    return airsim.Pose(position_val=airsim_pos, orientation_val=airsim_ori)

def npimg_to_ros2imgmsg(image: np.ndarray, header: Header, encoding)-> Image:
    bridge = CvBridge()
    img_msg: Image = bridge.cv2_to_imgmsg(image, encoding=encoding)
    img_msg.header = header
    return img_msg

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", help="input bag path (folder or filepath) to read from"
    )
    parser.add_argument(
        "output", help="output bag path (folder or filepath) to write to"
    )

    args = parser.parse_args()

    # Initialize the reader
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=args.input, storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening input bag file '{args.input}': {e}")
        return # Exit if input bag cannot be opened


    # Get topic metadata from the input bag
    all_topic_metadata = reader.get_all_topics_and_types()
    topic_type_map = {meta.name: meta.type for meta in all_topic_metadata}
    registered_topics_writer = set() # Keep track of topics registered in writer

    # --- Initialize ROS Bag Writer ---
    writer = rosbag2_py.SequentialWriter()
    storage_options_out = rosbag2_py.StorageOptions(uri=args.output, storage_id="mcap")
    converter_options_out = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    try:
        writer.open(storage_options_out, converter_options_out)
    except Exception as e:
        print(f"Error opening output bag file '{args.output}': {e}")
        del reader # Clean up reader before exiting
        return
    
    # --- Register ALL Topics from Input Bag with Writer ---
    print(f"Registering {len(all_topic_metadata)} topics from input bag '{args.input}':")
    for topic_meta in all_topic_metadata:
        try:
            # Use the metadata directly from the reader
            writer.create_topic(topic_meta)
            registered_topics_writer.add(topic_meta.name)
            print(f"  Registered: {topic_meta.name} ({topic_meta.type})")
        except Exception as e:
            print(f"  Failed to register topic '{topic_meta.name}': {e}")
            return

    # --- Register NEW AirSim Image Topics with Writer ---
    print("Registering new AirSim image topics:")
    try:
        image_msg_type_str = "sensor_msgs/msg/Image"
        path_msg_type_str = "nav_msgs/msg/Path"
        new_topics_meta = [
            rosbag2_py.TopicMetadata(name=FC_IMG_RGB_TOPIC, type=image_msg_type_str, serialization_format="cdr"),
            rosbag2_py.TopicMetadata(name=FC_IMG_MONO_TOPIC, type=image_msg_type_str, serialization_format="cdr"),
            rosbag2_py.TopicMetadata(name=FC_IMG_DEPTH_TOPIC, type=image_msg_type_str, serialization_format="cdr"),
            rosbag2_py.TopicMetadata(name=FC_IMG_DEPTHVIS_TOPIC, type=image_msg_type_str, serialization_format="cdr"),
            rosbag2_py.TopicMetadata(name=FR_IMG_MONO_TOPIC, type=image_msg_type_str, serialization_format="cdr"),
            rosbag2_py.TopicMetadata(name=ODOM_LOCAL_PATH_TOPIC, type=path_msg_type_str, serialization_format="cdr")
            
        ]
        for topic_meta in new_topics_meta:
             # Avoid re-registering if somehow names overlap (unlikely here)
             if topic_meta.name not in registered_topics_writer:
                 writer.create_topic(topic_meta)
                 registered_topics_writer.add(topic_meta.name)
                 print(f"  Registered: {topic_meta.name} ({topic_meta.type})")
             else:
                 print(f"  Skipping registration (already exists): {topic_meta.name}")
    except Exception as e:
        print(f"Error registering new AirSim topics: {e}")
        del writer
        del reader
        return
    
    def get_msg_type_from_name(topic_name):
        if topic_name in topic_type_map:
            msg_type_str = topic_type_map[topic_name]
            try:
                return get_message(msg_type_str)
            except ModuleNotFoundError:
                print(f"Error: Message type '{msg_type_str}' not found. Ensure ROS environment is sourced.")
                return None
            except Exception as e:
                print(f"Error getting message type for {msg_type_str}: {e}")
                return None
        raise ValueError(f"topic {topic_name} not in bag or type map")

    # Connecting to airsim
    client = airsim.ComputerVisionClient() # Or MultiRotorClient if controlling flight
    print("Attempting to connect to AirSim...")
    client.confirmConnection()
    print("Connected to AirSim.")
    client.enableApiControl(True, "cv") # Ensure API control is enabled if needed
    client.armDisarm(True, "cv") # Arm if necessary for pose setting to stick sometimes

    last_capture_time_stamp = None # units is nanoseconds
    cumulative_path = Path()
    while reader.has_next():
        (topic, data, timestamp) = reader.read_next() # timestamp is in nanoseconds
        
        # Copy over every message in old ROSBAG
        writer.write(topic, data, timestamp)
        
        if topic == ODOM_LOCAL_TOPIC:
            try:
                msg_type = get_msg_type_from_name(topic)
                if msg_type is None:
                    print(f"Skipping message for topic {topic} due to message type issue.")
                    continue
                msg: Odometry = deserialize_message(data, msg_type)
                odom_local_header: Header = msg.header # Get the header from the odometry message
                odom_local_pose: Pose = msg.pose.pose # Use Pose directly
                
                # Handles image capture frequency
                if last_capture_time_stamp is None:
                    last_capture_time_stamp = timestamp
                if (timestamp - last_capture_time_stamp) <= ((1/IMAGE_CAPTURE_FREQ) * 1e9):
                    continue
                
                # create path for visualizing the GT odom local pose
                pose_stamped = PoseStamped()
                pose_stamped.header = odom_local_header # Use the odom header for the individual pose
                pose_stamped.pose = odom_local_pose
                cumulative_path.poses.append(pose_stamped)

                # --- Update the main Path header ---
                # Set the frame_id from the first odom message, then keep it
                if not cumulative_path.header.frame_id:
                    cumulative_path.header.frame_id = odom_local_header.frame_id
                # Update the timestamp to the latest pose's timestamp
                cumulative_path.header.stamp = odom_local_header.stamp

                # --- Write the updated Path message to the output bag ---
                serialized_path = serialize_message(cumulative_path)
                writer.write(ODOM_LOCAL_PATH_TOPIC, serialized_path, timestamp)
                
                
                last_capture_time_stamp = timestamp
                
                
                # Convert the ROS2 Pose to AirSim Pose
                vehicle_pose_airsim = convert_ros2_to_airsim_pose(odom_local_pose)
                client.simSetVehiclePose(vehicle_pose_airsim, True, "cv") # Use the correct vehicle name if not "cv"

                # Capture the images
                img_responses = client.simGetImages(
                    [airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
                     airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True, False),
                     airsim.ImageRequest("front_right", airsim.ImageType.Scene, False, False)],
                    "cv"
                )
                
                front_center_image_rgb = np.frombuffer(img_responses[0].image_data_uint8, dtype=np.uint8).reshape(img_responses[0].height,
                                                                                             img_responses[0].width, 3)
                front_right_image_mono = cv2.cvtColor(np.frombuffer(img_responses[2].image_data_uint8, dtype=np.uint8).reshape(img_responses[2].height,
                                                                                             img_responses[2].width, 3), cv2.COLOR_RGB2GRAY)
                
                front_center_image_mono = cv2.cvtColor(front_center_image_rgb, cv2.COLOR_RGB2GRAY)
                
                front_center_depth = (airsim.list_to_2d_float_array(img_responses[1].image_data_float, img_responses[1].width,
                                                           img_responses[1].height)
                             .reshape(img_responses[1].height, img_responses[1].width))  # in meters
                
                
                depth_vis = 255 - np.interp(front_center_depth, (MIN_DEPTH_METERS, MAX_DEPTH_METERS), (0, 255)).astype(np.uint8)

                newImgHeader = Header()
                newImgHeader.stamp = msg.header.stamp
                
                
                front_center_rgb_img_msg = npimg_to_ros2imgmsg(front_center_image_rgb, newImgHeader, encoding='8UC3')
                front_center_mono_img_msg = npimg_to_ros2imgmsg(front_center_image_mono, newImgHeader, encoding="mono8")
                front_center_depth_img_msg = npimg_to_ros2imgmsg(front_center_depth, newImgHeader, encoding="32FC1")
                front_center_depth_vis_img_msg = npimg_to_ros2imgmsg(depth_vis, newImgHeader, encoding="passthrough")
                front_right_mono_img_msg = npimg_to_ros2imgmsg(front_right_image_mono, newImgHeader, encoding='mono8')
                
                writer.write(FC_IMG_RGB_TOPIC, serialize_message(front_center_rgb_img_msg), timestamp)
                writer.write(FC_IMG_MONO_TOPIC, serialize_message(front_center_mono_img_msg), timestamp)
                writer.write(FC_IMG_DEPTH_TOPIC, serialize_message(front_center_depth_img_msg), timestamp)
                writer.write(FC_IMG_DEPTHVIS_TOPIC, serialize_message(front_center_depth_vis_img_msg), timestamp)
                writer.write(FR_IMG_MONO_TOPIC, serialize_message(front_right_mono_img_msg), timestamp)
                
                if VISUALIZE:
                    cv2.imshow("front_center_image_rgb", front_center_image_rgb)
                    cv2.imshow("front_center_image_mono", front_center_image_mono)
                    cv2.imshow("front_right_image_mono", front_right_image_mono)
                    cv2.imshow("depth_vis", depth_vis)
                    cv2.waitKey(0)
                # Add a small sleep to allow AirSim to process/render, otherwise it might be too fast
                #time.sleep(0.01)

            except Exception as e:
                print(f"Error processing message for topic {topic}: {e}")
    del reader

    print("Finished processing bag file.")
    # Optional: Disarm and disable API control if needed
    # client.armDisarm(False, "cv")
    # client.enableApiControl(False, "cv")

    # No need to explicitly delete reader with 'with' statement or relying on garbage collection


if __name__ == "__main__":
    main()