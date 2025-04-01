import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, TimerAction, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, FindExecutable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource

def generate_launch_description():

    # --- Declare Launch Arguments ---

    # Argument for the bag file path
    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        description='Path to the bag file to play.',
        # You could add a default value if needed:
        default_value='~/thesis_ws/src/bags/simple_test_bag/cv_modified/cv_modified_0.mcap'
    )

    vins_fusion_config_arg = DeclareLaunchArgument(
        'vf_config',
        description='Path to the config file to use for VINS-Fusion',
        # You could add a default value if needed:
        default_value='/home/codaero/thesis_ws/src/VINS-Fusion-ROS2-humble-arm/config/airsim/airsim_stereo_imu_config.yaml'
    )

    # Argument for optional delay before starting bag playback (in seconds)
    delay_arg = DeclareLaunchArgument(
        'delay',
        default_value='10.0', # Default to no delay
        description='Delay in seconds before starting bag playback.'
    )

    # --- Define Actions ---

    # Action to launch your specific node
    # ** REPLACE 'your_package_name' and 'your_node_executable' **
    vins_fusion = Node(
        package='vins',      # <-- Replace this
        executable='vins_node', # <-- Replace this
        name='VINS_Fusion',        # Optional: give your node a specific name
        output='screen',                  # Show node output in the 
        parameters=[{
            # Pass the config file path as a parameter
            # Use LaunchConfiguration to get the value from the declared argument
            'config_file': LaunchConfiguration('vf_config')
            # Or if using Method 1 or 2 directly:
            # 'config_file': vins_config_path
        }]
    )

    foxglove_bridge_pkg_dir = get_package_share_directory('foxglove_bridge')
    foxglove_bridge_launch_path = os.path.join(
        foxglove_bridge_pkg_dir,
        'launch',
        'foxglove_bridge_launch.xml' # The launch file provided by the package
    )

    # --- Define the action to include the foxglove_bridge launch file ---
    foxglove_bridge_launch = IncludeLaunchDescription(
        # Specify the source of the launch file
        XMLLaunchDescriptionSource(foxglove_bridge_launch_path)
    )
    
    foxglove_studio = ExecuteProcess(
        cmd=[
            FindExecutable(name='foxglove-studio')
        ],
        name='foxglove_studio', # Name for logging purposes
        # Ensures ros2 bag play stops if the launch file is terminated (e.g., Ctrl+C)
        shell=False, # Recommended for ExecuteProcess with specific commands
    )

    # Action to execute the ros2 bag play command
    # We use FindExecutable to make sure the 'ros2' command can be found
    # We use LaunchConfiguration to get the bag_path provided as an argument
    bag_play_action = ExecuteProcess(
        cmd=[
            FindExecutable(name='ros2'),
            'bag', 'play',
            LaunchConfiguration('bag_path')
            # Add any other ros2 bag play options here, e.g.:
            # '--loop',               # Loop playback
            # '-r', '0.5',            # Playback rate
            # '--start-offset', '5.0' # Start 5 seconds into the bag
        ],
        name='ros2_bag_play', # Name for logging purposes
        # Ensures ros2 bag play stops if the launch file is terminated (e.g., Ctrl+C)
        shell=True, # Recommended for ExecuteProcess with specific commands
    )

    # --- Assemble Launch Description ---

    # Create the LaunchDescription object
    ld = LaunchDescription()

    # Add arguments
    ld.add_action(bag_path_arg)
    ld.add_action(vins_fusion_config_arg)
    ld.add_action(delay_arg)

    ld.add_action(vins_fusion)
    
    # add foxglove studio
    ld.add_action(foxglove_bridge_launch)
    ld.add_action(foxglove_studio)
    

    # Add the action to play the bag, potentially delayed
    ld.add_action(
        TimerAction(
            period=LaunchConfiguration('delay'), # Delay is configurable via launch argument
            actions=[
                LogInfo(msg="Starting ros2 bag play..."),
                bag_play_action
            ]
        )
    )
    
    return ld