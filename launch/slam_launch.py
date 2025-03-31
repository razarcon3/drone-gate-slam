import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, TimerAction
from launch.substitutions import LaunchConfiguration, FindExecutable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # --- Declare Launch Arguments ---

    # Argument for the bag file path
    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        description='Path to the bag file to play.'
        # You could add a default value if needed:
        # default_value='/path/to/your/default_bag_file.mcap'
    )

    # Argument for optional delay before starting bag playback (in seconds)
    delay_arg = DeclareLaunchArgument(
        'delay',
        default_value='0.0', # Default to no delay
        description='Delay in seconds before starting bag playback.'
    )

    # --- Define Actions ---

    # Action to launch your specific node
    # ** REPLACE 'your_package_name' and 'your_node_executable' **
    your_node_action = Node(
        package='your_package_name',      # <-- Replace this
        executable='your_node_executable', # <-- Replace this
        name='my_processing_node',        # Optional: give your node a specific name
        output='screen',                  # Show node output in the terminal
        # Add parameters if your node needs them:
        # parameters=[{'param_name': 'value'},
        #             os.path.join(get_package_share_directory('your_package_name'), 'config', 'params.yaml')],
        # Add remappings if needed:
        # remappings=[
        #     ('/input/topic', '/actual/topic/from/bag'),
        # ]
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
        output='screen',
        name='ros2_bag_play', # Name for logging purposes
        # Ensures ros2 bag play stops if the launch file is terminated (e.g., Ctrl+C)
        shell=False, # Recommended for ExecuteProcess with specific commands
    )

    # --- Assemble Launch Description ---

    # Create the LaunchDescription object
    ld = LaunchDescription()

    # Add arguments
    ld.add_action(bag_path_arg)
    ld.add_action(delay_arg)

    # Add the action to start your node
    ld.add_action(your_node_action)

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

    # You could also add an event handler to stop bag playback if your_node exits:
    # from launch.event_handlers import OnProcessExit
    # ld.add_action(RegisterEventHandler(
    #     OnProcessExit(
    #         target_action=your_node_action,
    #         on_exit=[
    #             LogInfo(msg='Node exited, stopping bag play.'),
    #             # Send SIGINT (Ctrl+C) to the bag player process
    #             EmitEvent(event=SignalProcess(signal_number=signal.SIGINT, process_matcher=lambda proc: 'ros2_bag_play' in proc.name))
    #         ]
    #     )
    # ))
    # Note: Using the OnProcessExit handler requires importing RegisterEventHandler, EmitEvent, SignalProcess from launch.actions, and signal module.

    return ld