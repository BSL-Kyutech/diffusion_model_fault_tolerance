from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='diffusion_model_fault_tolerance',
            executable='gen_circle',
            name = 'gen_circle_node',
            output='screen'
        )
    ])