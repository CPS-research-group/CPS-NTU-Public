from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ecrts',
            executable='mock_camera_node:main',
            name='mock_camera_node',
            parameters=[
                {'fps': 20},
                {'height': 600},
                {'width': 800},
            ]
        )
    ])

