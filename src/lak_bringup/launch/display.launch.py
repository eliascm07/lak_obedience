import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # Ruta al URDF
    urdf_file = os.path.join(
        get_package_share_directory('lak_description'),
        'urdf', 'lak_obedience.urdf'
    )

    # Leer el URDF como string
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # Ruta a la configuración de RViz2
    rviz_config = os.path.join(
        get_package_share_directory('lak_bringup'),
        'rviz', 'lak_config.rviz'
    )

    return LaunchDescription([

        # Publica la descripción del robot y transforms
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}]
        ),

        # GUI con sliders para mover joints manualmente
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen',
        ),

        # Visualizador
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
        ),
    ])
