"""
vln_bringup.launch.py
======================
Single launch file. Starts everything:
  1. Gazebo with turtlebot3_home_service_challenge.world
  2. TurtleBot3 waffle spawn
  3. SLAM Toolbox (online async — builds map while exploring)
  4. Nav2 navigation stack
  5. object_detector_node   (delayed 6s)
  6. frontier_explorer_node (delayed 7s)
  7. vln_command_node       (delayed 7s)

Usage:
  ros2 launch vln_project vln_bringup.launch.py \
    checkpoint_dir:=/home/user/ros2_ws/src/vln_project/checkpoints

Args:
  checkpoint_dir  path to folder containing clip_finetuned.pt + meta.json
  map_file        path for object_map.json  (default: /tmp/object_map.json)
  use_rviz        whether to open RViz2     (default: true)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription,
    ExecuteProcess, TimerAction, SetEnvironmentVariable
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg         = get_package_share_directory("vln_project")
    nav2_pkg    = get_package_share_directory("nav2_bringup")
    slam_pkg    = get_package_share_directory("slam_toolbox")
    tb3_gz_pkg  = get_package_share_directory("turtlebot3_gazebo")

    # ── Declare args ──────────────────────────────────────────────────────────
    ckpt_arg    = DeclareLaunchArgument(
        "checkpoint_dir",
        default_value=os.path.join(pkg, "checkpoints"),
        description="Folder with clip_finetuned.pt + meta.json")

    map_arg     = DeclareLaunchArgument(
        "map_file",
        default_value="/tmp/object_map.json",
        description="Where to write/read object_map.json")

    rviz_arg    = DeclareLaunchArgument(
        "use_rviz", default_value="true",
        description="Open RViz2 for visualisation")

    ckpt_dir  = LaunchConfiguration("checkpoint_dir")
    map_file  = LaunchConfiguration("map_file")

    # ── Set TurtleBot3 model ──────────────────────────────────────────────────
    set_tb3_model = SetEnvironmentVariable(
        name="TURTLEBOT3_MODEL", value="waffle")

    # ── Gazebo with the home service challenge world ───────────────────────────
    # Copy your turtlebot3_home_service_challenge.world into
    # ~/ros2_ws/src/vln_project/worlds/
    world_file = os.path.join(pkg, "worlds",
                              "turtlebot3_home_service_challenge.world")

    gazebo = ExecuteProcess(
        cmd=[
            "gazebo", "--verbose", world_file,
            "-s", "libgazebo_ros_factory.so",
            "-s", "libgazebo_ros_init.so",
        ],
        output="screen"
    )

    # ── Spawn TurtleBot3 waffle at origin ─────────────────────────────────────
    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity",     "turtlebot3_waffle",
            "-file",        os.path.join(
                get_package_share_directory("turtlebot3_description"),
                "urdf", "turtlebot3_waffle.urdf"),
            "-x", "0", "-y", "0", "-z", "0.01",
        ],
        output="screen",
    )

    # ── Robot state publisher ─────────────────────────────────────────────────
    robot_state_pub = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gz_pkg, "launch",
                         "robot_state_publisher.launch.py")),
        launch_arguments={"use_sim_time": "true"}.items(),
    )

    # ── SLAM Toolbox ──────────────────────────────────────────────────────────
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_pkg, "launch",
                         "online_async_launch.py")),
        launch_arguments={
            "slam_params_file": os.path.join(pkg, "config", "slam_params.yaml"),
            "use_sim_time":     "true",
        }.items(),
    )

    # ── Nav2 ──────────────────────────────────────────────────────────────────
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_pkg, "launch",
                         "navigation_launch.py")),
        launch_arguments={
            "params_file":  os.path.join(pkg, "config", "nav2_params.yaml"),
            "use_sim_time": "true",
        }.items(),
    )

    # ── RViz2 ─────────────────────────────────────────────────────────────────
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d",
                   os.path.join(nav2_pkg, "rviz", "nav2_default_view.rviz")],
        output="screen",
    )

    # ── VLN nodes (delayed to let Gazebo + Nav2 come up first) ───────────────
    detector_node = TimerAction(period=6.0, actions=[
        Node(
            package="vln_project",
            executable="object_detector_node",
            name="object_detector",
            parameters=[{
                "checkpoint_dir":  ckpt_dir,
                "map_file":        map_file,
                "conf_threshold":  0.70,
                "detect_every_n":  3,
            }],
            output="screen",
        )
    ])

    explorer_node = TimerAction(period=7.0, actions=[
        Node(
            package="vln_project",
            executable="frontier_explorer_node",
            name="frontier_explorer",
            parameters=[{
                "map_file":           map_file,
                "frontier_min_cells": 5,
                "explore_enabled":    True,
            }],
            output="screen",
        )
    ])

    command_node = TimerAction(period=7.0, actions=[
        Node(
            package="vln_project",
            executable="vln_command_node",
            name="vln_command",
            parameters=[{"map_file": map_file}],
            output="screen",
        )
    ])

    return LaunchDescription([
        ckpt_arg, map_arg, rviz_arg,
        set_tb3_model,
        gazebo,
        robot_state_pub,
        spawn_robot,
        slam,
        nav2,
        rviz,
        detector_node,
        explorer_node,
        command_node,
    ])
