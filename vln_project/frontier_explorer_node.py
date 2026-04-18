#!/usr/bin/env python3
"""
frontier_explorer_node.py
==========================
ROS2 node. Autonomously explores the turtlebot3_home_service_challenge world
using frontier-based BFS search until all target objects are found.

Algorithm:
  1. Receive /map (OccupancyGrid from SLAM Toolbox)
  2. BFS from robot position across FREE cells
  3. Any FREE cell adjacent to UNKNOWN cell = frontier cell
  4. Cluster frontier cells with flood-fill
  5. Pick nearest cluster centroid as next goal
  6. Send to Nav2 NavigateToPose action
  7. Repeat until all 5 objects found in object_map.json

Subscribes:
  /map           nav_msgs/OccupancyGrid
  /odom          nav_msgs/Odometry
  /detections    std_msgs/String

Publishes:
  /explore_status  std_msgs/String
"""

import json
import math
from collections import deque
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler

FREE     =   0
OCCUPIED = 100
UNKNOWN  =  -1

TARGET_CLASSES = {
    "cafe_table",
    "double_cabinet",
    "first_2015_trash_can",
    "single_cabinet",
    "table",
}


class FrontierExplorerNode(Node):

    def __init__(self):
        super().__init__("frontier_explorer")

        self.declare_parameter("map_file",           "/tmp/object_map.json")
        self.declare_parameter("frontier_min_cells", 5)
        self.declare_parameter("explore_enabled",    True)

        self.map_file        = self.get_parameter("map_file").value
        self.min_frontier    = self.get_parameter("frontier_min_cells").value
        self.explore_enabled = self.get_parameter("explore_enabled").value

        # State
        self.grid        = None
        self.robot_x     = 0.0
        self.robot_y     = 0.0
        self.goal_active = False
        self.found_objs  = self._load_known()

        # Nav2 action client
        self._nav = ActionClient(self, NavigateToPose, "navigate_to_pose")

        self.create_subscription(OccupancyGrid, "/map",        self._map_cb,   10)
        self.create_subscription(Odometry,      "/odom",       self._odom_cb,  10)
        self.create_subscription(String,        "/detections", self._det_cb,   10)

        self.status_pub = self.create_publisher(String, "/explore_status", 10)

        # Tick every 3 seconds
        self.create_timer(3.0, self._tick)
        self.get_logger().info("FrontierExplorerNode ready")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _map_cb(self, msg):
        self.grid = msg

    def _odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def _det_cb(self, msg: String):
        d = json.loads(msg.data)
        obj = d.get("object")
        if obj and obj not in self.found_objs:
            self.found_objs.add(obj)
            n = len(self.found_objs)
            total = len(TARGET_CLASSES)
            self.get_logger().info(
                f"Found: {obj}  ({n}/{total})")
            if TARGET_CLASSES.issubset(self.found_objs):
                self.get_logger().info(
                    "All objects found! Exploration complete.")
                self.explore_enabled = False
                self.status_pub.publish(
                    String(data=json.dumps({
                        "status": "complete",
                        "found":  list(self.found_objs)
                    })))

    # ── Main tick ─────────────────────────────────────────────────────────────
    def _tick(self):
        if not self.explore_enabled:
            return
        if self.grid is None or self.goal_active:
            return
        if TARGET_CLASSES.issubset(self.found_objs):
            return

        frontiers = self._find_frontiers()
        if not frontiers:
            self.get_logger().info("No frontiers left — map fully explored.")
            self.status_pub.publish(
                String(data=json.dumps({"status": "map_complete"})))
            return

        gx, gy = self._nearest_frontier(frontiers)
        self.get_logger().info(
            f"Frontier goal → ({gx:.2f}, {gy:.2f})  "
            f"[{len(frontiers)} clusters  "
            f"{len(self.found_objs)}/{len(TARGET_CLASSES)} objects]")
        self._send_goal(gx, gy)

    # ── BFS frontier detection ─────────────────────────────────────────────────
    def _find_frontiers(self):
        g   = self.grid
        W   = g.info.width
        H   = g.info.height
        res = g.info.resolution
        ox  = g.info.origin.position.x
        oy  = g.info.origin.position.y
        data = np.array(g.data, dtype=np.int8).reshape(H, W)

        # Robot cell
        rx = int((self.robot_x - ox) / res)
        ry = int((self.robot_y - oy) / res)
        rx = max(0, min(W - 1, rx))
        ry = max(0, min(H - 1, ry))

        DIRS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        # BFS over FREE cells — mark cells adjacent to UNKNOWN as frontiers
        visited      = np.zeros((H, W), bool)
        frontier_map = np.zeros((H, W), bool)
        queue = deque([(rx, ry)])
        visited[ry, rx] = True

        while queue:
            cx, cy = queue.popleft()
            is_frontier = False
            for dx, dy in DIRS4:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                cell = int(data[ny, nx])
                if cell == UNKNOWN:
                    is_frontier = True
                elif cell == FREE and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((nx, ny))
            if is_frontier and data[cy, cx] == FREE:
                frontier_map[cy, cx] = True

        # Cluster frontier cells → centroid list
        clustered = np.zeros((H, W), bool)
        centroids = []
        for fy in range(H):
            for fx in range(W):
                if not (frontier_map[fy, fx] and not clustered[fy, fx]):
                    continue
                cells = []
                q2 = deque([(fx, fy)])
                clustered[fy, fx] = True
                while q2:
                    ex, ey = q2.popleft()
                    cells.append((ex, ey))
                    for dx, dy in DIRS4:
                        nx2, ny2 = ex + dx, ey + dy
                        if (0 <= nx2 < W and 0 <= ny2 < H
                                and frontier_map[ny2, nx2]
                                and not clustered[ny2, nx2]):
                            clustered[ny2, nx2] = True
                            q2.append((nx2, ny2))
                if len(cells) >= self.min_frontier:
                    cx = np.mean([c[0] for c in cells]) * res + ox
                    cy = np.mean([c[1] for c in cells]) * res + oy
                    centroids.append((float(cx), float(cy)))

        return centroids

    def _nearest_frontier(self, frontiers):
        return min(frontiers,
                   key=lambda f: math.hypot(
                       f[0] - self.robot_x, f[1] - self.robot_y))

    # ── Nav2 goal ──────────────────────────────────────────────────────────────
    def _send_goal(self, gx: float, gy: float):
        if not self._nav.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("Nav2 not available yet")
            return

        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id    = "map"
        pose.header.stamp       = self.get_clock().now().to_msg()
        pose.pose.position.x    = gx
        pose.pose.position.y    = gy
        pose.pose.position.z    = 0.0
        q = quaternion_from_euler(0, 0, 0)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        goal.pose = pose

        self.goal_active = True
        future = self._nav.send_goal_async(goal)
        future.add_done_callback(self._goal_accepted)

    def _goal_accepted(self, future):
        handle = future.result()
        if not handle.accepted:
            self.goal_active = False
            return
        handle.get_result_async().add_done_callback(self._goal_done)

    def _goal_done(self, future):
        self.goal_active = False
        self.get_logger().info("Frontier reached")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _load_known(self) -> set:
        p = Path(self.map_file)
        if p.exists():
            with open(p) as f:
                return set(json.load(f).keys())
        return set()


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
