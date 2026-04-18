#!/usr/bin/env python3
"""
vln_command_node.py
====================
ROS2 node. Parses natural language navigation commands and drives the
TurtleBot3 waffle to the target object using Nav2.

Supported command patterns:
  Simple:   "go to cafe_table"
            "navigate to the trash can"
            "find the cabinet"

  Spatial:  "go to table near trash can"
            "go to table next to the cabinet"
            "navigate to table beside cafe_table"

Pipeline:
  1. spaCy dependency parse → extract (target, spatial_relation, anchor)
  2. Look up object_map.json for all known positions
  3. If spatial relation: pick the target instance closest to anchor
  4. Compute spatial "near" relations between all objects in map
  5. A* path plan on /map occupancy grid
  6. Send NavigateToPose goal to Nav2

Subscribes:
  /vln_command    std_msgs/String   (natural language)
  /map            nav_msgs/OccupancyGrid
  /odom           nav_msgs/Odometry

Publishes:
  /vln_status     std_msgs/String   (JSON status updates)

Usage:
  ros2 topic pub /vln_command std_msgs/String \
    "data: 'go to table near trash can'" --once
"""

import json
import math
import heapq
from collections import defaultdict
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler, euler_from_quaternion

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
    USE_SPACY = True
except Exception:
    USE_SPACY = False
    print("[vln_command] spaCy not available, using keyword fallback")

DEVICE_NONE = None  # Nav only, no ML inference here


# ─────────────────────────────────────────────────────────────────────────────
# Class → aliases  (edit this when you add new objects)
# ─────────────────────────────────────────────────────────────────────────────
ALIASES: dict[str, list[str]] = {
    "cafe_table":            ["cafe table", "cafe_table", "coffee table", "cafe"],
    "double_cabinet":        ["double cabinet", "double_cabinet", "big cabinet",
                              "large cabinet"],
    "first_2015_trash_can":  ["trash can", "trash_can", "trashcan", "bin",
                              "garbage", "trash", "waste bin", "dustbin"],
    "single_cabinet":        ["single cabinet", "single_cabinet",
                              "small cabinet", "little cabinet"],
    "table":                 ["table", "dining table", "desk", "work table"],
}

# Spatial keywords that indicate a relation
SPATIAL_WORDS = {
    "near", "next", "beside", "by", "close", "adjacent",
    "alongside", "nearby", "next to", "close to",
}


# ─────────────────────────────────────────────────────────────────────────────
# NLP parser
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_alias(phrase: str) -> str | None:
    """Match a phrase to a canonical class name. Returns None if no match."""
    phrase = phrase.lower().strip()
    # Try longest aliases first (so "double cabinet" beats "cabinet")
    candidates = []
    for cls, aliases in ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if alias in phrase:
                candidates.append((len(alias), cls))
                break
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def parse_command(text: str) -> dict:
    """
    Returns:
      {"target": str, "relation": str|None, "anchor": str|None}

    Examples:
      "go to table near trash can"
        → {"target": "table", "relation": "near",
           "anchor": "first_2015_trash_can"}

      "go to cafe_table"
        → {"target": "cafe_table", "relation": None, "anchor": None}
    """
    text_lower = text.lower().strip()
    result = {"target": None, "relation": None, "anchor": None}

    if USE_SPACY:
        doc = _NLP(text_lower)

        # Find spatial relation keyword
        relation = None
        rel_token_i = -1
        for i, tok in enumerate(doc):
            if tok.text in SPATIAL_WORDS or tok.lemma_ in SPATIAL_WORDS:
                relation = tok.text
                rel_token_i = i
                break
        # Also catch two-word phrases: "next to", "close to"
        tokens = [t.text for t in doc]
        for phrase in ["next to", "close to"]:
            words = phrase.split()
            for i in range(len(tokens) - 1):
                if tokens[i] == words[0] and tokens[i+1] == words[1]:
                    relation = phrase
                    rel_token_i = i
                    break

        result["relation"] = relation

        if relation and rel_token_i >= 0:
            # Text before relation keyword = target phrase
            before = " ".join(tokens[:rel_token_i])
            # Text after relation keyword = anchor phrase
            skip = 2 if relation in ["next to", "close to"] else 1
            after  = " ".join(tokens[rel_token_i + skip:])
            result["target"] = _resolve_alias(before)
            result["anchor"] = _resolve_alias(after)
        else:
            result["target"] = _resolve_alias(text_lower)

    else:
        # Keyword fallback (no spaCy)
        # Check for spatial keywords
        for sw in sorted(SPATIAL_WORDS, key=len, reverse=True):
            if sw in text_lower:
                parts = text_lower.split(sw, 1)
                result["target"]   = _resolve_alias(parts[0])
                result["anchor"]   = _resolve_alias(parts[1]) if len(parts) > 1 else None
                result["relation"] = sw
                break
        if result["target"] is None:
            result["target"] = _resolve_alias(text_lower)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# A* on occupancy grid
# ─────────────────────────────────────────────────────────────────────────────
def _astar(data: np.ndarray, W: int, H: int,
           start: tuple, goal: tuple, inflate: int = 3) -> list:
    """
    Returns list of (col, row) from start to goal, or [] if unreachable.
    inflate: number of cells to expand obstacles (robot safety radius)
    """
    # Inflate obstacles
    from scipy.ndimage import binary_dilation
    occ      = (data == 100)
    struct   = np.ones((2 * inflate + 1, 2 * inflate + 1), bool)
    inflated = binary_dilation(occ, struct)

    def h(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    open_set   = [(0.0, start)]
    came_from  = {}
    g_score    = defaultdict(lambda: float("inf"))
    g_score[start] = 0.0
    DIRS8 = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            return path[::-1]
        cx, cy = cur
        for dx, dy in DIRS8:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if inflated[ny, nx]:
                continue
            step_cost  = 1.414 if (dx and dy) else 1.0
            tentative_g = g_score[cur] + step_cost
            nb = (nx, ny)
            if tentative_g < g_score[nb]:
                g_score[nb]  = tentative_g
                came_from[nb] = cur
                heapq.heappush(open_set,
                               (tentative_g + h(nb, goal), nb))
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Spatial relation builder
# ─────────────────────────────────────────────────────────────────────────────
NEAR_THRESHOLD = 1.0   # metres — objects within this distance are "near"

def compute_near_relations(obj_map: dict) -> dict:
    """
    For every object in obj_map, compute which other objects are within
    NEAR_THRESHOLD metres. Stores result in obj["near"] list.
    Returns updated map.
    """
    names = list(obj_map.keys())
    for n in names:
        obj_map[n]["near"] = []
    for i, a in enumerate(names):
        for b in names[i+1:]:
            da = obj_map[a]
            db = obj_map[b]
            dist = math.hypot(da["x"] - db["x"], da["y"] - db["y"])
            if dist <= NEAR_THRESHOLD:
                obj_map[a]["near"].append(b)
                obj_map[b]["near"].append(a)
    return obj_map


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 Node
# ─────────────────────────────────────────────────────────────────────────────
class VLNCommandNode(Node):

    def __init__(self):
        super().__init__("vln_command")

        self.declare_parameter("map_file", "/tmp/object_map.json")
        self.map_file = self.get_parameter("map_file").value

        self.grid      = None
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0

        self._nav = ActionClient(self, NavigateToPose, "navigate_to_pose")

        self.create_subscription(String,        "/vln_command", self._cmd_cb,  10)
        self.create_subscription(OccupancyGrid, "/map",         self._map_cb,  10)
        self.create_subscription(Odometry,      "/odom",        self._odom_cb, 10)
        self.status_pub = self.create_publisher(String, "/vln_status", 10)

        self.get_logger().info(
            "VLNCommandNode ready.\n"
            "  Publish commands to /vln_command\n"
            "  Examples:\n"
            "    'go to cafe_table'\n"
            "    'go to table near trash can'")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _map_cb(self, msg): self.grid = msg

    def _odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.robot_yaw = euler_from_quaternion(
            [q.x, q.y, q.z, q.w])

    def _cmd_cb(self, msg: String):
        command = msg.data.strip()
        self.get_logger().info(f"Command: '{command}'")
        self._publish("received", command=command)

        # 1. Parse
        parsed = parse_command(command)
        self.get_logger().info(f"Parsed: {parsed}")

        target   = parsed["target"]
        relation = parsed["relation"]
        anchor   = parsed["anchor"]

        if target is None:
            self._publish("error",
                          detail=f"Could not identify object in: '{command}'")
            return

        # 2. Load map
        obj_map = self._load_map()
        if not obj_map:
            self._publish("error",
                          detail="object_map.json is empty. Run exploration first.")
            return

        # 3. Compute / refresh near relations
        obj_map = compute_near_relations(obj_map)
        self._save_map(obj_map)

        # 4. Resolve target
        if target not in obj_map:
            self._publish("error",
                          detail=f"'{target}' not in map yet. Run exploration.")
            return

        goal_entry = obj_map[target]

        # 5. If spatial relation given, validate anchor is near target
        if relation and anchor:
            if anchor not in obj_map:
                self._publish("error",
                              detail=f"Anchor '{anchor}' not in map.")
                return
            anchor_entry = obj_map[anchor]
            dist = math.hypot(
                goal_entry["x"] - anchor_entry["x"],
                goal_entry["y"] - anchor_entry["y"])
            if dist > NEAR_THRESHOLD * 2:
                self.get_logger().warn(
                    f"'{target}' is {dist:.2f}m from '{anchor}' "
                    f"(threshold {NEAR_THRESHOLD}m) — navigating anyway.")
            self.get_logger().info(
                f"Spatial: {target} is {dist:.2f}m from {anchor}")

        gx = goal_entry["x"]
        gy = goal_entry["y"]
        color = goal_entry.get("color", "unknown")

        self.get_logger().info(
            f"Navigating to '{target}' @ ({gx:.2f}, {gy:.2f})  color={color}")
        self._publish("navigating",
                      target=target, goal_x=gx, goal_y=gy,
                      color=color, relation=relation, anchor=anchor)

        # 6. A* plan (informational — Nav2 does actual driving)
        if self.grid is not None:
            path = self._plan_path(gx, gy)
            self.get_logger().info(f"A* path: {len(path)} cells")
            self._publish("path_planned",
                          target=target, waypoints=len(path),
                          goal_x=gx, goal_y=gy)

        # 7. Navigate
        self._navigate(gx, gy, target)

    # ── A* ────────────────────────────────────────────────────────────────────
    def _plan_path(self, gx: float, gy: float) -> list:
        g   = self.grid
        W   = g.info.width
        H   = g.info.height
        res = g.info.resolution
        ox  = g.info.origin.position.x
        oy  = g.info.origin.position.y
        data = np.array(g.data, dtype=np.int8).reshape(H, W)

        def w2c(wx, wy):
            col = int((wx - ox) / res)
            row = int((wy - oy) / res)
            return (max(0, min(W-1, col)), max(0, min(H-1, row)))

        try:
            return _astar(data, W, H,
                          w2c(self.robot_x, self.robot_y),
                          w2c(gx, gy))
        except Exception as e:
            self.get_logger().warn(f"A* failed: {e}")
            return []

    # ── Nav2 ──────────────────────────────────────────────────────────────────
    def _navigate(self, gx: float, gy: float, target_name: str):
        if not self._nav.wait_for_server(timeout_sec=5.0):
            self._publish("error", detail="Nav2 action server not available")
            return

        goal_msg  = NavigateToPose.Goal()
        pose      = PoseStamped()
        pose.header.frame_id    = "map"
        pose.header.stamp       = self.get_clock().now().to_msg()
        pose.pose.position.x    = gx
        pose.pose.position.y    = gy
        pose.pose.position.z    = 0.0
        angle = math.atan2(gy - self.robot_y, gx - self.robot_x)
        q = quaternion_from_euler(0, 0, angle)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        goal_msg.pose = pose

        future = self._nav.send_goal_async(goal_msg)
        future.add_done_callback(
            lambda f: self._nav_accepted(f, target_name, gx, gy))

    def _nav_accepted(self, future, name, gx, gy):
        handle = future.result()
        if not handle.accepted:
            self._publish("error", detail="Nav2 rejected goal")
            return
        handle.get_result_async().add_done_callback(
            lambda f: self._nav_done(f, name, gx, gy))

    def _nav_done(self, future, name, gx, gy):
        self.get_logger().info(f"Arrived at '{name}'")
        self._publish("arrived", target=name, goal_x=gx, goal_y=gy)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _load_map(self) -> dict:
        p = Path(self.map_file)
        if not p.exists():
            return {}
        with open(p) as f:
            return json.load(f)

    def _save_map(self, obj_map: dict):
        with open(self.map_file, "w") as f:
            json.dump(obj_map, f, indent=2)

    def _publish(self, status: str, **kwargs):
        payload = {"status": status, **kwargs}
        self.status_pub.publish(String(data=json.dumps(payload)))
        self.get_logger().info(f"Status → {payload}")


def main(args=None):
    rclpy.init(args=args)
    node = VLNCommandNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
