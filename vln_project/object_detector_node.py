#!/usr/bin/env python3
"""
object_detector_node.py
========================
ROS2 node. Subscribes to the TurtleBot3 camera, classifies every Nth frame
using the fine-tuned CLIP model, extracts dominant colour, and records the
robot's world coordinates at the moment of detection.

Subscribes:
  /camera/image_raw      sensor_msgs/Image
  /odom                  nav_msgs/Odometry

Publishes:
  /detections            std_msgs/String  (JSON per detection)

Writes:
  object_map.json        persistent dict of all known objects

object_map.json entry:
{
  "cafe_table": {
    "x": 0.9, "y": 1.4, "z": 0.0,
    "color": "brown",
    "confidence": 0.94,
    "near": []        <- filled later by vln_command_node
  }
}
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from cv_bridge import CvBridge

try:
    import open_clip
    USE_OPEN_CLIP = True
except ImportError:
    import clip
    USE_OPEN_CLIP = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Model (must match clip_finetune.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, embed_dim: int, num_classes: int):
        super().__init__()
        self.clip_model = clip_model
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256),       nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feats = self.clip_model.encode_image(x).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return self.head(feats)


# ─────────────────────────────────────────────────────────────────────────────
# Dominant colour detection (HSV ranges)
# ─────────────────────────────────────────────────────────────────────────────
_COLOR_RANGES = {
    "red":    (np.array([0,  100, 100], np.uint8), np.array([10, 255, 255], np.uint8)),
    "orange": (np.array([10, 100, 100], np.uint8), np.array([25, 255, 255], np.uint8)),
    "yellow": (np.array([25, 100, 100], np.uint8), np.array([35, 255, 255], np.uint8)),
    "green":  (np.array([35,  50,  50], np.uint8), np.array([85, 255, 255], np.uint8)),
    "blue":   (np.array([85,  50,  50], np.uint8), np.array([130,255, 255], np.uint8)),
    "brown":  (np.array([0,   40,  30], np.uint8), np.array([20, 200, 150], np.uint8)),
    "gray":   (np.array([0,    0,  60], np.uint8), np.array([180, 40, 200], np.uint8)),
    "black":  (np.array([0,    0,   0], np.uint8), np.array([180, 255, 60], np.uint8)),
    "white":  (np.array([0,    0, 200], np.uint8), np.array([180,  30, 255],np.uint8)),
}

def get_dominant_color(bgr: np.ndarray) -> str:
    small = cv2.resize(bgr, (64, 64))
    hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    best_name, best_count = "unknown", 0
    for name, (lo, hi) in _COLOR_RANGES.items():
        count = int(cv2.inRange(hsv, lo, hi).sum() // 255)
        if count > best_count:
            best_count, best_name = count, name
    return best_name


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 Node
# ─────────────────────────────────────────────────────────────────────────────
class ObjectDetectorNode(Node):

    def __init__(self):
        super().__init__("object_detector")

        # Parameters
        self.declare_parameter("checkpoint_dir",  "checkpoints")
        self.declare_parameter("map_file",        "/tmp/object_map.json")
        self.declare_parameter("conf_threshold",  0.70)
        self.declare_parameter("detect_every_n",  3)

        ckpt_dir        = self.get_parameter("checkpoint_dir").value
        self.map_file   = self.get_parameter("map_file").value
        self.conf_thr   = self.get_parameter("conf_threshold").value
        self.every_n    = self.get_parameter("detect_every_n").value

        # Load model
        self.get_logger().info(f"Loading model from: {ckpt_dir}")
        meta_path = os.path.join(ckpt_dir, "meta.json")
        if not os.path.exists(meta_path):
            self.get_logger().error(f"meta.json not found at {meta_path}")
            self.get_logger().error("Run clip_finetune.py first!")
            raise RuntimeError("Model not found")

        with open(meta_path) as f:
            meta = json.load(f)
        self.classes   = meta["classes"]
        self.embed_dim = meta["embed_dim"]

        if USE_OPEN_CLIP:
            clip_base, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai")
        else:
            clip_base, self.preprocess = clip.load("ViT-B/32", device=DEVICE)
        clip_base = clip_base.to(DEVICE)

        self.model = CLIPClassifier(
            clip_base, self.embed_dim, len(self.classes)).to(DEVICE)
        self.model.load_state_dict(
            torch.load(os.path.join(ckpt_dir, "clip_finetuned.pt"),
                       map_location=DEVICE))
        self.model.eval()
        self.get_logger().info(
            f"Model ready | classes: {self.classes}")

        # State
        self.bridge  = CvBridge()
        self.frame_n = 0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_z = 0.0
        self.obj_map = self._load_map()

        # Subscriptions
        self.create_subscription(
            Image, "/camera/image_raw", self._image_cb, 10)
        self.create_subscription(
            Odometry, "/odom", self._odom_cb, 10)

        # Publisher
        self.det_pub = self.create_publisher(String, "/detections", 10)

        self.get_logger().info("ObjectDetectorNode ready")

    # ── Odom callback ─────────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_z = msg.pose.pose.position.z

    # ── Image callback ────────────────────────────────────────────────────────
    def _image_cb(self, msg: Image):
        self.frame_n += 1
        if self.frame_n % self.every_n != 0:
            return

        # Convert ROS image → numpy (BGR) + PIL (RGB)
        bgr     = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        pil_img = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        # Classify
        with torch.no_grad():
            tensor = self.preprocess(pil_img).unsqueeze(0).to(DEVICE)
            probs  = torch.softmax(self.model(tensor), dim=-1).squeeze(0).cpu().numpy()

        best_idx  = int(probs.argmax())
        best_conf = float(probs[best_idx])
        best_cls  = self.classes[best_idx]

        if best_conf < self.conf_thr:
            return

        color = get_dominant_color(bgr)

        detection = {
            "object":     best_cls,
            "confidence": round(best_conf, 4),
            "color":      color,
            "x":          round(self.robot_x, 3),
            "y":          round(self.robot_y, 3),
            "z":          round(self.robot_z, 3),
        }

        # Update map — keep highest-confidence sighting per class
        prev_conf = self.obj_map.get(best_cls, {}).get("confidence", 0.0)
        if best_conf > prev_conf:
            self.obj_map[best_cls] = {
                "x":          round(self.robot_x, 3),
                "y":          round(self.robot_y, 3),
                "z":          round(self.robot_z, 3),
                "color":      color,
                "confidence": round(best_conf, 4),
                "near":       [],
            }
            self._save_map()
            self.get_logger().info(
                f"MAP UPDATE: {best_cls} @ "
                f"({self.robot_x:.2f}, {self.robot_y:.2f})  "
                f"color={color}  conf={best_conf:.3f}")

        # Publish detection event
        self.det_pub.publish(String(data=json.dumps(detection)))

    # ── Map persistence ───────────────────────────────────────────────────────
    def _load_map(self) -> dict:
        p = Path(self.map_file)
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            self.get_logger().info(
                f"Loaded existing map: {len(data)} objects")
            return data
        return {}

    def _save_map(self):
        with open(self.map_file, "w") as f:
            json.dump(self.obj_map, f, indent=2)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectorNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
