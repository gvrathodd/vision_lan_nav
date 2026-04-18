# VLN Project — Complete Setup Guide
# TurtleBot3 Waffle + ROS2 Humble + turtlebot3_home_service_challenge

---

## Folder structure (final result)

```
~/ros2_ws/src/vln_project/
├── vln_project/
│   ├── __init__.py
│   ├── clip_finetune.py            ← run offline (Step 1)
│   ├── object_detector_node.py     ← ROS2 node
│   ├── frontier_explorer_node.py   ← ROS2 node
│   └── vln_command_node.py         ← ROS2 node
├── launch/
│   └── vln_bringup.launch.py
├── config/
│   ├── slam_params.yaml
│   └── nav2_params.yaml
├── worlds/
│   └── turtlebot3_home_service_challenge.world   ← copy yours here
├── checkpoints/                    ← created after Step 1
│   ├── clip_finetuned.pt
│   └── meta.json
├── resource/
│   └── vln_project
├── package.xml
└── setup.py
```

---

## STEP 0 — Install Python dependencies (once)

```bash
pip install open-clip-torch torch torchvision Pillow numpy opencv-python scipy
pip install spacy
python -m spacy download en_core_web_sm
```

---

## STEP 1 — Copy your world file

```bash
cp /path/to/turtlebot3_home_service_challenge.world \
   ~/ros2_ws/src/vln_project/worlds/
```

---

## STEP 2 — Fine-tune CLIP on your images (run once, offline)

```bash
cd ~/ros2_ws/src/vln_project

python vln_project/clip_finetune.py \
  --data_dir /path/to/dataset/sample_images \
  --output_dir checkpoints/ \
  --epochs 30

# Expected output:
#   checkpoints/clip_finetuned.pt
#   checkpoints/meta.json
#   Best val accuracy: 0.85+
```

Your dataset folder must look exactly like:
```
sample_images/
  cafe_table/           ← jpg/png photos
  double_cabinet/
  first_2015_trash_can/
  single_cabinet/
  table/
```

---

## STEP 3 — Build the ROS2 package

```bash
cd ~/ros2_ws
colcon build --packages-select vln_project --symlink-install
source install/setup.bash
```

Add to your ~/.bashrc:
```bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
echo "export TURTLEBOT3_MODEL=waffle" >> ~/.bashrc
source ~/.bashrc
```

---

## STEP 4 — Launch everything

```bash
ros2 launch vln_project vln_bringup.launch.py \
  checkpoint_dir:=$HOME/ros2_ws/src/vln_project/checkpoints
```

This starts (in order):
1. Gazebo + turtlebot3_home_service_challenge.world
2. TurtleBot3 waffle spawned at origin (0, 0)
3. SLAM Toolbox → builds /map topic as robot explores
4. Nav2 → handles all path planning + motion
5. RViz2 → shows map, robot, frontiers
6. object_detector_node (after 6s)
7. frontier_explorer_node (after 7s)
8. vln_command_node (after 7s)

---

## STEP 5 — Watch the robot explore

The frontier_explorer_node automatically drives the robot.
In the terminal you will see:

```
[object_detector]: MAP UPDATE: cafe_table @ (0.88, 1.41) color=brown conf=0.943
[object_detector]: MAP UPDATE: first_2015_trash_can @ (1.38, 0.01) color=black conf=0.971
[frontier_explorer]: Found: cafe_table  (1/5)
[frontier_explorer]: Found: first_2015_trash_can  (2/5)
...
[frontier_explorer]: All objects found! Exploration complete.
```

Check object_map.json at any time:
```bash
cat /tmp/object_map.json
```

---

## STEP 6 — Send navigation commands

Open a new terminal:
```bash
source ~/ros2_ws/install/setup.bash

# Simple command
ros2 topic pub /vln_command std_msgs/String \
  "data: 'go to cafe_table'" --once

# Spatial command
ros2 topic pub /vln_command std_msgs/String \
  "data: 'go to table near trash can'" --once

# More examples
ros2 topic pub /vln_command std_msgs/String \
  "data: 'navigate to double cabinet'" --once

ros2 topic pub /vln_command std_msgs/String \
  "data: 'find the table next to the cafe table'" --once

# Monitor status
ros2 topic echo /vln_status
```

Expected status output:
```json
{"status": "navigating", "target": "cafe_table", "goal_x": 0.9, "goal_y": 1.4, "color": "brown"}
{"status": "path_planned", "target": "cafe_table", "waypoints": 72}
{"status": "arrived", "target": "cafe_table", "goal_x": 0.9, "goal_y": 1.4}
```

---

## How to add new objects (e.g. window, chair)

1. Collect photos → put in `sample_images/window/`
2. In `clip_finetune.py` → add `"window"` to `CLASSES` list
3. In `vln_command_node.py` → add to `ALIASES` dict:
   ```python
   "window": ["window", "glass window", "the window"],
   ```
4. Retrain:
   ```bash
   python vln_project/clip_finetune.py \
     --data_dir sample_images/ --output_dir checkpoints/ --epochs 30
   ```
5. Rebuild:
   ```bash
   colcon build --packages-select vln_project
   ```

---

## Supported natural language commands

| You say | Navigates to |
|---|---|
| "go to cafe_table" | cafe_table |
| "go to the trash can" | first_2015_trash_can |
| "go to trash" | first_2015_trash_can |
| "navigate to double cabinet" | double_cabinet |
| "find the cabinet" | double_cabinet |
| "go to table near trash can" | table (closest to trash can) |
| "go to table next to cafe table" | table (closest to cafe_table) |

---

## Troubleshooting

**"meta.json not found"**
→ Run Step 2 (clip_finetune.py) first.

**"not in map yet"**
→ Object not seen during exploration. Let exploration run longer, or
  manually teleop the robot near that object:
  `ros2 run teleop_twist_keyboard teleop_twist_keyboard`

**Nav2 goal rejected**
→ Goal is in an obstacle. Check RViz2 — the goal marker should be in white area.

**Low classification accuracy**
→ Add more photos (aim for 30-100 per class), especially in different lighting.

**Gazebo fails to load world**
→ Make sure turtlebot3_home_service_challenge.world is in worlds/ folder.
   Also check turtlebot3_home_service_challenge package is installed:
   `sudo apt install ros-humble-turtlebot3-simulations`
