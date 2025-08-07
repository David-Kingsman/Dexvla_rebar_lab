# ARM CONTROL BASE: UFACTORY TELEOPERATION

**Experiments for Ufactory Xarm Robots**

## Overview

This repo contains code, datasets, and instructions to support the following use cases:
- Collecting Demonstration Data
- Data-post-precess preparation for training process


## Installation

```bash
conda create -n ufact python=3.9
conda activate ufact
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r ./requirements.txt
```

Install ufactory API, and download  from source code [XArm Python SDK](https://github.com/xArm-Developer/xArm-Python-SDK/tree/a54b2fd1922d3245f1f78c8e518871d4760ead1c)

```bash
git clone https://github.com/xArm-Developer/xArm-Python-SDK.git
cd xArm-Python-SDK
# Install according to official instructions
Install from pypi
```


## Teleoperation using Meta quest 2
### Installation

Headset should be plugged into PC, and the permissions prompt "USB debugging" should be accepted in the system of headset. Also, the controller has to be in view of the headset cameras. It detects the pose of the handle via infrared stickers on the handle. Teleoperation works by applying the changes to the oculus handle’s pose to the robot gripper’s pose.

### Core code

This repo supports Meta Quest 2 as the teleoperation device. To start, install [oculus_reader](https://github.com/rail-berkeley/oculus_reader/blob/main/oculus_reader/reader.py) according to the official instructions

["meta_quest2.py"](io_devices/meta_quest2.py)

### Operation instruction:
* Move: **Move controller**
* Grasp: **Right trigger**
* Over: **A**


## Project Structure and Pipeline Overview

| Directory / File         | Description |
| :---------------------: |:-----------------------------:|
| data/utils/             | Utility scripts for data conversion, statistics, and post-processing |
| data/                   | Raw and processed data storage, including collected demos and images|
| configs/                | Configuration files for data collection, observation, and policy parameters|
| scripts/                | Scripts for data collection and hardware control (e.g., camera nodes, data collection) |
| visualize_episodes_rebar.py | Visualization tools for episodes and training results |
| arm_control_base/       | Core modules for robot arm control and hardware interface |
| utils.py                | General utility functions |
| io_devices/             | Device interface modules (e.g., Meta Quest 2, spacemouse, virtual controller, Oculus reader) |

This table summarizes the main components and data flow in the repository. For details on each module, refer to the corresponding scripts and configuration files.

Currently available configurations:
All collection and post-processing configurations are in the `configs/` directory. Common configurations include:
- `data_collection.yaml`: data collection parameters
- `real_robot_observation_cfg.yml`: observation configuration


## Collecting demos

**Recommendation: `camera_node.py` and `data_collection_metaquest.py` need to be run in different terminals simultaneously. **

### Reset robot pose if need

```bash
 python scripts/reset_robot.py 
```

### Open cameras needed, such as

```bash
# check cameras path
v4l2-ctl --list-devices

# 2. Verify camera sequence. Start with the left web camera (LHS -- Webcam 1); Then proceed to the right web camera (RHS -- Webcam 2)
# Launch Left Webcam camera node
python scripts/camera_node.py --camera-ref webcam_1 --use-rgb --visualization --img-h 1080 --img-w 1920 --fps 30 --camera-address '/dev/video2'

# Launch Right Webcam camera node
python scripts/camera_node.py --camera-ref webcam_2 --use-rgb --visualization --img-h 1080 --img-w 1920 --fps 30 --camera-address '/dev/video0'  
```

** if you want to use RGB and depth data from RealSense camera, you can run the following command:**

```bash
# RealSense camera example with RGB and depth
python scripts/camera_node.py --camera-ref rs_1 --use-rgb --use-depth --visualization --img-h 480 --img-w 640 --fps 30
```

### Start collecting data with Meta Quest 2

```bash
# to collect data with Ufactory arms and Robotiq 2F85
python scripts/data_collection_metaquest.py
```

### Memory Monitoring and Optimization Tips

When collecting large amounts of images and sensor data, memory usage can increase rapidly, especially if `save2memory_first=True`. To avoid running out of memory, consider the following tips:

- **Monitor memory usage in real time**  
  
  Use the `htop` tool to monitor real-time memory usage: `htop`

- **Free system cache**  
  
  If you notice high memory usage, you can manually free the Linux system cache (note: this does **not** free memory used by the Python process itself, only the filesystem cache):

  ```bash
  sudo sync
  sudo sysctl -w vm.drop_caches=3
  ```

  Here, `sync` writes all in-memory data to disk, and `vm.drop_caches=3` frees all caches.

- **Optimization suggestions**  
  
  - For large datasets, it is recommended to set `save2memory_first` to `False` so that images are written directly to disk, reducing memory pressure.
  - Monitor memory regularly and free cache if necessary.
  - The fundamental solution is to optimize the data collection and saving logic to avoid keeping too much data in memory at once.

- **Clean up node processes**  
  
  To forcefully terminate camera node processes, run:

  ```bash
  pkill -9 -f scripts/camera_node.py
  ```

## Collecting Demonstration Data format

### Data Folder Structure

Collected data is stored in the `demos_collected` folder, with each run in a separate subfolder named `runXYZ`, where `XYZ` is the run number. Each run folder contains:

``` yaml
demos_collected/
├── images/                                   # All collected images are stored here
│   ├── webcam_webcam_1_xxxxxxxxxxxxx/        # Images from camera 1, folder named with timestamp
│   │   ├── color_000000xxx.jpg               # Single color image (incremental index)
│   │   └── ...                               # More images
│   └── webcam_webcam_2_xxxxxxxxxxxxx/        # Images from camera 2, folder named with timestamp
│       ├── color_000000xxx.jpg
│       └── ...
├── run001/                                   # Data folder for the 1st collection run
│   ├── config.json                           # Configuration for this run (controller type, observation config, etc.)
│   ├── demo_action.npz                       # Action sequence: [x, y, z, rx, ry, rz, grasp] per step
│   ├── demo_action_grasp.npz                 # Gripper action per step (open/close, usually -1 or 1)
│   ├── demo_action_hot.npz                   # Action hotkey/flag per step
│   ├── demo_camera_1.npz                     # Camera 1 image info and metadata (image path, timestamp, etc.)
│   ├── demo_camera_2.npz                     # Camera 2 image info and metadata
│   ├── demo_ee_states.npz                    # End-effector states per step
│   ├── demo_FT_processed.npz                 # (Optional) Processed force/torque sensor data per step
│   ├── demo_FT_raw.npz                       # (Optional) Raw force/torque sensor data per step
│   ├── demo_gripper_states.npz               # Gripper state per step (open/close)
│   └── demo_joint_states.npz                 # Robot joint states per step
│ 
├── runXYZ/                                   # Data folder for following runs
└── ...                                       # More collection runs (run003, run004, etc.)
```              

## Data Preparation

The collected raw data has already been converted to [hdf5](https://www.hdfgroup.org/solutions/hdf5/) to store the data for training. For related content, please refer to the [arm_control_base](../arm_control_base). Change the config in [`Convert_to_hdf5.py`](./data/utils/Convert_to_hdf5.py) to match your data collection setup and execute:

```bash
# you may need to modify the details in the below script
python data/utils/Convert_to_hdf5.py
# For specific trim purpose with ['data/trim_config.json'](./data/utils/Convert_to_hdf5._trim.py)
python data/utils/Convert_to_hdf5_trim.py
```

After postprocessing, you may see the following structure:

```yaml
├── action (300, 7) float32 # Action data for each step, [x, y, z, rx, ry, rz, gripper]
├── action_hot (300, 1) float32 # One-hot or other tags of the action (if any, may not be actually written)
├── target (1,) float32 # Target information (such as target number or position)
└── observations
    ├── FT_raw (300, 6) float32 # Raw force/torque sensor data [fx, fy, fz, tx, ty, tz]
    ├── FT_processed (300, 6) float32 # Processed force/torque data
    ├── qpos (300, 7) float32 # End effector pose and gripper status [x, y, z, rx, ry, rz, gripper]
    └── images
        ├── webcam_1 (300, 700, 520, 3) uint8 # RGB image sequence of camera 1, (number of frames, height, width, channels)
        └── webcam_2 (300, 700, 520, 3) uint8 # RGB image sequence of camera 2, (number of frames, height, width, channels)
        └── ...
```

> NOTE: The [hdf5](https://www.hdfgroup.org/solutions/hdf5/) dataset format is compatible with the UFACTORY XARM diffusion policy training pipeline and includes all necessary metadata for normalization and sequence sampling.        

### To visualize HDF5's state-action pair and visual observations 
```bash
# Example
python visualize_episodes_rebar.py --dataset_dir data/toytest1_FT_hdf5/1_insert --episode_idx 1 --video # Change dataset_dir to your own path
# Example
python visualize_episodes_rebar.py --dataset_dir data/toytest1_FT_hdf5/1_insert --video
```

## Common Problems

- **Memory overflow**: It is recommended to turn off `save2memory_first` and release the cache regularly

- **Camera node stuck**: Use `pkill -9 -f scripts/camera_node.py` to force shutdown
  python scripts/camera_node.py --camera-ref webcam_1 --use-rgb --visualization --img-h 1080 --img-w 1920 --fps 30 --camera-address '/dev/video2'

- **Data post-processing error**: Check the `demos_collected` structure and configuration file parameters


# Acknowledgement
Our repo was built on: 
- [Deoxys](https://github.com/UT-Austin-RPL/deoxys_control) 
- [Deoxys vision](https://github.com/UT-Austin-RPL/deoxys_vision), 
- [GRoot](https://github.com/NVIDIA/Isaac-GR00T)
- [oculus_reader](https://github.com/rail-berkeley/oculus_reader)