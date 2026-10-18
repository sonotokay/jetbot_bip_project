# JetBot BIP Project

Autonomous JetBot (Jetson Nano, ROS 1) that drives by **behavioural cloning**: a CNN predicts steering from the camera image, and a laptop runs YOLOv8 to detect people and trigger an alarm on the robot. Optional modules add LiDAR obstacle avoidance and Kalman-filtered IMU drift correction (not used in the final demo).

## Architecture

```
Camera ─► cnn_steering ──/steering_cmd──┐
   │                                     ├─► motor_fusion ──/cmd_vel──► motors
   │      LiDAR ─► lidar_avoidance ──────┤   optional: /lidar_override, /lidar_cmd_vel, /speed_multiplier
   │      IMU ─► kalman_filter ──────────┘   optional: /kalman_imu
   │
   └─(HTTP, 5 Hz)─► laptop server.py (YOLOv8n) ─(person)─► alarm_server.py on Jetson ─► alarm.wav
```

| Component | File | Role |
|---|---|---|
| CNN steering | [nodes/cnn_steering_node.py](nodes/cnn_steering_node.py) | Loads `models/checkpoints/best_model.pth`, predicts steering in [-1, 1], publishes `/steering_cmd` and `/cnn_ready`; sends frames to the laptop detector |
| LiDAR avoidance *(optional)* | [nodes/lidar_backup_avoidance_node.py](nodes/lidar_backup_avoidance_node.py) | State machine (NORMAL → BACKING_UP → ADJUSTING → WAITING) when an obstacle is closer than 0.3 m |
| Motor fusion | [nodes/motor_fusion_node.py](nodes/motor_fusion_node.py) | Converts CNN steering into `/cmd_vel` once the CNN is ready; applies LiDAR override, speed multiplier and IMU yaw correction when those modules run |
| Kalman filter *(optional)* | [kf-mpu9250/filter.py](kf-mpu9250/filter.py) | Filters MPU9250 IMU data (`/imu` → `/kalman_imu`) |
| Data collector | [nodes/data_collector_node.py](nodes/data_collector_node.py) | Records images + gamepad steering/throttle + IMU ω at 10 Hz into `data/training/` with `metadata.json` |
| Detection server | [server.py](server.py) | FastAPI + YOLOv8n on the laptop; `POST /detect` returns person/bottle boxes and calls the Jetson alarm on a person |
| Alarm server | [alarm_server.py](alarm_server.py) | FastAPI on the Jetson (port 9000); `POST /alarm` plays [ressources/alarm.wav](ressources/alarm.wav) with a cooldown, `GET /health` |
| Model | [core/model.py](core/model.py) | `SteeringCNN` (NVIDIA PilotNet-style, 5 conv + 4 FC, tanh output) and `LightweightSteeringCNN` |
| Config | [core/config.py](core/config.py) | Topics, image size (224×224), training hyperparameters, speeds, LiDAR thresholds, node rates, `USE_AUGMENTATION`, `LAPTOP_IP` / `JETSON_IP` (overridable via env) |

## Setup

**Robot (Jetson Nano):** ROS 1 with the `jetbot_pro` package (motors, CSI camera, LiDAR). Place this repo at `~/jetbot_project` as the `jetbot_project` package, then `pip install -r requirements-robot.txt` (PyTorch from NVIDIA's JetPack wheel). Set `LAPTOP_IP` and `JETSON_IP` as environment variables or in [core/config.py](core/config.py). For training on a PC, use `requirements-train.txt`.

**Laptop:** run the detector natively or with Docker:

```bash
pip install -r requirements-server.txt
JETSON_IP=<jetson-ip> uvicorn server:app --host 0.0.0.0 --port 8000
# or
docker build -t jetbot-yolo . && docker run -p 8000:8000 -e JETSON_IP=<jetson-ip> jetbot-yolo
```

## Usage

**1. Collect data** — drive manually with a gamepad while recording:

```bash
roslaunch jetbot_project base.launch        # motors, camera, joystick teleop
rosrun jetbot_project data_collector_node.py
```

**2. Clean and prepare data** ([scripts/](scripts/)):

| Script | Purpose |
|---|---|
| `merge_datasets.py <out> <src1> <src2> ...` | Merge recording runs into one dataset |
| `delete_by_timestamps.py <dataset> -r START END [-r ...]` | Remove bad frames (e.g. hand corrections) by HHMMSS ranges |
| `images_to_video.py <dataset> [out.mp4] [fps]` | Render a dataset to video with steering overlay for inspection |
| `generate_mock_data.py` | Create a synthetic dataset to test the pipeline |
| `grayscale_viewer.py` | Flask stream (port 5001) of the camera as preprocessed for the CNN |
| `merge_images.py` | Combine training-curve plots side by side |
| `read_battery_status.py --pid <PID>` | Read battery voltage by tracing the JetBot driver's serial reads (`strace`) |

**3. Train** (offline, GPU recommended) — uses `DATA_DIR` and the hyperparameters in `core/config.py` (MSE loss, Adam, LR-on-plateau, 50 epochs, 20 % random validation split). Data augmentation (flip, brightness, shadow) is implemented but disabled via `USE_AUGMENTATION`: with the small datasets and short training runs it hurt performance.

```bash
python scripts/train.py
```

Outputs `best_model.pth`, `final_model.pth`, checkpoints every 10 epochs and `training_history.png` in `models/checkpoints/`, which is where the steering node loads the model from. Datasets and trained weights are not included in this repository.

**4. Drive autonomously:**

```bash
python3 alarm_server.py                                  # on the Jetson
roslaunch jetbot_project autonomous.launch use_imu:=false # base + LiDAR + CNN + avoidance + fusion (use_imu:=true adds the Kalman filter)
```

## Repository layout

```
core/            model and configuration
nodes/           ROS nodes
kf-mpu9250/      IMU Kalman filter node
launch/          base, autonomous and kalman_filter launch files
scripts/         training and dataset utilities
ressources/      alarm sound
```

See [PORTFOLIO.md](PORTFOLIO.md) for diagrams and a longer overview.

## License

[MIT](LICENSE)
