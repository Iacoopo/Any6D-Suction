# Robotic Picking Demo

This directory contains the deploy-oriented version of the Isaac Sim + ROS2 + Any6D robotic picking demo.

## Scope

The demo is structured as a two-process system:

- an Isaac Sim process that builds the scene, captures observations, queries Any6D over ROS2, and executes the pick-and-place routine with a UR10;
- a Linux ROS2 process that runs Any6D pose estimation and returns a 6D pose prediction in the Isaac world frame.

The current implementation supports multi-view acquisition on the Isaac side and view selection based on a pose-derived surface suitability metric for suction picking.

## Directory Layout

- `isaac/ycb_pick_suction_v4.py`
  - Isaac Sim entrypoint for scene setup, multi-view capture, ROS2 request publishing, pose-driven target selection, and UR10 control.
- `ros2/ros2_any6d_pose_server.py`
  - ROS2 Any6D pose server entrypoint for Linux.
- `config/objects_config.json`
  - Object-specific demo configuration used by the Isaac script.

## Supported Objects

The demo configuration currently includes:

- `003_cracker_box`
- `006_mustard_bottle`

Additional objects should be added to `config/objects_config.json` together with the required object metadata.

## Runtime Architecture

### Isaac Sim Process

The Isaac process is responsible for:

- loading the UR10 and the target object;
- spawning distractors and camera viewpoints;
- capturing `rgb`, `depth`, `mask`, and camera intrinsics;
- publishing a ROS2 request containing:
  - `object_name`
  - `t_w_c`
  - `object_frame_correction`
- receiving Any6D pose responses;
- selecting the best view candidate;
- generating a suction pick target and executing the controller.

### Any6D ROS2 Process

The Linux Any6D process is responsible for:

- subscribing to observation topics;
- resolving the target object estimator;
- running `estimator.register(...)`;
- converting the predicted pose into the Isaac world frame;
- publishing either:
  - a successful response with `position` and `orientation_xyzw`, or
  - an error response with `status=error`.

## Prerequisites

### Linux Side

- ROS2 Humble installed and sourceable via `/opt/ros/humble/setup.bash`
- a working Any6D runtime environment
- access to the `cv_final` workspace
- GPU runtime compatible with the installed PyTorch / CUDA stack

### Isaac Side

- Isaac Sim installed on Windows or Linux
- access to the same `cv_final` workspace content, or an equivalent mirrored copy
- ROS2 bridge enabled through the Isaac Python runtime

### Networking

- Isaac Sim host and Linux Any6D host must share the same `ROS_DOMAIN_ID`
- DDS discovery must be allowed by the local network and firewall configuration

## Launch Procedure

The demo is intended to be launched manually from two terminals.

### 1. Start the Any6D ROS2 Server on Linux

Generic form:

```bash
source /opt/ros/humble/setup.bash
python3 /home/iacopo/cv_final/demo/ros2/ros2_any6d_pose_server.py \
  --cv-final-root /home/iacopo/cv_final \
  --object-name 006_mustard_bottle
```

Example for `003_cracker_box`:

```bash
source /opt/ros/humble/setup.bash
python3 /home/iacopo/cv_final/demo/ros2/ros2_any6d_pose_server.py \
  --cv-final-root /home/iacopo/cv_final \
  --object-name 003_cracker_box
```

### 2. Start Isaac Sim with the Isaac Python Interpreter

The Isaac command should always be expressed in the following form:

```text
<Isaac Python executable> <path to Isaac demo script> --object-name <object> --num-views <N> --distractor-count <M>
```

Windows example:

```powershell
C:\isaac-sim\python.bat C:\Users\tomin\cv_final\demo\isaac\ycb_pick_suction_v4.py --object-name 006_mustard_bottle --num-views 4 --distractor-count 100
```

Linux example:

```bash
/opt/isaac-sim/python.sh /home/iacopo/cv_final/demo/isaac/ycb_pick_suction_v4.py --object-name 006_mustard_bottle --num-views 4 --distractor-count 100
```

Example for `003_cracker_box`:

```bash
/opt/isaac-sim/python.sh /home/iacopo/cv_final/demo/isaac/ycb_pick_suction_v4.py --object-name 003_cracker_box --num-views 2 --distractor-count 10
```

## Expected Runtime Signals

If the system is configured correctly, the logs should show the following sequence.

### Isaac Sim

- `Simulation started. ROS2 Any6D bridge enabled. object=...`
- `Pubblicata richiesta Any6D request_id=...`
- per-view logs including:
  - camera pose
  - mask pixel count
  - pose error diagnostics
- a final selection log such as:
  - `Best-surface su ... view ...`

### Any6D Server

- runtime initialization
- estimator preload for the selected object
- successful responses:
  - `Published pose for request_id=...`
- or explicit error responses:
  - `status=error`

## Object Configuration

`config/objects_config.json` is currently the object metadata source used by the Isaac demo script.

Each object entry should define:

- `label`
- `prim_path`
- `frame_correction`
- `extents_m`

These fields are used for:

- semantic label matching in segmentation;
- target prim resolution inside Isaac;
- frame conversion between Any6D and Isaac;
- surface scoring for suction-oriented view selection.

## Deployment Notes

- The source of truth for the demo is `demo/`.
- The Isaac script is parameterized by object name, number of views, and distractor count.
- The ROS2 Any6D server is parameterized by object name and workspace root.
- The Linux ROS2 server resolves the workspace root relative to its own location unless overridden explicitly.

## Operational Notes

- The demo assumes a top-down suction picking policy.
- The final pick target is derived from the selected predicted world position and a vertical raycast.
- View selection is currently based on a surface suitability score computed from the predicted 6D pose, with mask support used as a secondary signal.

## Extending the Demo

To add a new object:

1. add the object entry to `config/objects_config.json`;
2. ensure the corresponding mesh and Any6D estimator assets exist in the workspace;
3. verify the object-frame correction used in the ROS2 request is correct;
4. validate spawn behavior and suction suitability for the new geometry.
