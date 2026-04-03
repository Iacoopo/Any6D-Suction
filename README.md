# Any6D-Suction

This repository packages three components under a single project root:

- a modified copy of the upstream `Any6D` codebase;
- a deploy-oriented Isaac Sim + ROS2 robotic picking demo;
- experiment and analysis code used to evaluate Any6D under task-relevant conditions.

The project extends the original Any6D pipeline toward suction-based robotic picking in cluttered scenes. The demo uses Isaac Sim for scene construction and robot control, and a Linux ROS2 node for Any6D inference.

## Repository Layout

- `Any6D/`
  - modified upstream Any6D codebase and third-party components (`foundationpose`, `sam2`, `instantmesh`, `bop_toolkit`)
- `demo/`
  - robotic demo assets and entrypoints
- `experiments/`
  - evaluation and metric scripts
- `datasets/`
  - local datasets, generated outputs, and experiment assets
- `data_generation/`
  - utilities and assets used for synthetic data generation

## Scope of This README

This document explains how to:

1. create a working Conda environment comparable to the locally tested `any6d_blackwell` environment;
2. build the project extensions;
3. install or download the required checkpoints;
4. understand the ROS2 dependency used by the demo server;
5. run the manual launch flow for the robotic demo.

The setup below is based on:

- the upstream Any6D installation instructions in `Any6D/README.md`;
- the FoundationPose installation notes in `Any6D/foundationpose/readme.md`;
- the official Isaac Sim 5.1.0 documentation: <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html>;
- the project-specific environment currently used for the mixed-platform demo.

## Tested Environment

The current local demo environment was validated with the following core runtime:

- Python `3.10.20`
- PyTorch `2.7.0+cu128`
- CUDA runtime reported by PyTorch: `12.8`
- `nvdiffrast` import successful
- `pytorch3d` import successful

Important note:

- the upstream Any6D README targets Python `3.9` with PyTorch `2.4.1+cu121`;
- the local `any6d_blackwell` environment is a newer adaptation for a newer-class GPU;
- therefore, the setup documented here should be treated as the **project-tested environment**, not as a byte-for-byte copy of the upstream instructions.

## System Prerequisites

On Ubuntu, install the usual build and runtime dependencies before creating the Conda environment:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  ninja-build \
  git \
  wget \
  curl \
  unzip \
  ffmpeg \
  libgl1 \
  libglib2.0-0
```

For the demo server, ROS2 Humble must also be installed on the Linux machine.

For Isaac-side installation, Python runtime usage, and ROS2 integration, refer to the official Isaac Sim 5.1.0 documentation:

- Isaac Sim 5.1.0 documentation: <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html>
- Installation section: <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/index.html>
- Python environment installation: <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html>
- ROS 2 installation and bridge documentation: <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_ros.html>
- ROS 2 tutorials: <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/index.html>

## Conda Environment Setup

### 1. Create and activate the environment

```bash
conda create -n any6d_blackwell python=3.10 -y
conda activate any6d_blackwell
python -m pip install --upgrade pip
```

### 2. Install Eigen inside the Conda environment

This follows the logic used by the upstream Any6D / FoundationPose instructions:

```bash
conda install -c conda-forge eigen=3.4.0 -y
export CMAKE_PREFIX_PATH="$CONDA_PREFIX:$CMAKE_PREFIX_PATH"
```

### 3. Install the PyTorch stack

For the Blackwell-oriented environment currently used in this project, install a recent CUDA 12.8 PyTorch build:

```bash
python -m pip install \
  torch==2.7.0 \
  torchvision \
  torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

If you are not targeting a Blackwell-class GPU, you may instead follow the upstream `cu121` instructions from `Any6D/README.md`.

### 4. Install the common Python dependencies

The repository contains a project-specific dependency list for the Blackwell setup:

```bash
python -m pip install -r Any6D/requirements-core-bw.txt
```

If you want the exact upstream dependency set instead, use:

```bash
python -m pip install -r Any6D/requirements.txt
```

Do not install both blindly unless you know why you need to merge them.

## Additional Core Dependencies

### NVDiffRast

```bash
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
```

### Kaolin

The upstream Any6D README uses:

```bash
python -m pip install --no-cache-dir kaolin==0.16.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
```

However, on the current Blackwell-oriented environment, Kaolin compatibility may depend on the exact PyTorch/CUDA combination. In particular, Kaolin can fail at import time if its `warp` integration is not aligned with the installed `warp-lang` version.

Practical recommendation:

- first complete the environment without treating Kaolin as the primary success criterion;
- then test whether the specific Any6D pipeline you need actually imports and uses Kaolin correctly;
- if Kaolin import failures appear, treat them as an environment compatibility issue rather than a repository issue.

### PyTorch3D

The upstream README uses:

```bash
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.4.1cu121
```

In the current local environment, `pytorch3d` is importable. If you are rebuilding the environment from scratch on a different GPU or CUDA combination, you may need to choose a wheel that matches your PyTorch runtime rather than copying the `cu121` command literally.

## Build Steps

### 1. Build FoundationPose extensions

```bash
cd /home/iacopo/any6d/Any6D
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/pybind11/share/cmake/pybind11 \
  bash foundationpose/build_all_conda.sh
```

If your `pybind11` CMake path differs, adjust the path accordingly.

### 2. Build SAM2

```bash
cd /home/iacopo/any6d/Any6D/sam2
pip install -e .
```

The upstream SAM2 package may try to build CUDA extensions. In many cases this succeeds automatically; if it does not, consult the notes in `Any6D/sam2/README.md`.

### 3. Build InstantMesh

```bash
cd /home/iacopo/any6d/Any6D/instantmesh
pip install -e .
```

### 4. Build BOP Toolkit

```bash
cd /home/iacopo/any6d/Any6D/bop_toolkit
python setup.py install
```

## Checkpoints and Weights

This repository should not ship large pretrained weights. They must be downloaded separately.

### FoundationPose Weights

Download the FoundationPose network weights from the upstream links referenced in:

- `Any6D/README.md`
- `Any6D/foundationpose/readme.md`

Expected directory structure:

```text
Any6D/foundationpose/weights/
├── 2024-01-11-20-02-45/
└── 2023-10-28-18-33-37/
```

These are the scorer and refiner checkpoints used by the released Any6D code.

### SAM2 Checkpoints

From `Any6D/sam2`:

```bash
cd /home/iacopo/any6d/Any6D/sam2/checkpoints
./download_ckpts.sh
```

The project has used the large SAM2 checkpoint path documented by the upstream README.

### InstantMesh Checkpoints

Download the InstantMesh checkpoints referenced in the upstream Any6D README and place them under:

```text
Any6D/instantmesh/ckpts/
├── diffusion_pytorch_model.bin
└── instant_mesh_large.ckpt
```

## Optional Dataset Assets

The upstream Any6D README references:

- YCBV models
- HO3D evaluation files
- Any6D anchor results

For this repository, these assets are not expected to be versioned in Git. They should be downloaded or generated locally as needed and placed under the workspace directories expected by the scripts.

## ROS2 Dependency for the Demo Server

The robotic demo server uses `rclpy`, but in the current project layout `rclpy` is not installed directly inside the Conda environment. Instead, it is provided by the system ROS2 installation after sourcing ROS2.

Before launching the ROS2 Any6D server, always run:

```bash
source /opt/ros/humble/setup.bash
```

Then launch the server from the active Conda environment:

```bash
conda activate any6d_blackwell
source /opt/ros/humble/setup.bash
python3 /home/iacopo/any6d/demo/ros2/ros2_any6d_pose_server.py \
  --cv-final-root /home/iacopo/any6d \
  --object-name 003_cracker_box
```

This mixed setup is expected in the current deployment model.

## Demo Launch Model

The robotic demo is intentionally launched from two manual terminals.

### Linux terminal: Any6D ROS2 server

```bash
conda activate any6d_blackwell
source /opt/ros/humble/setup.bash
python3 /home/iacopo/any6d/demo/ros2/ros2_any6d_pose_server.py \
  --cv-final-root /home/iacopo/any6d \
  --object-name 006_mustard_bottle
```

### Isaac Sim terminal

The Isaac launch command is always of the form:

```text
<Isaac Python executable> <path to demo/isaac/ycb_pick_suction_v4.py> --object-name <object> --num-views <N> --distractor-count <M>
```

Windows example:

```powershell
C:\isaac-sim\python.bat C:\Users\tomin\any6d\demo\isaac\ycb_pick_suction_v4.py --object-name 006_mustard_bottle --num-views 4 --distractor-count 100
```

Linux example:

```bash
/opt/isaac-sim/python.sh /home/iacopo/any6d/demo/isaac/ycb_pick_suction_v4.py --object-name 006_mustard_bottle --num-views 4 --distractor-count 100
```

## Sanity Checks

Before attempting the full demo, validate these steps independently:

### Python runtime

```bash
conda activate any6d_blackwell
python -V
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
python -c "import nvdiffrast.torch as dr; print('nvdiffrast ok')"
python -c "import pytorch3d; print('pytorch3d ok')"
```

### Demo server import path

```bash
conda activate any6d_blackwell
source /opt/ros/humble/setup.bash
python3 /home/iacopo/any6d/demo/ros2/ros2_any6d_pose_server.py --object-name 003_cracker_box --mock
```

If this runs, the ROS2 bridge and script path are correct even without running full Any6D inference.

## Known Caveats

- Upstream Any6D installation instructions target an older, more conservative dependency stack than the local environment.

## Source References

The setup instructions above were derived from:

- `Any6D/README.md`
- `Any6D/foundationpose/readme.md`
- the local `requirements-core-bw.txt`
- the currently validated `any6d_blackwell` runtime used during demo development
