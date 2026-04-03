import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Isaac Sim ROS2 Any6D suction picking.")
    parser.add_argument(
        "--object-name",
        type=str,
        default="006_mustard_bottle",
        help="YCB object name, for example 003_cracker_box or 006_mustard_bottle.",
    )
    parser.add_argument("--num-views", type=int, default=4, help="Number of camera views to evaluate.")
    parser.add_argument("--distractor-count", type=int, default=100, help="Number of simple distractor cuboids.")
    return parser.parse_args()


CLI_ARGS = parse_args()
DEMO_ROOT = Path(__file__).resolve().parents[1]
OBJECTS_CONFIG_PATH = DEMO_ROOT / "config" / "objects_config.json"


def configure_internal_ros2_bridge():
    isaac_root = Path(__file__).resolve().parents[1]
    ros_distro = os.environ.get("ROS_DISTRO", "humble")
    ros_lib_dir = isaac_root / "exts" / "isaacsim.ros2.bridge" / ros_distro / "lib"

    os.environ.setdefault("ROS_DISTRO", ros_distro)
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    if ros_lib_dir.exists():
        current_path = os.environ.get("PATH", "")
        ros_lib_dir_str = str(ros_lib_dir)
        if ros_lib_dir_str.lower() not in current_path.lower():
            os.environ["PATH"] = f"{current_path};{ros_lib_dir_str}" if current_path else ros_lib_dir_str


configure_internal_ros2_bridge()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import json
import importlib.util
import random
import time

import carb
import numpy as np
import omni.physx
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import RigidPrim, SingleRigidPrim
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.semantics import add_labels
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.examples.universal_robots.controllers.pick_place_controller import (
    PickPlaceController,
)
from isaacsim.robot.manipulators.grippers import SurfaceGripper
from isaacsim.storage.native import get_assets_root_path
from pxr import Sdf


enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

if importlib.util.find_spec("rclpy") is None:
    raise RuntimeError(
        "rclpy is not available inside Isaac Sim after enabling isaacsim.ros2.bridge. "
        "Check ROS2 bridge installation and internal humble runtime setup."
    )

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


RGB_TOPIC = "/any6d/rgb"
DEPTH_TOPIC = "/any6d/depth"
MASK_TOPIC = "/any6d/mask"
CAMERA_INFO_TOPIC = "/any6d/camera_info"
REQUEST_TOPIC = "/any6d/request"
POSE_TOPIC = "/any6d/pred_pose"
POSE_JSON_TOPIC = "/any6d/pred_pose_json"
CAMERA_FRAME_ID = "any6d_camera"
WORLD_FRAME_ID = "world"
POSE_TIMEOUT_SEC = 15.0
MAX_CAMERA_REFRAMES = 8
NUM_VIEWS = CLI_ARGS.num_views
VIEW_AZIMUTH_DELTA_DEG = 90.0

with open(OBJECTS_CONFIG_PATH, "r", encoding="utf-8") as f:
    OBJECT_CONFIGS = json.load(f)

if CLI_ARGS.object_name not in OBJECT_CONFIGS:
    raise ValueError(
        f"Unsupported object '{CLI_ARGS.object_name}'. Supported objects: {sorted(OBJECT_CONFIGS)}"
    )

OBJECT_NAME = CLI_ARGS.object_name
OBJECT_LABEL = OBJECT_CONFIGS[OBJECT_NAME]["label"]
OBJECT_PRIM_PATH = OBJECT_CONFIGS[OBJECT_NAME]["prim_path"]
OBJECT_FRAME_CORRECTION = OBJECT_CONFIGS[OBJECT_NAME]["frame_correction"]
DISTRACTOR_COUNT = CLI_ARGS.distractor_count
TARGET_REACHABLE_X = (0.46, 0.56)
TARGET_REACHABLE_Y = (-0.12, 0.12)
TARGET_SAFE_RADIUS_XY = 0.18
OBJECT_EXTENTS_M = np.array(OBJECT_CONFIGS[OBJECT_NAME]["extents_m"], dtype=np.float32)


def now_stamp(clock):
    return clock.now().to_msg()


def quat_xyzw_to_rotmat(quat_xyzw):
    qx, qy, qz, qw = [float(x) for x in quat_xyzw]
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def compute_surface_pick_metrics(quat_xyzw, object_extents_m):
    rotation = quat_xyzw_to_rotmat(quat_xyzw)
    extents = np.asarray(object_extents_m, dtype=np.float32)
    face_areas = np.array(
        [
            extents[1] * extents[2],
            extents[0] * extents[2],
            extents[0] * extents[1],
        ],
        dtype=np.float32,
    )
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    best_axis = 0
    best_alignment = -1.0
    best_area = 0.0
    best_score = -1.0
    for axis_idx in range(3):
        axis_world = rotation[:, axis_idx]
        alignment = float(abs(np.dot(axis_world, world_up)))
        area = float(face_areas[axis_idx])
        score = alignment * area
        if score > best_score:
            best_axis = axis_idx
            best_alignment = alignment
            best_area = area
            best_score = score

    return {
        "best_axis": int(best_axis),
        "top_alignment": float(best_alignment),
        "face_area_m2": float(best_area),
        "surface_pick_score": float(best_score),
    }


def make_image_message(array, encoding, frame_id, stamp):
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(array.shape[0])
    msg.width = int(array.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = False
    if array.ndim == 2:
        msg.step = int(array.strides[0])
    else:
        msg.step = int(array.strides[0])
    msg.data = array.tobytes()
    return msg


def make_camera_info_message(width, height, intrinsic, frame_id, stamp):
    msg = CameraInfo()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.width = int(width)
    msg.height = int(height)
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0] * 5
    msg.k = [
        float(intrinsic[0, 0]),
        0.0,
        float(intrinsic[0, 2]),
        0.0,
        float(intrinsic[1, 1]),
        float(intrinsic[1, 2]),
        0.0,
        0.0,
        1.0,
    ]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [
        float(intrinsic[0, 0]),
        0.0,
        float(intrinsic[0, 2]),
        0.0,
        0.0,
        float(intrinsic[1, 1]),
        float(intrinsic[1, 2]),
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    return msg


class Any6DBridgeNode(Node):
    def __init__(self):
        super().__init__("any6d_pick_bridge")
        self.rgb_pub = self.create_publisher(Image, RGB_TOPIC, 10)
        self.depth_pub = self.create_publisher(Image, DEPTH_TOPIC, 10)
        self.mask_pub = self.create_publisher(Image, MASK_TOPIC, 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, CAMERA_INFO_TOPIC, 10)
        self.request_pub = self.create_publisher(String, REQUEST_TOPIC, 10)
        self.pose_json_sub = self.create_subscription(String, POSE_JSON_TOPIC, self._pose_json_callback, 10)
        self.latest_pose = None
        self.latest_pose_time = None

    def _pose_json_callback(self, msg):
        payload = json.loads(msg.data)
        self.latest_pose = payload
        self.latest_pose_time = time.time()
        if payload.get("status") == "ok" and "position" in payload:
            position = payload["position"]
            self.get_logger().info(
                f"Received response request_id={payload['request_id']} x={position[0]:.3f} y={position[1]:.3f} z={position[2]:.3f}"
            )
        else:
            self.get_logger().error(
                f"Received error response request_id={payload.get('request_id', -1)} "
                f"message={payload.get('message', 'unknown')}"
            )

    def publish_observation(self, rgb, depth_m, mask, intrinsic, t_w_c, request_id):
        stamp = now_stamp(self.get_clock())
        rgb_msg = make_image_message(np.ascontiguousarray(rgb.astype(np.uint8)), "rgb8", CAMERA_FRAME_ID, stamp)
        depth_mm = np.ascontiguousarray(np.clip(np.round(depth_m * 1000.0), 0.0, 65535.0).astype(np.uint16))
        depth_msg = make_image_message(depth_mm, "16UC1", CAMERA_FRAME_ID, stamp)
        mask_msg = make_image_message(np.ascontiguousarray(mask.astype(np.uint8)), "mono8", CAMERA_FRAME_ID, stamp)
        camera_info_msg = make_camera_info_message(rgb.shape[1], rgb.shape[0], intrinsic, CAMERA_FRAME_ID, stamp)

        request_msg = String()
        request_msg.data = json.dumps(
            {
                "request_id": request_id,
                "object_name": OBJECT_NAME,
                "camera_frame_id": CAMERA_FRAME_ID,
                "world_frame_id": WORLD_FRAME_ID,
                "rgb_topic": RGB_TOPIC,
                "depth_topic": DEPTH_TOPIC,
                "mask_topic": MASK_TOPIC,
                "camera_info_topic": CAMERA_INFO_TOPIC,
                "pose_topic": POSE_TOPIC,
                "t_w_c": np.asarray(t_w_c, dtype=np.float32).tolist(),
                "object_frame_correction": OBJECT_FRAME_CORRECTION,
            }
        )

        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
        self.mask_pub.publish(mask_msg)
        self.camera_info_pub.publish(camera_info_msg)
        self.request_pub.publish(request_msg)


def randomize_object(rigid_prim):
    target_x = random.uniform(*TARGET_REACHABLE_X)
    target_y = random.uniform(*TARGET_REACHABLE_Y)
    if random.random() > 0.5:
        target_z = 0.10
        target_quat = euler_angles_to_quat(np.array([0.0, np.pi / 2.0, random.uniform(0.0, np.pi)]))
    else:
        target_z = 0.03
        target_quat = euler_angles_to_quat(np.array([0.0, 0.0, random.uniform(0.0, np.pi)]))

    rigid_prim.set_world_poses(
        positions=np.array([[target_x, target_y, target_z]], dtype=np.float32),
        orientations=np.array([target_quat], dtype=np.float32),
    )
    print(f"Oggetto randomizzato a: x={target_x:.2f}, y={target_y:.2f}", flush=True)
    return np.array([target_x, target_y, target_z], dtype=np.float32)


def randomize_distractors(distractors, target_position):
    target_xy = np.asarray(target_position[:2], dtype=np.float32)
    for index, distractor in enumerate(distractors):
        distractor_x = None
        distractor_y = None
        for _ in range(100):
            candidate_x = random.uniform(0.20, 0.85)
            candidate_y = random.uniform(-0.45, 0.45)
            if np.linalg.norm(np.array([candidate_x, candidate_y], dtype=np.float32) - target_xy) < TARGET_SAFE_RADIUS_XY:
                continue
            if TARGET_REACHABLE_X[0] <= candidate_x <= TARGET_REACHABLE_X[1] and TARGET_REACHABLE_Y[0] <= candidate_y <= TARGET_REACHABLE_Y[1]:
                continue
            distractor_x = candidate_x
            distractor_y = candidate_y
            break
        if distractor_x is None or distractor_y is None:
            distractor_x = 0.20 + 0.04 * (index % 10)
            distractor_y = -0.42 + 0.09 * (index // 10)

        distractor_z = random.uniform(0.08, 0.28)
        distractor_quat = euler_angles_to_quat(np.array([0.0, 0.0, random.uniform(0.0, 2.0 * np.pi)], dtype=np.float32))
        distractor.set_world_pose(
            position=np.array([distractor_x, distractor_y, distractor_z], dtype=np.float32),
            orientation=np.array(distractor_quat, dtype=np.float32),
        )


def isaac_usd_to_cv_transform():
    conversion = np.eye(4, dtype=np.float32)
    conversion[1, 1] = -1.0
    conversion[2, 2] = -1.0
    return conversion


def row_major_transform_to_standard(matrix):
    matrix = np.array(matrix, dtype=np.float64, copy=True)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform matrix, got {matrix.shape}")
    return matrix.T


def k_from_camera_params(camera_params):
    aperture = np.array(camera_params["cameraAperture"], dtype=np.float64)
    resolution = np.array(camera_params["renderProductResolution"], dtype=np.float64)
    aperture_offset = np.array(camera_params["cameraApertureOffset"], dtype=np.float64)
    focal_length = float(camera_params["cameraFocalLength"])
    pixel_size = aperture[0] / resolution[0]
    fx = focal_length / pixel_size
    fy = focal_length / pixel_size
    cx = resolution[0] / 2.0 + aperture_offset[0]
    cy = resolution[1] / 2.0 + aperture_offset[1]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def t_w_c_from_camera_params(camera_params):
    world_to_camera_row_major = np.array(camera_params["cameraViewTransform"], dtype=np.float64).reshape(4, 4)
    world_to_camera_standard = row_major_transform_to_standard(world_to_camera_row_major)
    t_w_c_isaac_usd = np.linalg.inv(world_to_camera_standard).astype(np.float32)
    return t_w_c_isaac_usd @ isaac_usd_to_cv_transform()


def build_camera_views(target_prim):
    target_positions, _ = target_prim.get_world_poses()
    target_position = np.asarray(target_positions[0], dtype=np.float32)
    base_radius = random.uniform(0.82, 0.95)
    base_height = random.uniform(0.62, 0.78)
    base_azimuth_deg = random.uniform(-20.0, 20.0)
    azimuths_deg = [base_azimuth_deg + VIEW_AZIMUTH_DELTA_DEG * view_index for view_index in range(NUM_VIEWS)]

    views = []
    for azimuth_deg in azimuths_deg:
        azimuth = np.deg2rad(azimuth_deg)
        camera_position = np.array(
            [
                target_position[0] + base_radius * np.cos(azimuth),
                target_position[1] + base_radius * np.sin(azimuth),
                target_position[2] + base_height,
            ],
            dtype=np.float32,
        )
        look_at_target = target_position + np.array([0.0, 0.0, 0.03], dtype=np.float32)
        views.append(
            {
                "position": camera_position,
                "look_at": look_at_target,
                "azimuth_deg": float(azimuth_deg),
            }
        )
    return views


def apply_camera_view(camera, view):
    position = np.asarray(view["position"], dtype=np.float32)
    look_at = np.asarray(view["look_at"], dtype=np.float32)
    with camera:
        rep.modify.pose(position=tuple(float(x) for x in position), look_at=tuple(float(x) for x in look_at))
    print(
        "Camera view impostata: "
        f"azimuth={view.get('azimuth_deg', float('nan')):.2f} deg "
        f"pos={position.tolist()} look_at={look_at.tolist()}",
        flush=True,
    )


def capture_observation(rgb_annot, depth_annot, inst_seg_annot, camera_params_annot):
    rep.orchestrator.step(rt_subframes=8, delta_time=0.0, pause_timeline=False)
    rgb = rgb_annot.get_data()[:, :, :3].astype(np.uint8)
    depth_m = depth_annot.get_data().astype(np.float32)
    depth_m[~np.isfinite(depth_m)] = 0.0
    seg = inst_seg_annot.get_data()
    camera_params = camera_params_annot.get_data()

    target_id = None
    for object_id, labels in seg["info"]["idToLabels"].items():
        class_value = labels.get("class", "") if isinstance(labels, dict) else labels
        class_text = " ".join(class_value) if isinstance(class_value, list) else str(class_value)
        normalized = class_text.lower()
        if (
            OBJECT_LABEL.lower() in normalized
            or OBJECT_NAME.lower() in normalized
            or OBJECT_PRIM_PATH.lower() in normalized
        ):
            target_id = int(object_id)
            break

    if target_id is None:
        print(f"Instance segmentation labels disponibili: {seg['info'].get('idToLabels', {})}", flush=True)
        return None

    mask = np.zeros_like(seg["data"], dtype=np.uint8)
    mask[seg["data"] == target_id] = 255
    intrinsic = k_from_camera_params(camera_params)
    t_w_c = t_w_c_from_camera_params(camera_params)
    return rgb, depth_m, mask, intrinsic, t_w_c


def capture_observation_with_retries(camera, view, rgb_annot, depth_annot, inst_seg_annot, camera_params_annot):
    for attempt in range(1, MAX_CAMERA_REFRAMES + 1):
        observation = capture_observation(
            rgb_annot,
            depth_annot,
            inst_seg_annot,
            camera_params_annot,
        )
        if observation is not None:
            if attempt > 1:
                print(f"Target rientrato nel frame dopo {attempt} tentativi camera.", flush=True)
            return observation
        perturbed_view = {
            "position": np.asarray(view["position"], dtype=np.float32)
            + np.array(
                [
                    random.uniform(-0.03, 0.03),
                    random.uniform(-0.03, 0.03),
                    random.uniform(-0.02, 0.03),
                ],
                dtype=np.float32,
            ),
            "look_at": np.asarray(view["look_at"], dtype=np.float32),
        }
        apply_camera_view(camera, perturbed_view)
        for _ in range(4):
            simulation_app.update()
        print(f"Target non visibile, riposiziono la camera (tentativo {attempt}/{MAX_CAMERA_REFRAMES}).", flush=True)
    return None


if __name__ == "__main__":
    rclpy.init()
    ros_node = Any6DBridgeNode()

    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        ros_node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()
        sys.exit()

    asset_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
    robot_prim_path = "/World/UR10"
    robot_name = "my_ur10"

    robot_node = add_reference_to_stage(usd_path=asset_path, prim_path=robot_prim_path)
    robot_node.GetVariantSet("Gripper").SetVariantSelection("Short_Suction")

    suction_tip_path = f"{robot_prim_path}/ee_link/SurfaceGripper"
    gripper = SurfaceGripper(
        end_effector_prim_path=f"{robot_prim_path}/ee_link",
        surface_gripper_path=suction_tip_path,
    )

    stage = omni.usd.get_context().get_stage()
    gripper_prim = stage.GetPrimAtPath(suction_tip_path)
    if not gripper_prim.HasAttribute("isaac:maxGripDistance"):
        gripper_prim.CreateAttribute("isaac:maxGripDistance", Sdf.ValueTypeNames.Float).Set(0.45)
    else:
        gripper_prim.GetAttribute("isaac:maxGripDistance").Set(0.45)

    my_robot = my_world.scene.add(
        SingleManipulator(
            prim_path=robot_prim_path,
            name=robot_name,
            end_effector_prim_path=f"{robot_prim_path}/ee_link",
            gripper=gripper,
        )
    )
    my_robot.set_joints_default_state(
        positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0.0], dtype=np.float32)
    )

    ycb_asset_path = assets_root_path + f"/Isaac/Props/YCB/Axis_Aligned_Physics/{OBJECT_NAME}.usd"
    obj_prim_path = OBJECT_PRIM_PATH
    add_reference_to_stage(usd_path=ycb_asset_path, prim_path=obj_prim_path)
    my_obj = my_world.scene.add(RigidPrim(prim_paths_expr=obj_prim_path, name=OBJECT_LABEL))
    add_labels(stage.GetPrimAtPath(obj_prim_path), [OBJECT_LABEL, OBJECT_NAME], "class")

    distractors = []
    distractor_colors = [
        np.array([0.85, 0.25, 0.25]),
        np.array([0.20, 0.65, 0.90]),
        np.array([0.90, 0.75, 0.20]),
        np.array([0.40, 0.80, 0.45]),
        np.array([0.75, 0.45, 0.90]),
    ]
    for index in range(DISTRACTOR_COUNT):
        distractor = my_world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/Distractor_{index}",
                name=f"distractor_{index}",
                position=np.array([0.2 + 0.08 * index, -0.35 + 0.14 * index, 0.04], dtype=np.float32),
                scale=np.array(
                    [
                        random.uniform(0.05, 0.10),
                        random.uniform(0.05, 0.10),
                        random.uniform(0.05, 0.14),
                    ],
                    dtype=np.float32,
                ),
                size=1.0,
                color=distractor_colors[index % len(distractor_colors)] * 255.0,
            )
        )
        distractors.append(distractor)

    camera = rep.create.camera(
        name="Any6DROS2Camera",
        focal_length=24.0,
        horizontal_aperture=36.0,
        position=(2.0, 0.0, 1.5),
    )

    render_product = rep.create.render_product(camera, (640, 480))
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    inst_seg_annot = rep.AnnotatorRegistry.get_annotator("instance_segmentation")
    camera_params_annot = rep.AnnotatorRegistry.get_annotator("camera_params")
    rgb_annot.attach([render_product])
    depth_annot.attach([render_product])
    inst_seg_annot.attach([render_product])
    camera_params_annot.attach([render_product])

    my_controller = PickPlaceController(
        name="pick_place_controller",
        gripper=my_robot.gripper,
        robot_articulation=my_robot,
    )
    my_controller.reset(end_effector_initial_height=0.7)
    articulation_controller = my_robot.get_articulation_controller()

    my_world.reset()
    my_robot.gripper.set_default_state(opened=True)
    target_position = randomize_object(my_obj)
    randomize_distractors(distractors, target_position)
    camera_views = build_camera_views(my_obj)
    current_view_index = 0
    view_candidates = []
    apply_camera_view(camera, camera_views[current_view_index])

    print(
        f"Simulation started. ROS2 Any6D bridge enabled. object={OBJECT_NAME} num_views={NUM_VIEWS} distractors={DISTRACTOR_COUNT}",
        flush=True,
    )

    reset_needed = False
    task_completed = False
    waiting_for_pose = False
    active_request_id = None
    request_sent_at = None
    active_request_mask_pixels = None
    request_id = 0
    settle_frames = 0
    picking_target = None

    while simulation_app.is_running():
        my_world.step(render=True)
        rclpy.spin_once(ros_node, timeout_sec=0.0)

        if my_world.is_stopped() and not reset_needed:
            reset_needed = True
            task_completed = False

        if my_world.is_playing():
            if reset_needed:
                my_world.reset()
                my_controller.reset(end_effector_initial_height=0.7)
                target_position = randomize_object(my_obj)
                randomize_distractors(distractors, target_position)
                camera_views = build_camera_views(my_obj)
                current_view_index = 0
                view_candidates = []
                apply_camera_view(camera, camera_views[current_view_index])
                reset_needed = False
                task_completed = False
                waiting_for_pose = False
                picking_target = None
                settle_frames = 0
                ros_node.latest_pose = None
                active_request_id = None
                request_sent_at = None
                active_request_mask_pixels = None

            if task_completed:
                continue

            if not waiting_for_pose and picking_target is None:
                settle_frames += 1
                if settle_frames < 15:
                    continue

                active_view = camera_views[current_view_index]
                observation = capture_observation_with_retries(
                    camera,
                    active_view,
                    rgb_annot,
                    depth_annot,
                    inst_seg_annot,
                    camera_params_annot,
                )
                if observation is None:
                    print("Impossibile inquadrare il target dopo vari tentativi camera, riparto dal prossimo ciclo.", flush=True)
                    if current_view_index + 1 < len(camera_views):
                        current_view_index += 1
                        apply_camera_view(camera, camera_views[current_view_index])
                        settle_frames = 0
                        continue
                    camera_views = build_camera_views(my_obj)
                    current_view_index = 0
                    view_candidates = []
                    apply_camera_view(camera, camera_views[current_view_index])
                    settle_frames = 0
                    continue

                rgb, depth_m, mask, intrinsic, t_w_c = observation
                mask_pixels = int(np.count_nonzero(mask))
                camera_world_pos = np.asarray(t_w_c[:3, 3], dtype=np.float32)
                print(
                    f"View {current_view_index + 1}/{len(camera_views)} | "
                    f"T_W_C camera_pos={camera_world_pos.tolist()} mask_pixels={mask_pixels}",
                    flush=True,
                )
                if mask_pixels == 0:
                    print("Mask vuota, salto la richiesta Any6D.", flush=True)
                    continue

                request_id += 1
                ros_node.publish_observation(rgb, depth_m, mask, intrinsic, t_w_c, request_id)
                waiting_for_pose = True
                active_request_id = request_id
                request_sent_at = time.time()
                active_request_mask_pixels = mask_pixels
                print(f"Pubblicata richiesta Any6D request_id={request_id}", flush=True)
                continue

            if waiting_for_pose and request_sent_at is not None and (time.time() - request_sent_at) > POSE_TIMEOUT_SEC:
                waiting_for_pose = False
                active_request_id = None
                request_sent_at = None
                ros_node.latest_pose = None
                settle_frames = 0
                active_request_mask_pixels = None
                print("Timeout attesa posa Any6D, riprovo con una nuova richiesta.", flush=True)
                continue

            if waiting_for_pose and ros_node.latest_pose is not None:
                pose_payload = ros_node.latest_pose
                if pose_payload.get("request_id") != active_request_id:
                    continue
                if pose_payload.get("status") != "ok":
                    waiting_for_pose = False
                    active_request_id = None
                    request_sent_at = None
                    ros_node.latest_pose = None
                    settle_frames = 0
                    active_request_mask_pixels = None
                    print(f"Errore Any6D: {pose_payload.get('message', 'unknown')}", flush=True)
                    continue

                pred_world_pos = np.array(pose_payload["position"], dtype=np.float32)
                pred_world_quat_xyzw = np.array(pose_payload["orientation_xyzw"], dtype=np.float32)
                surface_metrics = compute_surface_pick_metrics(pred_world_quat_xyzw, OBJECT_EXTENTS_M)
                gt_world_pos_batch, _ = my_obj.get_world_poses()
                gt_world_pos = np.asarray(gt_world_pos_batch[0], dtype=np.float32)
                gt_translation_error_cm = float(np.linalg.norm(pred_world_pos - gt_world_pos) * 100.0)
                print(
                    f"View {current_view_index + 1}/{len(camera_views)} | "
                    f"Errore traslazione pred-vs-gt: {gt_translation_error_cm:.2f} cm | "
                    f"mask_pixels={active_request_mask_pixels} | "
                    f"surface_pick_score={surface_metrics['surface_pick_score']:.5f} | "
                    f"top_alignment={surface_metrics['top_alignment']:.3f} | "
                    f"pred={pred_world_pos.tolist()} gt={gt_world_pos.tolist()}",
                    flush=True,
                )

                view_candidates.append(
                    {
                        "view_index": current_view_index,
                        "mask_pixels": int(active_request_mask_pixels or 0),
                        "pred_world_pos": pred_world_pos.copy(),
                        "pred_world_quat_xyzw": pred_world_quat_xyzw.copy(),
                        "error_cm": gt_translation_error_cm,
                        "surface_metrics": surface_metrics,
                    }
                )

                if current_view_index + 1 < len(camera_views):
                    current_view_index += 1
                    apply_camera_view(camera, camera_views[current_view_index])
                    waiting_for_pose = False
                    active_request_id = None
                    request_sent_at = None
                    ros_node.latest_pose = None
                    active_request_mask_pixels = None
                    settle_frames = 0
                    print(f"Passo alla view {current_view_index + 1}/{len(camera_views)}.", flush=True)
                    continue

                best_candidate = max(
                    view_candidates,
                    key=lambda item: (
                        item["surface_metrics"]["surface_pick_score"],
                        item["surface_metrics"]["top_alignment"],
                        item["mask_pixels"],
                    ),
                )
                selected_pred_world_pos = np.asarray(best_candidate["pred_world_pos"], dtype=np.float32)
                selected_translation_error_cm = float(np.linalg.norm(selected_pred_world_pos - gt_world_pos) * 100.0)
                print(
                    f"Best-surface su {len(view_candidates)} view | "
                    f"view_scelta={best_candidate['view_index'] + 1} "
                    f"mask_pixels={best_candidate['mask_pixels']} "
                    f"surface_pick_score={best_candidate['surface_metrics']['surface_pick_score']:.5f} "
                    f"top_alignment={best_candidate['surface_metrics']['top_alignment']:.3f} "
                    f"face_area_m2={best_candidate['surface_metrics']['face_area_m2']:.5f} "
                    f"errore_traslazione={selected_translation_error_cm:.2f} cm | "
                    f"pred_best={selected_pred_world_pos.tolist()}",
                    flush=True,
                )

                physx_query_interface = omni.physx.get_physx_scene_query_interface()
                ray_origin = np.array([selected_pred_world_pos[0], selected_pred_world_pos[1], 1.5], dtype=np.float32)
                hit = physx_query_interface.raycast_closest(ray_origin, np.array([0.0, 0.0, -1.0], dtype=np.float32), 2.0)

                suction_cup_length = 0.067
                gap = 0.005
                if hit["hit"] and obj_prim_path in hit["rigidBody"]:
                    picking_target = np.asarray(hit["position"], dtype=np.float32) + np.array(
                        [0.0, 0.0, suction_cup_length + gap],
                        dtype=np.float32,
                    )
                else:
                    picking_target = selected_pred_world_pos + np.array([0.0, 0.0, suction_cup_length + gap], dtype=np.float32)

                waiting_for_pose = False
                active_request_id = None
                request_sent_at = None
                ros_node.latest_pose = None
                active_request_mask_pixels = None
                current_view_index = 0
                view_candidates = []
                print(f"Ricevuta posa Any6D, target pick = {picking_target.tolist()}", flush=True)

            if picking_target is not None:
                actions = my_controller.forward(
                    picking_position=picking_target,
                    placing_position=np.array([0.0, 0.6, 0.35], dtype=np.float32),
                    current_joint_positions=my_robot.get_joint_positions(),
                    end_effector_orientation=euler_angles_to_quat(np.array([0.0, np.pi / 2.0, 0.0], dtype=np.float32)),
                    end_effector_offset=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                )

                if my_controller.is_done() and not task_completed:
                    print("Successfully validated picking with ROS2 Any6D pose!", flush=True)
                    task_completed = True

                articulation_controller.apply_action(actions)

    ros_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()
