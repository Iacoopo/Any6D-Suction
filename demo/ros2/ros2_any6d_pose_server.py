import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import rclpy
import trimesh
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

DEMO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CV_FINAL_ROOT = DEMO_ROOT.parent


RGB_TOPIC = "/any6d/rgb"
DEPTH_TOPIC = "/any6d/depth"
MASK_TOPIC = "/any6d/mask"
CAMERA_INFO_TOPIC = "/any6d/camera_info"
REQUEST_TOPIC = "/any6d/request"
POSE_TOPIC = "/any6d/pred_pose"
POSE_JSON_TOPIC = "/any6d/pred_pose_json"
WORLD_FRAME_ID = "world"


def parse_args():
    parser = argparse.ArgumentParser(description="ROS2 Any6D pose server")
    parser.add_argument("--cv-final-root", type=str, default=str(DEFAULT_CV_FINAL_ROOT))
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=5)
    parser.add_argument("--mock", action="store_true", help="Skip Any6D and publish a deterministic mock pose.")
    parser.add_argument(
        "--object-name",
        type=str,
        default="003_cracker_box",
        help="Object to preload at startup when running the real Any6D server.",
    )
    parser.add_argument(
        "--preload-object",
        type=str,
        default=None,
        help="Deprecated alias for --object-name.",
    )
    return parser.parse_args()


def quat_xyzw_from_rotmat(rotation):
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float32)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quat / norm


def load_any6d_estimator(any6d_root: Path, mesh: trimesh.Trimesh, debug: int):
    if str(any6d_root) not in sys.path:
        sys.path.insert(0, str(any6d_root))

    import nvdiffrast.torch as dr
    from estimater import Any6D
    from foundationpose.learning.training.predict_pose_refine import PoseRefinePredictor
    from foundationpose.learning.training.predict_score import ScorePredictor

    glctx = dr.RasterizeCudaContext()
    return Any6D(
        mesh=mesh,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        glctx=glctx,
        debug=debug,
        debug_dir=str(any6d_root / "results" / "ros2_pose_server"),
    )


def verify_any6d_runtime(any6d_root: Path):
    if str(any6d_root) not in sys.path:
        sys.path.insert(0, str(any6d_root))

    import cv2  # noqa: F401
    import nvdiffrast.torch as dr  # noqa: F401
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Any6D runtime environment")

    weights_dir = any6d_root / "foundationpose" / "weights"
    if not weights_dir.exists():
        raise FileNotFoundError(f"Any6D weights directory not found: {weights_dir}")

    return {
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "weights_dir": str(weights_dir),
    }


class Ros2Any6DPoseServer(Node):
    def __init__(self, cv_final_root: Path, debug: int, iteration: int, mock: bool):
        super().__init__("any6d_pose_server")
        self.cv_final_root = cv_final_root
        self.debug = debug
        self.iteration = iteration
        self.mock = mock
        self.rgb = None
        self.depth = None
        self.mask = None
        self.camera_info = None
        self.estimators = {}
        self.mesh_root = cv_final_root / "datasets" / "ho3d" / "YCB_Video_Models" / "models"

        self.create_subscription(Image, RGB_TOPIC, self._rgb_callback, 10)
        self.create_subscription(Image, DEPTH_TOPIC, self._depth_callback, 10)
        self.create_subscription(Image, MASK_TOPIC, self._mask_callback, 10)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self._camera_info_callback, 10)
        self.create_subscription(String, REQUEST_TOPIC, self._request_callback, 10)
        self.pose_json_pub = self.create_publisher(String, POSE_JSON_TOPIC, 10)

    def _rgb_callback(self, msg):
        self.rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()

    def _depth_callback(self, msg):
        depth_mm = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).copy()
        self.depth = depth_mm.astype(np.float32) / 1000.0

    def _mask_callback(self, msg):
        self.mask = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width).copy() > 0

    def _camera_info_callback(self, msg):
        intrinsic = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self.camera_info = intrinsic

    def _get_estimator(self, object_name: str):
        if object_name in self.estimators:
            return self.estimators[object_name]

        mesh_path = self.mesh_root / object_name / "textured_simple.obj"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found for {object_name}: {mesh_path}")

        mesh = trimesh.load(mesh_path)
        estimator = load_any6d_estimator(self.cv_final_root / "Any6D", mesh, self.debug)
        self.estimators[object_name] = estimator
        return estimator

    def _request_callback(self, msg):
        if self.rgb is None or self.depth is None or self.mask is None or self.camera_info is None:
            self._publish_error(json.loads(msg.data), "Observation incomplete")
            return

        payload = json.loads(msg.data)
        object_name = payload["object_name"]
        t_w_c = np.asarray(payload["t_w_c"], dtype=np.float32)
        object_frame_correction = np.asarray(payload["object_frame_correction"], dtype=np.float32)
        object_frame_correction_inv = np.linalg.inv(object_frame_correction)
        try:
            if self.mock:
                pred_pose = self._build_mock_pose(self.depth, self.mask)
            else:
                estimator = self._get_estimator(object_name)
                pred_pose = estimator.register(
                    K=self.camera_info,
                    rgb=self.rgb,
                    depth=self.depth,
                    ob_mask=self.mask,
                    iteration=self.iteration,
                    name=f"ros2_{object_name}",
                ).astype(np.float32)
        except Exception as exc:
            self._publish_error(payload, str(exc))
            return

        pred_world = t_w_c @ pred_pose @ object_frame_correction_inv
        quat = quat_xyzw_from_rotmat(pred_world[:3, :3])
        response = {
            "request_id": payload["request_id"],
            "status": "ok",
            "world_frame_id": payload.get("world_frame_id", WORLD_FRAME_ID),
            "position": [float(pred_world[0, 3]), float(pred_world[1, 3]), float(pred_world[2, 3])],
            "orientation_xyzw": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
            "mock": self.mock,
            "timestamp": time.time(),
        }
        out_msg = String()
        out_msg.data = json.dumps(response)
        self.pose_json_pub.publish(out_msg)
        self.get_logger().info(
            f"Published pose for request_id={payload['request_id']} object={object_name}: "
            f"{pred_world[0, 3]:.3f}, {pred_world[1, 3]:.3f}, {pred_world[2, 3]:.3f}"
        )

    def _build_mock_pose(self, depth, mask):
        masked_depth = depth[mask]
        valid_depth = masked_depth[masked_depth > 0.0]
        z = float(np.median(valid_depth)) if valid_depth.size > 0 else 0.55
        pred_pose = np.eye(4, dtype=np.float32)
        pred_pose[2, 3] = z
        return pred_pose

    def _publish_error(self, payload, message):
        response = {
            "request_id": payload.get("request_id", -1),
            "status": "error",
            "message": message,
            "timestamp": time.time(),
        }
        out_msg = String()
        out_msg.data = json.dumps(response)
        self.pose_json_pub.publish(out_msg)
        self.get_logger().error(message)


def main():
    args = parse_args()
    if args.preload_object is not None:
        args.object_name = args.preload_object
    cv_final_root = Path(args.cv_final_root).resolve()
    any6d_root = cv_final_root / "Any6D"

    if args.mock:
        print("[Any6D ROS2] Running in mock mode.", flush=True)
    else:
        runtime_info = verify_any6d_runtime(any6d_root)
        print(
            "[Any6D ROS2] Runtime ready with "
            f"{runtime_info['cuda_device_count']} CUDA device(s), "
            f"device='{runtime_info['cuda_device_name']}', "
            f"weights='{runtime_info['weights_dir']}'",
            flush=True,
        )

    rclpy.init()
    node = Ros2Any6DPoseServer(cv_final_root, args.debug, args.iteration, args.mock)
    if not args.mock and args.object_name:
        print(f"[Any6D ROS2] Preloading estimator for '{args.object_name}'...", flush=True)
        node._get_estimator(args.object_name)
        print(f"[Any6D ROS2] Estimator for '{args.object_name}' loaded.", flush=True)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
