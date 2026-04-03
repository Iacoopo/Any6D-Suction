"""
Spec-first scaffold for the Isaac Sim -> Any6D synthetic data generator.

This file is intentionally structured as an implementation proposal in code.
It establishes the architecture, dataset contracts, factor registry, and
export/naming logic before we lock down the full Replicator capture pipeline.

Design choices reflected here are based on:
- local project spec in `any6d/ISAAC_ANY6D_DATA_SPEC.md`
- local project context in `any6d/PROJECT_CONTEXT.md`
- existing local prototypes in `my_scripts/dataset_generation_v3.py` and
  `my_scripts/isaac_any6d_generator.py`
- Isaac Sim / Replicator patterns from the local codebase, including:
  - `test_sdg_getting_started.py`
  - `test_sdg_useful_snippets.py`
  - `pose_writer.py`
  - `custom_fps_writer_annotator.py`

Current scope:
- define a clean, modular architecture
- encode naming and metadata rules required by Any6D
- provide deterministic sample planning for anchors and queries
- prepare capture/export interfaces for the next implementation phase

Deferred to the next step:
- full scene setup
- annotator attachment
- dual-pass capture
- on-disk image/depth/mask writing from live Isaac data
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import random
import sys
import traceback
from typing import Any, Iterable, Literal

import numpy as np
from PIL import Image


SampleType = Literal["anchor", "query"]
SplitName = Literal["anchors", "queries"]
FactorGroup = Literal["none", "single_factor", "combined_factor"]


LOCAL_YCB_AXIS_ALIGNED_DIR = Path("C:/isaac-sim/any6d/YCB_Axis_Aligned")
LOCAL_YCB_AXIS_ALIGNED_PHYSICS_DIR = Path("C:/isaac-sim/any6d/YCB_Axis_Aligned_Physics")


def _resolve_default_target_objects() -> dict[str, str]:
    object_names = (
        "003_cracker_box",
        "005_tomato_soup_can",
        "006_mustard_bottle",
        "010_potted_meat_can",
        "021_bleach_cleanser",
    )
    resolved: dict[str, str] = {}
    for object_name in object_names:
        local_physics = LOCAL_YCB_AXIS_ALIGNED_PHYSICS_DIR / f"{object_name}.usd"
        local_axis_aligned = LOCAL_YCB_AXIS_ALIGNED_DIR / f"{object_name}.usd"
        # Local Axis_Aligned assets are the most reliable package in this workspace.
        # The local Axis_Aligned_Physics USDs can reference "../Axis_Aligned/...":
        # if that sibling folder is not present with that exact name, they load as empty prims.
        if local_axis_aligned.exists():
            resolved[object_name] = str(local_axis_aligned)
        elif local_physics.exists():
            resolved[object_name] = str(local_physics)
        else:
            resolved[object_name] = f"/Isaac/Props/YCB/Axis_Aligned_Physics/{object_name}.usd"
    return resolved


DEFAULT_TARGET_OBJECTS: dict[str, str] = _resolve_default_target_objects()

REPLICATOR_TEST_OBJECTS_ROOT = (
    "C:/isaac-sim/extscache/omni.replicator.core-1.12.27+107.3.3.wx64.r.cp311/"
    "omni/replicator/core/tests/data/objects"
)

DATASET_PROFILES: dict[str, dict[str, int]] = {
    "smoke": {
        "anchors": 1,
        "viewpoint_queries": 2,
        "distance_queries": 2,
        "lighting_queries": 2,
        "viewpoint_lighting_queries": 1,
        "viewpoint_lighting_clutter_queries": 1,
        "viewpoint_lighting_occlusion_queries": 1,
        "viewpoint_lighting_clutter_occlusion_queries": 1,
        "clutter_queries": 2,
        "occlusion_queries": 4,
    },
    "pilot_any6d_v1": {
        "anchors": 12,
        "viewpoint_queries": 48,
        "distance_queries": 12,
        "lighting_queries": 6,
        "viewpoint_lighting_queries": 6,
        "viewpoint_lighting_clutter_queries": 6,
        "viewpoint_lighting_occlusion_queries": 6,
        "viewpoint_lighting_clutter_occlusion_queries": 6,
        "clutter_queries": 16,
        "occlusion_queries": 24,
    },
    "analysis_any6d": {
        "anchors": 10,
        "viewpoint_queries": 20,
        "distance_queries": 10,
        "lighting_queries": 20,
        "viewpoint_lighting_queries": 30,
        "viewpoint_lighting_clutter_queries": 30,
        "viewpoint_lighting_occlusion_queries": 30,
        "viewpoint_lighting_clutter_occlusion_queries": 30,
        "clutter_queries": 20,
        "occlusion_queries": 20,
    },
    "rq2_balanced_n24_v1": {
        "anchors": 16,
        "viewpoint_queries": 24,
        "distance_queries": 24,
        "lighting_queries": 24,
        "viewpoint_lighting_queries": 24,
        "viewpoint_lighting_clutter_queries": 24,
        "viewpoint_lighting_occlusion_queries": 24,
        "viewpoint_lighting_clutter_occlusion_queries": 24,
        "clutter_queries": 24,
        "occlusion_queries": 24,
    },
}


DEFAULT_DISTRACTOR_OBJECTS: tuple[str, ...] = (
    "/Isaac/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
    "/Isaac/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
    "/Isaac/Props/YCB/Axis_Aligned_Physics/008_pudding_box.usd",
    "/Isaac/Props/YCB/Axis_Aligned_Physics/009_gelatin_box.usd",
    "/Isaac/Props/YCB/Axis_Aligned_Physics/011_banana.usd",
    "/Isaac/Props/YCB/Axis_Aligned_Physics/019_pitcher_base.usd",
)


MANDATORY_FILES = (
    "rgb.png",
    "depth.png",
    "mask_target.png",
    "mask_visib_target.png",
    "K.txt",
    "meta.json",
)


def log(message: str) -> None:
    print(f"[Any6D] {message}", flush=True)


@dataclass(frozen=True)
class Resolution:
    width: int = 640
    height: int = 480


@dataclass(frozen=True)
class CameraConfig:
    resolution: Resolution = field(default_factory=Resolution)
    focal_length_mm: float = 24.0
    horizontal_aperture_mm: float = 36.0
    near_m: float = 0.01
    far_m: float = 100.0


@dataclass(frozen=True)
class OutputConfig:
    root_dir: Path
    depth_scale: float = 1000.0


@dataclass(frozen=True)
class SceneConfig:
    include_robot: bool = False
    include_ground_plane: bool = True
    target_mount_position_m: tuple[float, float, float] = (0.5, 0.0, 0.05)
    max_distractors: int = 10
    assets_root_override: str | None = None
    target_objects: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TARGET_OBJECTS))
    distractor_primitive_types: tuple[str, ...] = ("Cube", "Cylinder", "Sphere", "Cone")
    distractor_ground_clearance_m: float = 0.002


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    sample_type: SampleType
    factor_group: FactorGroup
    description: str
    requires_distractors: bool = False
    requires_occluders: bool = False
    requires_depth_postprocess: bool = False
    requires_mask_postprocess: bool = False


@dataclass(frozen=True)
class SampleRequest:
    object_name: str
    sample_type: SampleType
    split: SplitName
    index: int
    factor_name: str
    factor_group: FactorGroup
    factor_value: dict[str, Any]


@dataclass(frozen=True)
class SamplePaths:
    sample_dir: Path
    rgb_path: Path
    depth_path: Path
    mask_target_path: Path
    mask_visib_target_path: Path
    k_path: Path
    meta_path: Path
    sample_id: str


@dataclass
class SceneHandles:
    """Holds runtime handles once the Isaac scene exists."""

    target_prim_path: str = "/World/Target"
    camera_prim_path: str = "/World/Any6D_Camera"
    robot_prim_path: str = "/World/UR10"
    dome_light_prim_path: str = "/World/Any6D_DomeLight"
    key_light_prim_path: str = "/World/Any6D_KeyLight"
    fill_light_prim_path: str = "/World/Any6D_FillLight"
    distractor_paths: list[str] = field(default_factory=list)
    occluder_paths: list[str] = field(default_factory=list)
    target_handle: Any | None = None
    camera_handle: Any | None = None
    camera_rep: Any | None = None
    dome_light_rep: Any | None = None
    key_light_rep: Any | None = None
    fill_light_rep: Any | None = None
    robot_handle: Any | None = None
    distractor_handles: dict[str, Any] = field(default_factory=dict)
    target_base_position_m: np.ndarray | None = None


@dataclass
class CaptureArtifacts:
    """
    Normalized container for live data extracted from Isaac/Replicator.

    The next implementation phase will populate this using annotators:
    rgb, distance_to_image_plane, instance_segmentation,
    bounding_box_2d_tight, camera_params, optionally bounding_box_3d_fast.
    """

    rgb_rgba: np.ndarray | None = None
    depth_m: np.ndarray | None = None
    mask_target: np.ndarray | None = None
    mask_visib_target: np.ndarray | None = None
    k_matrix: np.ndarray | None = None
    t_w_c: np.ndarray | None = None
    t_w_o: np.ndarray | None = None
    t_c_o: np.ndarray | None = None
    t_w_c_raw_isaac_usd: np.ndarray | None = None
    t_c_o_raw_isaac_usd: np.ndarray | None = None
    bbox_2d_tight_xyxy: list[int] | None = None
    visibility_ratio: float | None = None
    occlusion_ratio: float | None = None
    target_total_mask_pixel_count: int | None = None
    target_visible_pixel_count: int | None = None
    target_depth_valid_pixel_count: int | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GeneratorConfig:
    output: OutputConfig
    camera: CameraConfig = field(default_factory=CameraConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    object_frame_convention_raw: str = "isaac_asset"
    object_frame_convention_eval: str = "any6d_ref"
    object_frame_corrections: dict[str, np.ndarray] = field(default_factory=dict)
    random_seed: int = 7
    debug_logging: bool = False


def build_default_factor_registry() -> dict[str, FactorDefinition]:
    return {
        "anchor_clean": FactorDefinition(
            name="anchor_clean",
            sample_type="anchor",
            factor_group="none",
            description="Reference anchor with minimal clutter, minimal occlusion, and controlled lighting.",
        ),
        "viewpoint": FactorDefinition(
            name="viewpoint",
            sample_type="query",
            factor_group="single_factor",
            description="Vary camera azimuth/elevation around the object while keeping scene complexity limited.",
        ),
        "distance": FactorDefinition(
            name="distance",
            sample_type="query",
            factor_group="single_factor",
            description="Vary camera-target distance while preserving object observability.",
        ),
        "lighting": FactorDefinition(
            name="lighting",
            sample_type="query",
            factor_group="single_factor",
            description="Vary dome and local light configuration while keeping geometry fixed.",
        ),
        "viewpoint_lighting": FactorDefinition(
            name="viewpoint_lighting",
            sample_type="query",
            factor_group="combined_factor",
            description="Jointly vary camera viewpoint and lighting configuration.",
        ),
        "viewpoint_lighting_clutter": FactorDefinition(
            name="viewpoint_lighting_clutter",
            sample_type="query",
            factor_group="combined_factor",
            description="Jointly vary camera viewpoint, lighting, and distractor clutter.",
            requires_distractors=True,
        ),
        "viewpoint_lighting_occlusion": FactorDefinition(
            name="viewpoint_lighting_occlusion",
            sample_type="query",
            factor_group="combined_factor",
            description="Jointly vary camera viewpoint, lighting, and explicit occlusion.",
            requires_distractors=True,
            requires_occluders=True,
        ),
        "viewpoint_lighting_clutter_occlusion": FactorDefinition(
            name="viewpoint_lighting_clutter_occlusion",
            sample_type="query",
            factor_group="combined_factor",
            description="Jointly vary camera viewpoint, lighting, distractor clutter, and explicit occlusion.",
            requires_distractors=True,
            requires_occluders=True,
        ),
        "clutter": FactorDefinition(
            name="clutter",
            sample_type="query",
            factor_group="single_factor",
            description="Vary the number and arrangement of distractor objects.",
            requires_distractors=True,
        ),
        "occlusion": FactorDefinition(
            name="occlusion",
            sample_type="query",
            factor_group="single_factor",
            description="Introduce target occlusion using explicit occluders or distractors.",
            requires_distractors=True,
            requires_occluders=True,
        ),
        "depth_quality": FactorDefinition(
            name="depth_quality",
            sample_type="query",
            factor_group="single_factor",
            description="Control noise and missing-data patterns in depth.",
            requires_depth_postprocess=True,
        ),
        "mask_quality": FactorDefinition(
            name="mask_quality",
            sample_type="query",
            factor_group="single_factor",
            description="Apply deterministic degradations to the target mask after oracle export.",
            requires_mask_postprocess=True,
        ),
        "background_material": FactorDefinition(
            name="background_material",
            sample_type="query",
            factor_group="single_factor",
            description="Vary supporting surface and background appearance.",
        ),
    }


class SamplePlanner:
    """
    Deterministic planner for dataset requests.

    This planner deliberately separates "which sample should exist" from
    "how Isaac renders it". That keeps the benchmark design testable without
    requiring Isaac Sim to be running.
    """

    def __init__(self, config: GeneratorConfig, factor_registry: dict[str, FactorDefinition] | None = None):
        self.config = config
        self.factor_registry = factor_registry or build_default_factor_registry()
        self._rng = random.Random(config.random_seed)

    def plan_object_requests(self, object_name: str, counts: dict[str, int]) -> list[SampleRequest]:
        requests: list[SampleRequest] = []
        anchor_count = int(counts.get("anchors", 0))
        query_pose_count = max(
            int(counts.get("viewpoint_queries", 0)),
            int(counts.get("distance_queries", 0)),
            int(counts.get("lighting_queries", 0)),
            int(counts.get("viewpoint_lighting_queries", 0)),
            int(counts.get("viewpoint_lighting_clutter_queries", 0)),
            int(counts.get("viewpoint_lighting_occlusion_queries", 0)),
            int(counts.get("viewpoint_lighting_clutter_occlusion_queries", 0)),
            int(counts.get("clutter_queries", 0)),
            int(counts.get("occlusion_queries", 0)),
        )
        anchor_pose_bank = [
            self._sample_random_object_pose(sample_index=i, object_name=object_name, pose_role="anchor")
            for i in range(anchor_count)
        ]
        query_pose_bank = [
            self._sample_random_object_pose(sample_index=i, object_name=object_name, pose_role="query")
            for i in range(query_pose_count)
        ]

        requests.extend(self.plan_anchor_requests(object_name, anchor_count, pose_bank=anchor_pose_bank))
        requests.extend(
            self.plan_viewpoint_requests(
                object_name,
                int(counts.get("viewpoint_queries", 0)),
                pose_bank=query_pose_bank,
            )
        )
        requests.extend(
            self.plan_distance_requests(
                object_name,
                int(counts.get("distance_queries", 0)),
                pose_bank=query_pose_bank,
            )
        )
        requests.extend(
            self.plan_lighting_requests(
                object_name,
                int(counts.get("lighting_queries", 0)),
                pose_bank=query_pose_bank,
            )
        )
        requests.extend(
            self.plan_viewpoint_lighting_requests(
                object_name,
                int(counts.get("viewpoint_lighting_queries", 0)),
                pose_bank=query_pose_bank,
            )
        )
        requests.extend(
            self.plan_viewpoint_lighting_clutter_requests(
                object_name,
                int(counts.get("viewpoint_lighting_clutter_queries", 0)),
                pose_bank=query_pose_bank,
            )
        )
        requests.extend(
            self.plan_viewpoint_lighting_occlusion_requests(
                object_name,
                int(counts.get("viewpoint_lighting_occlusion_queries", 0)),
                pose_bank=query_pose_bank,
            )
        )
        requests.extend(
            self.plan_viewpoint_lighting_clutter_occlusion_requests(
                object_name,
                int(counts.get("viewpoint_lighting_clutter_occlusion_queries", 0)),
                pose_bank=query_pose_bank,
            )
        )
        requests.extend(
            self.plan_clutter_requests(
                object_name,
                int(counts.get("clutter_queries", 0)),
                pose_bank=query_pose_bank,
            )
        )
        requests.extend(
            self.plan_occlusion_requests(
                object_name,
                int(counts.get("occlusion_queries", 0)),
                pose_bank=query_pose_bank,
            )
        )
        return requests

    def plan_anchor_requests(
        self,
        object_name: str,
        num_samples: int,
        pose_bank: list[dict[str, Any]] | None = None,
    ) -> list[SampleRequest]:
        return [
            SampleRequest(
                object_name=object_name,
                sample_type="anchor",
                split="anchors",
                index=i,
                factor_name="anchor_clean",
                factor_group="none",
                factor_value=self._merge_factor_pose(
                    pose_bank,
                    i,
                    {
                        "anchor_view_id": i,
                    },
                ),
            )
            for i in range(num_samples)
        ]

    def plan_viewpoint_requests(
        self,
        object_name: str,
        num_samples: int,
        pose_bank: list[dict[str, Any]] | None = None,
    ) -> list[SampleRequest]:
        requests: list[SampleRequest] = []
        for i in range(num_samples):
            requests.append(
                SampleRequest(
                    object_name=object_name,
                    sample_type="query",
                    split="queries",
                    index=i,
                    factor_name="viewpoint",
                    factor_group="single_factor",
                    factor_value=self._merge_factor_pose(
                        pose_bank,
                        i,
                        {
                            "azimuth_deg": round(self._rng.uniform(0.0, 360.0), 3),
                            "elevation_deg": round(self._rng.uniform(10.0, 80.0), 3),
                            "camera_target_distance_m": round(self._rng.uniform(0.8, 1.8), 4),
                        },
                    ),
                )
            )
        return requests

    def plan_distance_requests(
        self,
        object_name: str,
        num_samples: int,
        pose_bank: list[dict[str, Any]] | None = None,
    ) -> list[SampleRequest]:
        requests: list[SampleRequest] = []
        if num_samples <= 0:
            return requests
        min_distance = 0.55
        max_distance = 1.8
        if num_samples == 1:
            distance_values = [round((min_distance + max_distance) * 0.5, 4)]
        else:
            distance_values = [
                round(value, 4)
                for value in np.linspace(min_distance, max_distance, num=num_samples, endpoint=True).tolist()
            ]
        base_azimuth = 35.0
        base_elevation = 22.5
        for i, distance_m in enumerate(distance_values):
            requests.append(
                SampleRequest(
                    object_name=object_name,
                    sample_type="query",
                    split="queries",
                    index=i,
                    factor_name="distance",
                    factor_group="single_factor",
                    factor_value=self._merge_factor_pose(
                        pose_bank,
                        i,
                        {
                            "azimuth_deg": round(base_azimuth, 3),
                            "elevation_deg": round(base_elevation, 3),
                            "camera_target_distance_m": float(distance_m),
                        },
                    ),
                )
            )
        return requests

    def plan_lighting_requests(
        self,
        object_name: str,
        num_samples: int,
        pose_bank: list[dict[str, Any]] | None = None,
    ) -> list[SampleRequest]:
        requests: list[SampleRequest] = []
        for i in range(num_samples):
            requests.append(
                SampleRequest(
                    object_name=object_name,
                    sample_type="query",
                    split="queries",
                    index=i,
                    factor_name="lighting",
                    factor_group="single_factor",
                    factor_value=self._merge_factor_pose(pose_bank, i, self._sample_lighting_factor(i)),
                )
            )
        return requests

    def plan_viewpoint_lighting_requests(
        self,
        object_name: str,
        num_samples: int,
        pose_bank: list[dict[str, Any]] | None = None,
    ) -> list[SampleRequest]:
        requests: list[SampleRequest] = []
        for i in range(num_samples):
            factor_value = {
                "azimuth_deg": round(self._rng.uniform(0.0, 360.0), 3),
                "elevation_deg": round(self._rng.uniform(12.0, 78.0), 3),
                "camera_target_distance_m": round(self._rng.uniform(0.85, 1.75), 4),
            }
            factor_value.update(self._sample_lighting_factor(i))
            factor_value = self._merge_factor_pose(pose_bank, i, factor_value)
            requests.append(
                SampleRequest(
                    object_name=object_name,
                    sample_type="query",
                    split="queries",
                    index=i,
                    factor_name="viewpoint_lighting",
                    factor_group="combined_factor",
                    factor_value=factor_value,
                )
            )
        return requests

    def plan_viewpoint_lighting_clutter_requests(
        self,
        object_name: str,
        num_samples: int,
        pose_bank: list[dict[str, Any]] | None = None,
    ) -> list[SampleRequest]:
        requests: list[SampleRequest] = []
        if num_samples <= 0:
            return requests
        clutter_levels = [1, 2, 3, 5, 7]
        for i in range(num_samples):
            factor_value = {
                "azimuth_deg": round(self._rng.uniform(0.0, 360.0), 3),
                "elevation_deg": round(self._rng.uniform(12.0, 78.0), 3),
                "camera_target_distance_m": round(self._rng.uniform(0.85, 1.55), 4),
                "distractor_count": int(min(clutter_levels[i % len(clutter_levels)], self.config.scene.max_distractors)),
            }
            factor_value.update(self._sample_lighting_factor(i))
            factor_value = self._merge_factor_pose(pose_bank, i, factor_value)
            requests.append(
                SampleRequest(
                    object_name=object_name,
                    sample_type="query",
                    split="queries",
                    index=i,
                    factor_name="viewpoint_lighting_clutter",
                    factor_group="combined_factor",
                    factor_value=factor_value,
                )
            )
        return requests

    def plan_viewpoint_lighting_occlusion_requests(
        self,
        object_name: str,
        num_samples: int,
        pose_bank: list[dict[str, Any]] | None = None,
    ) -> list[SampleRequest]:
        requests: list[SampleRequest] = []
        if num_samples <= 0:
            return requests
        for i in range(num_samples):
            factor_value = {
                "azimuth_deg": round(self._rng.uniform(0.0, 360.0), 3),
                "elevation_deg": round(self._rng.uniform(12.0, 78.0), 3),
                "camera_target_distance_m": round(self._rng.uniform(0.85, 1.55), 4),
            }
            factor_value.update(self._sample_lighting_factor(i))
            factor_value.update(self._sample_occlusion_factor(i, num_samples))
            factor_value = self._merge_factor_pose(pose_bank, i, factor_value)
            requests.append(
                SampleRequest(
                    object_name=object_name,
                    sample_type="query",
                    split="queries",
                    index=i,
                    factor_name="viewpoint_lighting_occlusion",
                    factor_group="combined_factor",
                    factor_value=factor_value,
                )
            )
        return requests

    def plan_viewpoint_lighting_clutter_occlusion_requests(
        self,
        object_name: str,
        num_samples: int,
        pose_bank: list[dict[str, Any]] | None = None,
    ) -> list[SampleRequest]:
        requests: list[SampleRequest] = []
        if num_samples <= 0:
            return requests
        clutter_levels = [1, 2, 3, 5, 7]
        for i in range(num_samples):
            factor_value = {
                "azimuth_deg": round(self._rng.uniform(0.0, 360.0), 3),
                "elevation_deg": round(self._rng.uniform(12.0, 78.0), 3),
                "camera_target_distance_m": round(self._rng.uniform(0.85, 1.45), 4),
                "distractor_count": int(min(clutter_levels[i % len(clutter_levels)], self.config.scene.max_distractors)),
            }
            factor_value.update(self._sample_lighting_factor(i))
            factor_value.update(self._sample_occlusion_factor(i, num_samples))
            factor_value = self._merge_factor_pose(pose_bank, i, factor_value)
            requests.append(
                SampleRequest(
                    object_name=object_name,
                    sample_type="query",
                    split="queries",
                    index=i,
                    factor_name="viewpoint_lighting_clutter_occlusion",
                    factor_group="combined_factor",
                    factor_value=factor_value,
                )
            )
        return requests

    def _sample_random_object_pose(
        self,
        sample_index: int,
        object_name: str,
        pose_role: str,
    ) -> dict[str, Any]:
        role_seed = self._pose_stream_seed(object_name=object_name, pose_role=pose_role, sample_index=sample_index)
        rng = random.Random(role_seed)
        mount_x, mount_y, _ = self.config.scene.target_mount_position_m
        translation_x = float(mount_x + rng.uniform(-0.08, 0.08))
        translation_y = float(mount_y + rng.uniform(-0.16, 0.16))
        regime = rng.random()
        if regime < 0.5:
            roll_deg = rng.uniform(-12.0, 12.0)
            pitch_deg = rng.uniform(-12.0, 12.0)
        elif regime < 0.82:
            roll_deg = rng.uniform(-28.0, 28.0)
            pitch_deg = rng.uniform(-28.0, 28.0)
        else:
            roll_deg = rng.uniform(-55.0, 55.0)
            pitch_deg = rng.uniform(-55.0, 55.0)
        yaw_deg = rng.uniform(-180.0, 180.0)
        return {
            "episode_pose_id": int(sample_index),
            "target_pose_sampling_mode": "episode_random_6d_v2",
            "target_pose_role": pose_role,
            "target_position_xy_m": [
                round(translation_x, 4),
                round(translation_y, 4),
            ],
            "target_orientation_rpy_deg": [
                round(float(roll_deg), 3),
                round(float(pitch_deg), 3),
                round(float(yaw_deg), 3),
            ],
        }

    def _pose_stream_seed(self, object_name: str, pose_role: str, sample_index: int) -> int:
        role_offset = 17 if pose_role == "anchor" else 53
        object_hash = sum((idx + 1) * ord(ch) for idx, ch in enumerate(object_name))
        return int(self.config.random_seed * 1009 + object_hash * 97 + role_offset * 7919 + sample_index * 104729)

    def _merge_factor_pose(
        self,
        pose_bank: list[dict[str, Any]] | None,
        sample_index: int,
        factor_value: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(factor_value)
        if pose_bank is not None and sample_index < len(pose_bank):
            merged.update(dict(pose_bank[sample_index]))
        return merged

    def _sample_lighting_factor(self, sample_index: int) -> dict[str, Any]:
        families = ("soft_box", "hard_spot", "cool_lab", "sunset_split", "neon")
        family = families[sample_index % len(families)]
        # Keep the light schemas fixed during large campaigns. Changing the light
        # type forces stage churn (`DeletePrims` + `Define`) and becomes one of the
        # main throughput bottlenecks on long runs.
        key_light_type = "Rect"
        fill_light_type = "Sphere"

        if family == "soft_box":
            dome_intensity = self._rng.uniform(10.0, 45.0)
            dome_color = self._sample_light_color(0.48, 0.78, 0.08, 0.45, 0.82, 1.0)
            key_intensity = self._rng.uniform(25000.0, 85000.0)
            fill_intensity = self._rng.uniform(6000.0, 26000.0)
            key_scale = self._rng.uniform(0.22, 0.42)
            fill_scale = self._rng.uniform(0.14, 0.28)
        elif family == "hard_spot":
            dome_intensity = self._rng.uniform(0.05, 3.0)
            dome_color = self._sample_light_color(0.02, 0.18, 0.25, 0.78, 0.72, 1.0)
            key_intensity = self._rng.uniform(110000.0, 260000.0)
            fill_intensity = self._rng.uniform(500.0, 9000.0)
            key_scale = self._rng.uniform(0.04, 0.18)
            fill_scale = self._rng.uniform(0.08, 0.24)
        elif family == "cool_lab":
            dome_intensity = self._rng.uniform(2.0, 14.0)
            dome_color = self._sample_light_color(0.46, 0.72, 0.25, 0.82, 0.72, 1.0)
            key_intensity = self._rng.uniform(65000.0, 180000.0)
            fill_intensity = self._rng.uniform(1500.0, 22000.0)
            key_scale = self._rng.uniform(0.16, 0.34)
            fill_scale = self._rng.uniform(0.1, 0.24)
        elif family == "sunset_split":
            dome_intensity = self._rng.uniform(0.1, 6.0)
            dome_color = self._sample_light_color(0.0, 0.16, 0.35, 0.9, 0.72, 1.0)
            key_intensity = self._rng.uniform(70000.0, 210000.0)
            fill_intensity = self._rng.uniform(4000.0, 42000.0)
            key_scale = self._rng.uniform(0.06, 0.24)
            fill_scale = self._rng.uniform(0.12, 0.34)
        else:
            dome_intensity = self._rng.uniform(0.0, 1.0)
            dome_color = self._sample_light_color(0.0, 1.0, 0.35, 1.0, 0.62, 1.0)
            key_intensity = self._rng.uniform(90000.0, 240000.0)
            fill_intensity = self._rng.uniform(20000.0, 140000.0)
            key_scale = self._rng.uniform(0.05, 0.22)
            fill_scale = self._rng.uniform(0.05, 0.22)

        key_scale = self._stabilize_light_scale(key_light_type, key_scale)
        fill_scale = self._stabilize_light_scale(fill_light_type, fill_scale)
        key_position = self._sample_light_position(
            radius_xy=(0.9, 2.4),
            z_range=(1.15, 2.3),
            avoid_view_corridor=True,
            corridor_clearance_m=0.48,
        )
        fill_position = self._sample_light_position(
            radius_xy=(0.35, 2.0),
            z_range=(0.85, 1.9),
            avoid_view_corridor=True,
            corridor_clearance_m=0.36,
        )
        key_color = self._sample_family_key_color(family)
        fill_color = self._sample_family_fill_color(family, key_color)
        if self._rng.random() < 0.4:
            dome_color = self._sample_light_color(0.0, 1.0, 0.25, 0.95, 0.72, 1.0)
        if self._rng.random() < 0.55:
            key_color = self._sample_light_color(0.0, 1.0, 0.6, 1.0, 0.78, 1.0)
        if self._rng.random() < 0.55:
            fill_color = self._sample_light_color(0.0, 1.0, 0.55, 1.0, 0.74, 1.0)
        if self._rng.random() < 0.38:
            monochrome_hue = self._sample_monochrome_hue()
            dome_color, key_color, fill_color = self._sample_monochrome_palette(monochrome_hue)

        return {
            "lighting_profile": family,
            "dome_intensity": round(dome_intensity, 4),
            "dome_color": [round(v, 4) for v in dome_color],
            "key_light_type": key_light_type,
            "key_intensity": round(key_intensity, 4),
            "key_color": [round(v, 4) for v in key_color],
            "key_position": [round(v, 4) for v in key_position],
            "key_scale": round(key_scale, 4),
            "fill_light_type": fill_light_type,
            "fill_intensity": round(fill_intensity, 4),
            "fill_color": [round(v, 4) for v in fill_color],
            "fill_position": [round(v, 4) for v in fill_position],
            "fill_scale": round(fill_scale, 4),
        }

    def _sample_monochrome_hue(self) -> float:
        anchors = (
            0.0,   # red
            0.08,  # orange
            0.16,  # yellow-green
            0.33,  # green
            0.52,  # cyan
            0.66,  # blue
            0.78,  # violet
            0.9,   # magenta
        )
        base_hue = self._rng.choice(anchors)
        return float((base_hue + self._rng.uniform(-0.025, 0.025)) % 1.0)

    def _sample_monochrome_palette(
        self,
        base_hue: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        dome = self._hsv_to_rgb_local(
            base_hue + self._rng.uniform(-0.015, 0.015),
            self._rng.uniform(0.8, 1.0),
            self._rng.uniform(0.72, 0.96),
        )
        key = self._hsv_to_rgb_local(
            base_hue + self._rng.uniform(-0.01, 0.01),
            self._rng.uniform(0.9, 1.0),
            self._rng.uniform(0.88, 1.0),
        )
        fill = self._hsv_to_rgb_local(
            base_hue + self._rng.uniform(-0.02, 0.02),
            self._rng.uniform(0.82, 1.0),
            self._rng.uniform(0.72, 0.94),
        )
        return dome, key, fill

    def _sample_light_position(
        self,
        radius_xy: tuple[float, float],
        z_range: tuple[float, float],
        avoid_view_corridor: bool,
        corridor_clearance_m: float,
    ) -> list[float]:
        target = np.array([0.5, 0.0, 0.09], dtype=np.float64)
        camera = self._lighting_reference_camera_position(target)
        for _ in range(48):
            azimuth = self._rng.uniform(0.0, 2.0 * np.pi)
            radius = self._rng.uniform(radius_xy[0], radius_xy[1])
            z_value = self._rng.uniform(z_range[0], z_range[1])
            position = np.array(
                [
                    0.5 + radius * float(np.cos(azimuth)),
                    radius * float(np.sin(azimuth)),
                    z_value,
                ],
                dtype=np.float64,
            )
            if not avoid_view_corridor:
                return position.tolist()
            if self._distance_point_to_segment(position, camera, target) < corridor_clearance_m:
                continue
            view_dir = camera - target
            light_dir = position - target
            if np.linalg.norm(light_dir[:2]) < 1e-6:
                continue
            view_dir_xy = view_dir[:2] / max(np.linalg.norm(view_dir[:2]), 1e-8)
            light_dir_xy = light_dir[:2] / max(np.linalg.norm(light_dir[:2]), 1e-8)
            if float(np.dot(view_dir_xy, light_dir_xy)) > 0.72:
                continue
            return position.tolist()
        fallback_azimuth = np.radians(35.0 + 135.0)
        fallback_radius = max(radius_xy[0], min(radius_xy[1], 1.25))
        fallback_z = max(z_range[0], min(z_range[1], 1.1))
        return [
            0.5 + fallback_radius * float(np.cos(fallback_azimuth)),
            fallback_radius * float(np.sin(fallback_azimuth)),
            fallback_z,
        ]

    def _stabilize_light_scale(self, light_type: str, scale: float) -> float:
        light_type = str(light_type).strip().title()
        if light_type == "Rect":
            return float(np.clip(scale, 0.08, 0.26))
        if light_type == "Disk":
            return float(np.clip(scale, 0.06, 0.2))
        if light_type == "Sphere":
            return float(np.clip(scale, 0.05, 0.18))
        if light_type == "Distant":
            return float(np.clip(scale, 0.04, 0.14))
        return float(scale)

    def _lighting_reference_camera_position(self, target: np.ndarray) -> np.ndarray:
        radius = 0.68
        azimuth = np.radians(35.0)
        elevation = np.radians(22.5)
        return np.array(
            [
                target[0] + radius * np.cos(elevation) * np.cos(azimuth),
                target[1] + radius * np.cos(elevation) * np.sin(azimuth),
                target[2] + radius * np.sin(elevation),
            ],
            dtype=np.float64,
        )

    def _distance_point_to_segment(
        self,
        point: np.ndarray,
        seg_a: np.ndarray,
        seg_b: np.ndarray,
    ) -> float:
        point = np.asarray(point, dtype=np.float64)
        seg_a = np.asarray(seg_a, dtype=np.float64)
        seg_b = np.asarray(seg_b, dtype=np.float64)
        seg = seg_b - seg_a
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq <= 1e-8:
            return float(np.linalg.norm(point - seg_a))
        t = float(np.dot(point - seg_a, seg) / seg_len_sq)
        t = float(np.clip(t, 0.0, 1.0))
        closest = seg_a + t * seg
        return float(np.linalg.norm(point - closest))

    def _sample_family_key_color(self, family: str) -> tuple[float, float, float]:
        if family == "soft_box":
            return self._sample_light_color(0.0, 0.22, 0.08, 0.5, 0.88, 1.0)
        if family == "hard_spot":
            return self._sample_light_color(0.0, 0.18, 0.55, 1.0, 0.84, 1.0)
        if family == "cool_lab":
            return self._sample_light_color(0.48, 0.78, 0.45, 1.0, 0.76, 1.0)
        if family == "sunset_split":
            return self._sample_light_color(0.0, 0.12, 0.68, 1.0, 0.78, 1.0)
        return self._sample_light_color(0.76, 1.0, 0.8, 1.0, 0.72, 1.0)

    def _sample_family_fill_color(
        self,
        family: str,
        key_color: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        if family == "soft_box":
            return self._sample_light_color(0.42, 0.72, 0.12, 0.55, 0.82, 1.0)
        if family == "hard_spot":
            return self._sample_light_color(0.02, 0.18, 0.28, 0.8, 0.68, 1.0)
        if family == "cool_lab":
            return self._sample_light_color(0.4, 0.82, 0.32, 0.95, 0.74, 1.0)
        if family == "sunset_split":
            return self._sample_light_color(0.44, 0.7, 0.62, 1.0, 0.72, 1.0)
        # Neon family: encourage strong complementary contrast.
        key_h, _, _ = self._rgb_to_hsv(*key_color)
        complementary_h = (key_h + 0.48 + self._rng.uniform(-0.12, 0.12)) % 1.0
        return self._hsv_to_rgb_local(complementary_h, self._rng.uniform(0.82, 1.0), self._rng.uniform(0.82, 1.0))

    def _sample_light_color(
        self,
        hue_min: float,
        hue_max: float,
        sat_min: float,
        sat_max: float,
        val_min: float,
        val_max: float,
    ) -> tuple[float, float, float]:
        hue = self._rng.uniform(hue_min, hue_max)
        sat = self._rng.uniform(sat_min, sat_max)
        val = self._rng.uniform(val_min, val_max)
        return self._hsv_to_rgb_local(hue, sat, val)

    def _hsv_to_rgb_local(self, h: float, s: float, v: float) -> tuple[float, float, float]:
        h = float(h % 1.0)
        s = float(np.clip(s, 0.0, 1.0))
        v = float(np.clip(v, 0.0, 1.0))
        i = int(h * 6.0)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = i % 6
        if i == 0:
            return (v, t, p)
        if i == 1:
            return (q, v, p)
        if i == 2:
            return (p, v, t)
        if i == 3:
            return (p, q, v)
        if i == 4:
            return (t, p, v)
        return (v, p, q)

    def _rgb_to_hsv(self, r: float, g: float, b: float) -> tuple[float, float, float]:
        r = float(np.clip(r, 0.0, 1.0))
        g = float(np.clip(g, 0.0, 1.0))
        b = float(np.clip(b, 0.0, 1.0))
        mx = max(r, g, b)
        mn = min(r, g, b)
        delta = mx - mn
        if delta <= 1e-8:
            h = 0.0
        elif mx == r:
            h = ((g - b) / delta) % 6.0
        elif mx == g:
            h = ((b - r) / delta) + 2.0
        else:
            h = ((r - g) / delta) + 4.0
        h = (h / 6.0) % 1.0
        s = 0.0 if mx <= 1e-8 else delta / mx
        return (h, s, mx)

    def plan_clutter_requests(
        self,
        object_name: str,
        num_samples: int,
        pose_bank: list[dict[str, Any]] | None = None,
    ) -> list[SampleRequest]:
        requests: list[SampleRequest] = []
        if num_samples <= 0:
            return requests
        clutter_levels = [1, 2, 3, 5, 7]
        for i in range(num_samples):
            distractor_count = clutter_levels[i % len(clutter_levels)]
            requests.append(
                SampleRequest(
                    object_name=object_name,
                    sample_type="query",
                    split="queries",
                    index=i,
                    factor_name="clutter",
                    factor_group="single_factor",
                    factor_value=self._merge_factor_pose(
                        pose_bank,
                        i,
                        {
                            "distractor_count": int(min(distractor_count, self.config.scene.max_distractors)),
                            "azimuth_deg": 35.0,
                            "elevation_deg": 22.5,
                            "camera_target_distance_m": 0.95,
                        },
                    ),
                )
            )
        return requests

    def plan_occlusion_requests(
        self,
        object_name: str,
        num_samples: int,
        pose_bank: list[dict[str, Any]] | None = None,
    ) -> list[SampleRequest]:
        requests: list[SampleRequest] = []
        if num_samples <= 0:
            return requests
        for i in range(num_samples):
            factor_value = self._sample_occlusion_factor(i, num_samples)
            factor_value.update(
                {
                    "azimuth_deg": 35.0,
                    "elevation_deg": 22.5,
                    "camera_target_distance_m": 0.95,
                }
            )
            factor_value = self._merge_factor_pose(pose_bank, i, factor_value)
            requests.append(
                SampleRequest(
                    object_name=object_name,
                    sample_type="query",
                    split="queries",
                    index=i,
                    factor_name="occlusion",
                    factor_group="single_factor",
                    factor_value=factor_value,
                )
            )
        return requests

    def _sample_occlusion_factor(self, sample_index: int, num_samples: int) -> dict[str, Any]:
        normalized_index = (sample_index + 0.5) / num_samples
        if normalized_index < (1.0 / 3.0):
            local_progress = normalized_index / (1.0 / 3.0)
            overlap_bias = 0.02 + 0.24 * local_progress
        elif normalized_index < (2.0 / 3.0):
            local_progress = (normalized_index - (1.0 / 3.0)) / (1.0 / 3.0)
            overlap_bias = 0.28 + 0.28 * local_progress
        else:
            local_progress = (normalized_index - (2.0 / 3.0)) / (1.0 / 3.0)
            overlap_bias = 0.62 + 0.28 * local_progress
        overlap_bias = float(np.clip(overlap_bias + self._rng.uniform(-0.05, 0.05), 0.0, 0.95))
        occluder_count = 1
        if overlap_bias > 0.26:
            occluder_count = 2
        if overlap_bias > 0.58:
            occluder_count = 3
        if overlap_bias > 0.86:
            occluder_count = 4
        occluder_count = min(occluder_count, self.config.scene.max_distractors)
        occluders: list[dict[str, float]] = []
        lateral_base = 0.08 - 0.06 * overlap_bias
        depth_base = 0.19 - 0.08 * overlap_bias
        scale_base = 0.78 + 0.5 * overlap_bias
        height_base = 0.052 + 0.014 * overlap_bias
        for occ_idx in range(occluder_count):
            side = -1.0 if occ_idx % 2 == 0 else 1.0
            center_pull = max(0.18, 1.0 - min(0.75, overlap_bias))
            if occ_idx == 0:
                lateral = side * max(0.0, lateral_base * center_pull + self._rng.uniform(-0.01, 0.01))
                depth = max(0.075, depth_base + self._rng.uniform(-0.012, 0.012))
                scale = max(0.72, scale_base + self._rng.uniform(-0.08, 0.08))
            else:
                lateral_multiplier = 0.55 - 0.12 * min(occ_idx, 2) + 0.18 * (1.0 - overlap_bias)
                lateral = side * max(
                    0.0,
                    lateral_base * max(0.1, lateral_multiplier) * center_pull + self._rng.uniform(-0.012, 0.012),
                )
                depth = max(0.075, depth_base + 0.014 * occ_idx + self._rng.uniform(-0.012, 0.012))
                scale = max(
                    0.72,
                    scale_base * (0.92 - 0.09 * min(occ_idx, 3)) + self._rng.uniform(-0.06, 0.06),
                )
            occluders.append(
                {
                    "lateral": round(float(lateral), 4),
                    "depth": round(float(depth), 4),
                    "scale": round(float(scale), 4),
                    "height": round(float(height_base + self._rng.uniform(-0.006, 0.006)), 4),
                    "yaw_deg": round(float(self._rng.uniform(0.0, 65.0)), 3),
                }
            )
        return {
            "occlusion_sampling_mode": "distributional_geometry_v1",
            "sample_overlap_bias": round(overlap_bias, 4),
            "occluder_count": int(occluder_count),
            "occluders": occluders,
        }

    def plan_dataset(
        self,
        anchors_per_object: int,
        viewpoint_queries_per_object: int,
        distance_queries_per_object: int = 0,
        lighting_queries_per_object: int = 0,
        viewpoint_lighting_queries_per_object: int = 0,
        viewpoint_lighting_clutter_queries_per_object: int = 0,
        viewpoint_lighting_occlusion_queries_per_object: int = 0,
        viewpoint_lighting_clutter_occlusion_queries_per_object: int = 0,
        clutter_queries_per_object: int = 0,
        occlusion_queries_per_object: int = 0,
    ) -> list[SampleRequest]:
        requests: list[SampleRequest] = []
        for object_name in self.config.scene.target_objects:
            requests.extend(
                self.plan_object_requests(
                    object_name,
                    {
                        "anchors": anchors_per_object,
                        "viewpoint_queries": viewpoint_queries_per_object,
                        "distance_queries": distance_queries_per_object,
                        "lighting_queries": lighting_queries_per_object,
                        "viewpoint_lighting_queries": viewpoint_lighting_queries_per_object,
                        "viewpoint_lighting_clutter_queries": viewpoint_lighting_clutter_queries_per_object,
                        "viewpoint_lighting_occlusion_queries": viewpoint_lighting_occlusion_queries_per_object,
                        "viewpoint_lighting_clutter_occlusion_queries": viewpoint_lighting_clutter_occlusion_queries_per_object,
                        "clutter_queries": clutter_queries_per_object,
                        "occlusion_queries": occlusion_queries_per_object,
                    },
                )
            )
        return requests


class Any6DPathBuilder:
    """Enforces the folder structure and naming scheme required by the spec."""

    def __init__(self, output_root: Path):
        self.output_root = output_root

    def build(self, request: SampleRequest) -> SamplePaths:
        if request.sample_type == "anchor":
            sample_dir = self.output_root / "anchors" / request.object_name / f"{request.index:03d}"
            sample_id = f"anchors/{request.object_name}/{request.index:03d}"
        else:
            sample_dir = self.output_root / "queries" / request.factor_name / request.object_name / f"{request.index:06d}"
            sample_id = f"queries/{request.factor_name}/{request.object_name}/{request.index:06d}"

        return SamplePaths(
            sample_dir=sample_dir,
            rgb_path=sample_dir / "rgb.png",
            depth_path=sample_dir / "depth.png",
            mask_target_path=sample_dir / "mask_target.png",
            mask_visib_target_path=sample_dir / "mask_visib_target.png",
            k_path=sample_dir / "K.txt",
            meta_path=sample_dir / "meta.json",
            sample_id=sample_id,
        )


class MetadataBuilder:
    """Builds Any6D-compliant `meta.json` from normalized capture artifacts."""

    def __init__(self, config: GeneratorConfig):
        self.config = config

    def build(self, request: SampleRequest, paths: SamplePaths, artifacts: CaptureArtifacts, object_usd_path: str) -> dict[str, Any]:
        self._validate_artifacts(artifacts)
        object_frame_correction = self._get_object_frame_correction(request.object_name)
        t_w_o_eval = artifacts.t_w_o
        t_c_o_eval = artifacts.t_c_o
        if object_frame_correction is not None:
            t_w_o_eval = (artifacts.t_w_o @ object_frame_correction).astype(np.float32)
            t_c_o_eval = (artifacts.t_c_o @ object_frame_correction).astype(np.float32)

        metadata: dict[str, Any] = {
            "sample_id": paths.sample_id,
            "sample_type": request.sample_type,
            "split": request.split,
            "factor_group": request.factor_group,
            "factor_name": request.factor_name,
            "factor_value": request.factor_value,
            "object_name": request.object_name,
            "object_usd_path": object_usd_path,
            "camera_convention": "cv",
            "raw_camera_convention": "isaac_usd",
            "object_frame_convention_raw": self.config.object_frame_convention_raw,
            "object_frame_convention_eval": self.config.object_frame_convention_eval,
            "image_width_px": self.config.camera.resolution.width,
            "image_height_px": self.config.camera.resolution.height,
            "depth_scale": self.config.output.depth_scale,
            "K": artifacts.k_matrix.tolist(),
            "T_W_C": artifacts.t_w_c.tolist(),
            "T_W_O": artifacts.t_w_o.tolist(),
            "T_C_O": artifacts.t_c_o.tolist(),
            "T_W_C_raw_isaac_usd": artifacts.t_w_c_raw_isaac_usd.tolist(),
            "T_C_O_raw_isaac_usd": artifacts.t_c_o_raw_isaac_usd.tolist(),
            "bbox_2d_tight_xyxy": artifacts.bbox_2d_tight_xyxy,
            "target_total_mask_pixel_count": artifacts.target_total_mask_pixel_count,
            "target_visible_pixel_count": artifacts.target_visible_pixel_count,
            "target_depth_valid_pixel_count": artifacts.target_depth_valid_pixel_count,
            "visibility_ratio": artifacts.visibility_ratio,
            "occlusion_ratio": artifacts.occlusion_ratio,
        }
        if object_frame_correction is not None:
            metadata["T_O_isaac_to_any6d_ref"] = object_frame_correction.tolist()
            metadata["T_W_O_any6d_ref"] = t_w_o_eval.tolist()
            metadata["T_C_O_any6d_ref"] = t_c_o_eval.tolist()
        else:
            metadata["T_O_isaac_to_any6d_ref"] = None
            metadata["T_W_O_any6d_ref"] = None
            metadata["T_C_O_any6d_ref"] = None

        return metadata

    def _get_object_frame_correction(self, object_name: str) -> np.ndarray | None:
        matrix = self.config.object_frame_corrections.get(object_name)
        if matrix is None:
            return None
        matrix = np.array(matrix, dtype=np.float32)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"Invalid object-frame correction shape for {object_name}: {matrix.shape}. Expected (4, 4)."
            )
        return matrix

    def _validate_artifacts(self, artifacts: CaptureArtifacts) -> None:
        required_arrays = {
            "mask_target": artifacts.mask_target,
            "mask_visib_target": artifacts.mask_visib_target,
            "k_matrix": artifacts.k_matrix,
            "t_w_c": artifacts.t_w_c,
            "t_w_o": artifacts.t_w_o,
            "t_c_o": artifacts.t_c_o,
            "t_w_c_raw_isaac_usd": artifacts.t_w_c_raw_isaac_usd,
            "t_c_o_raw_isaac_usd": artifacts.t_c_o_raw_isaac_usd,
        }
        for name, value in required_arrays.items():
            if value is None:
                raise ValueError(f"Missing required artifact: {name}")

        if artifacts.k_matrix.shape != (3, 3):
            raise ValueError(f"Invalid K shape: {artifacts.k_matrix.shape}")
        for matrix_name, matrix in (
            ("T_W_C", artifacts.t_w_c),
            ("T_W_O", artifacts.t_w_o),
            ("T_C_O", artifacts.t_c_o),
            ("T_W_C_raw_isaac_usd", artifacts.t_w_c_raw_isaac_usd),
            ("T_C_O_raw_isaac_usd", artifacts.t_c_o_raw_isaac_usd),
        ):
            if matrix.shape != (4, 4):
                raise ValueError(f"Invalid {matrix_name} shape: {matrix.shape}")
            if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=matrix.dtype), atol=1e-4):
                raise ValueError(f"Invalid {matrix_name} homogeneous last row: {matrix[3].tolist()}")

        if artifacts.bbox_2d_tight_xyxy is None:
            raise ValueError("Missing required artifact: bbox_2d_tight_xyxy")
        if len(artifacts.bbox_2d_tight_xyxy) != 4:
            raise ValueError(f"Invalid bbox_2d_tight_xyxy length: {artifacts.bbox_2d_tight_xyxy}")
        x_min, y_min, x_max, y_max = (int(v) for v in artifacts.bbox_2d_tight_xyxy)
        width = self.config.camera.resolution.width
        height = self.config.camera.resolution.height
        if x_min < 0 or y_min < 0 or x_max >= width or y_max >= height:
            raise ValueError(f"Bounding box out of image bounds: {artifacts.bbox_2d_tight_xyxy}")
        if x_max < x_min or y_max < y_min:
            raise ValueError(f"Degenerate bounding box: {artifacts.bbox_2d_tight_xyxy}")
        if artifacts.target_total_mask_pixel_count is None or artifacts.target_visible_pixel_count is None:
            raise ValueError("Missing target pixel counts")
        if artifacts.target_depth_valid_pixel_count is None:
            raise ValueError("Missing target depth valid pixel count")
        if int(artifacts.target_total_mask_pixel_count) <= 0:
            raise ValueError("target_total_mask_pixel_count must be > 0")
        if int(artifacts.target_visible_pixel_count) <= 0:
            raise ValueError("target_visible_pixel_count must be > 0")
        if int(artifacts.target_visible_pixel_count) > int(artifacts.target_total_mask_pixel_count):
            raise ValueError(
                "target_visible_pixel_count cannot exceed target_total_mask_pixel_count"
            )
        if int(artifacts.target_depth_valid_pixel_count) <= 0:
            raise ValueError("target_depth_valid_pixel_count must be > 0")
        if artifacts.visibility_ratio is None or artifacts.occlusion_ratio is None:
            raise ValueError("Missing visibility / occlusion ratios")
        if not (0.0 <= float(artifacts.visibility_ratio) <= 1.0):
            raise ValueError(f"Invalid visibility_ratio: {artifacts.visibility_ratio}")
        if not (0.0 <= float(artifacts.occlusion_ratio) <= 1.0):
            raise ValueError(f"Invalid occlusion_ratio: {artifacts.occlusion_ratio}")
        if not np.allclose(
            float(artifacts.visibility_ratio) + float(artifacts.occlusion_ratio),
            1.0,
            atol=1e-4,
        ):
            raise ValueError(
                "visibility_ratio and occlusion_ratio must sum to 1 within tolerance"
            )
        expected_shape = (
            self.config.camera.resolution.height,
            self.config.camera.resolution.width,
        )
        for mask_name, mask in (
            ("mask_target", artifacts.mask_target),
            ("mask_visib_target", artifacts.mask_visib_target),
        ):
            if mask.shape != expected_shape:
                raise ValueError(f"Invalid {mask_name} shape: {mask.shape}")


class Any6DDataGenerator:
    """
    Main orchestration facade.

    Implementation status:
    - dataset planning: ready
    - path and metadata contracts: ready
    - scene setup / capture / file emission: intentionally stubbed

    The next coding pass should implement:
    1. SimulationApp and Isaac imports
    2. stage + scene construction
    3. annotator attachment
    4. two-pass capture
    5. image/mask/depth writing
    6. dataset validation hooks
    """

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.factor_registry = build_default_factor_registry()
        self.path_builder = Any6DPathBuilder(config.output.root_dir)
        self.metadata_builder = MetadataBuilder(config)
        self.planner = SamplePlanner(config, self.factor_registry)
        self.scene_handles = SceneHandles(
            distractor_paths=[f"/World/Distractor_{i}" for i in range(config.scene.max_distractors)],
            occluder_paths=[f"/World/Occluder_{i}" for i in range(config.scene.max_distractors)],
        )
        self._rng = random.Random(config.random_seed)
        self._simulation_app = None
        self._world = None
        self._stage = None
        self._assets_root = None
        self._render_product = None
        self._annotators: dict[str, Any] = {}
        self._target_object_name: str | None = None
        self._distractor_style_map: dict[str, dict[str, Any]] = {}
        self._occluder_style_map: dict[str, dict[str, Any]] = {}
        self._runtime_imports: dict[str, Any] = {}

    def _debug_log(self, message: str) -> None:
        if self.config.debug_logging:
            log(message)

    def summarize_architecture(self) -> dict[str, Any]:
        """Small helper to make the proposal inspectable without running Isaac."""
        return {
            "output_root": str(self.config.output.root_dir),
            "resolution": {
                "width": self.config.camera.resolution.width,
                "height": self.config.camera.resolution.height,
            },
            "target_objects": list(self.config.scene.target_objects.keys()),
            "factor_registry": {name: definition.description for name, definition in self.factor_registry.items()},
            "include_robot": self.config.scene.include_robot,
            "max_distractors": self.config.scene.max_distractors,
        }

    def build_requests(self, anchors_per_object: int = 10, viewpoint_queries_per_object: int = 100) -> list[SampleRequest]:
        return self.planner.plan_dataset(
            anchors_per_object=anchors_per_object,
            viewpoint_queries_per_object=viewpoint_queries_per_object,
        )

    def ensure_sample_dir(self, paths: SamplePaths) -> None:
        paths.sample_dir.mkdir(parents=True, exist_ok=True)

    def write_metadata_preview(self, request: SampleRequest, artifacts: CaptureArtifacts) -> dict[str, Any]:
        """
        Useful during implementation: build a valid metadata dict before wiring
        real disk writes for images and matrices.
        """
        object_usd_path = self.config.scene.target_objects[request.object_name]
        paths = self.path_builder.build(request)
        return self.metadata_builder.build(request, paths, artifacts, object_usd_path)

    def export_plan(self, requests: Iterable[SampleRequest], output_path: Path | None = None) -> Path:
        plan = []
        for request in requests:
            paths = self.path_builder.build(request)
            plan.append(
                {
                    "sample_id": paths.sample_id,
                    "sample_dir": str(paths.sample_dir),
                    "object_name": request.object_name,
                    "sample_type": request.sample_type,
                    "factor_name": request.factor_name,
                    "factor_value": request.factor_value,
                }
            )

        output_path = output_path or (self.config.output.root_dir / "_generator_plan.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        return output_path

    def setup_scene(self) -> None:
        log("setup_scene: start")
        self._ensure_runtime()

        SimulationApp = self._runtime_imports["SimulationApp"]
        World = self._runtime_imports["World"]
        add_labels = self._runtime_imports["add_labels"]
        get_assets_root_path = self._runtime_imports["get_assets_root_path"]
        rep = self._runtime_imports["rep"]

        if self._simulation_app is None:
            self._simulation_app = SimulationApp({"headless": True})

        if self._world is not None:
            return

        self._world = World(stage_units_in_meters=1.0)
        self._stage = self._world.stage
        self._assets_root = self._resolve_assets_root(get_assets_root_path)

        if self.config.scene.include_ground_plane:
            self._world.scene.add_default_ground_plane()

        vertical_aperture_mm = (
            self.config.camera.horizontal_aperture_mm
            * self.config.camera.resolution.height
            / self.config.camera.resolution.width
        )

        target = np.array(self.config.scene.target_mount_position_m, dtype=np.float64)
        default_camera_position = target + np.array([1.0, 0.0, 0.5], dtype=np.float64)
        camera_rep = rep.create.camera(
            name="Any6D_Camera",
            focal_length=self.config.camera.focal_length_mm,
            horizontal_aperture=self.config.camera.horizontal_aperture_mm,
            clipping_range=(self.config.camera.near_m, self.config.camera.far_m),
            projection_type="pinhole",
            position=tuple(default_camera_position.tolist()),
            look_at=tuple(target.tolist()),
        )
        camera_prim = camera_rep.get_output_prims()["prims"][0]
        vertical_aperture_attr = camera_prim.GetAttribute("verticalAperture")
        if vertical_aperture_attr.IsValid():
            vertical_aperture_attr.Set(float(vertical_aperture_mm))
        self.scene_handles.camera_prim_path = str(camera_prim.GetPrimPath())
        self.scene_handles.camera_handle = None
        self.scene_handles.camera_rep = camera_rep
        self._render_product = rep.create.render_product(
            camera_rep,
            (self.config.camera.resolution.width, self.config.camera.resolution.height),
        )

        dome_light_rep = rep.create.light(
            light_type="Dome",
            name="Any6DDomeLight",
            intensity=80.0,
            color=(0.98, 0.99, 1.0),
        )
        self.scene_handles.dome_light_rep = dome_light_rep
        self.scene_handles.key_light_rep = None
        self.scene_handles.fill_light_rep = None
        self.scene_handles.dome_light_prim_path = str(dome_light_rep.get_output_prims()["prims"][0].GetPrimPath())
        self.scene_handles.key_light_prim_path = "/World/Any6D_KeyLight"
        self.scene_handles.fill_light_prim_path = "/World/Any6D_FillLight"
        self._apply_default_lighting()

        for index, path in enumerate(self.scene_handles.distractor_paths):
            if self._stage.GetPrimAtPath(path).IsValid():
                continue
            self._create_procedural_primitive(path, index=index, role="distractor")
        for index, path in enumerate(self.scene_handles.occluder_paths):
            if self._stage.GetPrimAtPath(path).IsValid():
                continue
            self._create_procedural_primitive(path, index=index, role="occluder")

        self._world.reset()
        self._step_updates(10)
        log("setup_scene: done")

    def setup_annotators(self) -> None:
        log("setup_annotators: start")
        self._ensure_runtime()
        rep = self._runtime_imports["rep"]

        if self._annotators:
            return

        annotator_names = {
            "rgb": "rgb",
            "depth": "distance_to_image_plane",
            "semantic_segmentation": "semantic_segmentation",
            "instance_segmentation": "instance_segmentation",
            "bounding_box_2d_tight": "bounding_box_2d_tight",
            "camera_params": "camera_params",
        }
        for key, annot_name in annotator_names.items():
            if annot_name == "semantic_segmentation":
                annotator = rep.AnnotatorRegistry.get_annotator(annot_name, init_params={"colorize": False})
            else:
                annotator = rep.AnnotatorRegistry.get_annotator(annot_name)
            annotator.attach([self._render_product])
            self._annotators[key] = annotator
        log(f"setup_annotators: attached {list(self._annotators.keys())}")

    def warmup_renderer(self) -> None:
        log("warmup_renderer: start")
        self._ensure_runtime()
        rep = self._runtime_imports["rep"]
        self._world.reset()
        self._step_updates(30)
        for _ in range(6):
            rep.orchestrator.step(rt_subframes=32)
        log("warmup_renderer: done")

    def apply_request_to_scene(self, request: SampleRequest) -> None:
        log(f"apply_request_to_scene: {request.sample_type}/{request.factor_name}/{request.object_name}/{request.index}")
        self._ensure_target_object(request.object_name)
        self._hide_all_procedural_objects()
        self._place_target_base()

        factor_name = request.factor_name
        if factor_name == "anchor_clean":
            self._apply_anchor_scene_state(request)
        elif factor_name == "viewpoint":
            self._apply_viewpoint_scene_state(request)
        elif factor_name == "distance":
            self._apply_distance_scene_state(request)
        elif factor_name == "lighting":
            self._apply_lighting_scene_state(request)
        elif factor_name == "viewpoint_lighting":
            self._apply_viewpoint_lighting_scene_state(request)
        elif factor_name == "viewpoint_lighting_clutter":
            self._apply_viewpoint_lighting_clutter_scene_state(request)
        elif factor_name == "viewpoint_lighting_occlusion":
            self._apply_viewpoint_lighting_occlusion_scene_state(request)
        elif factor_name == "viewpoint_lighting_clutter_occlusion":
            self._apply_viewpoint_lighting_clutter_occlusion_scene_state(request)
        elif factor_name == "clutter":
            self._apply_clutter_scene_state(request)
        elif factor_name == "occlusion":
            self._apply_occlusion_scene_state(request)
        else:
            raise NotImplementedError(f"Factor not implemented yet: {factor_name}")

        self._step_updates(4)
        log(f"apply_request_to_scene: done ({request.factor_name})")

    def capture_sample(self, request: SampleRequest) -> CaptureArtifacts:
        log(f"capture_sample: start ({request.factor_name}, idx={request.index})")
        self._ensure_runtime()
        rep = self._runtime_imports["rep"]

        rep.orchestrator.step(rt_subframes=2)

        rgb_rgba = np.array(self._annotators["rgb"].get_data(), copy=True)
        depth_m = np.array(self._annotators["depth"].get_data(), copy=True)
        depth_m[np.isinf(depth_m)] = 0.0

        sem_seg_full = self._annotators["semantic_segmentation"].get_data()
        inst_seg_full = self._annotators["instance_segmentation"].get_data()
        bbox_data = self._annotators["bounding_box_2d_tight"].get_data()
        cam_params = self._annotators["camera_params"].get_data()

        self._log_segmentation_debug("semantic_segmentation", sem_seg_full)
        self._log_segmentation_debug("instance_segmentation", inst_seg_full)

        mask_visib = self._extract_target_mask(
            semantic_data=sem_seg_full,
            instance_data=inst_seg_full,
            object_name=request.object_name,
        )
        bbox_xyxy = self._extract_target_bbox_xyxy(bbox_data, request.object_name)

        hidden_paths = self._collect_non_target_visible_paths()
        for path in hidden_paths:
            self._set_prim_visibility(path, False)

        rep.orchestrator.step(rt_subframes=1)
        sem_seg_target = self._annotators["semantic_segmentation"].get_data()
        inst_seg_target = self._annotators["instance_segmentation"].get_data()
        mask_target = self._extract_target_mask(
            semantic_data=sem_seg_target,
            instance_data=inst_seg_target,
            object_name=request.object_name,
        )
        mask_visib = np.where(mask_target > 0, mask_visib, 0).astype(np.uint8)

        for path in hidden_paths:
            self._set_prim_visibility(path, True)

        k_matrix = self._k_from_camera_params(cam_params)
        t_w_c_raw_isaac_usd = self._t_w_c_from_camera_params(cam_params)
        t_w_o = self._t_w_o_from_target()
        t_c_o_raw_isaac_usd = np.linalg.inv(t_w_c_raw_isaac_usd) @ t_w_o
        t_w_c = self._camera_pose_isaac_usd_to_cv(t_w_c_raw_isaac_usd)
        t_c_o = self._camera_frame_isaac_usd_to_cv(t_c_o_raw_isaac_usd)

        total_pixels = int(np.count_nonzero(mask_target))
        visible_pixels = int(np.count_nonzero(mask_visib))
        valid_depth_mask = (mask_target > 0) & np.isfinite(depth_m) & (depth_m > 0.0)
        valid_depth_pixels = int(np.count_nonzero(valid_depth_mask))
        if total_pixels <= 0:
            raise ValueError("target_total_mask_pixel_count == 0")
        if visible_pixels <= 0:
            raise ValueError("target_visible_pixel_count == 0")
        if valid_depth_pixels <= 0:
            raise ValueError("target_depth_valid_pixel_count == 0")
        if total_pixels > 0 and visible_pixels > total_pixels:
            self._debug_log(
                f"capture_sample: clamping visible pixels from {visible_pixels} to {total_pixels}"
            )
            visible_pixels = total_pixels
        visibility_ratio = float(visible_pixels / total_pixels) if total_pixels > 0 else 0.0
        visibility_ratio = float(np.clip(visibility_ratio, 0.0, 1.0))
        occlusion_ratio = float(np.clip(1.0 - visibility_ratio, 0.0, 1.0))

        log(
            "capture_sample: done "
            f"(visible={visible_pixels}, total={total_pixels}, visibility={visibility_ratio:.4f})"
        )

        return CaptureArtifacts(
            rgb_rgba=rgb_rgba,
            depth_m=depth_m,
            mask_target=mask_target,
            mask_visib_target=mask_visib,
            k_matrix=k_matrix,
            t_w_c=t_w_c,
            t_w_o=t_w_o,
            t_c_o=t_c_o,
            t_w_c_raw_isaac_usd=t_w_c_raw_isaac_usd,
            t_c_o_raw_isaac_usd=t_c_o_raw_isaac_usd,
            bbox_2d_tight_xyxy=bbox_xyxy,
            visibility_ratio=visibility_ratio,
            occlusion_ratio=occlusion_ratio,
            target_total_mask_pixel_count=total_pixels,
            target_visible_pixel_count=visible_pixels,
            target_depth_valid_pixel_count=valid_depth_pixels,
        )
        

    def write_sample(self, request: SampleRequest, artifacts: CaptureArtifacts) -> SamplePaths:
        log(f"write_sample: start ({request.factor_name}, idx={request.index})")
        paths = self.path_builder.build(request)
        self.ensure_sample_dir(paths)

        if artifacts.rgb_rgba is None or artifacts.depth_m is None:
            raise ValueError("Artifacts do not contain image data")

        rgb = artifacts.rgb_rgba[..., :3]
        Image.fromarray(rgb).save(paths.rgb_path)

        depth_raw = np.clip(
            np.round(artifacts.depth_m * self.config.output.depth_scale),
            0,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)
        Image.fromarray(depth_raw).save(paths.depth_path)

        Image.fromarray(artifacts.mask_target.astype(np.uint8)).save(paths.mask_target_path)
        Image.fromarray(artifacts.mask_visib_target.astype(np.uint8)).save(paths.mask_visib_target_path)
        np.savetxt(paths.k_path, artifacts.k_matrix, fmt="%.8f")

        object_usd_path = self.config.scene.target_objects[request.object_name]
        metadata = self.metadata_builder.build(request, paths, artifacts, object_usd_path)
        paths.meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        log(f"write_sample: wrote {paths.sample_dir}")
        return paths

    def generate_requests(self, requests: Iterable[SampleRequest]) -> list[SamplePaths]:
        log("generate_requests: start")
        self.setup_scene()
        self.setup_annotators()
        self.warmup_renderer()

        written: list[SamplePaths] = []
        for request in requests:
            log(f"generate_requests: processing {request.sample_type}/{request.factor_name}/{request.object_name}/{request.index}")
            try:
                self.apply_request_to_scene(request)
                artifacts = self.capture_sample(request)
                written.append(self.write_sample(request, artifacts))
            except ValueError as exc:
                log(
                    "generate_requests: skipped invalid sample "
                    f"{request.sample_type}/{request.factor_name}/{request.object_name}/{request.index}: {exc}"
                )
        log("generate_requests: done")
        return written

    def close(self) -> None:
        if self._simulation_app is not None:
            log("close: shutting down SimulationApp")
            self._simulation_app.close()
            self._simulation_app = None
            log("close: done")

    def _ensure_runtime(self) -> None:
        if self._runtime_imports:
            return

        from isaacsim import SimulationApp

        if self._simulation_app is None:
            self._simulation_app = SimulationApp({"headless": True})

        import omni.replicator.core as rep
        from omni.physx import get_physx_scene_query_interface
        from isaacsim.core.api import World
        import isaacsim.core.utils.bounds as bounds_utils
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path
        from isaacsim.core.utils.semantics import add_labels
        from pxr import PhysxSchema, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics

        self._runtime_imports = {
            "SimulationApp": SimulationApp,
            "rep": rep,
            "get_physx_scene_query_interface": get_physx_scene_query_interface,
            "World": World,
            "bounds_utils": bounds_utils,
            "add_reference_to_stage": add_reference_to_stage,
            "get_assets_root_path": get_assets_root_path,
            "add_labels": add_labels,
            "PhysxSchema": PhysxSchema,
            "PhysicsSchemaTools": PhysicsSchemaTools,
            "Usd": Usd,
            "UsdGeom": UsdGeom,
            "UsdPhysics": UsdPhysics,
        }

    def _step_updates(self, num_updates: int) -> None:
        for _ in range(num_updates):
            self._simulation_app.update()

    def _create_procedural_primitive(self, prim_path: str, index: int, role: str) -> None:
        from pxr import Gf, UsdGeom

        primitive_type = self._rng.choice(self.config.scene.distractor_primitive_types)
        if primitive_type == "Cube":
            geom = UsdGeom.Cube.Define(self._stage, prim_path)
            geom.GetSizeAttr().Set(1.0)
        elif primitive_type == "Cylinder":
            geom = UsdGeom.Cylinder.Define(self._stage, prim_path)
            geom.GetRadiusAttr().Set(0.5)
            geom.GetHeightAttr().Set(1.0)
        elif primitive_type == "Sphere":
            geom = UsdGeom.Sphere.Define(self._stage, prim_path)
            geom.GetRadiusAttr().Set(0.5)
        elif primitive_type == "Cone":
            geom = UsdGeom.Cone.Define(self._stage, prim_path)
            geom.GetRadiusAttr().Set(0.5)
            geom.GetHeightAttr().Set(1.0)
        else:
            raise ValueError(f"Unsupported distractor primitive type: {primitive_type}")

        prim = geom.GetPrim()
        add_labels = self._runtime_imports["add_labels"]
        add_labels(prim, [f"procedural_{role}_{primitive_type.lower()}"], "class")
        self._enable_collision_query_on_subtree(prim_path)
        self._set_subtree_physics_enabled(prim_path, False)
        self._apply_random_visual_style(prim_path, index=index, role=role, refresh_material=True)
        self._set_prim_visibility(prim_path, False)

    def _apply_random_visual_style(
        self,
        prim_path: str,
        index: int,
        role: str,
        refresh_material: bool = False,
    ) -> None:
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return

        style_map = self._distractor_style_map if role == "distractor" else self._occluder_style_map
        style = style_map.get(prim_path)
        if style is None or refresh_material:
            if role == "distractor":
                hue = float((0.137 * (index + 1) + self._rng.uniform(0.0, 0.48)) % 1.0)
                sat = float(self._rng.uniform(0.35, 0.95))
                val = float(self._rng.uniform(0.45, 0.98))
                base_color = self._hsv_to_rgb(hue, sat, val)
                accent_color = self._hsv_to_rgb(
                    (hue + self._rng.uniform(0.12, 0.32)) % 1.0,
                    float(np.clip(sat - self._rng.uniform(0.08, 0.22), 0.2, 1.0)),
                    float(np.clip(val + self._rng.uniform(-0.02, 0.1), 0.25, 1.0)),
                )
                style = {
                    "base_color": base_color,
                    "accent_color": accent_color,
                    "roughness": float(np.clip(self._rng.uniform(0.18, 0.75), 0.05, 0.95)),
                    "metallic": float(np.clip(self._rng.uniform(0.0, 0.18), 0.0, 1.0)),
                    "emissive_strength": float(self._rng.uniform(0.02, 0.18)),
                }
            else:
                palette_mode = self._rng.choice(("blackout", "hazard", "gel", "toxic"))
                if palette_mode == "blackout":
                    base_color = tuple(float(v) for v in self._rng.choice(
                        [(0.03, 0.03, 0.04), (0.06, 0.08, 0.09), (0.1, 0.1, 0.12)]
                    ))
                    accent_color = tuple(float(v) for v in self._rng.choice(
                        [(0.95, 0.12, 0.18), (0.95, 0.55, 0.08), (0.2, 0.9, 0.82)]
                    ))
                    roughness = self._rng.uniform(0.72, 0.96)
                    metallic = self._rng.uniform(0.0, 0.05)
                    emissive_strength = self._rng.uniform(0.18, 0.42)
                elif palette_mode == "hazard":
                    hue = float(self._rng.choice([0.03, 0.12, 0.17, 0.29]))
                    base_color = self._hsv_to_rgb(hue, float(self._rng.uniform(0.8, 1.0)), float(self._rng.uniform(0.75, 1.0)))
                    accent_color = self._hsv_to_rgb((hue + 0.5) % 1.0, float(self._rng.uniform(0.7, 1.0)), float(self._rng.uniform(0.18, 0.35)))
                    roughness = self._rng.uniform(0.45, 0.78)
                    metallic = self._rng.uniform(0.02, 0.12)
                    emissive_strength = self._rng.uniform(0.1, 0.22)
                elif palette_mode == "gel":
                    mono_hue = float(self._rng.uniform(0.0, 1.0))
                    base_color = self._hsv_to_rgb(mono_hue, float(self._rng.uniform(0.88, 1.0)), float(self._rng.uniform(0.2, 0.45)))
                    accent_color = self._hsv_to_rgb(mono_hue, float(self._rng.uniform(0.95, 1.0)), float(self._rng.uniform(0.85, 1.0)))
                    roughness = self._rng.uniform(0.12, 0.35)
                    metallic = self._rng.uniform(0.0, 0.08)
                    emissive_strength = self._rng.uniform(0.28, 0.62)
                else:
                    hue = float(self._rng.choice([0.24, 0.32, 0.47, 0.8]))
                    base_color = self._hsv_to_rgb(hue, float(self._rng.uniform(0.82, 1.0)), float(self._rng.uniform(0.18, 0.42)))
                    accent_color = self._hsv_to_rgb((hue + self._rng.uniform(0.08, 0.18)) % 1.0, 1.0, float(self._rng.uniform(0.92, 1.0)))
                    roughness = self._rng.uniform(0.2, 0.5)
                    metallic = self._rng.uniform(0.0, 0.15)
                    emissive_strength = self._rng.uniform(0.22, 0.55)
                style = {
                    "base_color": base_color,
                    "accent_color": accent_color,
                    "roughness": float(np.clip(roughness, 0.05, 0.98)),
                    "metallic": float(np.clip(metallic, 0.0, 1.0)),
                    "emissive_strength": float(np.clip(emissive_strength, 0.0, 1.0)),
                }
            style_map[prim_path] = style

        gprim = UsdGeom.Gprim(prim)
        if gprim:
            gprim.CreateDisplayColorPrimvar().Set([Gf.Vec3f(*style["base_color"])])

        material_path = f"{prim_path}/Looks/ProceduralMaterial"
        material = UsdShade.Material.Define(self._stage, material_path)
        shader = UsdShade.Shader.Define(self._stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*style["base_color"]))
        emissive = tuple(float(v) * float(style["emissive_strength"]) for v in style["accent_color"])
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(style["roughness"]))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(style["metallic"]))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(prim).Bind(material)

    def _ensure_target_object(self, object_name: str) -> None:
        self._ensure_runtime()
        add_reference_to_stage = self._runtime_imports["add_reference_to_stage"]

        if self._target_object_name == object_name and self._stage.GetPrimAtPath(self.scene_handles.target_prim_path).IsValid():
            return

        target_prim = self._stage.GetPrimAtPath(self.scene_handles.target_prim_path)
        if target_prim.IsValid():
            import omni.kit.commands

            omni.kit.commands.execute("DeletePrims", paths=[self.scene_handles.target_prim_path])
            self._step_updates(2)

        usd_path = self.config.scene.target_objects[object_name]
        add_reference_to_stage(
            usd_path=self._join_asset_path(self._assets_root, usd_path),
            prim_path=self.scene_handles.target_prim_path,
        )
        target_prim = self._stage.GetPrimAtPath(self.scene_handles.target_prim_path)
        self._apply_semantics_recursive(target_prim, object_name)
        self._enable_collision_query_on_subtree(self.scene_handles.target_prim_path)
        self.scene_handles.target_handle = target_prim
        self._target_object_name = object_name
        self._step_updates(4)

    def _place_target_base(self) -> None:
        target_mount = np.array(self.config.scene.target_mount_position_m, dtype=np.float64)
        orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._set_world_pose(self.scene_handles.target_prim_path, target_mount, orientation)
        self._step_updates(2)
        grounded_position = self._ground_target_to_mount(target_mount)
        self.scene_handles.target_base_position_m = grounded_position.astype(np.float64)

    def _ground_target_to_mount(
        self,
        target_mount: np.ndarray,
        orientation_wxyz: np.ndarray | None = None,
    ) -> np.ndarray:
        center, extents, min_pt, _ = self._target_world_bbox_stats()
        grounded_position = target_mount.copy()
        grounded_position[0] += float(target_mount[0] - center[0])
        grounded_position[1] += float(target_mount[1] - center[1])
        desired_bottom_z = float(target_mount[2])
        grounded_position[2] += float(desired_bottom_z - min_pt[2] + self.config.scene.distractor_ground_clearance_m)
        if orientation_wxyz is None:
            orientation_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._set_world_pose(
            self.scene_handles.target_prim_path,
            grounded_position,
            np.array(orientation_wxyz, dtype=np.float64),
        )
        self._step_updates(2)
        grounded_center, grounded_extents, grounded_min, grounded_max = self._target_world_bbox_stats()
        self._debug_log(
            "ground_target_to_mount: "
            f"center={grounded_center.tolist()} extents={grounded_extents.tolist()} "
            f"min={grounded_min.tolist()} max={grounded_max.tolist()} "
            f"position={grounded_position.tolist()}"
        )
        return grounded_position

    def _apply_anchor_scene_state(self, request: SampleRequest) -> None:
        factor = request.factor_value
        self._apply_default_lighting()
        self._apply_target_pose_from_factor(factor)
        anchor_id = int(factor.get("anchor_view_id", request.index))
        azimuth = (anchor_id * 37.0) % 360.0
        elevation = 25.0 + (anchor_id % 4) * 10.0
        self._frame_camera_on_target(azimuth, elevation, radius_scale=2.2 + 0.15 * (anchor_id % 3))

    def _apply_viewpoint_scene_state(self, request: SampleRequest) -> None:
        factor = request.factor_value
        self._apply_default_lighting()
        self._apply_target_pose_from_factor(factor)
        self._frame_camera_on_target(
            azimuth_deg=float(factor["azimuth_deg"]),
            elevation_deg=float(factor["elevation_deg"]),
            radius_m=float(factor["camera_target_distance_m"]),
            radius_scale=1.0,
        )

    def _apply_distance_scene_state(self, request: SampleRequest) -> None:
        factor = request.factor_value
        self._apply_default_lighting()
        self._apply_target_pose_from_factor(factor)
        self._frame_camera_on_target(
            azimuth_deg=float(factor["azimuth_deg"]),
            elevation_deg=float(factor["elevation_deg"]),
            radius_m=float(factor["camera_target_distance_m"]),
            radius_scale=1.0,
        )

    def _apply_lighting_scene_state(self, request: SampleRequest) -> None:
        factor = request.factor_value
        self._set_light_parameters(
            dome_intensity=float(factor["dome_intensity"]),
            dome_color=tuple(float(v) for v in factor["dome_color"]),
            key_light_type=str(factor.get("key_light_type", "Sphere")),
            key_intensity=float(factor["key_intensity"]),
            key_color=tuple(float(v) for v in factor["key_color"]),
            key_position=tuple(float(v) for v in factor["key_position"]),
            key_scale=float(factor.get("key_scale", 0.15)),
            fill_light_type=str(factor.get("fill_light_type", "Sphere")),
            fill_intensity=float(factor["fill_intensity"]),
            fill_color=tuple(float(v) for v in factor["fill_color"]),
            fill_position=tuple(float(v) for v in factor["fill_position"]),
            fill_scale=float(factor.get("fill_scale", 0.15)),
        )
        self._apply_target_pose_from_factor(factor)
        self._frame_camera_on_target(35.0, 22.5, radius_scale=2.35)

    def _apply_viewpoint_lighting_scene_state(self, request: SampleRequest) -> None:
        factor = request.factor_value
        self._set_light_parameters(
            dome_intensity=float(factor["dome_intensity"]),
            dome_color=tuple(float(v) for v in factor["dome_color"]),
            key_light_type=str(factor.get("key_light_type", "Sphere")),
            key_intensity=float(factor["key_intensity"]),
            key_color=tuple(float(v) for v in factor["key_color"]),
            key_position=tuple(float(v) for v in factor["key_position"]),
            key_scale=float(factor.get("key_scale", 0.15)),
            fill_light_type=str(factor.get("fill_light_type", "Sphere")),
            fill_intensity=float(factor["fill_intensity"]),
            fill_color=tuple(float(v) for v in factor["fill_color"]),
            fill_position=tuple(float(v) for v in factor["fill_position"]),
            fill_scale=float(factor.get("fill_scale", 0.15)),
        )
        self._apply_target_pose_from_factor(factor)
        self._frame_camera_on_target(
            azimuth_deg=float(factor["azimuth_deg"]),
            elevation_deg=float(factor["elevation_deg"]),
            radius_m=float(factor["camera_target_distance_m"]),
            radius_scale=1.0,
        )

    def _apply_viewpoint_lighting_clutter_scene_state(self, request: SampleRequest) -> None:
        factor = request.factor_value
        self._set_light_parameters(
            dome_intensity=float(factor["dome_intensity"]),
            dome_color=tuple(float(v) for v in factor["dome_color"]),
            key_light_type=str(factor.get("key_light_type", "Sphere")),
            key_intensity=float(factor["key_intensity"]),
            key_color=tuple(float(v) for v in factor["key_color"]),
            key_position=tuple(float(v) for v in factor["key_position"]),
            key_scale=float(factor.get("key_scale", 0.15)),
            fill_light_type=str(factor.get("fill_light_type", "Sphere")),
            fill_intensity=float(factor["fill_intensity"]),
            fill_color=tuple(float(v) for v in factor["fill_color"]),
            fill_position=tuple(float(v) for v in factor["fill_position"]),
            fill_scale=float(factor.get("fill_scale", 0.15)),
        )
        self._apply_target_pose_from_factor(factor)
        self._frame_camera_on_target(
            azimuth_deg=float(factor["azimuth_deg"]),
            elevation_deg=float(factor["elevation_deg"]),
            radius_m=float(factor["camera_target_distance_m"]),
            radius_scale=1.0,
        )
        self._place_distractors(int(factor["distractor_count"]))
        self._step_updates(2)

    def _apply_viewpoint_lighting_occlusion_scene_state(self, request: SampleRequest) -> None:
        factor = request.factor_value
        self._set_light_parameters(
            dome_intensity=float(factor["dome_intensity"]),
            dome_color=tuple(float(v) for v in factor["dome_color"]),
            key_light_type=str(factor.get("key_light_type", "Sphere")),
            key_intensity=float(factor["key_intensity"]),
            key_color=tuple(float(v) for v in factor["key_color"]),
            key_position=tuple(float(v) for v in factor["key_position"]),
            key_scale=float(factor.get("key_scale", 0.15)),
            fill_light_type=str(factor.get("fill_light_type", "Sphere")),
            fill_intensity=float(factor["fill_intensity"]),
            fill_color=tuple(float(v) for v in factor["fill_color"]),
            fill_position=tuple(float(v) for v in factor["fill_position"]),
            fill_scale=float(factor.get("fill_scale", 0.15)),
        )
        self._apply_target_pose_from_factor(factor)
        self._frame_camera_on_target(
            azimuth_deg=float(factor["azimuth_deg"]),
            elevation_deg=float(factor["elevation_deg"]),
            radius_m=float(factor["camera_target_distance_m"]),
            radius_scale=1.0,
        )
        self._step_updates(2)
        self._place_occluders(occluders=list(factor["occluders"]))

    def _apply_viewpoint_lighting_clutter_occlusion_scene_state(self, request: SampleRequest) -> None:
        factor = request.factor_value
        self._set_light_parameters(
            dome_intensity=float(factor["dome_intensity"]),
            dome_color=tuple(float(v) for v in factor["dome_color"]),
            key_light_type=str(factor.get("key_light_type", "Sphere")),
            key_intensity=float(factor["key_intensity"]),
            key_color=tuple(float(v) for v in factor["key_color"]),
            key_position=tuple(float(v) for v in factor["key_position"]),
            key_scale=float(factor.get("key_scale", 0.15)),
            fill_light_type=str(factor.get("fill_light_type", "Sphere")),
            fill_intensity=float(factor["fill_intensity"]),
            fill_color=tuple(float(v) for v in factor["fill_color"]),
            fill_position=tuple(float(v) for v in factor["fill_position"]),
            fill_scale=float(factor.get("fill_scale", 0.15)),
        )
        self._apply_target_pose_from_factor(factor)
        self._frame_camera_on_target(
            azimuth_deg=float(factor["azimuth_deg"]),
            elevation_deg=float(factor["elevation_deg"]),
            radius_m=float(factor["camera_target_distance_m"]),
            radius_scale=1.0,
        )
        self._place_distractors(int(factor["distractor_count"]))
        self._step_updates(2)
        self._place_occluders(occluders=list(factor["occluders"]))

    def _apply_clutter_scene_state(self, request: SampleRequest) -> None:
        factor = request.factor_value
        self._apply_default_lighting()
        self._apply_target_pose_from_factor(factor)
        self._place_distractors(int(factor["distractor_count"]))
        self._step_updates(2)
        self._frame_camera_on_target(
            azimuth_deg=float(factor["azimuth_deg"]),
            elevation_deg=float(factor["elevation_deg"]),
            radius_m=float(factor["camera_target_distance_m"]),
            radius_scale=1.0,
        )

    def _apply_occlusion_scene_state(self, request: SampleRequest) -> None:
        factor = request.factor_value
        self._apply_default_lighting()
        self._apply_target_pose_from_factor(factor)
        self._frame_camera_on_target(
            azimuth_deg=float(factor["azimuth_deg"]),
            elevation_deg=float(factor["elevation_deg"]),
            radius_m=float(factor["camera_target_distance_m"]),
            radius_scale=1.0,
        )
        self._step_updates(2)
        self._place_occluders(
            occluders=list(factor["occluders"]),
        )

    def _apply_target_pose_from_factor(self, factor: dict[str, Any]) -> None:
        position_xy = factor.get("target_position_xy_m")
        orientation_rpy = factor.get("target_orientation_rpy_deg")
        if position_xy is None or orientation_rpy is None:
            return
        mount_z = float(self.config.scene.target_mount_position_m[2])
        target_mount = np.array(
            [float(position_xy[0]), float(position_xy[1]), mount_z],
            dtype=np.float64,
        )
        orientation = self._quat_wxyz_from_euler_xyz_deg(
            float(orientation_rpy[0]),
            float(orientation_rpy[1]),
            float(orientation_rpy[2]),
        )
        self._set_world_pose(self.scene_handles.target_prim_path, target_mount, orientation)
        self._step_updates(2)
        grounded_position = self._ground_target_to_mount(target_mount, orientation_wxyz=orientation)
        self.scene_handles.target_base_position_m = grounded_position.astype(np.float64)

    def _set_camera_pose_spherical(self, radius: float, azimuth_deg: float, elevation_deg: float) -> None:
        target = self._target_base_position()
        azimuth = np.radians(azimuth_deg)
        elevation = np.radians(elevation_deg)
        x = radius * np.cos(elevation) * np.cos(azimuth)
        y = radius * np.cos(elevation) * np.sin(azimuth)
        z = radius * np.sin(elevation)
        position = np.array([target[0] + x, target[1] + y, target[2] + z], dtype=np.float64)
        self._set_camera_pose(position, target)

    def _frame_camera_on_target(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        radius_scale: float = 2.5,
        radius_m: float | None = None,
    ) -> None:
        center, extents = self._target_world_bbox_center_and_size()
        bbox_diag = float(np.linalg.norm(extents))
        bbox_max_extent = float(np.max(extents))
        auto_radius = max(0.25, bbox_diag * radius_scale, bbox_max_extent * 2.5)
        camera_radius = float(radius_m) if radius_m is not None else auto_radius
        self._debug_log(
            "frame_camera_on_target: "
            f"center={center.tolist()} extents={extents.tolist()} radius={camera_radius:.4f} "
            f"az={azimuth_deg:.2f} el={elevation_deg:.2f}"
        )
        self._set_camera_pose_spherical_look_at(camera_radius, azimuth_deg, elevation_deg, center)

    def _set_camera_pose_spherical_look_at(
        self,
        radius: float,
        azimuth_deg: float,
        elevation_deg: float,
        look_at_target: np.ndarray,
    ) -> None:
        target = np.array(look_at_target, dtype=np.float64)
        azimuth = np.radians(azimuth_deg)
        elevation = np.radians(elevation_deg)
        x = radius * np.cos(elevation) * np.cos(azimuth)
        y = radius * np.cos(elevation) * np.sin(azimuth)
        z = radius * np.sin(elevation)
        position = np.array([target[0] + x, target[1] + y, target[2] + z], dtype=np.float64)
        self._set_camera_pose(position, target)
        self._debug_log(
            "set_camera_pose: "
            f"position={position.tolist()} look_at={target.tolist()}"
        )

    def _set_camera_pose(self, position: np.ndarray, look_at_target: np.ndarray) -> None:
        rep = self._runtime_imports["rep"]
        camera_rep = self.scene_handles.camera_rep
        if camera_rep is None:
            raise RuntimeError("Camera Replicator handle is not initialized")
        with camera_rep:
            rep.modify.pose(
                position=tuple(float(v) for v in position.tolist()),
                look_at=tuple(float(v) for v in look_at_target.tolist()),
            )
        self._step_updates(2)

    def _target_base_position(self) -> np.ndarray:
        if self.scene_handles.target_base_position_m is not None:
            return np.array(self.scene_handles.target_base_position_m, dtype=np.float64)
        return np.array(self.config.scene.target_mount_position_m, dtype=np.float64)

    def _apply_default_lighting(self) -> None:
        self._set_light_parameters(
            dome_intensity=20.0,
            dome_color=(0.98, 0.99, 1.0),
            key_light_type="Rect",
            key_intensity=26000.0,
            key_color=(1.0, 0.98, 0.95),
            key_position=(1.0, -0.5, 1.5),
            key_scale=0.34,
            fill_light_type="Sphere",
            fill_intensity=12000.0,
            fill_color=(0.9, 0.95, 1.0),
            fill_position=(-1.1, 0.9, 1.2),
            fill_scale=0.30,
        )

    def _set_light_parameters(
        self,
        dome_intensity: float,
        dome_color: tuple[float, float, float],
        key_light_type: str,
        key_intensity: float,
        key_color: tuple[float, float, float],
        key_position: tuple[float, float, float],
        key_scale: float,
        fill_light_type: str,
        fill_intensity: float,
        fill_color: tuple[float, float, float],
        fill_position: tuple[float, float, float],
        fill_scale: float,
    ) -> None:
        dome_prim = self._stage.GetPrimAtPath(self.scene_handles.dome_light_prim_path)
        if dome_prim.IsValid():
            self._set_float_attribute(dome_prim, "inputs:intensity", dome_intensity)
            self._set_color_attribute(dome_prim, "inputs:color", dome_color)
        target_position = self._lighting_look_at_target()
        self._configure_local_light(
            prim_path=self.scene_handles.key_light_prim_path,
            light_type=key_light_type,
            intensity=key_intensity,
            color=key_color,
            position=np.array(key_position, dtype=np.float64),
            scale=key_scale,
            look_at_target=target_position,
        )
        self._configure_local_light(
            prim_path=self.scene_handles.fill_light_prim_path,
            light_type=fill_light_type,
            intensity=fill_intensity,
            color=fill_color,
            position=np.array(fill_position, dtype=np.float64),
            scale=fill_scale,
            look_at_target=target_position,
        )
        self._step_updates(2)

    def _lighting_look_at_target(self) -> np.ndarray:
        target_prim = self._stage.GetPrimAtPath(self.scene_handles.target_prim_path)
        if target_prim.IsValid():
            try:
                return self._target_world_bbox_center_and_size()[0]
            except Exception:
                pass
        return self._target_base_position()

    def _configure_local_light(
        self,
        prim_path: str,
        light_type: str,
        intensity: float,
        color: tuple[float, float, float],
        position: np.ndarray,
        scale: float,
        look_at_target: np.ndarray,
    ) -> None:
        light_type = str(light_type).strip().title()
        prim = self._ensure_local_light_prim(prim_path, light_type)
        self._set_float_attribute(prim, "inputs:intensity", intensity)
        self._set_color_attribute(prim, "inputs:color", color)
        orientation = self._quat_wxyz_look_at(position, np.array(look_at_target, dtype=np.float64))
        self._set_world_pose(prim_path, np.array(position, dtype=np.float64), orientation)
        self._set_local_scale(prim_path, np.array([1.0, 1.0, 1.0], dtype=np.float64))
        self._set_light_shape_attributes(prim, light_type, float(scale))

    def _ensure_local_light_prim(self, prim_path: str, light_type: str) -> Any:
        from pxr import UsdLux
        import omni.kit.commands

        schema_map = {
            "Sphere": UsdLux.SphereLight,
            "Rect": UsdLux.RectLight,
            "Disk": UsdLux.DiskLight,
            "Distant": UsdLux.DistantLight,
        }
        schema = schema_map.get(light_type)
        if schema is None:
            raise ValueError(f"Unsupported local light type: {light_type}")

        prim = self._stage.GetPrimAtPath(prim_path)
        if prim.IsValid() and prim.GetTypeName() != f"{light_type}Light":
            omni.kit.commands.execute("DeletePrims", paths=[prim_path])
            self._step_updates(1)
            prim = self._stage.GetPrimAtPath(prim_path)

        if not prim.IsValid():
            prim = schema.Define(self._stage, prim_path).GetPrim()
            self._step_updates(1)
        return prim

    def _set_light_shape_attributes(self, prim: Any, light_type: str, scale: float) -> None:
        scale = float(max(scale, 0.02))
        if light_type == "Sphere":
            self._set_float_attribute(prim, "inputs:radius", scale)
        elif light_type == "Disk":
            self._set_float_attribute(prim, "inputs:radius", scale)
        elif light_type == "Rect":
            self._set_float_attribute(prim, "inputs:width", scale * 2.4)
            self._set_float_attribute(prim, "inputs:height", scale * 1.6)
        elif light_type == "Distant":
            self._set_float_attribute(prim, "inputs:angle", max(0.2, scale * 12.0))

    def _set_float_attribute(self, prim: Any, attr_name: str, value: float) -> None:
        attr = prim.GetAttribute(attr_name)
        if attr.IsValid():
            attr.Set(float(value))

    def _set_color_attribute(self, prim: Any, attr_name: str, color: tuple[float, float, float]) -> None:
        from pxr import Gf

        attr = prim.GetAttribute(attr_name)
        if attr.IsValid():
            attr.Set(Gf.Vec3f(float(color[0]), float(color[1]), float(color[2])))

    def _hide_all_procedural_objects(self) -> None:
        for path in self.scene_handles.distractor_paths:
            self._set_prim_visibility(path, False)
        for path in self.scene_handles.occluder_paths:
            self._set_prim_visibility(path, False)

    def _place_distractors(self, distractor_count: int) -> None:
        target = self._target_base_position()
        target_center, target_extents = self._target_world_bbox_center_and_size()
        placed_obstacles = [
            {
                "position": target_center.astype(np.float64),
                "radius_xy": self._xy_footprint_radius(target_extents),
                "z_half_extent": float(max(target_extents[2] * 0.5, 0.01)),
            }
        ]
        obstacle_paths = [self.scene_handles.target_prim_path]
        active_count = min(distractor_count, len(self.scene_handles.distractor_paths))
        camera_position = self._camera_world_position()
        view_dir = target - camera_position
        view_dir_norm = np.linalg.norm(view_dir)
        if view_dir_norm > 1e-8:
            view_dir = view_dir / view_dir_norm
        else:
            view_dir = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        lateral = np.array([-view_dir[1], view_dir[0], 0.0], dtype=np.float64)
        lateral_norm = np.linalg.norm(lateral)
        if lateral_norm > 1e-8:
            lateral = lateral / lateral_norm
        else:
            lateral = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        for index, path in enumerate(self.scene_handles.distractor_paths):
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            if index >= active_count:
                self._set_prim_visibility(path, False)
                self._set_local_scale(path, np.array([1.0, 1.0, 1.0], dtype=np.float64))
                continue
            if index < max(1, active_count // 2):
                depth_offset = 0.08 + 0.05 * index
                lateral_offset = (-1.0 if index % 2 == 0 else 1.0) * (0.05 + 0.03 * index)
                height = 0.04 + 0.01 * (index % 2)
                position = (
                    target
                    - view_dir * depth_offset
                    + lateral * lateral_offset
                    + np.array([0.0, 0.0, height - target[2]], dtype=np.float64)
                )
            else:
                ring_index = index - max(1, active_count // 2)
                ring_count = max(1, active_count - max(1, active_count // 2))
                angle = np.radians((360.0 / ring_count) * ring_index + 22.5 * (ring_index % 2))
                radius = 0.24 + 0.07 * (ring_index % 3)
                height = 0.045 + 0.012 * (ring_index % 2)
                position = np.array(
                    [
                        target[0] + radius * np.cos(angle),
                        target[1] + radius * np.sin(angle),
                        height,
                    ],
                    dtype=np.float64,
                )
            yaw_deg = (index * 57.0) % 360.0
            scale_xyz = self._random_distractor_scale(path=path, occluder_mode=False)
            position = self._resolve_non_interpenetrating_position(
                candidate_position=position,
                scale_xyz=scale_xyz,
                placed_obstacles=placed_obstacles,
                target=target_center,
                lateral=lateral,
                view_dir=view_dir,
                occluder_mode=False,
            )
            orientation = self._quat_wxyz_from_euler_xyz_deg(0.0, 0.0, yaw_deg)
            position = self._resolve_geometry_safe_position(
                prim_path=path,
                candidate_position=position,
                orientation_wxyz=orientation,
                scale_xyz=scale_xyz,
                obstacle_paths=obstacle_paths,
                lateral=lateral,
                view_dir=view_dir,
                occluder_mode=False,
            )
            self._set_world_pose(
                path,
                position,
                orientation,
            )
            self._set_local_scale(path, scale_xyz)
            self._apply_random_visual_style(path, index=index, role="distractor")
            self._set_prim_visibility(path, True)
            obstacle_paths.append(path)
            placed_obstacles.append(
                {
                    "position": position.astype(np.float64),
                    "radius_xy": self._xy_footprint_radius(scale_xyz),
                    "z_half_extent": self._z_half_extent(scale_xyz),
                }
            )

    def _place_occluders(self, occluders: list[dict[str, Any]]) -> None:
        target = self._target_base_position()
        target_center, target_extents = self._target_world_bbox_center_and_size()
        placed_obstacles = [
            {
                "position": target_center.astype(np.float64),
                "radius_xy": self._xy_footprint_radius(target_extents),
                "z_half_extent": float(max(target_extents[2] * 0.5, 0.01)),
            }
        ]
        obstacle_paths = [self.scene_handles.target_prim_path]
        obstacle_paths.extend(self._visible_prim_paths(self.scene_handles.distractor_paths))
        camera_position = self._camera_world_position()
        view_dir = target - camera_position
        view_dir_norm = np.linalg.norm(view_dir)
        if view_dir_norm > 1e-8:
            view_dir = view_dir / view_dir_norm
        else:
            view_dir = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        lateral = np.array([-view_dir[1], view_dir[0], 0.0], dtype=np.float64)
        lateral_norm = np.linalg.norm(lateral)
        if lateral_norm > 1e-8:
            lateral = lateral / lateral_norm
        else:
            lateral = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        active_count = min(len(occluders), len(self.scene_handles.occluder_paths))
        for index, path in enumerate(self.scene_handles.occluder_paths):
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            if index >= active_count:
                self._set_prim_visibility(path, False)
                self._set_local_scale(path, np.array([1.0, 1.0, 1.0], dtype=np.float64))
                continue
            spec = occluders[index]
            depth_offset = float(spec["depth"])
            lateral_offset = float(spec["lateral"])
            height = float(spec["height"])
            position = (
                target
                - view_dir * depth_offset
                + lateral * lateral_offset
                + np.array([0.0, 0.0, height - target[2]], dtype=np.float64)
            )
            yaw_deg = float(spec.get("yaw_deg", (index * 33.0) % 360.0))
            scale_value = float(spec["scale"])
            scale_xyz = self._random_distractor_scale(path=path, occluder_mode=True, scale_value=scale_value)
            position = self._resolve_non_interpenetrating_position(
                candidate_position=position,
                scale_xyz=scale_xyz,
                placed_obstacles=placed_obstacles,
                target=target_center,
                lateral=lateral,
                view_dir=view_dir,
                occluder_mode=True,
            )
            orientation = self._quat_wxyz_from_euler_xyz_deg(0.0, 0.0, yaw_deg)
            position = self._resolve_geometry_safe_position(
                prim_path=path,
                candidate_position=position,
                orientation_wxyz=orientation,
                scale_xyz=scale_xyz,
                obstacle_paths=obstacle_paths,
                lateral=lateral,
                view_dir=view_dir,
                occluder_mode=True,
            )
            self._set_world_pose(
                path,
                position,
                orientation,
            )
            self._set_local_scale(path, scale_xyz)
            self._apply_random_visual_style(path, index=index, role="occluder")
            self._set_prim_visibility(path, True)
            obstacle_paths.append(path)
            placed_obstacles.append(
                {
                    "position": position.astype(np.float64),
                    "radius_xy": self._xy_footprint_radius(scale_xyz),
                    "z_half_extent": self._z_half_extent(scale_xyz),
                }
            )

    def _camera_world_position(self) -> np.ndarray:
        t_w_c = self._world_transform_matrix(self.scene_handles.camera_prim_path)
        return np.array(t_w_c[:3, 3], dtype=np.float64)

    def _random_distractor_scale(
        self,
        path: str,
        occluder_mode: bool,
        scale_value: float | None = None,
    ) -> np.ndarray:
        prim = self._stage.GetPrimAtPath(path)
        prim_type = prim.GetTypeName() if prim.IsValid() else "Cube"
        base = float(scale_value) if scale_value is not None else float(self._rng.uniform(0.05, 0.18))
        if occluder_mode:
            base = float(np.clip(base, 0.08, 0.32))
        type_gain = {
            "Sphere": 1.0,
            "Cube": 1.0,
            "Cylinder": 1.12 if occluder_mode else 1.2,
            "Cone": 1.16 if occluder_mode else 1.24,
        }.get(prim_type, 1.0)
        uniform_scale = base * type_gain
        return np.array([uniform_scale, uniform_scale, uniform_scale], dtype=np.float64)

    def _xy_footprint_radius(self, extents_xyz: np.ndarray) -> float:
        extents_xyz = np.asarray(extents_xyz, dtype=np.float64)
        return float(max(0.01, 0.5 * np.linalg.norm(extents_xyz[:2])))

    def _z_half_extent(self, extents_xyz: np.ndarray) -> float:
        extents_xyz = np.asarray(extents_xyz, dtype=np.float64)
        return float(max(0.01, 0.5 * abs(extents_xyz[2])))

    def _overlaps_approx(
        self,
        position_a: np.ndarray,
        radius_xy_a: float,
        z_half_extent_a: float,
        position_b: np.ndarray,
        radius_xy_b: float,
        z_half_extent_b: float,
        margin_xy: float,
        margin_z: float,
    ) -> bool:
        delta_xy = np.asarray(position_a[:2], dtype=np.float64) - np.asarray(position_b[:2], dtype=np.float64)
        xy_dist = float(np.linalg.norm(delta_xy))
        z_dist = float(abs(float(position_a[2]) - float(position_b[2])))
        xy_overlap = xy_dist < (radius_xy_a + radius_xy_b + margin_xy)
        z_overlap = z_dist < (z_half_extent_a + z_half_extent_b + margin_z)
        return xy_overlap and z_overlap

    def _resolve_non_interpenetrating_position(
        self,
        candidate_position: np.ndarray,
        scale_xyz: np.ndarray,
        placed_obstacles: list[dict[str, Any]],
        target: np.ndarray,
        lateral: np.ndarray,
        view_dir: np.ndarray,
        occluder_mode: bool,
    ) -> np.ndarray:
        position = np.array(candidate_position, dtype=np.float64)
        radius_xy = self._xy_footprint_radius(scale_xyz)
        z_half_extent = self._z_half_extent(scale_xyz)
        lateral = np.array(lateral, dtype=np.float64)
        view_dir = np.array(view_dir, dtype=np.float64)
        margin_xy = 0.012 if occluder_mode else 0.02
        margin_z = 0.008

        for attempt in range(16):
            overlap = False
            for obstacle in placed_obstacles:
                if self._overlaps_approx(
                    position,
                    radius_xy,
                    z_half_extent,
                    np.asarray(obstacle["position"], dtype=np.float64),
                    float(obstacle["radius_xy"]),
                    float(obstacle["z_half_extent"]),
                    margin_xy=margin_xy,
                    margin_z=margin_z,
                ):
                    overlap = True
                    push_dir = position - np.asarray(obstacle["position"], dtype=np.float64)
                    push_dir[2] = 0.0
                    push_norm = float(np.linalg.norm(push_dir[:2]))
                    if push_norm < 1e-6:
                        if occluder_mode:
                            push_dir = -view_dir.copy()
                        else:
                            side = -1.0 if attempt % 2 == 0 else 1.0
                            push_dir = side * lateral.copy()
                        push_dir[2] = 0.0
                        push_norm = float(np.linalg.norm(push_dir[:2]))
                    if push_norm < 1e-6:
                        push_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                        push_norm = 1.0
                    push_dir = push_dir / push_norm
                    required_xy = (
                        radius_xy
                        + float(obstacle["radius_xy"])
                        + margin_xy
                        - float(np.linalg.norm(position[:2] - np.asarray(obstacle["position"], dtype=np.float64)[:2]))
                    )
                    required_xy = max(required_xy, 0.015 if occluder_mode else 0.025)
                    position[:2] += push_dir[:2] * required_xy
                    if occluder_mode:
                        position[:2] += lateral[:2] * (0.01 if attempt % 2 == 0 else -0.01)
                    else:
                        position[:2] += lateral[:2] * (0.015 if attempt % 2 == 0 else -0.015)
                    break
            if not overlap:
                break

        position[2] = max(position[2], self.config.scene.distractor_ground_clearance_m + z_half_extent)
        return position

    def _active_prim_paths(self, prim_paths: list[str]) -> list[str]:
        active: list[str] = []
        for prim_path in prim_paths:
            prim = self._stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                active.append(prim_path)
        return active

    def _visible_prim_paths(self, prim_paths: list[str]) -> list[str]:
        self._ensure_runtime()
        Usd = self._runtime_imports["Usd"]
        UsdGeom = self._runtime_imports["UsdGeom"]
        visible: list[str] = []
        for prim_path in prim_paths:
            prim = self._stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            imageable = UsdGeom.Imageable(prim)
            if not imageable:
                continue
            if imageable.ComputeVisibility(Usd.TimeCode.Default()) != UsdGeom.Tokens.invisible:
                visible.append(prim_path)
        return visible

    def _collision_query_shape_paths(self, prim_path: str) -> list[str]:
        self._ensure_runtime()
        UsdGeom = self._runtime_imports["UsdGeom"]
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return []
        shape_paths: list[str] = []
        for subtree_prim in self._iter_prim_subtree(prim):
            if subtree_prim.IsA(UsdGeom.Gprim):
                shape_paths.append(subtree_prim.GetPath().pathString)
        return shape_paths

    def _enable_collision_query_on_subtree(self, prim_path: str) -> None:
        self._ensure_runtime()
        UsdGeom = self._runtime_imports["UsdGeom"]
        UsdPhysics = self._runtime_imports["UsdPhysics"]
        PhysxSchema = self._runtime_imports["PhysxSchema"]
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return

        for subtree_prim in self._iter_prim_subtree(prim):
            if not subtree_prim.IsA(UsdGeom.Gprim):
                continue

            if subtree_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI(subtree_prim)
            else:
                collision_api = UsdPhysics.CollisionAPI.Apply(subtree_prim)
            collision_attr = collision_api.GetCollisionEnabledAttr()
            if collision_attr.IsValid():
                collision_attr.Set(True)

            if subtree_prim.IsA(UsdGeom.Mesh):
                if subtree_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                    mesh_collision_api = UsdPhysics.MeshCollisionAPI(subtree_prim)
                else:
                    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(subtree_prim)
                approximation_attr = mesh_collision_api.GetApproximationAttr()
                if approximation_attr.IsValid():
                    approximation_attr.Set("convexHull")

            if subtree_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                physx_collision_api = PhysxSchema.PhysxCollisionAPI(subtree_prim)
            else:
                physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(subtree_prim)
            contact_attr = physx_collision_api.GetContactOffsetAttr()
            if contact_attr.IsValid():
                contact_attr.Set(0.001)
            rest_attr = physx_collision_api.GetRestOffsetAttr()
            if rest_attr.IsValid():
                rest_attr.Set(0.0)

    def _resolve_geometry_safe_position(
        self,
        prim_path: str,
        candidate_position: np.ndarray,
        orientation_wxyz: np.ndarray,
        scale_xyz: np.ndarray,
        obstacle_paths: list[str],
        lateral: np.ndarray,
        view_dir: np.ndarray,
        occluder_mode: bool,
    ) -> np.ndarray:
        position = np.array(candidate_position, dtype=np.float64)
        lateral = np.array(lateral, dtype=np.float64)
        view_dir = np.array(view_dir, dtype=np.float64)
        for attempt in range(12):
            self._set_world_pose(prim_path, position, orientation_wxyz)
            self._set_local_scale(prim_path, scale_xyz)
            self._step_updates(1)
            if not self._prim_overlaps_any_obstacle(prim_path, obstacle_paths):
                return position

            side = -1.0 if attempt % 2 == 0 else 1.0
            lateral_step = (0.02 + 0.01 * attempt) * side
            depth_step = 0.008 + 0.004 * attempt
            lift_step = 0.004 + 0.002 * min(attempt, 4)
            position[:2] += lateral[:2] * lateral_step
            if occluder_mode:
                position[:2] += -view_dir[:2] * depth_step
            else:
                position[:2] += lateral[:2] * (0.006 * side)
            position[2] += lift_step

        return position

    def _prim_overlaps_any_obstacle(self, prim_path: str, obstacle_paths: list[str]) -> bool:
        physx_overlap = self._prim_overlaps_any_obstacle_physx(prim_path, obstacle_paths)
        if physx_overlap is not None:
            return physx_overlap
        candidate_obb = self._compute_prim_obb(prim_path)
        if candidate_obb is None:
            return False
        for obstacle_path in obstacle_paths:
            if obstacle_path == prim_path:
                continue
            obstacle_obb = self._compute_prim_obb(obstacle_path)
            if obstacle_obb is None:
                continue
            if self._obb_intersects(candidate_obb, obstacle_obb):
                return True
        return False

    def _prim_overlaps_any_obstacle_physx(self, prim_path: str, obstacle_paths: list[str]) -> bool | None:
        self._ensure_runtime()
        PhysicsSchemaTools = self._runtime_imports["PhysicsSchemaTools"]
        get_physx_scene_query_interface = self._runtime_imports["get_physx_scene_query_interface"]
        candidate_shape_paths = self._collision_query_shape_paths(prim_path)
        if not candidate_shape_paths:
            return None
        obstacle_prefixes = tuple(path for path in obstacle_paths if path != prim_path)
        if not obstacle_prefixes:
            return False

        scene_query_interface = get_physx_scene_query_interface()
        if scene_query_interface is None:
            return None

        try:
            for shape_path in candidate_shape_paths:
                hit_found = {"value": False}
                path_hi, path_lo = PhysicsSchemaTools.encodeSdfPath(shape_path)

                def _report_hit(hit) -> bool:
                    collision_path = str(getattr(hit, "collision", "") or "")
                    rigid_body_path = str(
                        getattr(hit, "rigid_body", "") or getattr(hit, "rigidBody", "") or ""
                    )
                    if collision_path.startswith(prim_path) or rigid_body_path.startswith(prim_path):
                        return True
                    for obstacle_prefix in obstacle_prefixes:
                        if collision_path.startswith(obstacle_prefix) or rigid_body_path.startswith(obstacle_prefix):
                            hit_found["value"] = True
                            return False
                    return True

                scene_query_interface.overlap_shape(path_hi, path_lo, _report_hit, False)
                if hit_found["value"]:
                    return True
        except Exception:
            return None
        return False

    def _compute_prim_obb(self, prim_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        self._ensure_runtime()
        bounds_utils = self._runtime_imports["bounds_utils"]
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        try:
            bbox_cache = bounds_utils.create_bbox_cache()
            centroid, axes, half_extent = bounds_utils.compute_obb(bbox_cache, prim_path)
        except Exception:
            return None

        centroid = np.asarray(centroid, dtype=np.float64)
        axes = np.asarray(axes, dtype=np.float64)
        half_extent = np.asarray(half_extent, dtype=np.float64)
        axis_norms = np.linalg.norm(axes, axis=1)
        axis_norms = np.where(axis_norms > 1e-8, axis_norms, 1.0)
        axes = axes / axis_norms[:, None]
        half_extent = half_extent * axis_norms
        return centroid, axes, half_extent

    def _obb_intersects(
        self,
        obb_a: tuple[np.ndarray, np.ndarray, np.ndarray],
        obb_b: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> bool:
        center_a, axes_a, half_a = obb_a
        center_b, axes_b, half_b = obb_b
        eps = 1e-6

        rotation = np.zeros((3, 3), dtype=np.float64)
        abs_rotation = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                rotation[i, j] = float(np.dot(axes_a[i], axes_b[j]))
                abs_rotation[i, j] = abs(rotation[i, j]) + eps

        translation_world = center_b - center_a
        translation = np.array([np.dot(translation_world, axes_a[i]) for i in range(3)], dtype=np.float64)

        for i in range(3):
            ra = half_a[i]
            rb = float(np.dot(half_b, abs_rotation[i, :]))
            if abs(translation[i]) > ra + rb:
                return False

        for j in range(3):
            ra = float(np.dot(half_a, abs_rotation[:, j]))
            rb = half_b[j]
            t = abs(float(np.dot(translation, rotation[:, j])))
            if t > ra + rb:
                return False

        for i in range(3):
            for j in range(3):
                ra = half_a[(i + 1) % 3] * abs_rotation[(i + 2) % 3, j] + half_a[(i + 2) % 3] * abs_rotation[(i + 1) % 3, j]
                rb = half_b[(j + 1) % 3] * abs_rotation[i, (j + 2) % 3] + half_b[(j + 2) % 3] * abs_rotation[i, (j + 1) % 3]
                t = abs(
                    translation[(i + 2) % 3] * rotation[(i + 1) % 3, j]
                    - translation[(i + 1) % 3] * rotation[(i + 2) % 3, j]
                )
                if t > ra + rb:
                    return False

        return True

    def _collect_non_target_visible_paths(self) -> list[str]:
        paths: list[str] = []
        for path in self.scene_handles.distractor_paths:
            prim = self._stage.GetPrimAtPath(path)
            if prim.IsValid():
                paths.append(path)
        for path in self.scene_handles.occluder_paths:
            prim = self._stage.GetPrimAtPath(path)
            if prim.IsValid():
                paths.append(path)
        if self.config.scene.include_robot:
            robot_prim = self._stage.GetPrimAtPath(self.scene_handles.robot_prim_path)
            if robot_prim.IsValid():
                paths.append(self.scene_handles.robot_prim_path)
        return paths

    def _set_prim_visibility(self, prim_path: str, visible: bool) -> None:
        self._ensure_runtime()
        UsdGeom = self._runtime_imports["UsdGeom"]
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        imageable = UsdGeom.Imageable(prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def _set_subtree_physics_enabled(self, prim_path: str, enabled: bool) -> None:
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        for subtree_prim in self._iter_prim_subtree(prim):
            for attr_name in ("physics:rigidBodyEnabled", "physics:collisionEnabled"):
                attr = subtree_prim.GetAttribute(attr_name)
                if attr.IsValid():
                    attr.Set(bool(enabled))

    def _extract_target_mask(
        self,
        semantic_data: dict[str, Any] | None,
        instance_data: dict[str, Any] | None,
        object_name: str,
    ) -> np.ndarray:
        if semantic_data is not None:
            mask = self._extract_target_mask_from_segmentation_payload(semantic_data, object_name)
            if int(np.count_nonzero(mask)) > 0:
                return mask
        if instance_data is not None:
            return self._extract_target_mask_from_segmentation_payload(instance_data, object_name)
        return np.zeros(
            (self.config.camera.resolution.height, self.config.camera.resolution.width),
            dtype=np.uint8,
        )

    def _extract_target_mask_from_segmentation_payload(self, segmentation_data: dict[str, Any], object_name: str) -> np.ndarray:
        seg = np.array(segmentation_data["data"], copy=False)
        mask = np.zeros(seg.shape, dtype=np.uint8)
        id_to_labels = segmentation_data.get("info", {}).get("idToLabels", {})
        target_ids = [
            int(instance_id)
            for instance_id, labels in id_to_labels.items()
            if object_name in self._labels_to_class_string(labels)
        ]
        if not target_ids:
            return mask
        for target_id in target_ids:
            mask[seg == target_id] = 255
        return mask

    def _extract_target_bbox_xyxy(self, bbox_data: dict[str, Any], object_name: str) -> list[int]:
        candidates: list[tuple[int, int, int, int, int]] = []
        id_to_labels = bbox_data.get("info", {}).get("idToLabels", {})
        for bbox in bbox_data.get("data", []):
            semantic_id = bbox["semanticId"]
            labels = id_to_labels.get(semantic_id, id_to_labels.get(str(semantic_id), {}))
            if object_name not in self._labels_to_class_string(labels):
                continue
            x_min = int(round(float(bbox["x_min"])))
            y_min = int(round(float(bbox["y_min"])))
            x_max = int(round(float(bbox["x_max"])))
            y_max = int(round(float(bbox["y_max"])))
            area = max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)
            candidates.append((area, x_min, y_min, x_max, y_max))
        if not candidates:
            return [0, 0, 0, 0]
        _, x_min, y_min, x_max, y_max = max(candidates, key=lambda item: item[0])
        return [x_min, y_min, x_max, y_max]

    def _k_from_camera_params(self, camera_params: dict[str, Any]) -> np.ndarray:
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

    def _t_w_c_from_camera_params(self, camera_params: dict[str, Any]) -> np.ndarray:
        world_to_camera_row_major = np.array(camera_params["cameraViewTransform"], dtype=np.float64).reshape(4, 4)
        world_to_camera_standard = self._row_major_transform_to_standard(world_to_camera_row_major)
        return np.linalg.inv(world_to_camera_standard).astype(np.float32)

    def _t_w_o_from_target(self) -> np.ndarray:
        matrix = self._world_transform_matrix(self.scene_handles.target_prim_path)
        return matrix.astype(np.float32)

    def _world_transform_matrix(self, prim_path: str) -> np.ndarray:
        from pxr import UsdGeom

        prim = self._stage.GetPrimAtPath(prim_path)
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
        matrix_np = np.array(matrix, dtype=np.float64)
        return self._row_major_transform_to_standard(matrix_np)

    def _row_major_transform_to_standard(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.array(matrix, dtype=np.float64, copy=True)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected 4x4 transform matrix, got {matrix.shape}")
        return matrix.T

    def _isaac_usd_to_cv_transform(self) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _camera_frame_isaac_usd_to_cv(self, t_c_o_isaac_usd: np.ndarray) -> np.ndarray:
        conversion = self._isaac_usd_to_cv_transform()
        return (conversion @ t_c_o_isaac_usd).astype(np.float32)

    def _camera_pose_isaac_usd_to_cv(self, t_w_c_isaac_usd: np.ndarray) -> np.ndarray:
        conversion = self._isaac_usd_to_cv_transform()
        return (t_w_c_isaac_usd @ conversion).astype(np.float32)

    def _target_world_bbox_stats(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from pxr import Gf, Usd, UsdGeom

        prim = self._stage.GetPrimAtPath(self.scene_handles.target_prim_path)
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        bbox = bbox_cache.ComputeWorldBound(prim)
        aligned = bbox.ComputeAlignedBox()
        min_pt = np.array(aligned.GetMin(), dtype=np.float64)
        max_pt = np.array(aligned.GetMax(), dtype=np.float64)
        if np.any(~np.isfinite(min_pt)) or np.any(~np.isfinite(max_pt)):
            fallback_center = np.array(self.config.scene.target_mount_position_m, dtype=np.float64)
            fallback_extents = np.array([0.1, 0.1, 0.1], dtype=np.float64)
            self._debug_log("target_world_bbox: invalid bounds, using fallback extents")
            return fallback_center, fallback_extents, fallback_center - 0.5 * fallback_extents, fallback_center + 0.5 * fallback_extents
        center = 0.5 * (min_pt + max_pt)
        extents = np.maximum(max_pt - min_pt, np.array([1e-3, 1e-3, 1e-3], dtype=np.float64))
        self._debug_log(f"target_world_bbox: min={min_pt.tolist()} max={max_pt.tolist()}")
        return center, extents, min_pt, max_pt

    def _target_world_bbox_center_and_size(self) -> tuple[np.ndarray, np.ndarray]:
        center, extents, _, _ = self._target_world_bbox_stats()
        return center, extents

    def _apply_semantics_recursive(self, root_prim: Any, class_name: str) -> None:
        add_labels = self._runtime_imports["add_labels"]
        applied = 0
        for prim in self._iter_prim_subtree(root_prim):
            if not prim.IsValid():
                continue
            add_labels(prim, [class_name], "class")
            applied += 1
        self._debug_log(f"apply_semantics_recursive: class={class_name} prims={applied}")

    def _iter_prim_subtree(self, root_prim: Any) -> Iterable[Any]:
        stack = [root_prim]
        while stack:
            prim = stack.pop()
            yield prim
            children = list(prim.GetChildren())
            children.reverse()
            stack.extend(children)

    def _set_world_pose(self, prim_path: str, position: np.ndarray, orientation_wxyz: np.ndarray) -> None:
        from pxr import Gf, UsdGeom

        prim = self._stage.GetPrimAtPath(prim_path)
        xform = UsdGeom.Xformable(prim)
        translate_op = None
        orient_op = None
        for op in xform.GetOrderedXformOps():
            op_name = op.GetOpName()
            if op_name == "xformOp:translate":
                translate_op = op
            elif op_name == "xformOp:orient":
                orient_op = op
        if translate_op is None:
            translate_op = xform.AddTranslateOp()
        if orient_op is None:
            orient_op = xform.AddOrientOp()

        translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
        precision = orient_op.GetPrecision()
        if precision == UsdGeom.XformOp.PrecisionFloat:
            orient_value = Gf.Quatf(
                float(orientation_wxyz[0]),
                Gf.Vec3f(float(orientation_wxyz[1]), float(orientation_wxyz[2]), float(orientation_wxyz[3])),
            )
        else:
            orient_value = Gf.Quatd(
                float(orientation_wxyz[0]),
                Gf.Vec3d(float(orientation_wxyz[1]), float(orientation_wxyz[2]), float(orientation_wxyz[3])),
            )
        orient_op.Set(orient_value)

    def _set_local_scale(self, prim_path: str, scale_xyz: np.ndarray) -> None:
        from pxr import Gf, UsdGeom

        prim = self._stage.GetPrimAtPath(prim_path)
        xform = UsdGeom.Xformable(prim)
        scale_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:scale":
                scale_op = op
                break
        if scale_op is None:
            scale_op = xform.AddScaleOp()
        scale_op.Set(Gf.Vec3f(float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])))

    def _quat_wxyz_from_euler_xyz_deg(self, roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
        r = np.radians(roll_deg) / 2.0
        p = np.radians(pitch_deg) / 2.0
        y = np.radians(yaw_deg) / 2.0

        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        yv = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return np.array([w, x, yv, z], dtype=np.float64)

    def _quat_wxyz_look_at(
        self,
        origin: np.ndarray,
        target: np.ndarray,
        world_up: np.ndarray | None = None,
    ) -> np.ndarray:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64) if world_up is None else world_up.astype(np.float64)

        forward = target - origin
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        forward = forward / forward_norm

        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-8:
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            right = np.cross(forward, world_up)
            right_norm = np.linalg.norm(right)
            if right_norm < 1e-8:
                return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        right = right / right_norm

        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        rotation = np.eye(3, dtype=np.float64)
        rotation[:, 0] = right
        rotation[:, 1] = up
        rotation[:, 2] = -forward
        return self._quat_wxyz_from_rotation_matrix(rotation)

    def _quat_wxyz_from_rotation_matrix(self, rotation: np.ndarray) -> np.ndarray:
        trace = float(np.trace(rotation))
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (rotation[2, 1] - rotation[1, 2]) / s
            y = (rotation[0, 2] - rotation[2, 0]) / s
            z = (rotation[1, 0] - rotation[0, 1]) / s
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif rotation[1, 1] > rotation[2, 2]:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s
        quat = np.array([w, x, y, z], dtype=np.float64)
        quat /= np.linalg.norm(quat)
        return quat

    def _asset_name_from_usd(self, usd_path: str) -> str:
        return Path(usd_path).stem

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> tuple[float, float, float]:
        h = float(h % 1.0)
        s = float(np.clip(s, 0.0, 1.0))
        v = float(np.clip(v, 0.0, 1.0))
        i = int(h * 6.0)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = i % 6
        if i == 0:
            return (v, t, p)
        if i == 1:
            return (q, v, p)
        if i == 2:
            return (p, v, t)
        if i == 3:
            return (p, q, v)
        if i == 4:
            return (t, p, v)
        return (v, p, q)

    def _labels_to_class_string(self, labels: Any) -> str:
        if isinstance(labels, dict):
            class_value = labels.get("class", "")
            if isinstance(class_value, str):
                return class_value
            if isinstance(class_value, (list, tuple)):
                return " ".join(str(item) for item in class_value)
            return str(class_value)
        if isinstance(labels, str):
            return labels
        if isinstance(labels, (list, tuple)):
            return " ".join(str(item) for item in labels)
        return str(labels)

    def _log_segmentation_debug(self, name: str, segmentation_data: dict[str, Any]) -> None:
        info = segmentation_data.get("info", {})
        id_to_labels = info.get("idToLabels", {})
        seg = np.array(segmentation_data.get("data"), copy=False)
        unique_ids = [int(value) for value in np.unique(seg)[:12]]
        label_preview = {
            str(key): self._labels_to_class_string(value)
            for key, value in list(id_to_labels.items())[:8]
        }
        self._debug_log(f"{name}: unique_ids={unique_ids} labels={label_preview}")

    def _resolve_assets_root(self, get_assets_root_path: Any) -> str:
        candidates: list[str] = []

        first_target = next(iter(self.config.scene.target_objects.values()))
        if Path(first_target).is_absolute():
            parent = str(Path(first_target).parent)
            if self._asset_exists(parent, first_target):
                return parent

        if self.config.scene.assets_root_override:
            candidates.append(self.config.scene.assets_root_override)

        env_override = os.environ.get("ANY6D_ISAAC_ASSETS_ROOT")
        if env_override:
            candidates.append(env_override)

        resolved_root = get_assets_root_path()
        if resolved_root:
            candidates.append(resolved_root)

        for candidate in candidates:
            if self._asset_exists(candidate, first_target):
                return candidate.rstrip("/\\")

        raise RuntimeError(
            "Isaac asset root is not usable. The generator currently supports either "
            "`SceneConfig.assets_root_override`, the env var `ANY6D_ISAAC_ASSETS_ROOT`, or a working "
            "`get_assets_root_path()` resolver. None of these exposed the YCB asset library."
        )

    def _asset_exists(self, assets_root: str, relative_asset_path: str) -> bool:
        full_path = self._join_asset_path(assets_root, relative_asset_path)

        if full_path.startswith(("http://", "https://")):
            return True

        normalized = full_path.replace("/", "\\")
        return Path(normalized).exists()

    def _join_asset_path(self, assets_root: str, relative_asset_path: str) -> str:
        if Path(relative_asset_path).is_absolute():
            return str(Path(relative_asset_path))
        return assets_root.rstrip("/\\") + relative_asset_path


def create_stub_artifacts(config: GeneratorConfig) -> CaptureArtifacts:
    """
    Creates fake but shape-valid artifacts.

    This is useful right now to exercise pathing and metadata logic before
    Isaac-specific capture code exists.
    """

    resolution = config.camera.resolution
    width = resolution.width
    height = resolution.height

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[100:200, 220:320] = 255

    k_matrix = np.array(
        [
            [615.0, 0.0, width / 2.0],
            [0.0, 615.0, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    t_w_c_raw_isaac_usd = np.eye(4, dtype=np.float32)
    t_w_o = np.eye(4, dtype=np.float32)
    t_w_o[:3, 3] = np.array([0.5, 0.0, 0.05], dtype=np.float32)
    isaac_usd_to_cv = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    t_c_o_raw_isaac_usd = np.linalg.inv(t_w_c_raw_isaac_usd) @ t_w_o
    t_w_c = t_w_c_raw_isaac_usd @ isaac_usd_to_cv
    t_c_o = isaac_usd_to_cv @ t_c_o_raw_isaac_usd

    total_pixels = int(np.count_nonzero(mask))

    return CaptureArtifacts(
        rgb_rgba=np.zeros((height, width, 4), dtype=np.uint8),
        depth_m=np.zeros((height, width), dtype=np.float32),
        mask_target=mask.copy(),
        mask_visib_target=mask.copy(),
        k_matrix=k_matrix,
        t_w_c=t_w_c,
        t_w_o=t_w_o,
        t_c_o=t_c_o,
        t_w_c_raw_isaac_usd=t_w_c_raw_isaac_usd,
        t_c_o_raw_isaac_usd=t_c_o_raw_isaac_usd,
        bbox_2d_tight_xyxy=[220, 100, 319, 199],
        visibility_ratio=1.0,
        occlusion_ratio=0.0,
        target_total_mask_pixel_count=total_pixels,
        target_visible_pixel_count=total_pixels,
        notes=["stub artifacts for metadata and planning validation"],
    )


def build_default_generator(
    output_root: str | Path = "C:/isaac-sim/any6d/_out_any6d_spec",
    debug_logging: bool = False,
    object_frame_corrections: dict[str, np.ndarray] | None = None,
) -> Any6DDataGenerator:
    config = GeneratorConfig(
        output=OutputConfig(root_dir=Path(output_root)),
        scene=SceneConfig(assets_root_override=os.environ.get("ANY6D_ISAAC_ASSETS_ROOT")),
        object_frame_corrections=dict(object_frame_corrections or {}),
        debug_logging=debug_logging,
    )
    return Any6DDataGenerator(config)


def build_local_test_generator(
    output_root: str | Path = "C:/isaac-sim/any6d/_out_any6d_spec_localtest",
    debug_logging: bool = False,
    object_frame_corrections: dict[str, np.ndarray] | None = None,
) -> Any6DDataGenerator:
    scene = SceneConfig(
        include_ground_plane=False,
        assets_root_override=REPLICATOR_TEST_OBJECTS_ROOT,
        target_objects={
            "003_cracker_box": str(Path(REPLICATOR_TEST_OBJECTS_ROOT) / "003_cracker_box.usd"),
        },
        distractor_primitive_types=("Cube", "Sphere"),
        max_distractors=0,
    )
    config = GeneratorConfig(
        output=OutputConfig(root_dir=Path(output_root)),
        scene=scene,
        object_frame_corrections=dict(object_frame_corrections or {}),
        debug_logging=debug_logging,
    )
    return Any6DDataGenerator(config)


def load_object_frame_corrections(corrections_path: str | Path | None) -> dict[str, np.ndarray]:
    if corrections_path is None:
        return {}
    path = Path(corrections_path)
    if not path.exists():
        log(f"object-frame corrections file not found, continuing without it: {path}")
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    corrections: dict[str, np.ndarray] = {}
    for object_name, entry in payload.items():
        if object_name.startswith("_") or not isinstance(entry, dict):
            continue
        matrix = entry.get("T_O_isaac_to_ycb")
        if matrix is None:
            continue
        matrix_np = np.array(matrix, dtype=np.float32)
        if matrix_np.shape != (4, 4):
            raise ValueError(
                f"Invalid correction matrix for {object_name}: {matrix_np.shape}. Expected (4, 4)."
            )
        corrections[object_name] = matrix_np
    return corrections


def _resolve_object_names(generator: Any6DDataGenerator, args: argparse.Namespace) -> list[str]:
    if args.all_objects:
        return list(generator.config.scene.target_objects.keys())
    if args.object_name not in generator.config.scene.target_objects:
        available = ", ".join(generator.config.scene.target_objects.keys())
        raise ValueError(f"Object '{args.object_name}' is not available in this mode. Available: {available}")
    return [args.object_name]


def _resolve_generation_counts(args: argparse.Namespace) -> dict[str, int]:
    counts = {
        "anchors": int(args.anchors),
        "viewpoint_queries": int(args.viewpoint_queries),
        "distance_queries": int(args.distance_queries),
        "lighting_queries": int(args.lighting_queries),
        "viewpoint_lighting_queries": int(args.viewpoint_lighting_queries),
        "viewpoint_lighting_clutter_queries": int(args.viewpoint_lighting_clutter_queries),
        "viewpoint_lighting_occlusion_queries": int(args.viewpoint_lighting_occlusion_queries),
        "viewpoint_lighting_clutter_occlusion_queries": int(args.viewpoint_lighting_clutter_occlusion_queries),
        "clutter_queries": int(args.clutter_queries),
        "occlusion_queries": int(args.occlusion_queries),
    }
    if args.dataset_profile is None:
        return counts
    profile = DATASET_PROFILES[args.dataset_profile]
    return {
        "anchors": int(profile["anchors"]),
        "viewpoint_queries": int(profile["viewpoint_queries"]),
        "distance_queries": int(profile["distance_queries"]),
        "lighting_queries": int(profile["lighting_queries"]),
        "viewpoint_lighting_queries": int(profile.get("viewpoint_lighting_queries", 0)),
        "viewpoint_lighting_clutter_queries": int(profile.get("viewpoint_lighting_clutter_queries", 0)),
        "viewpoint_lighting_occlusion_queries": int(profile.get("viewpoint_lighting_occlusion_queries", 0)),
        "viewpoint_lighting_clutter_occlusion_queries": int(
            profile.get("viewpoint_lighting_clutter_occlusion_queries", 0)
        ),
        "clutter_queries": int(profile["clutter_queries"]),
        "occlusion_queries": int(profile["occlusion_queries"]),
    }


def _build_requests_from_args(generator: Any6DDataGenerator, args: argparse.Namespace) -> list[SampleRequest]:
    counts = _resolve_generation_counts(args)
    requests: list[SampleRequest] = []
    for object_name in _resolve_object_names(generator, args):
        requests.extend(generator.planner.plan_object_requests(object_name, counts))
    return requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isaac Sim -> Any6D data generator")
    parser.add_argument(
        "--mode",
        choices=("default", "local-test"),
        default="default",
        help="Asset source mode. 'default' uses the standard Isaac asset layout, 'local-test' uses the local cracker-box fallback asset.",
    )
    parser.add_argument(
        "--object",
        dest="object_name",
        default="003_cracker_box",
        help="Object name to generate. Must exist in the selected generator config.",
    )
    parser.add_argument(
        "--all-objects",
        action="store_true",
        help="Generate samples for all objects available in the selected generator config.",
    )
    parser.add_argument(
        "--dataset-profile",
        choices=tuple(DATASET_PROFILES.keys()),
        default=None,
        help="Optional preset generation profile. If set, overrides per-factor counts.",
    )
    parser.add_argument(
        "--anchors",
        type=int,
        default=1,
        help="Number of anchor samples to generate.",
    )
    parser.add_argument(
        "--viewpoint-queries",
        type=int,
        default=1,
        help="Number of viewpoint query samples to generate.",
    )
    parser.add_argument(
        "--distance-queries",
        type=int,
        default=0,
        help="Number of distance query samples to generate.",
    )
    parser.add_argument(
        "--lighting-queries",
        type=int,
        default=0,
        help="Number of lighting query samples to generate.",
    )
    parser.add_argument(
        "--viewpoint-lighting-queries",
        type=int,
        default=0,
        help="Number of combined viewpoint+lighting query samples to generate.",
    )
    parser.add_argument(
        "--viewpoint-lighting-clutter-queries",
        type=int,
        default=0,
        help="Number of combined viewpoint+lighting+clutter query samples to generate.",
    )
    parser.add_argument(
        "--viewpoint-lighting-occlusion-queries",
        type=int,
        default=0,
        help="Number of combined viewpoint+lighting+occlusion query samples to generate.",
    )
    parser.add_argument(
        "--viewpoint-lighting-clutter-occlusion-queries",
        type=int,
        default=0,
        help="Number of combined viewpoint+lighting+clutter+occlusion query samples to generate.",
    )
    parser.add_argument(
        "--clutter-queries",
        type=int,
        default=0,
        help="Number of clutter query samples to generate.",
    )
    parser.add_argument(
        "--occlusion-queries",
        type=int,
        default=0,
        help="Number of occlusion query samples to generate.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--object-frame-corrections-json",
        type=str,
        default="C:/isaac-sim/any6d/object_frame_corrections.initial_est.json",
        help="Optional JSON map of per-object transforms from Isaac object frame to Any6D reference object frame.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only export the dataset plan, without starting Isaac Sim capture.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose diagnostic logging for camera framing, semantics, and segmentation payloads.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    object_frame_corrections = load_object_frame_corrections(args.object_frame_corrections_json)

    if args.mode == "local-test":
        generator = build_local_test_generator(
            args.output_root or "C:/isaac-sim/any6d/_out_any6d_spec_localtest",
            debug_logging=args.debug,
            object_frame_corrections=object_frame_corrections,
        )
    else:
        generator = build_default_generator(
            args.output_root or "C:/isaac-sim/any6d/_out_any6d_spec",
            debug_logging=args.debug,
            object_frame_corrections=object_frame_corrections,
        )

    requests = _build_requests_from_args(generator, args)
    plan_path = generator.export_plan(requests)

    log("generator ready")
    print(json.dumps(generator.summarize_architecture(), indent=2), flush=True)
    log(f"planned {len(requests)} samples")
    log(f"plan written to: {plan_path}")

    if args.plan_only:
        raise SystemExit(0)

    exit_code = 0
    try:
        written_paths = generator.generate_requests(requests)
        log("generated samples:")
        for path in written_paths:
            print(f"  - {path.sample_dir}", flush=True)
    except Exception as exc:
        exit_code = 1
        log(f"fatal error: {exc}")
        traceback.print_exc()
    finally:
        generator.close()
        log(f"process exiting with code {exit_code}")
        sys.stdout.flush()
        sys.stderr.flush()
        raise SystemExit(exit_code)
