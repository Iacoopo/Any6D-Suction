import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from itertools import permutations, product

import cv2
import numpy as np
import pandas as pd
import trimesh
from pytorch_lightning import seed_everything
from tqdm import tqdm

import nvdiffrast.torch as dr

from estimater import Any6D
from foundationpose.learning.training.predict_pose_refine import PoseRefinePredictor
from foundationpose.learning.training.predict_score import ScorePredictor
from metrics import compute_RT_distances, compute_add, compute_adds


# Configurazione dei Grasp Frames canonici per RQ3 (nel frame Any6D_ref dell'oggetto)
# Point: [x, y, z] in metri, Normal: [nx, ny, nz] (vettore unitario di approccio)
GRASP_CONFIG = {
    "003_cracker_box":      {"point": [0.0, 0.0, 0.03],  "normal": [0.0, 0.0, 1.0]},
    "005_tomato_soup_can":  {"point": [0.0, 0.05, 0.0],  "normal": [0.0, 1.0, 0.0]},
    "006_mustard_bottle":   {"point": [0.0, 0.0, 0.025], "normal": [0.0, 0.0, 1.0]},
    "010_potted_meat_can":  {"point": [0.0, 0.04, 0.0],  "normal": [0.0, 1.0, 0.0]},
    "021_bleach_cleanser":  {"point": [0.0, 0.0, 0.035], "normal": [0.0, 0.0, 1.0]}
}

def get_angle(v1, v2):
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return 0.0
    v1_u = v1 / v1_norm
    v2_u = v2 / v2_norm
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Any6D on Isaac-generated query samples.")
    parser.add_argument(
        "--name",
        type=str,
        default="isaac_any6d_eval",
        help="Experiment name used for the results directory.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/iacopo/cv_final/datasets/_out_any6d_spec",
        help="Root of the Isaac Any6D dataset export.",
    )
    parser.add_argument(
        "--ycb_model_path",
        type=str,
        default="/home/iacopo/cv_final/datasets/ho3d/YCB_Video_Models",
        help="Root containing YCB meshes under models/<object>/textured_simple.obj.",
    )
    parser.add_argument(
        "--factor",
        type=str,
        default=None,
        help="Optional factor filter, e.g. viewpoint, occlusion, lighting.",
    )
    parser.add_argument(
        "--object_name",
        type=str,
        default=None,
        help="Optional object filter, e.g. 003_cracker_box.",
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        choices=["visible", "full"],
        default="visible",
        help="Use visible mask or full silhouette mask for inference.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=5,
        help="Number of Any6D refinement iterations.",
    )
    parser.add_argument(
        "--max_samples_per_factor",
        type=int,
        default=None,
        help="Optional cap on processed samples for each factor/object pair.",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="Any6D debug level.",
    )
    parser.add_argument(
        "--gt_camera_convention",
        type=str,
        choices=["cv", "isaac_usd"],
        default="cv",
        help="Convention used by T_C_O stored in meta.json. Use isaac_usd to convert Isaac/USD camera frame to CV frame before evaluation.",
    )
    parser.add_argument(
        "--gt_pose_field",
        type=str,
        choices=["auto", "T_C_O", "T_C_O_any6d_ref"],
        default="auto",
        help="Which GT pose field from meta.json to evaluate against. 'auto' prefers T_C_O_any6d_ref when available.",
    )
    parser.add_argument(
        "--auto_object_frame_correction",
        action="store_true",
        help="Search a discrete set of rigid object-frame rotations and report the best correction for GT evaluation.",
    )
    parser.add_argument(
        "--object_frame_search_mode",
        type=str,
        choices=["symmetry_aware", "strict", "adds_only"],
        default="symmetry_aware",
        help="Ranking mode for object-frame correction search.",
    )
    parser.add_argument(
        "--object_frame_corrections_json",
        type=str,
        default=None,
        help="Optional JSON file mapping object_name to a 3x3 or 4x4 rigid correction matrix applied to GT object frames.",
    )
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    color_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")
    return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)


def load_depth(path: Path, depth_scale: float) -> np.ndarray:
    depth_raw = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    if depth_raw is None:
        raise FileNotFoundError(f"Could not read depth image: {path}")
    depth = depth_raw.astype(np.float32) / float(depth_scale)
    depth[np.isnan(depth)] = 0
    depth[np.isinf(depth)] = 0
    return depth


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask image: {path}")
    return mask > 0


def discover_query_samples(dataset_root: Path, factor_filter=None, object_filter=None, max_samples_per_factor=None):
    queries_root = dataset_root / "queries"
    if not queries_root.exists():
        raise FileNotFoundError(f"Queries root not found: {queries_root}")

    sample_dirs = []
    factors = [factor_filter] if factor_filter else sorted([p.name for p in queries_root.iterdir() if p.is_dir()])

    for factor in factors:
        factor_dir = queries_root / factor
        if not factor_dir.exists():
            continue
        objects = [object_filter] if object_filter else sorted([p.name for p in factor_dir.iterdir() if p.is_dir()])
        for obj_name in objects:
            obj_dir = factor_dir / obj_name
            if not obj_dir.exists():
                continue
            dirs = sorted([p for p in obj_dir.iterdir() if p.is_dir()])
            if max_samples_per_factor is not None:
                dirs = dirs[:max_samples_per_factor]
            sample_dirs.extend(dirs)
    return sample_dirs


def mesh_path_for_object(ycb_model_path: Path, object_name: str) -> Path:
    return ycb_model_path / "models" / object_name / "textured_simple.obj"


def flatten_factor_value(value):
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def convert_pose_isaac_usd_to_cv(t_c_o: np.ndarray) -> np.ndarray:
    # Isaac/USD camera convention is typically X right, Y up, -Z forward.
    # CV/OpenCV convention is X right, Y down, +Z forward.
    # This change of basis flips Y and Z in camera coordinates.
    usd_to_cv = np.eye(4, dtype=np.float32)
    usd_to_cv[1, 1] = -1.0
    usd_to_cv[2, 2] = -1.0
    return usd_to_cv @ t_c_o


def generate_discrete_object_frame_rotations():
    rots = []
    labels = []
    basis = np.eye(3, dtype=np.float32)
    for perm in permutations(range(3)):
        P = basis[:, perm]
        for signs in product([-1.0, 1.0], repeat=3):
            R = P @ np.diag(np.asarray(signs, dtype=np.float32))
            if np.linalg.det(R) > 0.5:
                if not any(np.allclose(R, existing) for existing in rots):
                    rots.append(R.astype(np.float32))
                    labels.append(
                        f"perm_{perm[0]}{perm[1]}{perm[2]}_sign_{int(signs[0]):+d}{int(signs[1]):+d}{int(signs[2]):+d}"
                    )
    return labels, rots


def apply_object_frame_correction(gt_pose: np.ndarray, rotation_correction: np.ndarray) -> np.ndarray:
    correction = np.eye(4, dtype=np.float32)
    rotation_correction = np.asarray(rotation_correction, dtype=np.float32)
    if rotation_correction.shape == (4, 4):
        correction = rotation_correction.astype(np.float32)
    else:
        correction[:3, :3] = rotation_correction
    return gt_pose @ correction


def load_object_frame_corrections(path: str | None):
    if path is None:
        return {}

    with open(path, "r") as f:
        raw = json.load(f)

    corrections = {}
    for object_name, matrix in raw.items():
        arr = np.asarray(matrix, dtype=np.float32)
        if arr.shape not in {(3, 3), (4, 4)}:
            raise ValueError(
                f"Invalid correction shape for {object_name}: {arr.shape}. Expected 3x3 or 4x4."
            )
        corrections[object_name] = arr
    return corrections


def resolve_gt_pose(meta: dict, gt_pose_field: str) -> tuple[np.ndarray, str]:
    if gt_pose_field == "auto":
        if "T_C_O_any6d_ref" in meta:
            return np.asarray(meta["T_C_O_any6d_ref"], dtype=np.float32), "T_C_O_any6d_ref"
        return np.asarray(meta["T_C_O"], dtype=np.float32), "T_C_O"

    if gt_pose_field not in meta:
        raise KeyError(f"GT pose field '{gt_pose_field}' not found in meta.json for sample {meta.get('sample_id')}")

    return np.asarray(meta[gt_pose_field], dtype=np.float32), gt_pose_field


def safe_mean(values):
    if not values:
        return np.nan
    return float(np.mean(values))


def summarize(rows, group_keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in group_keys)].append(row)

    summary_rows = []
    for key, items in grouped.items():
        out = {k: v for k, v in zip(group_keys, key)}
        out["num_samples"] = len(items)
        for metric in [
            "R_error_deg",
            "T_error_cm",
            "ADD_m",
            "ADD-S_m",
            "ADD_0.1d",
            "ADD-S_0.1d",
            "SCE_cm",
            "SNE_deg",
            "Z_err_cm",
            "success_SNE_15deg",
            "visibility_ratio",
            "occlusion_ratio",
        ]:
            out[metric] = safe_mean([item[metric] for item in items])
        summary_rows.append(out)

    return pd.DataFrame(summary_rows).sort_values(group_keys).reset_index(drop=True)


def evaluate_rows_with_object_frame_search(rows, ranking_mode="symmetry_aware"):
    candidate_labels, candidate_rots = generate_discrete_object_frame_rotations()
    candidate_rows = []

    for label, rot in zip(candidate_labels, candidate_rots):
        r_errs = []
        t_errs = []
        add_vals = []
        adds_vals = []
        add_hits = []
        adds_hits = []

        for row in rows:
            corrected_gt = apply_object_frame_correction(row["gt_pose_eval"], rot)
            err_R, err_T = compute_RT_distances(row["pred_pose"], corrected_gt)
            add = compute_add(row["mesh_vertices"], row["pred_pose"], corrected_gt)
            adds = compute_adds(row["mesh_vertices"], row["pred_pose"], corrected_gt)
            diameter = row["diameter_m"]

            r_errs.append(float(err_R[0]))
            t_errs.append(float(err_T[0]))
            add_hits.append(float(add <= diameter * 0.1))
            adds_hits.append(float(adds <= diameter * 0.1))
            adds_vals.append(float(adds))
            add_vals.append(float(add))

        candidate_rows.append(
            {
                "correction_label": label,
                "mean_R_error_deg": safe_mean(r_errs),
                "mean_T_error_cm": safe_mean(t_errs),
                "mean_ADD_m": safe_mean(add_vals),
                "mean_ADD-S_m": safe_mean(adds_vals),
                "mean_ADD_0.1d": safe_mean(add_hits),
                "mean_ADD-S_0.1d": safe_mean(adds_hits),
                "rotation_matrix": np.asarray(rot).tolist(),
            }
        )

    df = pd.DataFrame(candidate_rows)

    if ranking_mode == "adds_only":
        sort_keys = ["mean_ADD-S_m", "mean_T_error_cm", "mean_R_error_deg"]
    elif ranking_mode == "strict":
        sort_keys = ["mean_ADD_m", "mean_R_error_deg", "mean_ADD-S_m", "mean_T_error_cm"]
    else:
        # Symmetry-aware default:
        # 1. prefer solutions that are good under ADD-S
        # 2. then prefer low translational error
        # 3. break ties with ADD and only after that with rotation
        # This keeps the ranking robust for symmetric objects while still
        # discouraging obviously degenerate 180 deg alternatives when a
        # geometrically cleaner solution exists.
        sort_keys = ["mean_ADD-S_m", "mean_T_error_cm", "mean_ADD_m", "mean_R_error_deg"]

    df = df.sort_values(sort_keys).reset_index(drop=True)
    return df


def main():
    args = parse_args()
    seed_everything(0)

    dataset_root = Path(args.dataset_root).resolve()
    ycb_model_path = Path(args.ycb_model_path).resolve()
    object_frame_corrections = load_object_frame_corrections(args.object_frame_corrections_json)

    date_str = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
    save_root = Path("/home/iacopo/cv_final/Any6D/results/isaac_any6d") / args.name / date_str
    save_root.mkdir(parents=True, exist_ok=True)

    sample_dirs = discover_query_samples(
        dataset_root,
        factor_filter=args.factor,
        object_filter=args.object_name,
        max_samples_per_factor=args.max_samples_per_factor,
    )
    if not sample_dirs:
        raise RuntimeError("No query samples found with the current filters.")

    config = {
        "name": args.name,
        "dataset_root": str(dataset_root),
        "ycb_model_path": str(ycb_model_path),
        "factor": args.factor,
        "object_name": args.object_name,
        "mask_type": args.mask_type,
        "iteration": args.iteration,
        "max_samples_per_factor": args.max_samples_per_factor,
        "gt_camera_convention": args.gt_camera_convention,
        "gt_pose_field": args.gt_pose_field,
        "auto_object_frame_correction": args.auto_object_frame_correction,
        "object_frame_search_mode": args.object_frame_search_mode,
        "object_frame_corrections_json": args.object_frame_corrections_json,
        "objects_with_fixed_frame_correction": sorted(object_frame_corrections.keys()),
        "num_samples": len(sample_dirs),
    }
    with open(save_root / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    glctx = dr.RasterizeCudaContext()
    dummy_mesh = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4))
    est = Any6D(
        mesh=trimesh.Trimesh(vertices=dummy_mesh.vertices.copy(), faces=dummy_mesh.faces.copy()),
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        glctx=glctx,
        debug=args.debug,
        debug_dir=str(save_root / "debug"),
    )

    object_cache = {}
    rows = []

    for sample_dir in tqdm(sample_dirs, desc="Isaac query samples"):
        meta_path = sample_dir / "meta.json"
        rgb_path = sample_dir / "rgb.png"
        depth_path = sample_dir / "depth.png"
        k_path = sample_dir / "K.txt"
        mask_path = sample_dir / ("mask_visib_target.png" if args.mask_type == "visible" else "mask_target.png")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        object_name = meta["object_name"]
        factor_name = meta["factor_name"]
        mesh_path = mesh_path_for_object(ycb_model_path, object_name)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found for {object_name}: {mesh_path}")

        if object_name not in object_cache:
            mesh = trimesh.load(mesh_path)
            diameter = float(np.linalg.norm(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)))
            object_cache[object_name] = {"mesh": mesh, "diameter": diameter}
        est.reset_object(mesh=object_cache[object_name]["mesh"], symmetry_tfs=None)

        rgb = load_rgb(rgb_path)
        depth = load_depth(depth_path, meta["depth_scale"])
        mask = load_mask(mask_path)
        K = np.loadtxt(k_path).astype(np.float32)
        gt_pose_raw, gt_pose_field_used = resolve_gt_pose(meta, args.gt_pose_field)
        if args.gt_camera_convention == "isaac_usd":
            gt_pose = convert_pose_isaac_usd_to_cv(gt_pose_raw)
        else:
            gt_pose = gt_pose_raw

        if object_name in object_frame_corrections:
            gt_pose = apply_object_frame_correction(gt_pose, object_frame_corrections[object_name])

        if mask.sum() == 0:
            continue
        if np.count_nonzero(depth[mask] > 0) < 10:
            continue

        pred_pose = est.register(
            K=K,
            rgb=rgb,
            depth=depth,
            ob_mask=mask,
            iteration=args.iteration,
            name=meta["sample_id"].replace("/", "_"),
        )

        err_R, err_T = compute_RT_distances(pred_pose, gt_pose)
        add = compute_add(object_cache[object_name]["mesh"].vertices, pred_pose, gt_pose)
        adds = compute_adds(object_cache[object_name]["mesh"].vertices, pred_pose, gt_pose)
        diameter = object_cache[object_name]["diameter"]

        # Calcolo metriche operative per RQ3
        sce_cm = 0.0
        sne_deg = 0.0
        z_err_cm = 0.0
        success_sne_15 = 0.0
        
        grasp = GRASP_CONFIG.get(object_name)
        if grasp:
            p_obj = np.array(grasp["point"])
            n_obj = np.array(grasp["normal"])
            
            # Punto di contatto in Camera Space
            cp_gt = (gt_pose[:3, :3] @ p_obj) + gt_pose[:3, 3]
            cp_pred = (pred_pose[:3, :3] @ p_obj) + pred_pose[:3, 3]
            
            # Normale in Camera Space
            norm_gt = gt_pose[:3, :3] @ n_obj
            norm_pred = pred_pose[:3, :3] @ n_obj
            
            sce_cm = float(np.linalg.norm(cp_gt - cp_pred) * 100.0)
            sne_deg = float(get_angle(norm_gt, norm_pred))
            z_err_cm = float(abs(cp_gt[2] - cp_pred[2]) * 100.0)
            success_sne_15 = float(sne_deg < 15.0)

        row = {
            "sample_id": meta["sample_id"],
            "object_name": object_name,
            "factor_name": factor_name,
            "factor_value": flatten_factor_value(meta.get("factor_value")),
            "mask_type": args.mask_type,
            "gt_camera_convention": args.gt_camera_convention,
            "gt_pose_field_used": gt_pose_field_used,
            "has_fixed_object_frame_correction": object_name in object_frame_corrections,
            "visibility_ratio": float(meta.get("visibility_ratio", np.nan)),
            "occlusion_ratio": float(meta.get("occlusion_ratio", np.nan)),
            "gt_z_raw_m": float(gt_pose_raw[2, 3]),
            "gt_z_eval_m": float(gt_pose[2, 3]),
            "R_error_deg": float(err_R[0]),
            "T_error_cm": float(err_T[0]),
            "ADD_m": float(add),
            "ADD-S_m": float(adds),
            "diameter_m": float(diameter),
            "ADD_0.1d": float(add <= diameter * 0.1),
            "ADD-S_0.1d": float(adds <= diameter * 0.1),
            "SCE_cm": sce_cm,
            "SNE_deg": sne_deg,
            "Z_err_cm": z_err_cm,
            "success_SNE_15deg": success_sne_15,
            "bbox_2d_tight_xyxy": json.dumps(meta.get("bbox_2d_tight_xyxy", [])),
            "pred_pose": pred_pose.astype(np.float32),
            "gt_pose_eval": gt_pose.astype(np.float32),
            "mesh_vertices": np.asarray(object_cache[object_name]["mesh"].vertices, dtype=np.float32),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No valid samples were processed. Check masks, depth, and dataset structure.")

    export_rows = []
    for row in rows:
        export_row = dict(row)
        export_row.pop("pred_pose", None)
        export_row.pop("gt_pose_eval", None)
        export_row.pop("mesh_vertices", None)
        export_rows.append(export_row)

    per_sample_df = pd.DataFrame(export_rows).sort_values(["factor_name", "object_name", "sample_id"]).reset_index(drop=True)
    per_sample_df.to_csv(save_root / "per_sample_metrics.csv", index=False)

    summary_factor_object_df = summarize(rows, ["factor_name", "object_name"])
    summary_factor_object_df.to_csv(save_root / "summary_by_factor_object.csv", index=False)

    summary_factor_df = summarize(rows, ["factor_name"])
    summary_factor_df.to_csv(save_root / "summary_by_factor.csv", index=False)

    summary_object_df = summarize(rows, ["object_name"])
    summary_object_df.to_csv(save_root / "summary_by_object.csv", index=False)

    if args.auto_object_frame_correction:
        correction_df = evaluate_rows_with_object_frame_search(
            rows,
            ranking_mode=args.object_frame_search_mode,
        )
        correction_df.to_csv(save_root / "object_frame_correction_search.csv", index=False)
        best = correction_df.iloc[0].to_dict()
        with open(save_root / "best_object_frame_correction.json", "w") as f:
            json.dump(best, f, indent=2)

    with open(save_root / "run_summary.json", "w") as f:
        json.dump(
            {
                "num_samples_processed": len(rows),
                "num_sample_dirs_discovered": len(sample_dirs),
                "results_dir": str(save_root),
            },
            f,
            indent=2,
        )

    print(f"Results saved to: {save_root}")


if __name__ == "__main__":
    main()
