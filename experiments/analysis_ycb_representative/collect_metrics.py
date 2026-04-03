import csv
import json
from pathlib import Path


ROOT = Path(__file__).parent.resolve()
MANIFEST = ROOT / "experiments.json"
METRICS_DIR = ROOT / "metrics"
RESULTS_ROOT = Path("/home/iacopo/cv_final/Any6D/results/isaac_any6d")


def load_manifest():
    with MANIFEST.open("r") as f:
        return json.load(f)


def latest_result_dir(experiment_name: str):
    exp_root = RESULTS_ROOT / experiment_name
    if not exp_root.exists():
        return None
    dirs = sorted([p for p in exp_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not dirs:
        return None
    return dirs[-1]


def read_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def read_csv_rows(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def main():
    data = load_manifest()
    experiments = data["experiments"]
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    run_rows = []
    summary_rows = []

    for exp_name in experiments:
        latest_dir = latest_result_dir(exp_name)
        if latest_dir is None:
            run_rows.append(
                {
                    "experiment_name": exp_name,
                    "status": "missing",
                    "results_dir": "",
                }
            )
            continue

        latest_dir = Path(latest_dir)
        config_path = latest_dir / "config.json"
        run_summary_path = latest_dir / "run_summary.json"
        summary_path = latest_dir / "summary_by_factor_object.csv"

        if not (config_path.exists() and run_summary_path.exists() and summary_path.exists()):
            run_rows.append(
                {
                    "experiment_name": exp_name,
                    "status": "incomplete",
                    "results_dir": str(latest_dir),
                }
            )
            continue

        config = read_json(config_path)
        run_summary = read_json(run_summary_path)
        rows = read_csv_rows(summary_path)
        row = rows[0] if rows else {}

        run_rows.append(
            {
                "experiment_name": exp_name,
                "status": "ok",
                "results_dir": str(latest_dir),
                "factor": config.get("factor", ""),
                "object_name": config.get("object_name", ""),
                "dataset_root": config.get("dataset_root", ""),
                "gt_pose_field": config.get("gt_pose_field", ""),
                "num_samples_processed": run_summary.get("num_samples_processed", ""),
                "num_sample_dirs_discovered": run_summary.get("num_sample_dirs_discovered", ""),
            }
        )

        if row:
            summary_rows.append(
                {
                    "experiment_name": exp_name,
                    "results_dir": str(latest_dir),
                    "factor_name": row.get("factor_name", ""),
                    "object_name": row.get("object_name", ""),
                    "num_samples": row.get("num_samples", ""),
                    "R_error_deg": row.get("R_error_deg", ""),
                    "T_error_cm": row.get("T_error_cm", ""),
                    "ADD_m": row.get("ADD_m", ""),
                    "ADD-S_m": row.get("ADD-S_m", ""),
                    "ADD_0.1d": row.get("ADD_0.1d", ""),
                    "ADD-S_0.1d": row.get("ADD-S_0.1d", ""),
                    "SCE_cm": row.get("SCE_cm", ""),
                    "SNE_deg": row.get("SNE_deg", ""),
                    "Z_err_cm": row.get("Z_err_cm", ""),
                    "success_SNE_15deg": row.get("success_SNE_15deg", ""),
                    "visibility_ratio": row.get("visibility_ratio", ""),
                    "occlusion_ratio": row.get("occlusion_ratio", ""),
                }
            )

    latest_runs_path = METRICS_DIR / "latest_runs.csv"
    latest_summary_path = METRICS_DIR / "latest_summary.csv"

    with latest_runs_path.open("w", newline="") as f:
        fieldnames = [
            "experiment_name",
            "status",
            "results_dir",
            "factor",
            "object_name",
            "dataset_root",
            "gt_pose_field",
            "num_samples_processed",
            "num_sample_dirs_discovered",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_rows)

    with latest_summary_path.open("w", newline="") as f:
        fieldnames = [
            "experiment_name",
            "results_dir",
            "factor_name",
            "object_name",
            "num_samples",
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
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote: {latest_runs_path}")
    print(f"Wrote: {latest_summary_path}")


if __name__ == "__main__":
    main()
