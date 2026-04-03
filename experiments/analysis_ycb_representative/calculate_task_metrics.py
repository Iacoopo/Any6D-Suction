import csv
import json
import numpy as np
from pathlib import Path

# Configurazione dei Grasp Frames canonici (nel frame Any6D_ref dell'oggetto)
# Point: [x, y, z] in metri, Normal: [nx, ny, nz] (vettore unitario di approccio)
GRASP_CONFIG = {
    "003_cracker_box":      {"point": [0.0, 0.0, 0.03],  "normal": [0.0, 0.0, 1.0]},
    "005_tomato_soup_can":  {"point": [0.0, 0.05, 0.0],  "normal": [0.0, 1.0, 0.0]},
    "006_mustard_bottle":   {"point": [0.0, 0.0, 0.025], "normal": [0.0, 0.0, 1.0]},
    "010_potted_meat_can":  {"point": [0.0, 0.04, 0.0],  "normal": [0.0, 1.0, 0.0]},
    "021_bleach_cleanser":  {"point": [0.0, 0.0, 0.035], "normal": [0.0, 0.0, 1.0]}
}

RESULTS_ROOT = Path("/home/iacopo/cv_final/Any6D/results/isaac_any6d")
EXPERIMENTS_DIR = Path("/home/iacopo/cv_final/experiments/analysis_ycb_representative")
OUT_PATH = EXPERIMENTS_DIR / "metrics" / "rq3_task_aware_metrics.csv"

def get_angle(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def process_experiment(exp_name, obj_name):
    exp_dir = RESULTS_ROOT / exp_name
    if not exp_dir.exists(): return []
    
    # Prendo l'ultima run
    run_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
    if not run_dirs: return []
    latest_run = run_dirs[-1]
    
    metrics_file = latest_run / "per_sample_metrics.csv"
    if not metrics_file.exists(): return []
    
    grasp = GRASP_CONFIG.get(obj_name)
    if not grasp: return []
    
    p_obj = np.array(grasp["point"])
    n_obj = np.array(grasp["normal"])
    
    results = []
    with open(metrics_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Carico pose 4x4
            t_gt = np.array(json.loads(row["gt_pose"]))
            t_pred = np.array(json.loads(row["pred_pose"]))
            
            # Punto di contatto in Camera Space
            cp_gt = (t_gt[:3, :3] @ p_obj) + t_gt[:3, 3]
            cp_pred = (t_pred[:3, :3] @ p_obj) + t_pred[:3, 3]
            
            # Normale in Camera Space
            norm_gt = t_gt[:3, :3] @ n_obj
            norm_pred = t_pred[:3, :3] @ n_obj
            
            # Calcolo metriche
            sce = np.linalg.norm(cp_gt - cp_pred) * 100.0 # cm
            sne = get_angle(norm_gt, norm_pred) # deg
            z_err = abs(cp_gt[2] - cp_pred[2]) * 100.0 # cm
            
            results.append({
                "obj": obj_name,
                "factor": row["factor"],
                "SCE_cm": sce,
                "SNE_deg": sne,
                "Z_err_cm": z_err
            })
    return results

def main():
    all_results = []
    # Carico la lista esperimenti
    with open(EXPERIMENTS_DIR / "experiments.json", 'r') as f:
        config = json.load(f)
    
    for exp_name, details in config["experiments"].items():
        all_results.extend(process_experiment(exp_name, details["object_name"]))
    
    # Aggrego per (oggetto, fattore)
    summary = {}
    for r in all_results:
        key = (r["obj"], r["factor"])
        if key not in summary: summary[key] = {"sce": [], "sne": [], "z": []}
        summary[key]["sce"].append(r["SCE_cm"])
        summary[key]["sne"].append(r["SNE_deg"])
        summary[key]["z"].append(r["Z_err_cm"])
    
    with open(OUT_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["object_name", "factor", "mean_SCE_cm", "mean_SNE_deg", "mean_Z_err_cm", "success_SNE_15deg"])
        for (obj, factor), data in summary.items():
            sne_list = np.array(data["sne"])
            writer.writerow([
                obj, factor,
                np.mean(data["sce"]),
                np.mean(sne_list),
                np.mean(data["z"]),
                np.mean(sne_list < 15.0) # Recall per ventosa (allineamento < 15 gradi)
            ])
    print(f"Wrote Task-Aware metrics to {OUT_PATH}")

if __name__ == "__main__":
    main()
