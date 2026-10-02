# When Priors Meet Budgets — code, results, and data splits

This repository accompanies the manuscript **"When Priors Meet Budgets: A Controlled Study of Perspective Geometry for Real-Time Vehicle and Pedestrian Detection"** by Rakibul Alam, Md. Shohel Sayeed, and Siti Fatimah Binti Abdul Razak (Multimedia University), under review at the *Journal of Computational and Cognitive Engineering* (manuscript JCCE-11617).

It contains everything needed to check or reproduce the paper's numbers: the experiment notebook, the module code, the exact data splits, the full training configuration, the per-run results for all 47 runs, and scripts that rebuild the replication tables from those results without a GPU.

## Contents

| Path | What it is |
|---|---|
| `notebook/PGP_YOLO_KITTI_Q1_Experiment.ipynb` | The complete experiment, executed, with outputs. Runs top to bottom on free Google Colab and resumes across session limits. |
| `code/pgp_modules.py` | PGSA and PGCSF modules and the YOLO11 graph surgery (extracted from the notebook). |
| `code/regenerate_tables.py` | Rebuilds Tables 7, 8, and 9 (seed robustness, paired tests with Holm correction, BDD100K) from `results/`. CPU only, seconds. |
| `code/export_bdd100k_subset_lists.py` | Lists the exact BDD100K subset from the Drive cache (see below). |
| `configs/TRAINING_CONFIGURATION.md` | Every training and evaluation setting, including resolved framework defaults. |
| `configs/experiment_config.json` | The experiment configuration as run. |
| `configs/pgp_a1_yolo11s.yaml`, `pgp_a2_yolo11s.yaml` | Model definitions for the PGSA and PGSA+PGCSF variants. |
| `configs/logged_train_args.json` | Complete argument lists as logged by Ultralytics for five stage-2 runs. |
| `checkpoints/pgp_a1_init.pt`, `pgp_a2_init.pt` | Initial weights for the PGP variants (YOLO11s COCO weights transferred into the modified graph; new layers at identity). |
| `splits/kitti_train.txt`, `splits/kitti_val.txt` | The exact KITTI split (5,984 / 1,497 image IDs). |
| `results/*.json` | Per-run metrics for all 47 runs (precision, recall, mAP@0.5, mAP@0.5:0.95, per-class AP, latency, parameters, GFLOPs), plus cached predictions and stratified analyses. |
| `tables/`, `figures/`, `prior/` | Tables and figures as produced by the notebook, and the fitted perspective prior. |
| `env/requirements_frozen.txt`, `env/nvidia_smi.txt` | Frozen software environment and GPU. |

## Quick check without a GPU

```bash
pip install numpy scipy
python code/regenerate_tables.py
```

This prints and writes `tables/paper_table7_seed.csv`, `paper_table8_statistics.csv`, and `paper_table9_bdd100k.csv`, which match Tables 7–9 of the manuscript.

## Full reproduction

1. Open `notebook/PGP_YOLO_KITTI_Q1_Experiment.ipynb` in Google Colab with a T4 GPU.
2. Add your own Kaggle API token when the notebook asks (KITTI is downloaded from the Kaggle mirror `klemenko/kitti-dataset`, BDD100K from `a7madmostafa/bdd100k-yolo`).
3. Run all cells. Checkpoints are written to Google Drive, so after a session limit you simply run all cells again and training resumes. The full matrix is about 100 GPU-hours on a T4.

Software: Ultralytics 8.3.107, PyTorch 2.11.0+cu128, Python 3.12.13. The notebook pins Ultralytics; the rest of the environment is in `env/requirements_frozen.txt`.

## Data splits

**KITTI.** The notebook sorts the 7,481 labelled training image IDs, shuffles them with `random.Random(0)`, and takes the first 80% for training. `splits/kitti_*.txt` list the result. We confirmed that this procedure reproduces exactly the 1,497 validation images recorded in the cached predictions (`results/preds_val_*.json`).

**BDD100K.** The paper uses a fixed random subset of 6,000 training and 1,500 validation images drawn once from the Kaggle mirror and cached on Google Drive. The sampler shuffled files in filesystem order, so re-running it on another machine is not guaranteed to draw the same subset. The cached archive is therefore the authoritative record, and `code/export_bdd100k_subset_lists.py` lists it as `splits/bdd100k_train.txt` and `splits/bdd100k_val.txt`.

## Data licences

The KITTI and BDD100K images are not redistributed here; obtain them from their providers under their own licences. This repository contains only image IDs, derived metrics, and code.

## Run naming

| Pattern | Meaning |
|---|---|
| `yolo11s`, `yolov8s`, … | 80-epoch stage-1 base, seed 0 |
| `*_geo` / `*_stdft` | 25-epoch geometry-consistent fine-tune / equal-budget standard-augmentation control |
| `*_scr1`, `*_scr2` | KITTI replicates that re-run the entire pipeline from the COCO-pretrained weights with seeds 1 and 2 |
| `bdd_*_s1`, `bdd_*_s2` | BDD100K stage-2 seeds 1 and 2, branching from the shared seed-0 base |
| `pgp_A1_pgsa`, `pgp_A2_pgsa_pgcsf`, `pgp_yolo_full`, `pgp_v2`, `pgp_a1_geo` | ablation variants A1, A2, v1, v2, and A1 with the geometry-consistent fine-tune |
| `*_s1`, `*_s2` (KITTI) | an earlier, superseded replication that varied only stage 2 from one shared checkpoint; kept because the paper reports and retracts its result |

## Paper figures and tables

| Paper | File |
|---|---|
| Figures 1–13 | `F01_dataset_statistics`, `F03_perspective_prior`, `F05_architecture`, `F06_module_detail`, `F04_pacp_examples`, `F10_speed_accuracy`, `F15_ablation`, `F19_recipe_effect`, `F07_map_curves`, `F18_stratified_ap`, `F13_pr_curves`, `F17_learned_gates`, `F16_qualitative` (`figures/`; titles were cropped for the journal) |
| Table 1 | `tables/T1_dataset_summary.csv` |
| Table 2 | `configs/TRAINING_CONFIGURATION.md` |
| Tables 3, 4, 5, 6, 10 | `tables/T2_benchmark.csv`, `T4_per_class_ap50.csv`, `T3_ablation.csv`, `T6_recipe_effect.csv`, `T5_stratified.csv` |
| Tables 7, 8, 9 | `code/regenerate_tables.py` |

## Citation

See `CITATION.cff`. Please cite the article once it is published.
