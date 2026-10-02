# Training and evaluation configuration

Every run in the paper used **Ultralytics 8.3.107**, **PyTorch 2.11.0+cu128**, **Python 3.12.13** on a single **NVIDIA T4** (Google Colab). The full frozen environment is in `env/requirements_frozen.txt`; the experiment matrix and module settings are in `configs/experiment_config.json`.

Arguments not listed here are Ultralytics 8.3.107 defaults (`ultralytics/cfg/default.yaml` at tag `v8.3.107`). `configs/logged_train_args.json` holds five complete argument lists exactly as Ultralytics logged them, for the BDD100K stage-2 runs trained in the final session.

## Stage 1: base training (all models, 80 epochs)

The notebook passes only these arguments; everything else is the framework default:

| Argument | Value |
|---|---|
| `model` | COCO-pretrained checkpoint (`yolov5su.pt`, `yolov8s.pt`, `yolov10s.pt`, `yolo11s.pt`); PGP variants start from `checkpoints/pgp_a1_init.pt` or `pgp_a2_init.pt` (YOLO11s weights transferred into the modified graph) |
| `epochs` / `patience` | 80 / 25 |
| `imgsz` / `batch` / `workers` | 640 / 16 / 2 |
| `seed` | 0 (unsuffixed runs), 1 or 2 (`_scr1`, `_scr2` replicates); `deterministic=True` |

Resolved defaults that matter:

| Setting | Value |
|---|---|
| Optimizer | `optimizer='auto'`. In 8.3.107 the rule computes `ceil(N / max(batch, nbs)) × epochs` optimizer steps; at ≤ 10,000 it selects **AdamW** with `lr0 = round(0.002 × 5 / (4 + nc), 6)` and momentum 0.9, and sets `warmup_bias_lr = 0`. KITTI: 94 × 80 = 7,520 steps, nc = 6, so **AdamW, lr0 = 0.001**. With PA-CP images (7,719 images): 121 × 80 = 9,680 steps, still AdamW at 0.001. BDD100K: 94 × 80 = 7,520 steps, nc = 10, so **AdamW, lr0 ≈ 0.000714**. |
| Learning-rate schedule | linear decay to `lr0 × lrf`, `lrf = 0.01` |
| Weight decay | 0.0005 (scaled by `batch × accumulate / nbs` = 1) |
| Nominal batch | `nbs = 64`, so gradients accumulate over 4 batches of 16 |
| Warm-up | 3 epochs, warm-up momentum 0.8 |
| Loss gains | box 7.5, cls 0.5, dfl 1.5 |
| Augmentation | mosaic 1.0 with `close_mosaic = 10` (off for the last 10 epochs); scale 0.5; translate 0.1; fliplr 0.5; HSV (0.015, 0.7, 0.4); mixup 0; copy_paste 0 |
| Precision | AMP (mixed precision) |

## Stage 2: 25-epoch fine-tuning from the stage-1 `best.pt`

Both arms use the same optimizer and schedule and differ only in augmentation:

| Argument | Geo-FT (geometry-consistent) | Std-FT (equal-budget control) |
|---|---|---|
| `epochs` / `patience` | 25 / 15 | 25 / 15 |
| `optimizer`, `lr0`, `lrf` | SGD, 0.002, 0.05 | SGD, 0.002, 0.05 |
| momentum (logged) | 0.937 | 0.937 |
| `warmup_epochs` | 1.0 | 1.0 |
| `mosaic` | **0.0** | **1.0** with `close_mosaic = 0` (mosaic on for all 25 epochs) |
| `scale` / `translate` | **0.2 / 0.05** | 0.5 / 0.1 |
| `mixup` / `copy_paste` | 0 / 0 | 0 / 0 |
| `seed` | as the parent run (KITTI); 0, 1, 2 on a seed-0 parent (BDD100K `_s1`, `_s2`) | same |

## Checkpoint selection and evaluation

* **Reported checkpoint**: Ultralytics `best.pt`, which maximizes `0.1 × mAP@0.5 + 0.9 × mAP@0.5:0.95` on the validation split. KITTI has no labelled test split, so selection and evaluation share the validation split; this applies identically to every arm.
* **Accuracy**: `model.val(split="val", imgsz=640, batch=16)` with framework defaults: confidence 0.001, NMS IoU 0.7, at most 300 detections per image.
* **Latency**: 220 `predict` calls on one in-memory validation image; the first 20 are discarded as warm-up and the median of the remaining 200 is reported (FP16, batch 1, confidence 0.25, NMS IoU 0.7).
* **Height-stratified AP**: confidence 0.001, at most 300 detections, IoU 0.5 matching; strata by ground-truth box height: < 25 px, 25–40 px, ≥ 40 px.
* **Confusion matrices**: confidence 0.25, IoU 0.5, explicit greedy matcher.

## PGP module and augmentation settings

| Component | Setting |
|---|---|
| PGSA | after each of the three detection-head inputs; MLP 7 → 32 → 8 (8 channel groups), input = row embedding (r, r², sin πr, cos πr) + pyramid-level one-hot; last layer zero-initialized (identity at start); class-agnostic |
| PGCSF | after each neck concatenation; MLP 4 → 16 → 2, softmax re-weighting of the two concatenated pathways; zero-initialized; class-agnostic |
| Parameter overhead | +2,016 parameters and +0.0034 GFLOPs over YOLO11s (9,415,122 parameters) |
| Perspective prior | 24 ground-contact-row bins; per-class median and interquartile range of normalized box height (`prior/perspective_prior.json`) |
| PA-CP | 30% of training images get one augmented copy (1,735 images); 1–3 pasted Pedestrian/Cyclist instances from a crop bank of training instances with normalized height > 0.07; ground-contact row ~ Uniform(horizon, 0.96); height = per-class prior median × Uniform(0.85, 1.20); overlapping or out-of-frame pastes rejected; used only by the v1 and v2 models |
| Plausibility filter | drops detections whose height deviates from the per-class prior at their row by more than a log-ratio threshold τ, tuned on 400 training images |
