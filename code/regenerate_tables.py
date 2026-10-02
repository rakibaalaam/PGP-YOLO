# -*- coding: utf-8 -*-
"""Regenerate the manuscript's replication tables from the per-run result files,
with no GPU and no dataset:

  Table 7  seed robustness over three independent end-to-end replicates (KITTI)
  Table 8  paired significance tests (t-test, 95% CI, d_z, Holm correction)
  Table 9  external validation on BDD100K

Usage:  python code/regenerate_tables.py        (from the repository root)
Writes: tables/paper_table7_seed.csv, paper_table8_statistics.csv,
        paper_table9_bdd100k.csv, and prints them.

Run naming: KITTI seed 0 is the unsuffixed run and seeds 1-2 carry `_scr1`/`_scr2`
(each re-trains the whole pipeline from the COCO-pretrained weights). BDD100K seeds
1-2 carry `_s1`/`_s2` and branch from one shared seed-0 base."""
import csv, json, os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, OUT = os.path.join(ROOT, "results"), os.path.join(ROOT, "tables")

def m5095(name):
    with open(os.path.join(RES, name + ".json"), encoding="utf-8") as f:
        return json.load(f)["map5095"]

def m50(name):
    with open(os.path.join(RES, name + ".json"), encoding="utf-8") as f:
        return json.load(f)["map50"]

def fam(base, suffix, metric=m5095):
    return [metric(base), metric(f"{base}{suffix}1"), metric(f"{base}{suffix}2")]

def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"\n{name}"); print("  " + " | ".join(header))
    for r in rows: print("  " + " | ".join(str(x) for x in r))

# ---- Table 7: seed robustness (KITTI, independent end-to-end replicates)
T7 = [("YOLO11s (80-ep base)", "yolo11s"), ("YOLO11s (std-ft ctrl)", "yolo11s_stdft"),
      ("YOLO11s (geo-ft)", "yolo11s_geo"), ("YOLOv8s (80-ep base)", "yolov8s"),
      ("YOLOv8s (std-ft ctrl)", "yolov8s_stdft"), ("YOLOv8s (geo-ft)", "yolov8s_geo"),
      ("PGP-YOLO-A1 (modules only)", "pgp_A1_pgsa"), ("PGP-YOLO-A1 (geo-ft)", "pgp_a1_geo")]
rows = []
for label, base in T7:
    v = fam(base, "_scr")
    rows.append([label, f"{np.mean(v):.4f}", f"{np.std(v, ddof=1):.4f}"] + [f"{x:.4f}" for x in v])
write("paper_table7_seed.csv", ["Configuration", "Mean", "SD", "Run 1", "Run 2", "Run 3"], rows)

# ---- Table 8: paired tests across seeds, Holm-corrected over all seven comparisons
T8 = [("KITTI YOLO11s: Geo-FT vs Std-FT", ("yolo11s_geo", "_scr"), ("yolo11s_stdft", "_scr")),
      ("KITTI YOLOv8s: Geo-FT vs Std-FT", ("yolov8s_geo", "_scr"), ("yolov8s_stdft", "_scr")),
      ("KITTI YOLO11s: Geo-FT vs base", ("yolo11s_geo", "_scr"), ("yolo11s", "_scr")),
      ("KITTI PGSA vs YOLO11s (80 ep)", ("pgp_A1_pgsa", "_scr"), ("yolo11s", "_scr")),
      ("KITTI PGSA+Geo-FT vs YOLO11s+Geo-FT", ("pgp_a1_geo", "_scr"), ("yolo11s_geo", "_scr")),
      ("BDD100K YOLO11s: Geo-FT vs Std-FT", ("bdd_yolo11s_geo", "_s"), ("bdd_yolo11s_stdft", "_s")),
      ("BDD100K YOLOv8s: Geo-FT vs Std-FT", ("bdd_yolov8s_geo", "_s"), ("bdd_yolov8s_stdft", "_s"))]
res = []
for label, a, b in T8:
    A, B = np.array(fam(*a)), np.array(fam(*b)); d = A - B; n = len(d)
    mean, sd = d.mean(), d.std(ddof=1); half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    res.append(dict(label=label, mean=mean, lo=mean - half, hi=mean + half,
                    p=stats.ttest_rel(A, B).pvalue, wp=stats.wilcoxon(A, B).pvalue, dz=mean / sd))
p = np.array([r["p"] for r in res]); order = np.argsort(p); holm = np.empty(len(p)); run = 0.0
for rank, i in enumerate(order):                       # Holm step-down
    run = max(run, min(1.0, (len(p) - rank) * p[i])); holm[i] = run
rows = []
for r, ph in zip(res, holm):
    verdict = "est." if (ph < 0.05 and abs(r["mean"]) >= 0.001) else ("nom." if r["p"] < 0.05 else "n.d.")
    rows.append([r["label"], f"{r['mean']:+.4f}", f"[{r['lo']:+.4f}, {r['hi']:+.4f}]",
                 f"{r['p']:.3f}", f"{ph:.3f}", f"{r['dz']:.2f}", f"{r['wp']:.2f}", verdict])
write("paper_table8_statistics.csv",
      ["Comparison", "Mean diff", "95% CI", "p (paired t)", "Holm p", "d_z", "Wilcoxon p", "Verdict"], rows)

# ---- Table 9: BDD100K external validation (three fine-tuning seeds off one base)
rows = []
for arch in ("yolo11s", "yolov8s"):
    base = f"bdd_{arch}"; nice = "YOLO11s" if arch == "yolo11s" else "YOLOv8s"
    rows.append([f"{nice} (80-ep base)", f"{m50(base):.4f}", "-", f"{m5095(base):.4f}", "-"])
    for tag, lab in (("stdft", "std-ft ctrl"), ("geo", "geo-ft")):
        v50, v95 = fam(f"{base}_{tag}", "_s", m50), fam(f"{base}_{tag}", "_s")
        rows.append([f"{nice} ({lab})", f"{np.mean(v50):.4f}", f"{np.std(v50, ddof=1):.4f}",
                     f"{np.mean(v95):.4f}", f"{np.std(v95, ddof=1):.4f}"])
write("paper_table9_bdd100k.csv", ["Configuration", "mAP@0.5", "SD", "mAP@0.5:0.95", "SD"], rows)
