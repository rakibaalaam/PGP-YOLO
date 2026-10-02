# -*- coding: utf-8 -*-
"""Export the exact BDD100K subset used in the paper (6,000 train / 1,500 val images)
as two text files of image IDs. Run ONCE in Google Colab, in the same Drive account
that ran the experiment notebook; it needs no GPU and takes about a minute.

Why this is needed: the subset was drawn once from the Kaggle mirror and cached on
Drive (data/bdd_yolo.zip). The sampler shuffled the mirror's files in filesystem
order, so re-running it elsewhere is not guaranteed to reproduce the same subset;
the cached archive is the authoritative record, and this script lists it.

Output: <Drive>/PGP-YOLO/splits/bdd100k_train.txt and bdd100k_val.txt
Copy both files into splits/ in this repository."""
import shutil
import zipfile
from pathlib import Path

from google.colab import drive          # noqa: E402  (Colab only)

drive.mount("/content/drive")
DRIVE_ROOT = Path("/content/drive/MyDrive/PGP-YOLO")
CACHE = DRIVE_ROOT / "data" / "bdd_yolo.zip"
assert CACHE.exists(), f"cached BDD100K subset not found at {CACHE}"

out = DRIVE_ROOT / "splits"
out.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(CACHE) as z:
    names = [n[2:] if n.startswith("./") else n for n in z.namelist()]
for split, expected in (("train", 6000), ("val", 1500)):
    stems = sorted(Path(n).stem for n in names
                   if n.startswith(f"images/{split}/") and n.lower().endswith(".jpg"))
    (out / f"bdd100k_{split}.txt").write_text("\n".join(stems) + "\n")
    print(f"{split}: {len(stems)} images (expected {expected}) -> {out / f'bdd100k_{split}.txt'}")
