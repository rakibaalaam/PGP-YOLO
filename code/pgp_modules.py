# ============================================================================
# pgp_modules.py - PGP-YOLO custom modules + model builder.
# This file must exist on disk in every session: Ultralytics checkpoints pickle
# references to these classes, so resuming after a Colab reconnect re-imports it.
# ============================================================================
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml as pyyaml


def _row_feats(H, device, dtype):
    """Fourier-style embedding of the normalized image row r in [0,1] -> (H,4)."""
    r = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)
    return torch.stack([r, r * r, torch.sin(math.pi * r), torch.cos(math.pi * r)], dim=1)


class PGSA(nn.Module):
    """Perspective-Guided Scale Attention (contribution C1).

    Content-INDEPENDENT gating: a tiny MLP maps (row-embedding, pyramid-level
    one-hot) -> per-row gains for `groups` channel groups. Zero-init => identity
    at the start of training. The gate depends only on WHERE (image row), not on
    WHAT (features) - it is a learned camera-geometry prior.
    """

    def __init__(self, groups=8, level=0, hidden=32):
        super().__init__()
        self.k, self.level = int(groups), int(level)
        self.mlp = nn.Sequential(nn.Linear(7, hidden), nn.SiLU(),
                                 nn.Linear(hidden, self.k))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        B, C, H, W = x.shape
        # The gate is content-independent -> at inference it is a CONSTANT per
        # feature-map size. Cache it (getattr guard keeps checkpoints pickled
        # with older versions of this class loadable). Identical math, removes
        # the per-forward MLP overhead from the latency benchmark.
        if not self.training:
            key = (H, C, str(x.dtype), str(x.device))
            c = getattr(self, "_gcache", None)
            if c is not None and c[0] == key:
                return x * c[1]
        f = _row_feats(H, x.device, x.dtype)                    # (H,4)
        lv = f.new_zeros(H, 3)
        lv[:, self.level] = 1.0
        g = torch.tanh(self.mlp(torch.cat([f, lv], dim=1)))     # (H,k) in (-1,1)
        k = self.k if C % self.k == 0 else 1
        if k == 1:
            g = g.mean(dim=1, keepdim=True)
        gate = g.repeat_interleave(C // k, dim=1)               # (H,C)
        gate = 1.0 + gate.t().unsqueeze(0).unsqueeze(-1)        # (1,C,H,1)
        if not self.training:
            self._gcache = (key, gate.detach())
        return x * gate


class PGCSF(nn.Module):
    """Perspective-Guided Cross-Scale Fusion (contribution C2).

    Placed right after a neck `Concat`. Splits the fused tensor back into its
    source pathways (deep/semantic vs shallow/detail) and re-weights them per
    image row with a softmax gate (sum preserved). Zero-init => equal weights =>
    identity. `base_splits` are the unscaled channel counts of the concatenated
    sources; actual splits are recovered proportionally at runtime, so the same
    module definition works for every model scale.
    """

    def __init__(self, base_splits, hidden=16):
        super().__init__()
        self.base = [int(b) for b in base_splits]
        self.n = len(self.base)
        self.mlp = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(),
                                 nn.Linear(hidden, self.n))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        B, C, H, W = x.shape
        tot = sum(self.base)
        splits = [C * b // tot for b in self.base]
        splits[-1] = C - sum(splits[:-1])
        w = None
        if not self.training:                      # same eval-time cache as PGSA
            key = (H, str(x.dtype), str(x.device))
            c = getattr(self, "_wcache", None)
            if c is not None and c[0] == key:
                w = c[1]
        if w is None:
            w = torch.softmax(self.mlp(_row_feats(H, x.device, x.dtype)), dim=1) * self.n
            if not self.training:
                self._wcache = (key, w.detach())
        parts = torch.split(x, splits, dim=1)
        return torch.cat([p * w[:, i].view(1, 1, H, 1) for i, p in enumerate(parts)],
                         dim=1)


def model_complexity(m, imgsz=640):
    """Return (num_params, gflops) robustly across Ultralytics versions.

    `YOLO.info()` changed its return contract between releases (some return None),
    so we count parameters directly and fetch FLOPs via the stable util, guarded.
    `m` may be a YOLO wrapper or a raw nn.Module.
    """
    net = getattr(m, "model", m)
    params = sum(p.numel() for p in net.parameters())
    gflops = float("nan")
    try:
        from ultralytics.utils.torch_utils import get_flops
        gflops = float(get_flops(net, imgsz))
    except Exception:
        pass
    return params, gflops


def register_pgp_modules():
    """Make PGSA/PGCSF resolvable by Ultralytics' YAML parser and safe to unpickle."""
    import ultralytics.nn.tasks as _tasks
    import ultralytics.nn.modules as _mods
    for cls in (PGSA, PGCSF):
        setattr(_tasks, cls.__name__, cls)   # parse_model() resolves via tasks globals
        setattr(_mods, cls.__name__, cls)
    try:  # torch>=2.6 pickling allow-list (harmless no-op on older torch)
        torch.serialization.add_safe_globals([PGSA, PGCSF])
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Graph surgery: stock yolo11.yaml -> pgp-yolo yaml (+ index map for weights)
# ----------------------------------------------------------------------------
_CH_MODULES = ("Conv", "C3k2", "SPPF", "C2PSA", "C2f", "SPPELAN", "ELAN1")


def build_pgp_yaml(scale="s", nc=6, include_pgcsf=True,
                   groups=8, sa_hidden=32, csf_hidden=16):
    """Return (yaml_dict, old->new absolute-layer-index map)."""
    import ultralytics
    stock = Path(ultralytics.__file__).parent / "cfg/models/11/yolo11.yaml"
    assert stock.exists(), f"stock yolo11.yaml not found at {stock}"
    d = pyyaml.safe_load(stock.read_text())
    backbone, head = d["backbone"], d["head"]
    nb = len(backbone)

    def src_idx(f, i):                       # resolve a 'from' ref to absolute index
        return i - 1 if f == -1 else (f if f >= 0 else i + f)

    # base (unscaled) output channels of each ORIGINAL row - needed by PGCSF splits
    base_ch = {}
    for i, (f, n, m, args) in enumerate(backbone + head):
        if m in _CH_MODULES:
            base_ch[i] = args[0]
        elif m == "nn.Upsample":
            base_ch[i] = base_ch[src_idx(f, i)]
        elif m == "Concat":
            base_ch[i] = sum(base_ch[src_idx(x, i)] for x in f)
        else:                                # Detect etc.
            base_ch[i] = None

    detect_srcs = list(head[-1][0])
    assert head[-1][2] == "Detect", "unexpected stock head layout"

    old2new = {i: i for i in range(nb)}      # backbone indices are unchanged
    det_repl, new_head, cur = {}, [], nb

    def remap(f, old_i):
        if isinstance(f, list):
            return [remap(x, old_i) for x in f]
        return -1 if f == -1 else old2new[src_idx(f, old_i)]

    for j, (f, n, m, args) in enumerate(head):
        old_i = nb + j
        if m == "Detect":
            new_head.append([[det_repl.get(s, old2new[s]) for s in detect_srcs],
                             n, m, list(args)])
            old2new[old_i] = cur; cur += 1
            continue
        new_head.append([remap(f, old_i), n, m, list(args)])
        old2new[old_i] = cur; cur += 1
        if m == "Concat" and include_pgcsf:  # C2: row-conditioned fusion re-weighting
            srcs = [src_idx(x, old_i) for x in f]
            new_head.append([-1, 1, "PGCSF", [[base_ch[s] for s in srcs], csf_hidden]])
            cur += 1
        if old_i in detect_srcs:             # C1: row-conditioned level gating
            level = detect_srcs.index(old_i)
            new_head.append([-1, 1, "PGSA", [groups, level, sa_hidden]])
            det_repl[old_i] = cur; cur += 1

    d["head"], d["nc"], d["scale"] = new_head, nc, scale
    return d, old2new


def transfer_coco_weights(new_det_model, scale, old2new):
    """Copy COCO-pretrained YOLO11 weights through the index map (shape-checked)."""
    from ultralytics import YOLO
    donor_sd = YOLO(f"yolo11{scale}.pt").model.state_dict()
    remapped = {}
    for k, v in donor_sd.items():
        parts = k.split(".")
        if parts[0] == "model" and parts[1].isdigit():
            parts[1] = str(old2new.get(int(parts[1]), int(parts[1])))
        remapped[".".join(parts)] = v
    msd = new_det_model.state_dict()
    ok = {k: v for k, v in remapped.items() if k in msd and v.shape == msd[k].shape}
    new_det_model.load_state_dict(ok, strict=False)
    return len(ok), len(msd)


def save_init_ckpt(det_model, path):
    """Save an Ultralytics-schema checkpoint so YOLO(path).train() rebuilds
    the custom architecture (yaml dict travels inside the pickled model)."""
    import ultralytics
    m = deepcopy(det_model).half()
    for p in m.parameters():
        p.requires_grad = False
    torch.save({"epoch": -1, "best_fitness": None, "model": m, "ema": None,
                "updates": None, "optimizer": None, "train_args": {},
                "date": datetime.now().isoformat(),
                "version": ultralytics.__version__}, path)