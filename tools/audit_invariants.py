#!/usr/bin/env python
# Per-fold audit of the model's stated invariants. Run from the repo root.
"""Audit the dissertation's stated model invariants (O2, O3, O4) per fold.

Loads ablation checkpoints in the format written by
``src/run_ablation.py::train_ablation_fold`` (a dict with ``model_state_dict``,
``config_id``, ``fold``, ``seed``, ...), or freshly initialised models, runs them in
eval mode (float32, as trained) over each fold's *validation* window, and writes
one CSV row per (checkpoint, fold).

It never modifies the repository: it only imports from it.

Examples
--------
Real artefacts (OneDrive bundle)::

    cd /path/to/geometric-deep-learning-quantum-finance
    uv run python /path/to/check_invariants.py \
        --checkpoints '/path/to/bundle/checkpoints/ablation/A*/*.pt' \
        --parquet data/processed/fact_table.parquet \
        --out invariants_real.csv

Synthetic demo (what was run for the report)::

    uv run python check_invariants.py --checkpoints 'runs/demo/output/checkpoints/ablation/*/*.pt' \
        --synthetic factor --wf-train 200 --wf-val 60 --wf-step 60 --out demo.csv
    uv run python check_invariants.py --fresh A8 --folds 0,1 --synthetic factor ...

Checks (spec tolerance in brackets)
-----------------------------------
O2 hypersphere   : frac of val embeddings with | ||h||_2 - 1 | <= 1e-4        [>= 99%]
O2 uniformity    : repo uniformity loss on val embeddings                       [< -2.0]
O3 density matrix: symmetric, |Tr(rho)-1| <= 1e-8, lambda_min >= -1e-8 for rho and
                   rho' = U rho U^T                                             [1e-8]
O3 Cayley        : ||U U^T - I||_max (no tolerance stated; 1e-8 used for consistency)
O3 gate          : 1 < 1 + g < 2 strictly (observed and worst case over the simplex)
O4 attention     : max GATv2 attention weight per destination node <= 0.8       [0.8]
Structural       : p == diag(U rho U^T) == ||row_i(U L)||^2 / ||L||_F^2 ; predictions
                   independent of off-diagonal rho' ; gate effect on within-date ranks
Determinism      : two eval passes on the same date are bitwise identical
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import logging
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

_DEFAULT_REPO = Path(__file__).resolve().parent.parent
REPO = Path(os.environ.get("GDL_REPO", Path.cwd() if (Path.cwd() / "src" / "models.py").exists() else _DEFAULT_REPO))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from scipy import stats as sstats  # noqa: E402

from src.data_loader import FinancialGraphDataset, WalkForwardSplitter  # noqa: E402
from src.layers.graph_utils import build_batched_graphs_gpu  # noqa: E402
from src.layers.quantum import validate_density_matrix  # noqa: E402
from src.losses import uniformity_loss  # noqa: E402
from src.run_ablation import build_ablation_model  # noqa: E402

logger = logging.getLogger("check_invariants")

SPEC = {
    "sphere_tol": 1e-4,
    "sphere_frac": 0.99,
    "uniformity_max": -2.0,
    "dm_tol": 1e-8,
    "attn_max": 0.8,
    "min_projection_norm": 1e-6,  # src/layers/geometric.py _MIN_PROJECTION_NORM
    "eps": 1e-8,
}


# --------------------------------------------------------------------------- data


def load_config(path: Path, args) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg = copy.deepcopy(cfg)
    wf = cfg["training"]["walk_forward"]
    if args.wf_train:
        wf["train_window"] = args.wf_train
    if args.wf_val:
        wf["validation_window"] = args.wf_val
    if args.wf_step:
        wf["step_size"] = args.wf_step
    return cfg


def load_dataframe(args) -> pd.DataFrame:
    if args.parquet:
        logger.info("Reading parquet %s", args.parquet)
        return pd.read_parquet(args.parquet)
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from invariants_synthetic import make_synthetic_df  # synthetic-data helper (tools/)

    kind = args.synthetic or "factor"
    return make_synthetic_df(
        n_tickers=args.syn_tickers,
        n_dates=args.syn_dates,
        seed=args.syn_seed,
        factor=kind in ("factor", "heavy"),
        heavy=kind == "heavy",
    )


def fold_view(full_ds: FinancialGraphDataset, dates) -> FinancialGraphDataset:
    """Shallow copy of the full dataset restricted to `dates` (identical to
    constructing FinancialGraphDataset(dates=...) from the same frame, without
    re-reading/pivoting the 44M-row parquet per fold)."""
    view = copy.copy(full_ds)
    view.valid_dates = sorted(set(dates) & set(full_ds.valid_dates))
    return view


# --------------------------------------------------------------------------- hooks


class Recorder:
    """Forward hooks capturing intermediate tensors of any ablation model."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.handles = []
        self.cur: dict = {}
        self.gate_override = None  # callable(g) -> g'  (for the gate-effect test)
        geo = getattr(model, "geometric", None)
        if geo is not None:
            self.handles.append(geo.projection.register_forward_hook(self._z))
            self.handles.append(geo.register_forward_hook(self._h))
        qm = getattr(model, "quantum", None)
        if qm is not None:
            self.handles.append(qm.cholesky_map.register_forward_hook(self._alpha))
            self.handles.append(qm.register_forward_hook(self._q))
        gate = getattr(model, "regime_gate", None)
        if gate is not None:
            self.handles.append(gate.register_forward_hook(self._g))
        graph = getattr(model, "graph", None)
        if graph is not None and hasattr(graph, "gat"):
            self.handles.append(
                graph.gat.register_forward_pre_hook(self._gat_in, with_kwargs=True)
            )

    def _z(self, m, inp, out):
        self.cur["z"] = out.detach()

    def _h(self, m, inp, out):
        self.cur["h"] = out.detach()

    def _alpha(self, m, inp, out):
        self.cur["alpha"] = out.detach()

    def _q(self, m, inp, out):
        self.cur["p"] = out[0].detach()
        self.cur["rho_ev"] = out[1].detach()

    def _g(self, m, inp, out):
        self.cur["p_in_gate"] = inp[0].detach()
        self.cur["g"] = out.detach()
        if self.gate_override is not None:
            return self.gate_override(out)
        return None

    def _gat_in(self, m, args, kwargs):
        self.cur["gat_x"] = args[0].detach()
        self.cur["gat_ei"] = args[1].detach()
        self.cur["gat_ea"] = kwargs.get("edge_attr").detach()
        return None

    def reset(self):
        self.cur = {}

    def close(self):
        for h in self.handles:
            h.remove()


# --------------------------------------------------------------------------- helpers


def date_forward(model, data, cfg, device):
    """One validation date, identical to evaluation.evaluate_model_on_dataset."""
    gc = cfg["model"]["graph_construction"]
    data = data.to(device)
    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    ei, ew = build_batched_graphs_gpu(
        data.past_returns, batch, threshold=float(gc["correlation_threshold"]),
        max_neighbours=gc.get("max_neighbours"),
    )
    out = model(x=data.x, edge_index=ei, edge_weight=ew, context=data.context,
                batch=batch, x_seq=getattr(data, "x_seq", None))
    return out, data, ei, ew


def cayley64(qm) -> torch.Tensor:
    H = qm.H_raw.detach().double().cpu()
    A = (H - H.T) / 2
    I = torch.eye(H.shape[0], dtype=torch.float64)
    return torch.linalg.solve(I + A, I - A)


def tril_L(alpha: torch.Tensor, K: int) -> torch.Tensor:
    r, c = torch.tril_indices(K, K)
    L = alpha.new_zeros(alpha.shape[0], K, K)
    L[:, r, c] = alpha
    return L


def sigmoid_fp32_saturation():
    """Smallest |logit| at which 1+sigmoid(logit) in float32 hits exactly 1.0 / 2.0."""
    z = torch.linspace(0, 40, 400001, dtype=torch.float32)
    up = 1.0 + torch.sigmoid(z)
    lo = 1.0 + torch.sigmoid(-z)
    hi_thr = float(z[(up == 2.0)].min())
    lo_thr = float(z[(lo == 1.0)].min())
    return lo_thr, hi_thr


LO_THR, HI_THR = sigmoid_fp32_saturation()


def nanf(x):
    return float("nan") if x is None else float(x)


# --------------------------------------------------------------------------- audit


@torch.no_grad()
def audit_fold(model, cfg_id, ds, cfg, device, max_dates=None, pool_dates=64, seed=0):
    model.eval()
    rec = Recorder(model)
    K = model.quantum.num_regimes if hasattr(model, "quantum") else None
    has_geo, has_q = hasattr(model, "geometric"), hasattr(model, "quantum")
    has_gate = hasattr(model, "regime_gate")
    has_gat = hasattr(model, "graph") and hasattr(model.graph, "gat")

    acc = {k: [] for k in [
        "alpha", "p", "rho_ev", "g", "unif_date", "unif_wi_date",
        "rank_corr_gate_vs_const", "ic_actual", "ic_const_gate",
        "rho_ev_offdiag_grad", "pred_perturb_diff", "det_diff",
    ]}
    # streaming accumulators (real folds have ~20k nodes x 252 dates; nothing per-node is kept)
    sph = {"n": 0, "n_ok": 0, "dev64": 0.0, "dev32": 0.0, "zmin": float("inf"), "zmed": [],
           "n_repl": 0, "specdev": 0.0, "nonfinite": 0}
    att = {"nh": 0, "gt": 0, "nh1": 0, "gt1": 0, "nh5": 0, "gt5": 0, "amax": 0.0, "amax5": 0.0,
           "ent5": 0.0, "neff5": 0.0, "unif5": 0.0, "self": 0.0, "iso": 0, "nodes": 0, "degsum": 0}
    gsub = torch.Generator().manual_seed(seed)
    pooled = []   # per-date random subsample (<= pool_per_date rows) for the pooled uniformity
    pool_per_date = max(1, 4096 // pool_dates) * 4
    n_dates = len(ds) if max_dates is None else min(len(ds), max_dates)
    g_mean = None

    def sub(t, k):
        if t.shape[0] <= k:
            return t
        return t[torch.randperm(t.shape[0], generator=gsub)[:k]]

    # pass 1: stream over validation dates
    for idx in range(n_dates):
        data = ds[idx]
        if data.target_mask.sum() == 0:
            continue
        rec.reset()
        out, data_d, ei, ew = date_forward(model, data, cfg, device)
        cur = dict(rec.cur)
        preds = out["predictions"].squeeze(-1)
        m = data_d.target_mask
        if m.sum() >= 3:
            acc["ic_actual"].append(sstats.spearmanr(preds[m].cpu(), data_d.y[m].cpu()).correlation)
        if has_geo:
            h = cur["h"].cpu()
            z = cur["z"].cpu()
            n64 = torch.linalg.vector_norm(h.double(), dim=-1)
            n32 = torch.linalg.vector_norm(h, dim=-1)
            dev_ = (n64 - 1).abs()
            zn = torch.linalg.vector_norm(z.double(), dim=-1)
            sph["n"] += h.shape[0]
            sph["n_ok"] += int((dev_ <= SPEC["sphere_tol"]).sum())
            sph["dev64"] = max(sph["dev64"], float(torch.nan_to_num(dev_, nan=float("inf")).max()))
            sph["dev32"] = max(sph["dev32"], float(torch.nan_to_num((n32 - 1).abs(), nan=float("inf")).max()))
            sph["zmin"] = min(sph["zmin"], float(zn.min()))
            sph["zmed"].append(float(zn.median()))
            sph["n_repl"] += int((zn < SPEC["min_projection_norm"]).sum())
            sph["specdev"] = max(sph["specdev"], float((SPEC["eps"] / (zn + SPEC["eps"])).max()))
            sph["nonfinite"] += int((~torch.isfinite(h)).any(-1).sum())
            hs = sub(h, 4096)  # same cap the training loss uses
            acc["unif_date"].append(float(uniformity_loss(hs)))
            # canonical Wang & Isola (2020) eq. (5.10): log E exp(-2 ||x-y||^2)
            sq = torch.cdist(hs.double(), hs.double()).pow(2)
            nn_ = hs.shape[0]
            acc["unif_wi_date"].append(float(torch.log(torch.exp(-2 * sq[~torch.eye(nn_, dtype=torch.bool)]).mean())))
            pooled.append(sub(h, pool_per_date))
        if has_q:
            acc["alpha"].append(cur["alpha"].cpu())
            acc["p"].append(cur["p"].cpu())
            acc["rho_ev"].append(cur["rho_ev"].cpu())
        if has_gate:
            acc["g"].append(cur["g"].cpu())
        if has_gat:
            x, e_in, ea = cur["gat_x"], cur["gat_ei"], cur["gat_ea"]
            _, (ei_sl, alpha_att) = model.graph.gat(x, e_in, edge_attr=ea, return_attention_weights=True)
            ei_sl, alpha_att = ei_sl.cpu(), alpha_att.cpu()
            dst = ei_sl[1]
            N = x.shape[0]
            H = alpha_att.shape[1]
            not_self = ei_sl[0] != ei_sl[1]
            deg = torch.bincount(dst[not_self], minlength=N)  # in-degree excl. self-loop
            amax = torch.zeros(N, H).scatter_reduce(0, dst.unsqueeze(1).expand(-1, H), alpha_att, reduce="amax")
            ent = torch.zeros(N, H).index_add_(0, dst, -alpha_att * torch.log(alpha_att.clamp_min(1e-30)))
            selfw = torch.zeros(N, H).index_add_(0, dst[~not_self], alpha_att[~not_self])
            d1, d5 = deg >= 1, deg >= 5
            att["nh"] += amax.numel(); att["gt"] += int((amax > SPEC["attn_max"]).sum())
            att["nh1"] += amax[d1].numel(); att["gt1"] += int((amax[d1] > SPEC["attn_max"]).sum())
            att["nh5"] += amax[d5].numel(); att["gt5"] += int((amax[d5] > SPEC["attn_max"]).sum())
            att["amax"] = max(att["amax"], float(amax.max()))
            if d5.any():
                att["amax5"] = max(att["amax5"], float(amax[d5].max()))
                att["ent5"] += float((ent[d5] / torch.log(deg[d5].double() + 1).unsqueeze(1)).sum())
                att["neff5"] += float(torch.exp(ent[d5]).sum())
                att["unif5"] += float(((deg[d5].double() + 1).unsqueeze(1).expand(-1, H)).sum())
            att["self"] += float(selfw.sum())
            att["iso"] += int((deg == 0).sum()); att["nodes"] += N; att["degsum"] += int(deg.sum())
        # determinism: second identical eval pass
        out2, _, _, _ = date_forward(model, data, cfg, device)
        acc["det_diff"].append(float((out2["predictions"] - out["predictions"]).abs().max()))

    res: dict = {"config_id": cfg_id, "n_dates": n_dates}

    # ---------------- O2 hypersphere
    if has_geo:
        frac = sph["n_ok"] / max(sph["n"], 1)
        res.update({
            "sphere_n": sph["n"],
            "sphere_max_abs_dev_fp64norm": sph["dev64"],
            "sphere_max_abs_dev_fp32norm": sph["dev32"],
            "sphere_frac_within_1e-4": frac,
            "sphere_pass": bool(frac >= SPEC["sphere_frac"]),
            "sphere_prenorm_min": sph["zmin"],
            "sphere_prenorm_median_of_daily_medians": float(np.median(sph["zmed"])),
            "sphere_n_replaced_random": sph["n_repl"],
            "sphere_specformula_max_dev": sph["specdev"],
            "sphere_nonfinite": sph["nonfinite"],
        })
        # uniformity pooled like training: `pool_dates` dates per batch, <= 4096 points
        pooled_vals = []
        for i in range(0, len(pooled), pool_dates):
            P = sub(torch.cat(pooled[i:i + pool_dates]), 4096)
            pooled_vals.append(float(uniformity_loss(P)))
        res.update({
            "unif_repo_per_date_mean": float(np.mean(acc["unif_date"])),
            "unif_repo_per_date_max": float(np.max(acc["unif_date"])),
            "unif_repo_pooled_mean": float(np.mean(pooled_vals)),
            "unif_wang_isola_per_date_mean": float(np.mean(acc["unif_wi_date"])),
            "unif_pass_lt_-2": bool(np.mean(pooled_vals) < SPEC["uniformity_max"]),
        })

    # ---------------- O3 density matrices + Cayley + Born readout
    if has_q:
        qm = model.quantum
        alpha = torch.cat(acc["alpha"])            # (B,10) fp32, as computed in the model
        p = torch.cat(acc["p"])                    # (B,K)  fp32
        rho_ev = torch.cat(acc["rho_ev"])          # (B,K,K) fp32
        qdev = qm.H_raw.device
        L = qm._build_cholesky_factor(alpha.to(qdev))  # identical fp32 ops to the forward
        rho = qm._build_density_matrix(L).cpu()
        U32 = qm._cayley_unitary().detach().cpu()
        U64 = cayley64(qm)
        I = torch.eye(K, dtype=torch.float64)

        def dm_stats(R, tag):
            R64 = R.double()
            tr32 = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)  # fp32 trace (as torch.trace would)
            tr64 = torch.diagonal(R64, dim1=-2, dim2=-1).sum(-1)  # exact-ish trace of stored fp32 matrix
            asym = (R - R.transpose(-2, -1)).abs().amax((-2, -1))
            ev32 = torch.linalg.eigvalsh(R)
            ev64 = torch.linalg.eigvalsh(R64)
            ok = (asym <= SPEC["dm_tol"]) & ((tr64 - 1).abs() <= SPEC["dm_tol"]) & (ev64.min(-1).values >= -SPEC["dm_tol"])
            return {
                f"{tag}_max_abs_trace_err_fp32": float((tr32.double() - 1).abs().max()),
                f"{tag}_max_abs_trace_err_fp64sum": float((tr64 - 1).abs().max()),
                f"{tag}_frac_trace_within_1e-8": float(((tr64 - 1).abs() <= SPEC["dm_tol"]).double().mean()),
                f"{tag}_max_asym": float(asym.max()),
                f"{tag}_min_eig_fp32": float(ev32.min()),
                f"{tag}_min_eig_fp64": float(ev64.min()),
                f"{tag}_frac_eig_ge_-1e-8": float((ev64.min(-1).values >= -SPEC["dm_tol"]).double().mean()),
                f"{tag}_frac_all3_within_1e-8": float(ok.double().mean()),
                f"{tag}_frac_all3_within_1e-6": float(((asym <= 1e-6) & ((tr64 - 1).abs() <= 1e-6) & (ev64.min(-1).values >= -1e-6)).double().mean()),
            }

        res.update(dm_stats(rho, "rho"))
        res.update(dm_stats(rho_ev, "rhoev"))
        # repo's own validator: only inspects the LAST matrix in the batch, tol 1e-5
        v = validate_density_matrix(rho_ev)
        res["repo_validate_density_matrix_all_true"] = bool(all(v.values()))
        # exact-arithmetic reference (float64 from the same alpha)
        L64 = tril_L(alpha.double(), K)
        T = (L64 ** 2).sum((-2, -1))                       # Tr(L L^T) = ||alpha||^2
        rho64 = L64 @ L64.transpose(-2, -1) / T.clamp_min(SPEC["eps"])[:, None, None]
        res["rho_fp64_ref_max_trace_err"] = float((torch.diagonal(rho64, dim1=-2, dim2=-1).sum(-1) - 1).abs().max())
        res["trace_LLt_min"] = float(T.min())
        res["trace_LLt_frac_lt_1"] = float((T < 1).double().mean())
        res["specformula_trace_err_max"] = float((SPEC["eps"] / (T + SPEC["eps"])).max())  # eq (5.3) Tr/(Tr+eps)
        # Cayley
        res["U_orth_err_fp32"] = float((U32.double() @ U32.double().T - I).abs().max())
        res["U_orth_err_fp64"] = float((U64 @ U64.T - I).abs().max())
        res["U_dist_from_I_max"] = float((U64 - I).abs().max())
        res["U_input_dependent"] = False  # _cayley_unitary() takes no input: one global U per model
        # Born readout: p_i = ||row_i(U L)||^2 / ||L||_F^2  (float64, from the model's alpha)
        M = U64 @ L64
        p_formula = (M ** 2).sum(-1) / T[:, None]
        res["born_formula_max_abs_diff_vs_model_p"] = float((p.double() - p_formula).abs().max())
        # POVM form: p_i = a^T Q_i a, a = alpha/||alpha||, sum_i Q_i = I_10
        r_idx, c_idx = torch.tril_indices(K, K)
        n_a = r_idx.numel()
        Q = torch.zeros(K, n_a, n_a, dtype=torch.float64)
        for i in range(K):
            B = torch.zeros(K, n_a, dtype=torch.float64)  # row_i(U L)_j = sum_a B[j,a] alpha_a
            for a_i in range(n_a):
                B[c_idx[a_i], a_i] = U64[i, r_idx[a_i]]
            Q[i] = B.T @ B
        a = alpha.double() / alpha.double().norm(dim=-1, keepdim=True)
        p_povm = torch.einsum("ba,kac,bc->bk", a, Q, a)
        res["povm_sumQ_minus_I"] = float((Q.sum(0) - torch.eye(n_a, dtype=torch.float64)).abs().max())
        res["born_povm_max_abs_diff_vs_model_p"] = float((p.double() - p_povm).abs().max())
        res["p_min"] = float(p.min())
        res["p_sum_err_max"] = float((p.double().sum(-1) - 1).abs().max())
        res["p_std_across_dates_mean"] = float(p.std(0).mean()) if p.shape[0] > 1 else float("nan")
        # coherence magnitude (off-diagonal mass) of rho'
        off = rho_ev - torch.diag_embed(torch.diagonal(rho_ev, dim1=-2, dim2=-1))
        res["rhoev_offdiag_abs_mean"] = float(off.abs().mean())
        # the repo's "entropy regulariser" = H(p) - S(rho') = relative entropy of coherence >= 0
        ev = torch.linalg.eigvalsh(rho_ev.double()).clamp_min(1e-12)
        S_vn = -(ev * ev.log()).sum(-1)
        H_p = -(p.double() * p.double().clamp_min(1e-12).log()).sum(-1)
        res["H_p_mean"] = float(H_p.mean())
        res["S_rho_mean"] = float(S_vn.mean())
        res["entropy_reg_eq_Crel_mean"] = float((H_p - S_vn).mean())

    # ---------------- O3 gate multiplier
    if has_gate:
        g = torch.cat(acc["g"])
        mult = 1.0 + g  # fp32 exactly as in the model
        res["gate_obs_min_mult_minus_1"] = float(mult.min().double() - 1)
        res["gate_obs_2_minus_max_mult"] = float(2 - mult.max().double())
        res["gate_obs_n_eq_1"] = int((mult == 1.0).sum())
        res["gate_obs_n_eq_2"] = int((mult == 2.0).sum())
        res["gate_obs_strict_pass"] = bool(((mult > 1.0) & (mult < 2.0)).all())
        lin = model.regime_gate[0]
        W, b = lin.weight.detach().cpu(), lin.bias.detach().cpu()   # (G,K), (G,)
        vert = W + b[:, None]                            # logits at simplex vertices
        res["gate_logit_range_lo"] = float(vert.min())
        res["gate_logit_range_hi"] = float(vert.max())
        vm = 1.0 + torch.sigmoid(vert)
        res["gate_worstcase_strict_pass"] = bool(((vm > 1.0) & (vm < 2.0)).all())
        res["gate_fp32_saturation_logit_lo"] = -LO_THR
        res["gate_fp32_saturation_logit_hi"] = HI_THR
        res["gate_std_across_dates_mean"] = float(g.std(0).mean()) if g.shape[0] > 1 else float("nan")
        g_mean = g.mean(0)

    # ---------------- O4 attention
    if has_gat:
        nan = float("nan")
        res.update({
            "attn_n_node_heads": att["nh"],
            "attn_frac_isolated_nodes": att["iso"] / max(att["nodes"], 1),
            "attn_mean_indegree": att["degsum"] / max(att["nodes"], 1),
            "attn_max_weight_any": att["amax"],
            "attn_frac_nodehead_gt_0.8_all": att["gt"] / max(att["nh"], 1),
            "attn_frac_nodehead_gt_0.8_deg_ge1": att["gt1"] / att["nh1"] if att["nh1"] else nan,
            "attn_frac_nodehead_gt_0.8_deg_ge5": att["gt5"] / att["nh5"] if att["nh5"] else nan,
            "attn_max_weight_deg_ge5": att["amax5"] if att["nh5"] else nan,
            "attn_O4_literal_pass": bool(att["gt"] == 0),
            "attn_norm_entropy_mean_deg_ge5": att["ent5"] / att["nh5"] if att["nh5"] else nan,
            "attn_eff_neigh_mean_deg_ge5": att["neff5"] / att["nh5"] if att["nh5"] else nan,
            "attn_uniform_eff_neigh_mean_deg_ge5": att["unif5"] / att["nh5"] if att["nh5"] else nan,
            "attn_self_share_mean": att["self"] / max(att["nh"], 1),
        })

    # ---------------- structural tests on the first few dates (need grad / overrides)
    if has_q and has_gate:
        n_struct = min(n_dates, 20)
        for idx in range(n_struct):
            data = ds[idx]
            if data.target_mask.sum() < 3:
                continue
            # (i) off-diagonal gradient via a leaf rho'
            qm = model.quantum
            orig_fwd = qm.forward
            holder = {}

            def patched(emb, ctx, _orig=orig_fwd):
                p_, r_ = _orig(emb, ctx)
                leaf = r_.detach().clone().requires_grad_(True)
                holder["leaf"] = leaf
                return torch.diagonal(leaf, dim1=-2, dim2=-1), leaf

            with torch.enable_grad():
                qm.forward = patched
                try:
                    out, _, _, _ = date_forward(model, data, cfg, device)
                    out["predictions"].sum().backward()
                finally:
                    qm.forward = orig_fwd
            gr = holder["leaf"].grad[0].cpu()
            acc["rho_ev_offdiag_grad"].append(float((gr - torch.diag(torch.diagonal(gr))).abs().max()))
            # (ii) perturb off-diagonals of rho' (keep diagonal) -> predictions identical?
            def patched2(emb, ctx, _orig=orig_fwd):
                p_, r_ = _orig(emb, ctx)
                S = torch.randn_like(r_)
                S = (S + S.transpose(-2, -1)) / 2
                S = S - torch.diag_embed(torch.diagonal(S, dim1=-2, dim2=-1))
                r2 = r_ + 0.1 * S
                return torch.diagonal(r2, dim1=-2, dim2=-1), r2
            base, _, _, _ = date_forward(model, data, cfg, device)
            qm.forward = patched2
            try:
                pert, _, _, _ = date_forward(model, data, cfg, device)
            finally:
                qm.forward = orig_fwd
            acc["pred_perturb_diff"].append(float((pert["predictions"] - base["predictions"]).abs().max()))
            # (iii) gate effect on within-date ranking: actual g vs fold-mean g
            m = data.target_mask
            rec.gate_override = lambda gg: g_mean.to(device=gg.device, dtype=gg.dtype).expand_as(gg)
            try:
                const_out, dd, _, _ = date_forward(model, data, cfg, device)
            finally:
                rec.gate_override = None
            a_ = base["predictions"].squeeze(-1)[m].cpu()
            c_ = const_out["predictions"].squeeze(-1)[m].cpu()
            acc["rank_corr_gate_vs_const"].append(sstats.spearmanr(a_, c_).correlation)
            acc["ic_const_gate"].append(sstats.spearmanr(c_, dd.y[m].cpu()).correlation)
        res["pred_grad_wrt_rhoev_offdiag_max"] = float(np.max(acc["rho_ev_offdiag_grad"]))
        res["pred_change_when_offdiag_perturbed_max"] = float(np.max(acc["pred_perturb_diff"]))
        res["rank_corr_actual_vs_meangate_mean"] = float(np.nanmean(acc["rank_corr_gate_vs_const"]))
        res["rank_corr_actual_vs_meangate_min"] = float(np.nanmin(acc["rank_corr_gate_vs_const"]))
        res["ic_meangate_first_dates_mean"] = float(np.nanmean(acc["ic_const_gate"]))
        res["ic_actual_first_dates_mean"] = float(np.nanmean(acc["ic_actual"][: len(acc["ic_const_gate"])]))

    res["ic_actual_mean"] = float(np.nanmean(acc["ic_actual"])) if acc["ic_actual"] else float("nan")
    res["eval_repeat_max_abs_diff"] = float(np.max(acc["det_diff"]))
    res["eval_repeat_bitwise_identical"] = bool(np.max(acc["det_diff"]) == 0.0)
    rec.close()
    return res


# --------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="*", default=[], help="checkpoint paths/globs (.pt)")
    ap.add_argument("--fresh", default=None, help="audit a freshly initialised model of this config id (e.g. A8)")
    ap.add_argument("--fresh-seed", type=int, default=42)
    ap.add_argument("--folds", default=None, help="comma list of folds for --fresh (default: all)")
    ap.add_argument("--config", default=str(REPO / "config.yaml"))
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--synthetic", default=None, choices=["plain", "factor", "heavy"])
    ap.add_argument("--syn-tickers", type=int, default=40)
    ap.add_argument("--syn-dates", type=int, default=420)
    ap.add_argument("--syn-seed", type=int, default=42)
    ap.add_argument("--wf-train", type=int, default=None)
    ap.add_argument("--wf-val", type=int, default=None)
    ap.add_argument("--wf-step", type=int, default=None)
    ap.add_argument("--purge", type=int, default=20)
    ap.add_argument("--max-dates", type=int, default=None, help="cap validation dates per fold")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="invariants.csv")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    torch.set_grad_enabled(False)

    cfg = load_config(Path(args.config), args)
    df = load_dataframe(args)
    full_ds = FinancialGraphDataset(parquet_path="unused", config=cfg, dataframe=df)
    wf = cfg["training"]["walk_forward"]
    splitter = WalkForwardSplitter(full_ds.valid_dates, wf["train_window"], wf["validation_window"],
                                   wf["step_size"], purge_gap=args.purge)
    device = torch.device(args.device)

    jobs = []
    for pat in args.checkpoints:
        for path in sorted(glob.glob(pat)):
            jobs.append(("ckpt", path))
    if args.fresh:
        folds = range(len(splitter)) if args.folds is None else [int(f) for f in args.folds.split(",")]
        for f in folds:
            jobs.append(("fresh", f))

    rows = []
    for kind, item in jobs:
        if kind == "ckpt":
            ck = torch.load(item, map_location="cpu", weights_only=False)
            cid, fold = ck["config_id"], int(ck["fold"])
            model = build_ablation_model(cid, cfg, device)
            model.load_state_dict(ck["model_state_dict"])
            src = item
        else:
            cid, fold = args.fresh, item
            torch.manual_seed(args.fresh_seed)
            model = build_ablation_model(cid, cfg, device)
            src = f"fresh(seed={args.fresh_seed})"
        if cid == "A9":
            logger.info("Skipping A9 (LSTM): none of the audited invariants apply.")
            continue
        if fold >= len(splitter):
            logger.warning("fold %d not available in this dataset (%d folds); skipping %s", fold, len(splitter), src)
            continue
        _, val_dates = splitter.get_fold(fold)
        ds = fold_view(full_ds, val_dates)
        logger.info("%s fold %d: %d validation dates (%s)", cid, fold, len(ds), src)
        res = audit_fold(model, cid, ds, cfg, device, max_dates=args.max_dates)
        res = {"label": args.label, "source": src, "fold": fold, **res}
        rows.append(res)
        logger.info(json.dumps({k: v for k, v in res.items() if k in (
            "config_id", "fold", "sphere_frac_within_1e-4", "unif_repo_pooled_mean",
            "rhoev_max_abs_trace_err_fp32", "rhoev_min_eig_fp64", "gate_obs_strict_pass",
            "attn_frac_nodehead_gt_0.8_all", "born_formula_max_abs_diff_vs_model_p")}))

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info("Wrote %d rows to %s", len(rows), args.out)


if __name__ == "__main__":
    main()
