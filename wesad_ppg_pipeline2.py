#!/usr/bin/env python3
"""
wesad_ppg_pipeline2.py -- EXPLORATORY post-hoc refinement of wesad_ppg_pipeline.py (Pipeline 1).

Pipeline 1 is IMPORTED, never modified. Everything "Old" (HR/SDNN/RMSSD from peak tops,
DR/RI/AR, Morlet log10_Ew / E_ratio, QC thresholds, window grid, models, hyperparameters)
is Pipeline 1's own code, so the Old-vs-New comparison is like for like.

Pre-declared design (fixed here, run ONCE, no tuning):
  1. Beat anchor = steepest systolic rise (max dP/dt) on the 4x upsampled signal, parabolic
     sub-sample refinement. IBIs/HR/SDNN/RMSSD come from these anchors, not peak tops.
  2. APPG: per-beat second-derivative waves a-e -> b/a, c/a, d/a, e/a, AGI=(b-c-d-e)/a,
     replacing direct notch extraction. Same QC rules as P1 (SQI_MIN, MIN_BEATS, MIN_DIFFS,
     MIN_NOTCH_RATE-style gating on landmark detection rate).
  3. Z-scoring, within subject, label-free, identical transform for EVERY feature column:
       --zmode session (default): mean/SD over all 60 s windows (50% overlap) tiled across the
                                  whole recording, any label/condition, labels never used.
       --zmode calib            : mean/SD over windows in the first --calib_s seconds.
     NB with 'calib' the first minutes of WESAD are baseline, so baseline windows are partly
     normalised against themselves; 'session' is the primary specification.
  4. Evaluation windows = windows accepted by BOTH pipelines (same rows for every feature set).
     Old features are additionally evaluated on P1's original windows as a reproduction check.
  5. LOSO-CV with P1's models. One consolidated matrix:
       {HR only, Macro P1, Macro P2, Old, New} x {raw, z-scored} x models.

Usage (put this file next to wesad_ppg_pipeline.py, or pass --p1_dir):
    python wesad_ppg_pipeline2.py --data /path/to/WESAD --out results_p2
    python wesad_ppg_pipeline2.py --data ... --out results_p2 --p1_windows results/windows.csv
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.metrics import f1_score, roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="path to the WESAD folder")
ap.add_argument("--out", default="results_p2")
ap.add_argument("--p1_dir", default=str(Path(__file__).resolve().parent),
                help="folder containing wesad_ppg_pipeline.py")
ap.add_argument("--p1_windows", default=None,
                help="optional: P1's results/windows.csv, used only to verify Old features reproduce")
ap.add_argument("--zmode", choices=["session", "calib"], default="session")
ap.add_argument("--calib_s", type=int, default=300)
ap.add_argument("--recompute", action="store_true")
ARGS = ap.parse_args()

sys.path.insert(0, ARGS.p1_dir)
import wesad_ppg_pipeline as p1  # noqa: E402  (Pipeline 1, unmodified)

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Pipeline-2 constants (fixed before running)
# ----------------------------------------------------------------------------
UP = 4                       # 64 Hz -> 256 Hz for derivative work
FU = p1.FS * UP
SG_WIN, SG_POLY = 15, 3      # Savitzky-Golay on the 256 Hz signal (~59 ms)
APPG_PRE_S, APPG_POST_MAX_S = 0.10, 0.45
BOOT = 1000

APPG = ["appg_ba", "appg_ca", "appg_da", "appg_ea", "appg_agi"]
P1_MACRO = ["HR", "SDNN", "RMSSD"]
P2_MACRO = ["p2_HR", "p2_SDNN", "p2_RMSSD"]
P2_WAV = ["p2_log10_Ew", "p2_E_ratio"]
P1_SHAPE = ["DR", "RI", "AR"]
P1_WAV = ["log10_Ew", "E_ratio"]
FEATURE_SETS = {
    "HR only (P1 HR)": ["HR"],
    "Macro P1 (HR,SDNN,RMSSD)": P1_MACRO,
    "Macro P2 (HR,SDNN,RMSSD)": P2_MACRO,
    "Old = P1 (macro+notch+CWT)": P1_MACRO + P1_SHAPE + P1_WAV,
    "New = P2 (macro+APPG+CWT)": P2_MACRO + APPG + P2_WAV,
}
ALL_FEATS = sorted({f for v in FEATURE_SETS.values() for f in v})


# ----------------------------------------------------------------------------
# Pipeline-2 window features
# ----------------------------------------------------------------------------
def _resample(seg, n=100):
    r = np.interp(np.linspace(0, len(seg) - 1, n), np.arange(len(seg)), seg)
    rg = r.max() - r.min()
    return (r - r.min()) / rg if rg > 0 else None


def _appg_beat(d2, k0, ibi):
    """a..e waves of the second derivative around the anchor k0 (index at 256 Hz)."""
    lo = k0 - int(APPG_PRE_S * FU)
    hi = k0 + int(min(0.5 * ibi, APPG_POST_MAX_S) * FU)
    out = np.full(5, np.nan)
    if lo < 0 or hi >= len(d2):
        return out
    seg, kk = d2[lo:hi], k0 - lo
    pk, _ = signal.find_peaks(seg)
    tr, _ = signal.find_peaks(-seg)
    ac = pk[pk <= kk]
    if len(ac) == 0:
        return out
    a = seg[ac[np.argmax(seg[ac])]]
    if a <= 0:
        return out
    bc = tr[tr > kk]
    if len(bc) == 0:
        return out
    ib = bc[0]; b = seg[ib]
    out[0] = b / a
    cc = pk[pk > ib]
    if len(cc) == 0:
        return out
    ic = cc[0]; c = seg[ic]
    out[1] = c / a
    dc = tr[tr > ic]
    if len(dc) == 0:
        return out
    idd = dc[0]; d = seg[idd]
    out[2] = d / a
    ec = pk[pk > idd]
    if len(ec) == 0:
        return out
    e = seg[ec[0]]
    out[3] = e / a
    out[4] = (b - c - d - e) / a
    return out


def p2_window_features(x):
    """Pipeline-2 features for one filtered 60 s window, or None if QC rejects it."""
    xu = signal.resample_poly(x, UP, 1)
    d1 = signal.savgol_filter(xu, SG_WIN, SG_POLY, deriv=1, delta=1.0 / FU)
    d2 = signal.savgol_filter(xu, SG_WIN, SG_POLY, deriv=2, delta=1.0 / FU)
    k, _ = signal.find_peaks(d1, distance=int(0.35 * FU), prominence=0.5 * np.std(d1))
    k = k[(k > 0) & (k < len(d1) - 1)]
    if len(k) < p1.MIN_BEATS + 2:
        return None
    y0, y1, y2 = d1[k - 1], d1[k], d1[k + 1]
    den = y0 - 2 * y1 + y2
    delta = np.clip(np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0), -1, 1)
    tk = (k + delta) / FU                                   # refined anchor times (s)

    shapes = []
    for j in range(len(k) - 1):
        dur = tk[j + 1] - tk[j]
        shapes.append(_resample(xu[k[j]:k[j + 1] + 1]) if 0.35 <= dur <= 1.5 else None)
    good = [s for s in shapes if s is not None]
    if len(good) < p1.MIN_BEATS:
        return None
    template = np.median(good, axis=0)
    acc = [j for j, s in enumerate(shapes)
           if s is not None and np.corrcoef(s, template)[0, 1] >= p1.SQI_MIN]
    if len(acc) < p1.MIN_BEATS:
        return None
    accs = set(acc)
    ibi = {j: tk[j + 1] - tk[j] for j in acc}
    ibi_pair = {j: tk[j + 1] - tk[j] for j in acc if j + 1 in accs}   # as P1: consecutive accepted beats
    diffs = [ibi_pair[j + 1] - ibi_pair[j] for j in ibi_pair
             if j + 1 in ibi_pair and abs(ibi_pair[j + 1] - ibi_pair[j]) < 0.3]
    if len(diffs) < p1.MIN_DIFFS:
        return None
    ibi_arr = np.array(list(ibi_pair.values()))
    f0 = 1.0 / ibi_arr.mean()

    A = np.array([_appg_beat(d2, k[j], ibi[j]) for j in acc])
    feats = {"p2_HR": 60.0 * f0,
             "p2_SDNN": 1000.0 * float(np.std(ibi_arr, ddof=1)),
             "p2_RMSSD": 1000.0 * float(np.sqrt(np.mean(np.square(diffs))))}
    for i, name in enumerate(APPG):
        v = A[:, i]
        rate = np.isfinite(v).mean()
        feats[name] = float(np.median(v[np.isfinite(v)])) if rate >= p1.MIN_NOTCH_RATE else np.nan
    feats["appg_rate"] = float(np.isfinite(A[:, 0]).mean())          # b/a detection rate
    feats["appg_rate_full"] = float(np.isfinite(A[:, 4]).mean())     # all five waves found
    sc = p1.scalogram_features(x, f0)
    feats["p2_log10_Ew"], feats["p2_E_ratio"] = sc["log10_Ew"], sc["E_ratio"]
    feats["p2_n_beats"] = len(acc)
    return feats


# ----------------------------------------------------------------------------
# Per-subject processing
# ----------------------------------------------------------------------------
def both(x):
    o1 = p1.window_features(x)
    o2 = p2_window_features(x)
    return o1, o2


def make_row(sid, label, t0, o1, o2):
    row = dict(subject=sid, label=label, t0=t0, ok_p1=o1 is not None, ok_p2=o2 is not None)
    f1 = o1[0] if o1 is not None else {}
    row.update({k: f1.get(k, np.nan) for k in P1_MACRO + P1_SHAPE + P1_WAV + ["n_beats", "notch_rate"]})
    f2 = o2 if o2 is not None else {}
    row.update({k: f2.get(k, np.nan) for k in P2_MACRO + APPG + P2_WAV + ["appg_rate", "appg_rate_full", "p2_n_beats"]})
    return row


def process_subject(sid):
    bvp, lab = p1.load_subject(ARGS.data, sid)
    xf = p1.bandpass(bvp)
    lab_rows, ses_rows, totals = [], [], {0: 0, 1: 0}
    for cls, name in ((1, 0), (2, 1)):                       # identical grid to Pipeline 1
        m = (lab == cls).astype(int)
        edges = np.flatnonzero(np.diff(np.r_[0, m, 0]))
        for s, e in zip(edges[0::2], edges[1::2]):
            for w0 in range(s, e - p1.WIN + 1, p1.STEP):
                totals[name] += 1
                o1, o2 = both(xf[w0:w0 + p1.WIN])
                lab_rows.append(make_row(sid, name, w0 / p1.FS, o1, o2))
    for w0 in range(0, len(xf) - p1.WIN + 1, p1.STEP):       # label-free session grid
        o1, o2 = both(xf[w0:w0 + p1.WIN])
        if o1 is None and o2 is None:
            continue
        ses_rows.append(make_row(sid, -1, w0 / p1.FS, o1, o2))
    return lab_rows, ses_rows, totals


def zscore(df_lab, df_ses):
    """Within-subject z-score; one rule for every feature column; labels never used for the statistics."""
    out = df_lab.copy()
    for sid in out["subject"].unique():
        ref = df_ses[df_ses["subject"] == sid]
        if ARGS.zmode == "calib":
            ref = ref[ref["t0"] + p1.WIN / p1.FS <= ARGS.calib_s]
        m = (out["subject"] == sid)
        for f in ALL_FEATS:
            v = ref[f].to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v) < 3:
                out.loc[m, f] = np.nan
                continue
            sd = v.std(ddof=1)
            out.loc[m, f] = (out.loc[m, f] - v.mean()) / (sd if sd > 1e-9 else 1.0)
    return out


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def cluster_boot_auc(df, prob, rng):
    y, g = df["label"].to_numpy(), df["subject"].to_numpy()
    subs = np.unique(g)
    idx_by = {s: np.flatnonzero(g == s) for s in subs}
    d = []
    for _ in range(BOOT):
        pick = rng.choice(subs, len(subs), replace=True)
        idx = np.concatenate([idx_by[s] for s in pick])
        if len(np.unique(y[idx])) == 2:
            d.append(roc_auc_score(y[idx], prob[idx]))
    return np.percentile(d, [2.5, 97.5])


def evaluate(df, feats, model, rng):
    prob = p1.loso_predict(df, feats, model)
    y = df["label"].to_numpy()
    lo, hi = cluster_boot_auc(df, prob, rng)
    ps = p1.per_subject_auc(df, prob)
    return dict(AUC=roc_auc_score(y, prob), CI_lo=lo, CI_hi=hi,
                F1=f1_score(y, (prob >= 0.5).astype(int), zero_division=0),
                per_subject_AUC=ps.mean(), per_subject_SD=ps.std(ddof=1) if len(ps) > 1 else np.nan)


def main():
    out = Path(ARGS.out); out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print("[P2] EXPLORATORY post-hoc analysis; parameters fixed in the script; single run.")
    lab_csv, ses_csv, cnt_csv = out / "p2_windows.csv", out / "p2_session_windows.csv", out / "p2_counts_raw.csv"
    if lab_csv.exists() and ses_csv.exists() and cnt_csv.exists() and not ARGS.recompute:
        L, S, C = pd.read_csv(lab_csv), pd.read_csv(ses_csv), pd.read_csv(cnt_csv)
    else:
        lr, sr, cr = [], [], []
        for sid in p1.SUBJECTS:
            a, b, tot = process_subject(sid)
            lr += a; sr += b
            cr.append(dict(subject=sid, total_baseline=tot[0], total_stress=tot[1]))
            print(f"  S{sid}: {len(a)} labelled windows, {len(b)} session windows ({time.time()-t_start:.0f}s)", flush=True)
        L, S, C = pd.DataFrame(lr), pd.DataFrame(sr), pd.DataFrame(cr)
        L.to_csv(lab_csv, index=False); S.to_csv(ses_csv, index=False); C.to_csv(cnt_csv, index=False)

    # ---- reproduction check against the published P1 windows ---------------
    if ARGS.p1_windows:
        ref = pd.read_csv(ARGS.p1_windows)
        mine = L[L.ok_p1].merge(ref, on=["subject", "label", "t0"], suffixes=("", "_ref"))
        diffs = {f: float(np.nanmax(np.abs(mine[f] - mine[f + "_ref"]))) for f in P1_MACRO + P1_SHAPE + P1_WAV}
        print(f"[check] P1 windows matched: {len(mine)} of {len(ref)} in windows.csv; max |diff| per feature: "
              + json.dumps({k: round(v, 8) for k, v in diffs.items()}))

    # ---- retention ---------------------------------------------------------
    rows = []
    for sid in p1.SUBJECTS:
        g = L[L.subject == sid]; r = dict(subject=sid)
        for name, lv in (("baseline", 0), ("stress", 1)):
            gg = g[g.label == lv]
            r[f"{name}_total"] = len(gg)
            r[f"{name}_kept_P1"] = int(gg.ok_p1.sum())
            r[f"{name}_kept_P2"] = int(gg.ok_p2.sum())
            r[f"{name}_kept_both"] = int((gg.ok_p1 & gg.ok_p2).sum())
        gs = g[(g.label == 1) & g.ok_p2]
        gb = g[(g.label == 0) & g.ok_p2]
        r["appg_full_detect_stress"] = gs.appg_rate_full.mean() if len(gs) else np.nan
        r["appg_full_detect_base"] = gb.appg_rate_full.mean() if len(gb) else np.nan
        r["P1_notch_rate_stress"] = g[(g.label == 1) & g.ok_p1].notch_rate.mean()
        r["P1_notch_rate_base"] = g[(g.label == 0) & g.ok_p1].notch_rate.mean()
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(out / "p2_window_retention.csv", index=False)
    stress_subj = R[R.stress_kept_both > 0].subject.tolist()
    print("\n=== Window retention per subject (baseline/stress; kept by P1, P2, both) ===")
    print(R.round(2).to_string(index=False))
    print(f"\nSubjects with >=1 stress window retained by both pipelines: N={len(stress_subj)} -> {stress_subj}")

    # ---- timing-jitter diagnostic -----------------------------------------
    B = L[(L.label == 0) & L.ok_p1 & L.ok_p2]
    per = B.groupby("subject")[["RMSSD", "p2_RMSSD", "SDNN", "p2_SDNN"]].mean()
    print("\n=== Baseline RMSSD / SDNN (ms), subject means: P1 peak-top vs P2 max-dP/dt ===")
    print(per.round(1).to_string())
    print(f"Median over subjects: RMSSD P1 {per.RMSSD.median():.1f} -> P2 {per.p2_RMSSD.median():.1f} ms")

    # ---- datasets ----------------------------------------------------------
    D_both = L[L.ok_p1 & L.ok_p2].reset_index(drop=True)
    D_p1 = L[L.ok_p1].reset_index(drop=True)
    Z_both = zscore(D_both, S)
    Z_p1 = zscore(D_p1, S)
    print(f"\nModelling windows (accepted by both): {len(D_both)} "
          f"(baseline {(D_both.label==0).sum()}, stress {(D_both.label==1).sum()}); P1-only set: {len(D_p1)}")
    print(f"z-score mode: {ARGS.zmode}; statistics from {len(S)} label-free session windows")

    models = [m for m in p1.MODELS if m != "XGBoost" or p1.XGBClassifier is not None]
    if "XGBoost" not in models:
        print("[P2] WARNING: xgboost not installed -> XGBoost column skipped (as in Pipeline 1).")
    rng = np.random.default_rng(p1.SEED)
    res = []
    jobs = []
    for norm, data in (("raw", D_both), ("z-scored", Z_both)):
        for fs, feats in FEATURE_SETS.items():
            jobs.append((fs, norm, data, feats))
    for norm, data in (("raw", D_p1), ("z-scored", Z_p1)):
        jobs.append(("Old on P1's original windows (check)", norm, data, FEATURE_SETS["Old = P1 (macro+notch+CWT)"]))
    for fs, norm, data, feats in jobs:
        for m in models:
            r = evaluate(data, feats, m, rng)
            res.append(dict(feature_set=fs, normalisation=norm, model=m, n_windows=len(data), **r))
            print(f"  {norm:8s} | {fs:38s} | {m:19s} AUC {r['AUC']:.3f} [{r['CI_lo']:.3f}, {r['CI_hi']:.3f}]", flush=True)
    Res = pd.DataFrame(res)
    Res.to_csv(out / "p2_results_long.csv", index=False)

    Res["cell"] = Res.apply(lambda r: f"{r.AUC:.3f} [{r.CI_lo:.3f}, {r.CI_hi:.3f}]", axis=1)
    order = [(f, n) for f in list(FEATURE_SETS) + ["Old on P1's original windows (check)"] for n in ("raw", "z-scored")]
    mat = Res.pivot_table(index=["feature_set", "normalisation"], columns="model", values="cell", aggfunc="first")
    mat = mat.reindex(pd.MultiIndex.from_tuples(order, names=mat.index.names))[models]
    mat.to_csv(out / "p2_comparison_matrix.csv")
    ps = Res.pivot_table(index=["feature_set", "normalisation"], columns="model", values="per_subject_AUC")
    ps = ps.reindex(mat.index)[models].round(3)
    ps.to_csv(out / "p2_per_subject_auc_matrix.csv")
    (out / "p2_comparison_matrix.md").write_text(p1.md_table(mat.reset_index()) + "\n")
    print("\n=== CONSOLIDATED MATRIX: pooled LOSO AUC [95% subject-bootstrap CI] ===")
    print(mat.to_string())
    print("\n=== Mean per-subject AUC (not affected by between-subject score offsets) ===")
    print(ps.to_string())

    json.dump(dict(note="EXPLORATORY post-hoc; single run", zmode=ARGS.zmode, calib_s=ARGS.calib_s,
                   p2_constants=dict(UP=UP, SG_WIN=SG_WIN, SG_POLY=SG_POLY, APPG_PRE_S=APPG_PRE_S,
                                     APPG_POST_MAX_S=APPG_POST_MAX_S, BOOT=BOOT),
                   p1_constants=dict(SQI_MIN=p1.SQI_MIN, MIN_BEATS=p1.MIN_BEATS, MIN_DIFFS=p1.MIN_DIFFS,
                                     MIN_NOTCH_RATE=p1.MIN_NOTCH_RATE, WIN=p1.WIN, STEP=p1.STEP, SEED=p1.SEED),
                   stress_subjects=stress_subj, models=models, runtime_s=round(time.time() - t_start, 1)),
              open(out / "p2_run_log.json", "w"), indent=2)
    print(f"\nDone in {time.time()-t_start:.0f}s. Files in {out.resolve()}")


if __name__ == "__main__":
    main()
