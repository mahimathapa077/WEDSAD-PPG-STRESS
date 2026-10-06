#!/usr/bin/env python3
"""
wesad_ppg_pipeline2_extra.py -- one extra EXPLORATORY follow-up to Pipeline 2.

Question: was the New (P2) AUC bump driven by the APPG features or by the wavelet features?
Adds two rows to the Pipeline 2 matrix, using the SAME windows, SAME z-scoring, SAME models:
    Macro P2 + CWT   (p2_HR, p2_SDNN, p2_RMSSD, p2_log10_Ew, p2_E_ratio)
    Macro P2 + APPG  (p2_HR, p2_SDNN, p2_RMSSD, appg_ba..appg_agi)
and re-prints the existing Macro P2 and New rows as a reproduction check.
Needs no WESAD data: it reads the CSVs written by wesad_ppg_pipeline2.py.

Usage (same folder as wesad_ppg_pipeline.py):
    python wesad_ppg_pipeline2_extra.py --p2_dir results_p2
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--p2_dir", default="results_p2")
ap.add_argument("--p1_dir", default=str(Path(__file__).resolve().parent))
ARGS = ap.parse_args()
sys.path.insert(0, ARGS.p1_dir)
import wesad_ppg_pipeline as p1  # noqa: E402
warnings.filterwarnings("ignore")

M2 = ["p2_HR", "p2_SDNN", "p2_RMSSD"]
CWT = ["p2_log10_Ew", "p2_E_ratio"]
APPG = ["appg_ba", "appg_ca", "appg_da", "appg_ea", "appg_agi"]
SETS = {
    "Macro P2 (existing)": M2,
    "Macro P2 + CWT (NEW)": M2 + CWT,
    "Macro P2 + APPG (NEW)": M2 + APPG,
    "New = P2 all (existing)": M2 + APPG + CWT,
}
ALL = sorted({f for v in SETS.values() for f in v})
BOOT = 1000


def zscore(df, ses):
    out = df.copy()
    for sid in out["subject"].unique():
        ref = ses[ses["subject"] == sid]
        m = out["subject"] == sid
        for f in ALL:
            v = ref[f].to_numpy(float); v = v[np.isfinite(v)]
            if len(v) < 3:
                out.loc[m, f] = np.nan; continue
            sd = v.std(ddof=1)
            out.loc[m, f] = (out.loc[m, f] - v.mean()) / (sd if sd > 1e-9 else 1.0)
    return out


def boot(df, prob, rng):
    y, g = df["label"].to_numpy(), df["subject"].to_numpy()
    subs = np.unique(g); idx = {s: np.flatnonzero(g == s) for s in subs}
    d = []
    for _ in range(BOOT):
        pick = rng.choice(subs, len(subs), replace=True)
        i = np.concatenate([idx[s] for s in pick])
        if len(np.unique(y[i])) == 2:
            d.append(roc_auc_score(y[i], prob[i]))
    return np.percentile(d, [2.5, 97.5])


d = Path(ARGS.p2_dir)
L, S = pd.read_csv(d / "p2_windows.csv"), pd.read_csv(d / "p2_session_windows.csv")
D = L[L.ok_p1 & L.ok_p2].reset_index(drop=True)
Z = zscore(D, S)
models = [m for m in p1.MODELS if m != "XGBoost" or p1.XGBClassifier is not None]
rng = np.random.default_rng(p1.SEED)
rows = []
for norm, data in (("raw", D), ("z-scored", Z)):
    for name, feats in SETS.items():
        for m in models:
            prob = p1.loso_predict(data, feats, m)
            lo, hi = boot(data, prob, rng)
            ps = p1.per_subject_auc(data, prob)
            rows.append(dict(feature_set=name, normalisation=norm, model=m,
                             AUC=roc_auc_score(data["label"], prob), lo=lo, hi=hi, per_subject=ps.mean()))
R = pd.DataFrame(rows)
R["cell"] = R.apply(lambda r: f"{r.AUC:.3f} [{r.lo:.3f}, {r.hi:.3f}]", axis=1)
order = [(n, z) for n in SETS for z in ("raw", "z-scored")]
mat = R.pivot_table(index=["feature_set", "normalisation"], columns="model", values="cell", aggfunc="first")
mat = mat.reindex(pd.MultiIndex.from_tuples(order, names=mat.index.names))[models]
ps = R.pivot_table(index=["feature_set", "normalisation"], columns="model", values="per_subject").reindex(mat.index)[models].round(3)
print(f"Windows: {len(D)} (baseline {(D.label==0).sum()}, stress {(D.label==1).sum()}); stress subjects: {D[D.label==1].subject.nunique()}")
print("\nEXPLORATORY follow-up. Pooled LOSO AUC [95% subject-bootstrap CI]\n")
print(mat.to_string())
print("\nMean per-subject AUC\n")
print(ps.to_string())
mat.to_csv(d / "p2_extra_matrix.csv"); ps.to_csv(d / "p2_extra_per_subject.csv")
