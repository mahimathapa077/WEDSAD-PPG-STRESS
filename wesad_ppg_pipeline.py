#!/usr/bin/env python3
"""
WESAD wrist-PPG pipeline: baseline vs stress using Morlet CWT and beat-morphology features.

Usage:
    python wesad_ppg_pipeline.py --data /path/to/WESAD --out results

Expected data layout (as released with WESAD):
    WESAD/S2/S2.pkl, WESAD/S3/S3.pkl, ... (S1 and S12 are not part of the release)

Outputs (in --out):
    windows.csv            one row per accepted 60 s window with all features
    templates.npy          median normalised beat shape for each row of windows.csv
    window_counts.csv      windows kept vs rejected per subject
    table1_features.md/csv paired baseline vs stress statistics (N = 15 subjects)
    table2_models.md/csv   leave-one-subject-out classifier metrics
    table3_ablation.md/csv feature-group ablation (PRV only, morphology only, ...)
    table4_importance.md   random forest and permutation importance
    fig1_templates.png, fig2_scalograms.png, fig3_roc.png
"""
import argparse
import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:  # the script still runs without XGBoost
    XGBClassifier = None

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
trapz = getattr(np, "trapezoid", None) or np.trapz

# ----------------------------------------------------------------------------
# Constants (every value here should be reported in the Methods section)
# ----------------------------------------------------------------------------
SEED = 0
FS = 64                      # Empatica E4 BVP sampling rate (Hz)
LABEL_FS = 700               # WESAD label rate (Hz)
SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
WIN, STEP = 60 * FS, 30 * FS  # 60 s windows, 50% overlap
TRIM = 3 * FS                # discard 3 s at each window edge for the scalogram
MIN_BEATS = 30               # accepted beats needed per window
MIN_DIFFS = 15               # successive-interval differences needed for RMSSD
SQI_MIN = 0.80               # beat-template correlation threshold
MIN_NOTCH_RATE = 0.50        # fraction of beats with a detected notch to keep morphology
BAND = (0.5, 8.0)            # bandpass edges (Hz)
FB, FC = 1.5, 1.0            # Morlet bandwidth and centre frequency
FREQS = np.geomspace(BAND[0], BAND[1], 48)

FEATURES = ["HR", "SDNN", "RMSSD", "DR", "RI", "AR", "log10_Ew", "E_ratio"]
GROUPS = {
    "PRV only": ["HR", "SDNN", "RMSSD"],
    "Morphology only": ["DR", "RI", "AR"],
    "Wavelet only": ["log10_Ew", "E_ratio"],
    "All features": FEATURES,
}
TABLE1 = [
    ("RMSSD (ms)", "RMSSD"),
    ("DR", "DR"),
    ("RI", "RI"),
    ("log10 E_w", "log10_Ew"),
    ("AR", "AR"),
    ("E_ratio", "E_ratio"),
    ("HR (bpm)", "HR"),
]
MODELS = ["Logistic Regression", "SVM-RBF", "Random Forest", "XGBoost"]

SOS = signal.butter(2, BAND, btype="bandpass", fs=FS, output="sos")


# ----------------------------------------------------------------------------
# Loading and preprocessing
# ----------------------------------------------------------------------------
def load_subject(root, sid):
    path = Path(root) / f"S{sid}" / f"S{sid}.pkl"
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    bvp = np.asarray(d["signal"]["wrist"]["BVP"], dtype=float).ravel()
    lab = np.asarray(d["label"]).ravel()
    idx = np.minimum(np.round(np.arange(len(bvp)) * LABEL_FS / FS).astype(int), len(lab) - 1)
    return bvp, lab[idx]


def bandpass(x):
    """Zero-phase Butterworth bandpass, 0.5-8 Hz (2nd-order prototype, forward-backward)."""
    return signal.sosfiltfilt(SOS, x)


# ----------------------------------------------------------------------------
# Continuous wavelet transform (complex Morlet), implemented from the definition
# ----------------------------------------------------------------------------
def morlet_cwt(x, freqs=FREQS, fs=FS, fb=FB, fc=FC):
    """X_w(a,b) = a^(-1/2) * integral x(t) conj(psi((t-b)/a)) dt, with
    psi(u) = (pi fb)^(-1/2) exp(i 2 pi fc u) exp(-u^2 / fb) and a = fc / f (seconds).
    Because psi(-u) = conj(psi(u)), the transform is a convolution with psi(u/a)/sqrt(a)."""
    dt = 1.0 / fs
    scales = fc / np.asarray(freqs)
    out = np.empty((len(scales), len(x)), dtype=complex)
    for i, a in enumerate(scales):
        half = int(np.ceil(4.0 * np.sqrt(fb) * a * fs))
        u = np.arange(-half, half + 1) * dt / a
        psi = (np.pi * fb) ** -0.5 * np.exp(2j * np.pi * fc * u) * np.exp(-(u ** 2) / fb)
        out[i] = signal.fftconvolve(x, psi / np.sqrt(a), mode="same") * dt
    return out, scales


def scalogram_features(x, f0):
    """log10 of band energy E_w and harmonic-to-fundamental energy ratio.
    E_w = integral over ln f of < |X_w|^2 / a >_b  (the da db / a^2 measure in log-frequency)."""
    coefs, scales = morlet_cwt(x - x.mean())
    dens = (np.abs(coefs[:, TRIM:-TRIM]) ** 2).mean(axis=1) / scales
    lnf = np.log(FREQS)

    def band(lo, hi):
        m = (FREQS >= lo) & (FREQS <= hi)
        return trapz(dens[m], lnf[m]) if m.sum() >= 3 else np.nan

    e_total = band(*BAND)
    e_fund = band(0.75 * f0, 1.25 * f0)
    e_harm = band(1.75 * f0, BAND[1])
    return {"log10_Ew": np.log10(e_total), "E_ratio": e_harm / e_fund}


# ----------------------------------------------------------------------------
# Beat segmentation, landmarks, quality control
# ----------------------------------------------------------------------------
def detect_beats(x):
    peaks, _ = signal.find_peaks(x, distance=int(0.35 * FS), prominence=0.5 * np.std(x))
    if len(peaks) < 3:
        return []
    dx = np.gradient(x)
    feet = []
    for p0, p1 in zip(peaks[:-1], peaks[1:]):
        # Anchor the foot to the upstroke: find the steepest rise in the second half of the
        # inter-peak interval, then take the lowest point in the 0.3 s before it.
        lo = p0 + (p1 - p0) // 2
        u = lo + int(np.argmax(dx[lo:p1]))
        start = max(p0, u - int(0.3 * FS))
        feet.append(start + int(np.argmin(x[start:u + 1])))
    return [dict(k=j + 1, foot=feet[j], peak=int(peaks[j + 1]), end=feet[j + 1])
            for j in range(len(feet) - 1)]


def resample_beat(x, b, n=100):
    seg = x[b["foot"]:b["end"] + 1]
    r = np.interp(np.linspace(0, len(seg) - 1, n), np.arange(len(seg)), seg)
    rng = r.max() - r.min()
    return (r - r.min()) / rng if rng > 0 else None


def beat_landmarks(x, b):
    """DR, RI, AR for one beat. NaN when no diastolic peak is found in the decay limb."""
    f0, pk, f1 = b["foot"], b["peak"], b["end"]
    a_f, a_s = x[f0], x[pk]
    amp = a_s - a_f
    if amp <= 0:
        return None
    out = {"DR": np.nan, "RI": np.nan, "AR": np.nan}
    cand, _ = signal.find_peaks(x[pk:f1 + 1], prominence=0.02 * amp)
    if len(cand):
        d = pk + int(cand[0])                      # first local maximum after the systolic peak
        n = pk + int(np.argmin(x[pk:d + 1]))       # lowest point between them = dicrotic notch
        rise = trapz(x[f0:pk + 1] - a_f)
        decay = trapz(x[pk:n + 1] - a_f)
        out["DR"] = (a_s - x[n]) / amp
        out["RI"] = (x[d] - a_f) / amp
        out["AR"] = decay / rise if rise > 0 else np.nan
    return out


def window_features(x):
    """Features for one filtered 60 s window, or None if quality control rejects it."""
    beats = detect_beats(x)
    if len(beats) < MIN_BEATS:
        return None
    shapes = []
    for b in beats:
        dur = (b["end"] - b["foot"]) / FS
        shapes.append(resample_beat(x, b) if 0.35 <= dur <= 1.5 else None)
    good = [s for s in shapes if s is not None]
    if len(good) < MIN_BEATS:
        return None
    template = np.median(good, axis=0)
    accepted = [(b, s) for b, s in zip(beats, shapes)
                if s is not None and np.corrcoef(s, template)[0, 1] >= SQI_MIN]
    if len(accepted) < MIN_BEATS:
        return None

    peaks = {b["k"]: b["peak"] for b, _ in accepted}
    ibi = {k: (peaks[k + 1] - peaks[k]) / FS for k in peaks if k + 1 in peaks}
    diffs = [ibi[k + 1] - ibi[k] for k in ibi if k + 1 in ibi and abs(ibi[k + 1] - ibi[k]) < 0.3]
    if len(diffs) < MIN_DIFFS:
        return None
    ibi_arr = np.array(list(ibi.values()))
    f0 = 1.0 / ibi_arr.mean()

    lm = [m for m in (beat_landmarks(x, b) for b, _ in accepted) if m is not None]
    with_notch = [m for m in lm if not np.isnan(m["DR"])]
    notch_rate = len(with_notch) / len(accepted)
    morph = {k: np.nan for k in ("DR", "RI", "AR")}
    if notch_rate >= MIN_NOTCH_RATE:
        morph = {k: float(np.mean([m[k] for m in with_notch])) for k in morph}

    feats = {
        "HR": 60.0 * f0,
        "SDNN": 1000.0 * float(np.std(ibi_arr, ddof=1)),
        "RMSSD": 1000.0 * float(np.sqrt(np.mean(np.square(diffs)))),
        **morph,
        **scalogram_features(x, f0),
        "n_beats": len(accepted),
        "notch_rate": notch_rate,
    }
    return feats, np.median([s for _, s in accepted], axis=0)


def process_subject(root, sid):
    bvp, lab = load_subject(root, sid)
    xf = bandpass(bvp)
    rows, tmpl, n_total = [], [], 0
    for cls, name in ((1, 0), (2, 1)):                 # 1 = baseline, 2 = stress
        m = (lab == cls).astype(int)
        edges = np.flatnonzero(np.diff(np.r_[0, m, 0]))
        for s, e in zip(edges[0::2], edges[1::2]):
            for w0 in range(s, e - WIN + 1, STEP):
                n_total += 1
                out = window_features(xf[w0:w0 + WIN])
                if out is None:
                    continue
                feats, template = out
                rows.append(dict(subject=sid, label=name, t0=w0 / FS, **feats))
                tmpl.append(template)
    return rows, tmpl, n_total


# ----------------------------------------------------------------------------
# Statistics: paired baseline vs stress, Holm correction
# ----------------------------------------------------------------------------
def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def paired_table(df):
    cols = [c for _, c in TABLE1]
    per = df.groupby(["subject", "label"])[cols].mean().unstack("label")
    recs, pvals = [], []
    for name, col in TABLE1:
        b, s = per[(col, 0)], per[(col, 1)]
        ok = b.notna() & s.notna()
        b, s = b[ok], s[ok]
        diff = s - b
        t, p = stats.ttest_rel(s, b)
        dz = diff.mean() / diff.std(ddof=1)
        try:
            pw = stats.wilcoxon(s, b).pvalue
        except ValueError:
            pw = np.nan
        pvals.append(p)
        recs.append(dict(
            Metric=name,
            Baseline=f"{b.mean():.3f} ± {b.std(ddof=1):.3f}",
            Stress=f"{s.mean():.3f} ± {s.std(ddof=1):.3f}",
            Delta=f"{diff.mean():+.3f}",
            # log10 features: report the multiplicative change in linear units, 10^delta - 1
            Pct_change=(f"{100 * (10 ** diff.mean() - 1):+.1f}%" if col.startswith("log10")
                        else f"{100 * diff.mean() / b.mean():+.1f}%"),
            t=f"{t:.2f} (df={len(b) - 1})",
            p=fmt_p(p),
            Wilcoxon_p=fmt_p(pw) if not np.isnan(pw) else "n/a",
            d_z=f"{dz:.2f}",
            n=len(b),
        ))
    for r, pa in zip(recs, holm(np.array(pvals))):
        r["p_Holm"] = fmt_p(pa)
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------------
# Classification: leave-one-subject-out, fixed hyperparameters, subject bootstrap
# ----------------------------------------------------------------------------
def make_model(name, y_tr):
    if name == "Logistic Regression":
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    elif name == "SVM-RBF":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                  probability=True, random_state=SEED)
    elif name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                     class_weight="balanced", random_state=SEED, n_jobs=-1)
    elif name == "XGBoost":
        clf = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, eval_metric="logloss", random_state=SEED,
                            scale_pos_weight=(y_tr == 0).sum() / max(1, (y_tr == 1).sum()), n_jobs=1)
    else:
        raise ValueError(name)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf)


def loso_predict(df, feats, name):
    X, y, g = df[feats].to_numpy(), df["label"].to_numpy(), df["subject"].to_numpy()
    prob = np.zeros(len(df))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        model = make_model(name, y[tr]).fit(X[tr], y[tr])
        prob[te] = model.predict_proba(X[te])[:, 1]
    return prob


def metric_dict(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return dict(Precision=precision_score(y, pred, zero_division=0),
                Recall=recall_score(y, pred, zero_division=0),
                F1=f1_score(y, pred, zero_division=0),
                AUC=roc_auc_score(y, prob))


def bootstrap_ci(df, prob, n_boot=500):
    rng = np.random.default_rng(SEED)
    y, g = df["label"].to_numpy(), df["subject"].to_numpy()
    subs = np.unique(g)
    idx_by = {s: np.flatnonzero(g == s) for s in subs}
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(subs, len(subs), replace=True)
        idx = np.concatenate([idx_by[s] for s in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        draws.append(metric_dict(y[idx], prob[idx]))
    d = pd.DataFrame(draws)
    return {c: (d[c].quantile(0.025), d[c].quantile(0.975)) for c in d.columns}


def per_subject_auc(df, prob):
    out = []
    for s in np.unique(df["subject"]):
        m = (df["subject"] == s).to_numpy()
        if len(np.unique(df["label"][m])) == 2:
            out.append(roc_auc_score(df["label"][m], prob[m]))
    return np.array(out)


def summarise(df, prob):
    y = df["label"].to_numpy()
    m, ci = metric_dict(y, prob), bootstrap_ci(df, prob)
    ps = per_subject_auc(df, prob)
    row = {k: f"{m[k]:.3f} [{ci[k][0]:.3f}, {ci[k][1]:.3f}]" for k in ("Precision", "Recall", "F1", "AUC")}
    row["Per-subject AUC"] = f"{ps.mean():.3f} ± {ps.std(ddof=1):.3f}"
    return row


def run_models(df, out):
    avail = [m for m in MODELS if m != "XGBoost" or XGBClassifier is not None]
    if "XGBoost" not in avail:
        warnings.warn("xgboost not installed; skipping XGBoost (pip install xgboost)")
    probs, rows = {}, []
    for name in avail:
        probs[name] = loso_predict(df, FEATURES, name)
        rows.append({"Model": name, **summarise(df, probs[name])})
    t2 = pd.DataFrame(rows)
    t2.to_csv(out / "table2_models.csv", index=False)
    (out / "table2_models.md").write_text(md_table(t2) + "\n")

    abl = []
    for gname, feats in GROUPS.items():
        for name in ("Logistic Regression", "Random Forest"):
            abl.append({"Feature set": gname, "Model": name, **summarise(df, loso_predict(df, feats, name))})
    t3 = pd.DataFrame(abl)[["Feature set", "Model", "F1", "AUC", "Per-subject AUC"]]
    t3.to_csv(out / "table3_ablation.csv", index=False)
    (out / "table3_ablation.md").write_text(md_table(t3) + "\n")

    # Feature importance: RF impurity importance and permutation importance on held-out subjects
    X, y, g = df[FEATURES].to_numpy(), df["label"].to_numpy(), df["subject"].to_numpy()
    imp, perm = [], []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        if len(np.unique(y[te])) < 2:
            continue
        model = make_model("Random Forest", y[tr]).fit(X[tr], y[tr])
        imp.append(model[-1].feature_importances_)
        pi = permutation_importance(model, X[te], y[te], scoring="roc_auc", n_repeats=5,
                                    random_state=SEED)
        perm.append(pi.importances_mean)
    t4 = pd.DataFrame({
        "Feature": FEATURES,
        "RF impurity importance": [f"{v:.3f}" for v in np.mean(imp, axis=0)],
        "Permutation AUC drop": [f"{v:.3f}" for v in np.mean(perm, axis=0)],
    })
    t4["_s"] = np.mean(perm, axis=0)
    t4 = t4.sort_values("_s", ascending=False).drop(columns="_s")
    t4.to_csv(out / "table4_importance.csv", index=False)
    (out / "table4_importance.md").write_text(md_table(t4) + "\n")
    return probs


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_templates(df, templates, out):
    T = np.asarray(templates)
    subj, lab = df["subject"].to_numpy(), df["label"].to_numpy()
    phase = np.linspace(0, 100, T.shape[1])
    fig, ax = plt.subplots(figsize=(6, 4))
    for cls, name, color in ((0, "Baseline", "tab:blue"), (1, "Stress", "tab:red")):
        per = np.array([T[(subj == s) & (lab == cls)].mean(axis=0)
                        for s in np.unique(subj) if ((subj == s) & (lab == cls)).any()])
        mean, sem = per.mean(axis=0), per.std(axis=0, ddof=1) / np.sqrt(len(per))
        ax.plot(phase, mean, color=color, label=f"{name} (n={len(per)} subjects)")
        ax.fill_between(phase, mean - sem, mean + sem, color=color, alpha=0.2)
    ax.set_xlabel("Position within beat (%)")
    ax.set_ylabel("Normalised pulse amplitude")
    ax.set_title("Mean beat shape by condition (± SEM across subjects)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig1_templates.png", dpi=200)
    plt.close(fig)


def fig_scalograms(root, out, sid):
    bvp, lab = load_subject(root, sid)
    xf = bandpass(bvp)
    mags = []
    for cls in (1, 2):
        idx = np.flatnonzero(lab == cls)
        w0 = idx[0] + 30 * FS
        if not np.all(lab[w0:w0 + WIN] == cls) or w0 + WIN > len(xf):
            w0 = idx[0]
        seg = xf[w0:w0 + WIN]
        mags.append(np.abs(morlet_cwt(seg - seg.mean())[0]))
    vmax = max(m.max() for m in mags)
    t = np.arange(WIN) / FS
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, mag, title in zip(axes, mags, ("Baseline", "Stress")):
        pc = ax.pcolormesh(t, FREQS, mag, shading="auto", vmin=0, vmax=vmax)
        ax.set_yscale("log")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"S{sid}: {title}")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(pc, ax=axes, label="|X_w(a,b)| (shared colour scale)")
    fig.savefig(out / "fig2_scalograms.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_roc(df, probs, out):
    fig, ax = plt.subplots(figsize=(5, 5))
    y = df["label"].to_numpy()
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Leave-one-subject-out ROC (pooled)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig3_roc.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to the WESAD folder")
    ap.add_argument("--out", default="results")
    ap.add_argument("--recompute", action="store_true", help="ignore cached windows.csv")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cache, tcache = out / "windows.csv", out / "templates.npy"
    if cache.exists() and tcache.exists() and not args.recompute:
        df, templates = pd.read_csv(cache), np.load(tcache)
    else:
        rows, templates, counts = [], [], []
        for sid in SUBJECTS:
            r, t, n_total = process_subject(args.data, sid)
            rows += r
            templates += t
            counts.append(dict(subject=sid, windows_total=n_total, windows_kept=len(r)))
            print(f"S{sid}: kept {len(r)} of {n_total} windows", flush=True)
        df, templates = pd.DataFrame(rows), np.asarray(templates)
        df.to_csv(cache, index=False)
        np.save(tcache, templates)
        pd.DataFrame(counts).to_csv(out / "window_counts.csv", index=False)

    print(df.groupby("label").size().rename({0: "baseline windows", 1: "stress windows"}))

    t1 = paired_table(df)
    t1.to_csv(out / "table1_features.csv", index=False)
    (out / "table1_features.md").write_text(md_table(t1) + "\n")
    print(md_table(t1))

    probs = run_models(df, out)
    print((out / "table2_models.md").read_text())
    fig_templates(df, templates, out)
    fig_scalograms(args.data, out, SUBJECTS[0])
    fig_roc(df, probs, out)
    print(f"Done. Files written to {out.resolve()}")


if __name__ == "__main__":
    main()
