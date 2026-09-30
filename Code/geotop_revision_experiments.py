#!/usr/bin/env python3
"""
GeoTop -- experiments requested by Reviewer 3 (round 2), adapted to the folder layout
    data/{train,test}/{benign,malignant}

Step 1 (once):   python extract_features.py --data_root ./data --out geotop_features.npz [--user_extractor my_pipeline.py]
Step 2 (optional, leakage): python make_groups.py --order geotop_features_image_order.csv --out groups.npy
Step 3:          python geotop_revision_experiments.py --features geotop_features.npz --groups groups.npy --out results/

PROTOCOLS
  --protocol official  (default when the npz has is_test)  train on data/train, evaluate on data/test. ONE split, shared by
                       every model (also the CNN/ViT). CIs = bootstrap over test images; comparisons = paired bootstrap.
                       This is the cleanest answer to R3-pt.8: identical partition for all methods, no 500-resample figure.
  --protocol pooled    train+test pooled (N=3297), `--repeats` x 5-fold stratified (group-aware if --groups) partitions.
                       Use for the repeated-split analyses / the leakage-gap analysis.
Run BOTH; report `official` for the baseline table and `pooled` for the repeated-split ablations, and say which is which.

Deep baselines: fine-tune on data/train, predict data/test, save np.savez('cnn_preds.npz', probs=p) where p is either
(N_test,) in the row order of *_image_order.csv restricted to is_test rows, or (N,) with NaN outside the test rows. Then
    --external_preds cnn=cnn_preds.npz vit=vit_preds.npz

Layout ASSUMED for the 64+120 columns (edit LAYOUT / pass --layout json if your extraction differs):
    topo col = ch*16 + dim*8 + j      j: 0 bottleneck,1 W1,2 W2,3 landscape,4 Betti-curve,5 image,6 silhouette,7 entropy
    geo  col = ch*30 + fn*10 + curve*5 + stat    fn: 0 Area, 1 Perimeter, 2 Euler(L0); curve: 0 f, 1 df
    ch order: 0 grayscale, 1 R, 2 G, 3 B
`--simulate` = smoke test only; outputs are prefixed SIMULATED_ and must never be quoted.
"""
import argparse, json, os, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, precision_score
from sklearn.cross_decomposition import CCA

warnings.filterwarnings("ignore")

LAYOUT = dict(n_ch=4, topo_per_ch=16, geo_per_ch=30, ch_names=["gray", "R", "G", "B"],
              betti_j=4, fn_names=["A", "P", "E"])


# ----------------------------------------------------------------- column helpers
def topo_cols(ch=None, betti=None):
    L = LAYOUT
    cols = []
    for c in range(L["n_ch"]):
        if ch is not None and c not in ch:
            continue
        for d in range(2):
            for j in range(8):
                is_betti = (j == L["betti_j"])
                if betti is None or betti == is_betti:
                    cols.append(c * L["topo_per_ch"] + d * 8 + j)
    return np.array(cols, int)


def geo_cols(ch=None, fn=None):
    L = LAYOUT
    cols = []
    for c in range(L["n_ch"]):
        if ch is not None and c not in ch:
            continue
        for f in range(3):
            if fn is not None and f not in fn:
                continue
            cols += [c * L["geo_per_ch"] + f * 10 + k for k in range(10)]
    return np.array(cols, int)


# ----------------------------------------------------------------- splits & metrics
def make_splits(y, groups, repeats, seed):
    """`repeats` x 5-fold (stratified, group-aware if groups given). Each split = 80/20."""
    out = []
    for r in range(repeats):
        if groups is None:
            it = StratifiedKFold(5, shuffle=True, random_state=seed + r).split(np.zeros(len(y)), y)
        else:
            it = StratifiedGroupKFold(5, shuffle=True, random_state=seed + r).split(np.zeros(len(y)), y, groups)
        for k, (tr, te) in enumerate(it):
            out.append(dict(repeat=r, fold=k, train=tr, test=te))
    return out


def metrics(y, p, thr=0.5):
    """fast numpy implementation (bootstrap calls this thousands of times); validated against sklearn in __main__ selftest"""
    y = np.asarray(y); p = np.asarray(p, float); yh = p >= thr
    tp = int((yh & (y == 1)).sum()); fp = int((yh & (y == 0)).sum()); fn = int((~yh & (y == 1)).sum()); tn = int((~yh & (y == 0)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1); f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    n1, n0 = tp + fn, tn + fp
    auc = (stats.rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
    return dict(acc=(tp + tn) / len(y), sens=sens, spec=spec, bal_acc=(sens + spec) / 2, prec=prec, f1=f1, auroc=auc)


def rf(args, seed):
    return RandomForestClassifier(n_estimators=args.n_estimators, class_weight="balanced",
                                  n_jobs=-1, random_state=seed)


# ----------------------------------------------------------------- fusion models
# Every model: f(Xt, Xg, y, tr, te, args, seed) -> P(malignant) on te. ALL selection / weighting
# steps are fitted on the training fold only (no leakage from test fold).
def m_tda(Xt, Xg, y, tr, te, a, s):
    return rf(a, s).fit(Xt[tr], y[tr]).predict_proba(Xt[te])[:, 1]

def m_lkc(Xt, Xg, y, tr, te, a, s):
    return rf(a, s).fit(Xg[tr], y[tr]).predict_proba(Xg[te])[:, 1]

def m_concat(Xt, Xg, y, tr, te, a, s):
    X = np.hstack([Xt, Xg]); return rf(a, s).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]

def m_concat_mi(Xt, Xg, y, tr, te, a, s, k=100):
    X = np.hstack([Xt, Xg]); mi = mutual_info_classif(X[tr], y[tr], random_state=s)
    idx = np.argsort(mi)[::-1][:min(k, X.shape[1])]
    return rf(a, s).fit(X[tr][:, idx], y[tr]).predict_proba(X[te][:, idx])[:, 1]

def _inner_oof(Xb, y, tr, a, s):
    return cross_val_predict(rf(a, s), Xb[tr], y[tr], cv=StratifiedKFold(3, shuffle=True, random_state=s),
                             method="predict_proba")[:, 1]

def _branch_probs(Xt, Xg, y, tr, te, a, s):
    pt = rf(a, s).fit(Xt[tr], y[tr]).predict_proba(Xt[te])[:, 1]
    pg = rf(a, s).fit(Xg[tr], y[tr]).predict_proba(Xg[te])[:, 1]
    return pt, pg

def m_late_avg(Xt, Xg, y, tr, te, a, s):
    pt, pg = _branch_probs(Xt, Xg, y, tr, te, a, s); return (pt + pg) / 2

def m_late_weighted(Xt, Xg, y, tr, te, a, s):
    ot, og = _inner_oof(Xt, y, tr, a, s), _inner_oof(Xg, y, tr, a, s)
    ws = np.linspace(0, 1, 11)
    w = ws[np.argmax([roc_auc_score(y[tr], w * ot + (1 - w) * og) for w in ws])]
    pt, pg = _branch_probs(Xt, Xg, y, tr, te, a, s); return w * pt + (1 - w) * pg

def m_late_stack(Xt, Xg, y, tr, te, a, s):
    ot, og = _inner_oof(Xt, y, tr, a, s), _inner_oof(Xg, y, tr, a, s)
    lg = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
    meta = LogisticRegression(class_weight="balanced").fit(np.c_[lg(ot), lg(og)], y[tr])
    pt, pg = _branch_probs(Xt, Xg, y, tr, te, a, s)
    return meta.predict_proba(np.c_[lg(pt), lg(pg)])[:, 1]

def m_interaction(Xt, Xg, y, tr, te, a, s, k=10):
    """concat + all pairwise products between the top-k (by MI, train fold) topological and geometric features"""
    mt = np.argsort(mutual_info_classif(Xt[tr], y[tr], random_state=s))[::-1][:k]
    mg = np.argsort(mutual_info_classif(Xg[tr], y[tr], random_state=s))[::-1][:k]
    sc_t, sc_g = StandardScaler().fit(Xt[tr][:, mt]), StandardScaler().fit(Xg[tr][:, mg])
    def build(idx):
        zt, zg = sc_t.transform(Xt[idx][:, mt]), sc_g.transform(Xg[idx][:, mg])
        inter = (zt[:, :, None] * zg[:, None, :]).reshape(len(idx), -1)
        return np.hstack([Xt[idx], Xg[idx], inter])
    return rf(a, s).fit(build(tr), y[tr]).predict_proba(build(te))[:, 1]

def m_linear_concat(Xt, Xg, y, tr, te, a, s):
    X = np.hstack([Xt, Xg]); sc = StandardScaler().fit(X[tr])
    return LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr]) \
        .predict_proba(sc.transform(X[te]))[:, 1]

EXTRA = {}          # filled in main(): {"hog": X_hog}

def m_hog_svm(Xt, Xg, y, tr, te, a, s):
    from sklearn.svm import SVC
    H = EXTRA["hog"]; sc = StandardScaler().fit(H[tr])
    return SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=s).fit(sc.transform(H[tr]), y[tr]) \
        .predict_proba(sc.transform(H[te]))[:, 1]

def m_boost(Xt, Xg, y, tr, te, a, s):
    X = np.hstack([Xt, Xg])
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=s)     # fallback, say so in the paper
    return clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]

FUSION = {"TDA only": m_tda, "LKC only": m_lkc, "Early: concat (current GeoTop)": m_concat,
          "Early: concat + MI top-100 (in-fold)": m_concat_mi, "Late: mean of branch probs": m_late_avg,
          "Late: weighted (w tuned on inner CV)": m_late_weighted, "Late: stacked (logistic meta)": m_late_stack,
          "Interaction: concat + top-10 x top-10 products": m_interaction,
          "Linear: L2-logistic on concat": m_linear_concat}


def run_model(fn, Xt, Xg, y, splits, a):
    """returns (n_splits, N) array of probs, NaN outside each split's test set"""
    P = np.full((len(splits), len(y)), np.nan)
    for i, sp in enumerate(splits):
        P[i, sp["test"]] = fn(Xt, Xg, y, sp["train"], sp["test"], a, a.seed + i)
    return P


def summarise_single(P, y, sp, ref=None, B=2000, seed=0):
    """official protocol: ONE train/test split. Point estimate + percentile-bootstrap CI over test images;
    difference to `ref` = paired bootstrap over the same images (two-sided p = 2*min(P(d<=0), P(d>=0)))."""
    te = sp["test"]; yt = y[te]; p = P[0, te]; r = np.random.default_rng(seed); n = len(te)
    base = metrics(yt, p); boots = {k: [] for k in base}; dacc, dauc = [], []
    for _ in range(B):
        i = r.integers(0, n, n)
        if len(np.unique(yt[i])) < 2: continue
        m = metrics(yt[i], p[i])
        for k in m: boots[k].append(m[k])
        if ref is not None:
            m2 = metrics(yt[i], ref[0, te][i]); dacc.append(m["acc"] - m2["acc"]); dauc.append(m["auroc"] - m2["auroc"])
    out = {k: f"{base[k]:.3f} [{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]" for k, v in boots.items()}
    out["_mean_acc"] = base["acc"]
    if ref is not None:
        d = np.array(dacc); pv = 2 * min((d <= 0).mean(), (d >= 0).mean())
        out["Δacc vs ref [95% CI]"] = f"{d.mean():+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]"
        out["ΔAUROC vs ref [95% CI]"] = f"{np.mean(dauc):+.3f} [{np.percentile(dauc, 2.5):+.3f}, {np.percentile(dauc, 97.5):+.3f}]"
        out["p (paired bootstrap)"] = f"{min(pv, 1.0):.3g}"
    return out


def summarise(P, y, splits, ref=None):
    """per-repeat pooled out-of-fold metrics -> mean and 95% t-CI over repeats;
    plus Nadeau-Bengio corrected resampled t-test on per-split accuracy vs `ref`."""
    if len(splits) == 1:
        return summarise_single(P, y, splits[0], ref)
    reps = sorted({s["repeat"] for s in splits}); rows = []
    for r in reps:
        idx = [i for i, s in enumerate(splits) if s["repeat"] == r]
        te = np.concatenate([splits[i]["test"] for i in idx]); p = np.concatenate([P[i, splits[i]["test"]] for i in idx])
        rows.append(metrics(y[te], p))
    df = pd.DataFrame(rows); out = {}
    for c in df:
        m, sd, n = df[c].mean(), df[c].std(ddof=1), len(df)
        h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
        lo, hi = (m - h, m + h) if not np.isnan(h) else (np.nan, np.nan)
        out[c] = f"{m:.3f} [{max(lo, 0):.3f}, {min(hi, 1):.3f}]"   # CI over repeats; use >=10 repeats
    out["_mean_acc"] = df["acc"].mean()
    if ref is not None:
        d = np.array([(metrics(y[s["test"]], P[i, s["test"]])["acc"] - metrics(y[s["test"]], ref[i, s["test"]])["acc"])
                      for i, s in enumerate(splits)])
        J, ratio = len(d), len(splits[0]["test"]) / len(splits[0]["train"])
        t = d.mean() / np.sqrt((1 / J + ratio) * d.var(ddof=1) + 1e-12)
        out["Δacc vs ref"] = f"{d.mean():+.4f}"; out["p (NB-corrected t)"] = f"{2 * stats.t.sf(abs(t), J - 1):.3g}"
    return out



def paired_bootstrap(y, p_ref, p_other, B=2000, seed=0):
    """paired bootstrap over test images on pooled out-of-fold predictions (one repeat)"""
    r = np.random.default_rng(seed); n = len(y); da, dr = [], []
    for _ in range(B):
        i = r.integers(0, n, n)
        if len(np.unique(y[i])) < 2: continue
        m1, m2 = metrics(y[i], p_ref[i]), metrics(y[i], p_other[i]); da.append(m2["acc"] - m1["acc"]); dr.append(m2["auroc"] - m1["auroc"])
    q = lambda v: f"{np.mean(v):+.3f} [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]"
    return {"Δacc vs GeoTop (95% CI)": q(da), "ΔAUROC vs GeoTop (95% CI)": q(dr)}

def table(models, Xt, Xg, y, splits, a, ref_name):
    Ps = {n: run_model(f, Xt, Xg, y, splits, a) for n, f in models.items()}
    return pd.DataFrame({n: summarise(P, y, splits, None if n == ref_name else Ps[ref_name]) for n, P in Ps.items()}).T, Ps


# ----------------------------------------------------------------- redundancy analyses
def block_correlation(Xt, Xg, y):
    keep_t, keep_g = Xt.std(0) > 0, Xg.std(0) > 0
    rt, rg = stats.rankdata(Xt[:, keep_t], axis=0), stats.rankdata(Xg[:, keep_g], axis=0)
    C = np.corrcoef(np.hstack([rt, rg]).T); nt = rt.shape[1]; cross = np.abs(C[:nt, nt:])
    zt, zg = StandardScaler().fit_transform(Xt[:, keep_t]), StandardScaler().fit_transform(Xg[:, keep_g])
    k = 5; cc = CCA(n_components=k, max_iter=2000).fit(zt, zg); u, v = cc.transform(zt, zg)
    canon = [np.corrcoef(u[:, i], v[:, i])[0, 1] for i in range(k)]
    Z = StandardScaler().fit_transform(np.hstack([Xt[:, keep_t], Xg[:, keep_g]]))
    vif = np.diag(np.linalg.pinv(np.corrcoef(Z.T) + 1e-6 * np.eye(Z.shape[1])))  # ridge-stabilised VIF
    e_cols = geo_cols(fn=[2]); e_local = [np.where(np.where(keep_g)[0] == c)[0][0] + nt for c in e_cols if keep_g[c]]
    b_cols = topo_cols(betti=True); b_local = [np.where(np.where(keep_t)[0] == c)[0][0] for c in b_cols if keep_t[c]]
    return pd.Series({
        "max |Spearman| between any topo/geo pair": cross.max(),
        "share of geo features with |rho|>0.8 to some topo feature": (cross.max(0) > 0.8).mean(),
        "share of topo features with |rho|>0.8 to some geo feature": (cross.max(1) > 0.8).mean(),
        **{f"canonical corr #{i + 1}": c for i, c in enumerate(canon)},
        "median VIF (all 184)": np.median(vif), "max VIF": vif.max(), "share VIF>10": (vif > 10).mean(),
        "median VIF, Euler (L0) features": np.median(vif[e_local]) if e_local else np.nan,
        "median VIF, Betti-curve amplitudes": np.median(vif[b_local]) if b_local else np.nan,
        "|rho| max: Euler-L0 feats vs Betti-curve amps": np.abs(C[np.ix_(b_local, e_local)]).max() if e_local and b_local else np.nan})


def redundancy_ablation(Xt, Xg, y, splits, a, rng):
    cases = {
        "Full GeoTop (184)": (Xt, Xg),
        "Drop Euler (L0) from geometric block": (Xt, Xg[:, np.setdiff1d(np.arange(Xg.shape[1]), geo_cols(fn=[2]))]),
        "Drop Betti-curve amplitudes from topological block": (Xt[:, np.setdiff1d(np.arange(Xt.shape[1]), topo_cols(betti=True))], Xg),
        "Drop BOTH Euler-linked sets": (Xt[:, np.setdiff1d(np.arange(Xt.shape[1]), topo_cols(betti=True))],
                                        Xg[:, np.setdiff1d(np.arange(Xg.shape[1]), geo_cols(fn=[2]))]),
        "TDA + Area only": (Xt, Xg[:, geo_cols(fn=[0])]),
        "TDA + Perimeter only": (Xt, Xg[:, geo_cols(fn=[1])]),
        # controls that add 120 columns but (by construction) no NEW information:
        "Control: TDA + 120 column-permuted geo cols (dimension control)": (Xt, np.column_stack([rng.permutation(Xg[:, j]) for j in range(Xg.shape[1])])),
        "Control: TDA + duplicated TDA cols (pure redundancy)": (Xt, np.tile(Xt, (1, int(np.ceil(Xg.shape[1] / Xt.shape[1]))))[:, :Xg.shape[1]]),
        "TDA only": (Xt, None), "LKC only": (None, Xg)}
    Ps = {}
    for n, (t, g) in cases.items():
        if g is None: Ps[n] = run_model(lambda Xt_, Xg_, y_, tr, te, a_, s: m_concat(Xt_, Xg_[:, :0], y_, tr, te, a_, s), t, np.zeros((len(y), 0)), y, splits, a)
        elif t is None: Ps[n] = run_model(lambda Xt_, Xg_, y_, tr, te, a_, s: m_concat(Xt_[:, :0], Xg_, y_, tr, te, a_, s), np.zeros((len(y), 0)), g, y, splits, a)
        else: Ps[n] = run_model(m_concat, t, g, y, splits, a)
    return pd.DataFrame({n: summarise(P, y, splits, None if n.startswith("Full") else Ps["Full GeoTop (184)"]) for n, P in Ps.items()}).T


def channel_ablation(Xt, Xg, y, splits, a, extra):
    L = LAYOUT; cases = {}
    for name, ch in {"grayscale only": [0], "R only": [1], "G only": [2], "B only": [3], "RGB only (no grayscale)": [1, 2, 3],
                     "grayscale + RGB (current, 4 blocks)": [0, 1, 2, 3]}.items():
        cases[name] = (Xt[:, topo_cols(ch=ch)], Xg[:, geo_cols(ch=ch)])
    for name, (t, g) in extra.items(): cases[f"{name} (re-extracted)"] = (t, g)
    Ps = {n: run_model(m_concat, t, g, y, splits, a) for n, (t, g) in cases.items()}
    ref = "grayscale + RGB (current, 4 blocks)"
    return pd.DataFrame({n: summarise(P, y, splits, None if n == ref else Ps[ref]) for n, P in Ps.items()}).T


# ----------------------------------------------------------------- simulation (smoke test only)
def simulate(N=600, seed=0):
    r = np.random.default_rng(seed); y = (r.random(N) < 0.45).astype(int)
    lat = r.normal(size=(N, 6)) + y[:, None] * np.array([.6, .5, .4, .3, .2, .1])
    Xt = lat @ r.normal(size=(6, 64)) * .5 + r.normal(size=(N, 64))
    Xg = lat @ r.normal(size=(6, 120)) * .5 + 0.3 * y[:, None] * r.normal(size=(1, 120)) + r.normal(size=(N, 120))
    groups = np.repeat(np.arange(N // 2), 2)[:N]; is_test = np.arange(N) >= int(0.8 * N)
    return Xt, Xg, y, groups, is_test, r.normal(size=(N, 50)) + y[:, None] * 0.05   # pairs of near-duplicates


def load_npz(p):
    d = np.load(p, allow_pickle=True)
    return (d["X_topo"], d["X_geo"], d["y"].astype(int), (d["groups"] if "groups" in d.files else None),
            (d["is_test"].astype(bool) if "is_test" in d.files else None), (d["X_hog"] if "X_hog" in d.files else None))


def to_full(probs, y, is_test):
    """accept (N_test,) in image_order test-row order, or (N,) with NaN off-test; return (1,N) array"""
    probs = np.asarray(probs, float).ravel(); P = np.full((1, len(y)), np.nan)
    if len(probs) == len(y): P[0] = probs
    elif len(probs) == int(is_test.sum()): P[0, np.where(is_test)[0]] = probs
    else: raise ValueError(f"predictions have {len(probs)} entries; expected {len(y)} or {int(is_test.sum())}")
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features"); ap.add_argument("--out", default="results"); ap.add_argument("--groups")
    ap.add_argument("--protocol", choices=["official", "pooled"])
    ap.add_argument("--repeats", type=int, default=10, help="pooled protocol only"); ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--layout")
    ap.add_argument("--extra_features", nargs="*", default=[], help="name=file.npz (e.g. hsv=geotop_hsv.npz)")
    ap.add_argument("--external_preds", nargs="*", default=[], help="name=file.npz with key 'probs' (official protocol)")
    ap.add_argument("--simulate", action="store_true"); a = ap.parse_args()
    if a.layout: LAYOUT.update(json.load(open(a.layout)))
    if a.simulate:
        Xt, Xg, y, groups, is_test, hog = simulate(); pre = "SIMULATED_"
        print("=" * 78 + "\n SIMULATED DATA -- SMOKE TEST ONLY. DO NOT REPORT ANY NUMBER FROM THIS RUN.\n" + "=" * 78)
    else:
        Xt, Xg, y, groups, is_test, hog = load_npz(a.features); pre = ""
        if a.groups: groups = np.load(a.groups)
    proto = a.protocol or ("official" if is_test is not None else "pooled")
    if proto == "official" and is_test is None: raise SystemExit("official protocol needs is_test in the npz (use extract_features.py)")
    os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed); EXTRA["hog"] = hog
    assert Xt.shape[1] == LAYOUT["n_ch"] * LAYOUT["topo_per_ch"] and Xg.shape[1] == LAYOUT["n_ch"] * LAYOUT["geo_per_ch"], "unexpected feature counts"
    save = lambda df, n: (df.to_csv(os.path.join(a.out, pre + n + ".csv")), print(f"\n## [{proto}] {n}\n{df.drop(columns=['_mean_acc'], errors='ignore').to_string()}"))
    REF = "Early: concat (current GeoTop)"

    if proto == "official":
        splits = [dict(repeat=0, fold=0, train=np.where(~is_test)[0], test=np.where(is_test)[0])]
        print(f"official split: train {len(splits[0]['train'])} / test {len(splits[0]['test'])}; malignant share {y[~is_test].mean():.3f} / {y[is_test].mean():.3f}")
    else:
        splits = make_splits(y, None, a.repeats, a.seed)
    json.dump([{k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in sp.items()} for sp in splits], open(os.path.join(a.out, pre + "splits.json"), "w"))

    save(pd.DataFrame(block_correlation(Xt, Xg, y)).T, "block_correlation")
    fus, Ps = table(FUSION, Xt, Xg, y, splits, a, REF); save(fus, "fusion_strategies")                    # R3 pt.1
    save(fus[[c for c in ["acc", "sens", "spec", "bal_acc", "auroc"]]], "clinical_metrics")                # R3 pt.7
    save(redundancy_ablation(Xt, Xg, y, splits, a, rng), "redundancy_ablations")                            # R3 pt.4
    extra = {}
    for kv in a.extra_features:
        n, p = kv.split("="); t, g, yy = load_npz(p)[:3]; assert (yy == y).all(), "extra features must have the same row order"; extra[n] = (t, g)
    save(channel_ablation(Xt, Xg, y, splits, a, extra), "channel_ablation")                                # R3 pt.5

    # ---- R3 pt.6: leakage --------------------------------------------------------------------------
    if groups is not None and proto == "official":
        tr, te = splits[0]["train"], splits[0]["test"]; gtr, gte = set(groups[tr]), set(groups[te])
        cont = np.array([groups[i] in gtr for i in te]); print(f"\n[leakage] {int(cont.sum())}/{len(te)} test images ({100*cont.mean():.1f}%) have a near-duplicate/same-lesion partner in TRAIN")
        rows = {}
        for n in ["TDA only", "LKC only", REF]:
            for lab, msk in (("clean test images", ~cont), ("contaminated test images", cont)):
                if msk.sum() > 10 and len(np.unique(y[te][msk])) > 1:
                    o = summarise_single(Ps[n], y, dict(test=te[msk])); rows[f"{n} | {lab} (n={int(msk.sum())})"] = {k: o[k] for k in ("acc", "auroc")}
        tr_clean = np.array([i for i in tr if groups[i] not in gte])
        Pd = run_model(m_concat, Xt, Xg, y, [dict(repeat=0, fold=0, train=tr_clean, test=te)], a)
        o = summarise_single(Pd, y, splits[0], Ps[REF]); rows[f"GeoTop retrained on de-duplicated train (n_train={len(tr_clean)}), full test"] = {k: o[k] for k in ("acc", "auroc", "Δacc vs ref [95% CI]", "p (paired bootstrap)")}
        save(pd.DataFrame(rows).T, "leakage_official")
    elif groups is not None:
        gsp = make_splits(y, groups, a.repeats, a.seed); Pg = run_model(m_concat, Xt, Xg, y, gsp, a)
        save(pd.DataFrame({"random image-level splits": summarise(Ps[REF], y, splits), f"group-aware splits ({len(np.unique(groups))} groups / {len(y)} images)": summarise(Pg, y, gsp)}).T, "leakage_gap")
    else:
        print("\n[!] no groups supplied -> leakage analysis skipped (build them with make_groups.py)")

    # ---- R3 pt.8: matched baseline table (official protocol: same split for everyone) ----------------
    base = {"GeoTop (RF on 184)": m_concat, "GeoTop feats + boosted trees": m_boost}
    if hog is not None: base["HOG + SVM (RBF)"] = m_hog_svm
    tab, Pb = table(base, Xt, Xg, y, splits, a, "GeoTop (RF on 184)")
    rows = {n: dict(tab.loc[n]) for n in tab.index}
    if a.external_preds:
        if proto != "official": print("[!] --external_preds is only supported with the official protocol; skipped")
        else:
            for kv in a.external_preds:
                n, p = kv.split("="); P = to_full(np.load(p)["probs"], y, is_test); rows[n] = summarise_single(P, y, splits[0], Pb["GeoTop (RF on 184)"])
    save(pd.DataFrame(rows).T, "matched_baselines")


if __name__ == "__main__":
    if "--selftest" in os.sys.argv:                     # verifies the fast metrics against scikit-learn
        r = np.random.default_rng(0); yy = (r.random(300) < .4).astype(int); pp = np.round(np.clip(.3 * yy + r.random(300) * .7, 0, 1), 2)
        m = metrics(yy, pp); tn, fp, fn, tp = confusion_matrix(yy, pp >= .5).ravel()
        assert abs(m["auroc"] - roc_auc_score(yy, pp)) < 1e-12 and abs(m["f1"] - f1_score(yy, pp >= .5)) < 1e-12
        assert abs(m["prec"] - precision_score(yy, pp >= .5)) < 1e-12 and abs(m["sens"] - tp / (tp + fn)) < 1e-12
        print("metrics self-test OK (incl. tied scores)"); raise SystemExit
    main()
