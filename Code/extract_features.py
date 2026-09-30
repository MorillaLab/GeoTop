#!/usr/bin/env python3
"""
Build the GeoTop feature matrices from the image folders

    <data_root>/train/benign   <data_root>/train/malignant
    <data_root>/test/benign    <data_root>/test/malignant

    python extract_features.py --data_root ./data --out geotop_features.npz            # default extractor
    python extract_features.py --data_root ./data --out geotop_features.npz \
           --user_extractor my_pipeline.py                                              # YOUR original code (recommended)
    python extract_features.py --data_root ./data --out geotop_hsv.npz --colorspace hsv # for the colour-space ablation

Output npz: X_topo (N,64)  X_geo (N,120)  X_hog (N,d)  y (N,) [benign=0, malignant=1]  is_test (N,) bool
plus image_order.csv (path,label,is_test): row order of EVERY array here. Your CNN/ViT code must use this order.

--user_extractor FILE.py must define   extract(img_rgb_uint8) -> (topo (64,), geo (120,))
with the column layout documented in geotop_revision_experiments.py (per channel gray,R,G,B; topo 16/channel =
[H0: 7 amplitudes, entropy][H1: 7 amplitudes, entropy]; geo 30/channel = [Area,Perimeter,Euler] x [f, df] x 5 stats).
Wrap your feature_utils functions in that signature. Only your own extractor guarantees that the numbers correspond to
the ones already in the paper; the default extractor below re-implements the pipeline as DESCRIBED in the manuscript.

Default-extractor caveats (state them, or use your own code):
  * geometry (Area/Perimeter/Euler curves + 5 statistics) is fully implemented here and was unit-tested against
    skimage.measure.euler_number.
  * the topological block needs giotto-tda (gtda) and was NOT executed by the author of this script (library
    unavailable in the sandbox): run `--selftest` first, and compare a few rows against your existing features.
"""
import argparse, csv, importlib.util, os, sys
import numpy as np
from PIL import Image
from joblib import Parallel, delayed

_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy>=2 renamed trapz
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
AMPLITUDES = [("bottleneck", {}), ("wasserstein", {"p": 1}), ("wasserstein", {"p": 2}), ("landscape", {}),
              ("betti", {}), ("persistence_image", {}), ("silhouette", {})]      # order == LAYOUT (Betti-curve = index 4)


# ----------------------------------------------------------------------------- data listing
def list_images(root):
    rows = []
    for split in ("train", "test"):
        for lab, name in ((0, "benign"), (1, "malignant")):
            d = os.path.join(root, split, name)
            if not os.path.isdir(d):
                sys.exit(f"missing folder: {d}")
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(EXT):
                    rows.append((os.path.join(d, f), lab, split == "test"))
    return rows


def load_rgb(path, size=224):
    im = Image.open(path).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def channels(img, colorspace="rgb"):
    """gray + three channels of the chosen colour space, each float in [0,1]"""
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255
    if colorspace == "rgb":
        c = [img[..., i].astype(np.float64) / 255 for i in range(3)]
    elif colorspace == "hsv":
        h = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float64); c = [h[..., 0] / 179, h[..., 1] / 255, h[..., 2] / 255]
    elif colorspace == "lab":
        l = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64) / 255; c = [l[..., i] for i in range(3)]
    else:
        raise ValueError(colorspace)
    return [gray] + c


# ----------------------------------------------------------------------------- geometry (LKC) block
def area_perimeter_euler(mask):
    """mask: 2-D bool. Pixels = closed unit squares. Returns (area, boundary length, Euler characteristic chi=V-E+F).
    All sums are int64 (uint8 sums overflow on subtraction)."""
    Z = np.pad(mask, 1).astype(np.uint8); i64 = np.int64
    area = Z.sum(dtype=i64)
    per = (Z[:-1, 1:-1] ^ Z[1:, 1:-1]).sum(dtype=i64) + (Z[1:-1, :-1] ^ Z[1:-1, 1:]).sum(dtype=i64)        # boundary edges
    Eh = (Z[:-1, 1:-1] | Z[1:, 1:-1]).sum(dtype=i64); Ev = (Z[1:-1, :-1] | Z[1:-1, 1:]).sum(dtype=i64)       # covered unit edges
    V = (Z[:-1, :-1] | Z[:-1, 1:] | Z[1:, :-1] | Z[1:, 1:]).sum(dtype=i64)                                   # covered vertices
    return int(area), int(per), int(V - (Eh + Ev) + area)


def lkc_curves(ch, n_thr=200):
    """Area, Perimeter, Euler characteristic of the excursion sets X_t = {ch >= t}, t on 200 equidistant levels.
    Pixels are closed unit squares: chi = V - E + F (== 8-connectivity Euler number); perimeter = boundary edges."""
    ts = np.linspace(ch.min(), ch.max(), n_thr); A = np.empty(n_thr); P = np.empty(n_thr); E = np.empty(n_thr)
    for k, t in enumerate(ts):
        A[k], P[k], E[k] = area_perimeter_euler(ch >= t)
    m = ch.shape[0]
    return ts, {"A": A / m ** 2, "P": (0.5 * P) / m ** 2, "E": E / m ** 2}     # L2 = Area, L1 = V1 = Perimeter/2, L0 = chi; /m^2 scaling


def five_stats(c, t):
    a = np.abs(c); s = a.sum()
    p = a / s if s > 0 else np.ones_like(a) / len(a)
    ent = -np.sum(p[p > 0] * np.log(p[p > 0]))
    return [np.sqrt(_trapz(c ** 2, t)), _trapz(c, t), c.sum(), ent, float((a > 1e-12).sum())]   # L2 norm, integral, sum, entropy, #non-zero


def geo_features(chs):
    out = []
    for ch in chs:
        ts, cur = lkc_curves(ch)
        for name in ("A", "P", "E"):
            c = cur[name]; out += five_stats(c, ts) + five_stats(np.gradient(c, ts), ts)
    return np.array(out)                                                          # 4 x (3 x 2 x 5) = 120


# ----------------------------------------------------------------------------- topology block (needs giotto-tda)
def topo_features_batch(imgs_by_channel, normalize="zscore", filtration="sublevel", n_jobs=1):
    """imgs_by_channel: list of 4 arrays (N,H,W). returns (N,64)."""
    from gtda.homology import CubicalPersistence
    from gtda.diagrams import PersistenceEntropy, Amplitude
    feats = []
    for X in imgs_by_channel:
        X = X.astype(np.float64)
        if normalize == "zscore":
            X = (X - X.mean((1, 2), keepdims=True)) / (X.std((1, 2), keepdims=True) + 1e-12)
        if filtration == "superlevel":
            X = -X
        D = CubicalPersistence(homology_dimensions=(0, 1), n_jobs=n_jobs).fit_transform(X)
        ent = PersistenceEntropy().fit_transform(D)                                # (N,2)
        amps = [Amplitude(metric=m, metric_params=(p or None), n_jobs=n_jobs).fit_transform(D) for m, p in AMPLITUDES]   # 7 x (N,2)
        blk = np.hstack([np.column_stack([amps[j][:, d] for j in range(7)] + [ent[:, d]]) for d in range(2)])      # (N,16)
        feats.append(blk)
    return np.hstack(feats)


def _geo_one(path, colorspace):
    return geo_features(channels(load_rgb(path), colorspace))


def load_user_extractor(path):
    spec = importlib.util.spec_from_file_location("user_extractor", path); mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod); return mod.extract


def _user_one(path, extractor_path):
    t, g = load_user_extractor(extractor_path)(load_rgb(path)); return np.asarray(t, float), np.asarray(g, float)


def hog_one(path):
    from skimage.feature import hog
    import cv2
    g = cv2.cvtColor(load_rgb(path), cv2.COLOR_RGB2GRAY); g = cv2.resize(g, (128, 128))
    return hog(g, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))


def selftest():
    from skimage.measure import euler_number
    r = np.random.default_rng(0)
    for _ in range(5):
        Z = (r.random((40, 40)) > 0.55)
        chi = area_perimeter_euler(Z)[2]
        assert chi == euler_number(Z, connectivity=2), (chi, euler_number(Z, connectivity=2))
    Z = np.zeros((10, 10), bool); Z[2:8, 2:8] = True; Z[4:6, 4:6] = False        # square with a hole: chi = 0
    assert area_perimeter_euler(Z) == (32, 24 + 8, 0), area_perimeter_euler(Z)
    print("geometry self-test OK (Euler characteristic == skimage 8-connectivity).")
    try:
        from gtda.homology import CubicalPersistence  # noqa
        X = r.random((2, 32, 32)); f = topo_features_batch([X] * 4); assert f.shape == (2, 64); print("gtda topology self-test OK", f.shape)
    except ImportError:
        print("giotto-tda not installed: topology block cannot run here (pip install giotto-tda).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root"); ap.add_argument("--out", default="geotop_features.npz")
    ap.add_argument("--colorspace", default="rgb", choices=["rgb", "hsv", "lab"])
    ap.add_argument("--user_extractor"); ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--normalize", default="zscore", choices=["zscore", "none"])
    ap.add_argument("--filtration", default="sublevel", choices=["sublevel", "superlevel"])
    ap.add_argument("--limit", type=int, help="debug: only the first N images per folder")
    ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest:
        return selftest()
    rows = list_images(a.data_root)
    if a.limit:
        keep = []; cnt = {}
        for r in rows:
            k = (os.path.dirname(r[0])); cnt[k] = cnt.get(k, 0) + 1
            if cnt[k] <= a.limit: keep.append(r)
        rows = keep
    paths = [r[0] for r in rows]; y = np.array([r[1] for r in rows]); is_test = np.array([r[2] for r in rows])
    print(f"{len(rows)} images | train {int((~is_test).sum())} (malignant {int(y[~is_test].sum())}) | test {int(is_test.sum())} (malignant {int(y[is_test].sum())})")
    if a.user_extractor:
        res = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(_user_one)(p, a.user_extractor) for p in paths)
        Xt, Xg = np.vstack([r[0] for r in res]), np.vstack([r[1] for r in res])
    else:
        Xg = np.vstack(Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(_geo_one)(p, a.colorspace) for p in paths))
        chunks = [paths[i:i + 200] for i in range(0, len(paths), 200)]; Xt = []
        for ch in chunks:
            stacks = [[] for _ in range(4)]
            for p in ch:
                for k, c in enumerate(channels(load_rgb(p), a.colorspace)): stacks[k].append(c)
            Xt.append(topo_features_batch([np.stack(s) for s in stacks], a.normalize, a.filtration, n_jobs=1 if a.n_jobs == -1 else a.n_jobs))
        Xt = np.vstack(Xt)
    Xh = np.vstack(Parallel(n_jobs=a.n_jobs)(delayed(hog_one)(p) for p in paths))
    assert Xt.shape == (len(rows), 64) and Xg.shape == (len(rows), 120), (Xt.shape, Xg.shape)
    np.savez_compressed(a.out, X_topo=Xt, X_geo=Xg, X_hog=Xh, y=y, is_test=is_test, paths=np.array(paths))
    with open(os.path.splitext(a.out)[0] + "_image_order.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "label", "is_test"]); w.writerows(rows)
    print("saved", a.out, "and the image-order csv")


if __name__ == "__main__":
    main()
