#!/usr/bin/env python3
"""
Near-duplicate / same-lesion grouping for group-aware splits (proxy for patient-level splitting).

The Kaggle 'Skin Cancer: Malignant vs. Benign' images carry no patient or lesion IDs, so true
patient-level splitting is impossible. Next best, and cheap: detect visually near-identical images
(same lesion photographed twice, crops, flips, rotations, re-encodings) and keep every such cluster
on ONE side of every split.

    python make_groups.py --images data/all_images/ --labels labels.npy --out groups.npy [--max_hamming 24]

The image order must match the row order of your feature matrix (see --order file list).
Reports how many images sit in a cluster of size > 1: that number belongs in the paper.
"""
import argparse, csv, os, numpy as np
from PIL import Image


def dhash(path, size=16):
    im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16); return (a[:, 1:] > a[:, :-1]).ravel()          # size*size bits


def dihedral(a, k):                     # 8 rotations/flips of a (H,W) array
    a = np.rot90(a, k % 4); return np.fliplr(a) if k >= 4 else a


def hashes_all_transforms(path, size=16):
    im = Image.open(path).convert("L"); out = []
    for k in range(8):
        arr = dihedral(np.asarray(im), k)
        small = np.asarray(Image.fromarray(np.ascontiguousarray(arr)).resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
        out.append((small[:, 1:] > small[:, :-1]).ravel())
    return np.array(out)                # (8, bits)


def cluster(paths, max_hamming=24, size=16):
    H = np.array([hashes_all_transforms(p, size) for p in paths])         # (N, 8, bits)
    base = H[:, 0, :].astype(np.float32); n = len(paths); parent = np.arange(n)
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for k in range(8):
        T = H[:, k, :].astype(np.float32)
        D = base @ (1 - T).T + (1 - base) @ T.T                            # Hamming(base_i, transformed_j)
        ii, jj = np.where(np.triu(D <= max_hamming, 1))
        for i, j in zip(ii, jj):
            parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(n)]); _, gid = np.unique(roots, return_inverse=True); return gid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", required=True, help="*_image_order.csv written by extract_features.py (path,label,is_test) -- keeps row order identical")
    ap.add_argument("--out", default="groups.npy")
    ap.add_argument("--max_hamming", type=int, default=24, help="of 256 bits; 24 ~ 9%% of bits. Inspect clusters visually.")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.order))); paths = [r["path"] for r in rows]
    is_test = np.array([r["is_test"] in ("True", "true", "1") for r in rows]); y = np.array([int(r["label"]) for r in rows])
    g = cluster(paths, a.max_hamming); np.save(a.out, g); sizes = np.bincount(g)
    print(f"{len(paths)} images -> {len(sizes)} groups; {int((sizes[g] > 1).sum())} images ({100 * (sizes[g] > 1).mean():.1f}%) "
          f"are in a cluster of size > 1; largest cluster = {sizes.max()}")
    gtr = set(g[~is_test]); cont = np.array([gi in gtr for gi in g[is_test]])
    print(f"official split: {int(cont.sum())}/{int(is_test.sum())} TEST images ({100 * cont.mean():.1f}%) have a near-duplicate in TRAIN")
    mixed = [k for k in range(len(sizes)) if sizes[k] > 1 and len(set(y[g == k])) > 1]
    print(f"{len(mixed)} clusters mix benign and malignant labels (likely false merges: lower --max_hamming or inspect them)")
    with open(os.path.splitext(a.out)[0] + "_clusters.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["group", "size", "path", "label", "is_test"])
        for k in np.where(sizes > 1)[0]:
            for i in np.where(g == k)[0]: w.writerow([k, sizes[k], paths[i], y[i], is_test[i]])
    print("wrote", os.path.splitext(a.out)[0] + "_clusters.csv", "-> open a sample of these images and check them by eye before trusting the threshold.")
