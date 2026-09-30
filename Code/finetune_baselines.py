#!/usr/bin/env python3
"""
Fine-tune ResNet-18 or ViT-B/16 on data/train, evaluate on data/test, save predictions for
geotop_revision_experiments.py's --external_preds.

NOT TESTED in the environment that wrote this file (no torch/GPU, no internet to fetch pretrained
weights there). It follows the standard torchvision fine-tuning recipe; sanity-check the first run
with --epochs 1 --limit 40 before committing to a full run.

    pip install torch torchvision           # needs internet once, to download ImageNet weights

    python finetune_baselines.py --image_order geotop_features_image_order.csv --model resnet18 \\
           --epochs 15 --out resnet18_preds.npz
    python finetune_baselines.py --image_order geotop_features_image_order.csv --model vit_b_16 \\
           --epochs 15 --lr 3e-5 --batch_size 16 --out vit_preds.npz

Then:
    python geotop_revision_experiments.py --features geotop_features.npz --groups groups.npy \\
           --out results_official --external_preds resnet18=resnet18_preds.npz vit=vit_preds.npz

image_order.csv is the file extract_features.py writes (columns: path,label,is_test). Row order there
IS the row order of your feature matrix; this script trains/predicts using paths straight from that
file, so predictions line up with the right rows automatically -- do not re-sort or re-list images
yourself.

Output npz key "probs": P(malignant) for the TEST rows only, in the same order they appear in
image_order.csv (i.e. after filtering to is_test==True). geotop_revision_experiments.py accepts
exactly this shape (N_test,).
"""
import argparse, csv, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def read_order(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["label"] = int(r["label"]); r["is_test"] = r["is_test"] in ("True", "true", "1")
    return rows


class ImgDataset(Dataset):
    def __init__(self, rows, img_size, train):
        import torchvision.transforms as T
        self.rows = rows
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]          # ImageNet stats (torchvision pretrained weights)
        if train:
            self.tf = T.Compose([T.Resize((img_size, img_size)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                                  T.RandomRotation(20), T.ColorJitter(0.15, 0.15, 0.1),
                                  T.ToTensor(), T.Normalize(mean, std)])
        else:
            self.tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(mean, std)])

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]; img = Image.open(r["path"]).convert("RGB")
        return self.tf(img), r["label"]


def build_model(name, freeze_backbone):
    if name == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for p in m.parameters(): p.requires_grad_(False)
        m.fc = nn.Linear(m.fc.in_features, 2); img_size = 224
    elif name == "vit_b_16":
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for p in m.parameters(): p.requires_grad_(False)
        m.heads.head = nn.Linear(m.heads.head.in_features, 2); img_size = 224
    else:
        raise ValueError(name)
    return m, img_size


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    tot_loss = n = correct = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if train: optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(x); loss = criterion(out, y)
                if train:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                out = model(x); loss = criterion(out, y)
                if train: loss.backward(); optimizer.step()
            tot_loss += loss.item() * len(y); n += len(y); correct += (out.argmax(1) == y).sum().item()
    return tot_loss / n, correct / n


def predict_probs(model, loader, device):
    model.eval(); out = []
    with torch.no_grad():
        for x, _ in loader:
            p = torch.softmax(model(x.to(device)), dim=1)[:, 1]           # P(malignant); label 1 = malignant
            out.append(p.cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_order", required=True)
    ap.add_argument("--model", choices=["resnet18", "vit_b_16"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4, help="use a smaller LR (e.g. 3e-5) for vit_b_16")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.1, help="carved out of TRAIN only, for early stopping; test is never touched")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--freeze_backbone", action="store_true", help="linear probe only -- fast sanity check, not the reported baseline")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="debug: use only the first N train / N test rows")
    a = ap.parse_args()
    set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)
    if device == "cpu":
        print("[!] no GPU found: full fine-tuning will be slow. Consider --freeze_backbone for a quick check, "
              "or run on a machine with a GPU for the real baseline.")

    rows = read_order(a.image_order)
    train_rows = [r for r in rows if not r["is_test"]]; test_rows = [r for r in rows if r["is_test"]]
    if a.limit:
        train_rows, test_rows = train_rows[:a.limit], test_rows[:a.limit]
    rng = random.Random(a.seed); idx = list(range(len(train_rows))); rng.shuffle(idx)
    n_val = max(1, int(a.val_frac * len(idx))); val_idx, tr_idx = set(idx[:n_val]), idx[n_val:]
    tr_rows = [train_rows[i] for i in tr_idx]; val_rows = [train_rows[i] for i in val_idx]
    print(f"train {len(tr_rows)} | val (held out from train) {len(val_rows)} | test {len(test_rows)}")

    model, img_size = build_model(a.model, a.freeze_backbone); model.to(device)
    tr_ds, val_ds, te_ds = (ImgDataset(r, img_size, train=(r is tr_rows)) for r in (tr_rows, val_rows, test_rows))
    labels = np.array([r["label"] for r in tr_rows])
    w = np.where(labels == 1, (labels == 0).sum(), (labels == 1).sum()).astype(float)   # inverse-frequency sample weights
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    tr_loader = DataLoader(tr_ds, batch_size=a.batch_size, sampler=sampler, num_workers=a.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=a.batch_size, num_workers=a.num_workers)
    te_loader = DataLoader(te_ds, batch_size=a.batch_size, num_workers=a.num_workers)

    criterion = nn.CrossEntropyLoss()                                    # sampler already rebalances the classes
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=a.lr, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(a.epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, tr_loader, device, criterion, optimizer, scaler)
        val_loss, val_acc = run_epoch(model, val_loader, device, criterion)
        scheduler.step()
        print(f"epoch {ep+1}/{a.epochs}  train loss {tr_loss:.3f} acc {tr_acc:.3f} | val loss {val_loss:.3f} acc {val_acc:.3f} | {time.time()-t0:.0f}s")
        if val_loss < best_val - 1e-4:
            best_val, best_state, bad = val_loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= a.patience:
                print(f"early stopping at epoch {ep+1} (no val improvement for {a.patience} epochs)"); break
    if best_state is not None:
        model.load_state_dict(best_state)

    probs = predict_probs(model, te_loader, device)                      # order == test_rows == is_test rows of image_order.csv
    assert len(probs) == len(test_rows)
    np.savez(a.out, probs=probs)
    y_test = np.array([r["label"] for r in test_rows]); acc = ((probs >= 0.5).astype(int) == y_test).mean()
    print(f"saved {a.out} | test accuracy at 0.5 threshold: {acc:.3f} (rough check only -- the paper's CIs come from geotop_revision_experiments.py)")


if __name__ == "__main__":
    main()
