"""Inference: photo(s) of one sample + scale (px/mm of the stored file) -> valid grain-size curve.

Model = frozen MAE-B encoder (CLS + mean patch token, 4 dihedral views) + ridge head fitted on all labelled
tiles, the inference model of notebooks/cv-v3.ipynb (cell "inference"). The ridge head is fitted on the first
call and cached in --cache.

    python scripts/predict.py --photos a.jpg b.jpg --ppm 19.5 \
        --data-dir /kaggle/input/competitions/soil-grain-size-from-photos --cache /kaggle/working
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from soilgrain.config import load_config
from soilgrain.data import build_photo_table, load_labels
from soilgrain.metric import GRAIN_COLS, enforce_monotone
from soilgrain.tiling import load_at_ppm, make_tiles, tiles_of

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def dihedral_views(x):
    return [x, x.flip(-1), x.flip(-2), x.rot90(1, (-2, -1))]


def build_encoder(name):
    enc = timm.create_model(name, pretrained=True, num_classes=0, img_size=224).to(DEVICE).eval()
    pc = timm.models.get_pretrained_cfg(name)
    mean = torch.tensor(pc.mean, device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(pc.std, device=DEVICE).view(1, 3, 1, 1)
    return enc, mean, std


@torch.no_grad()
def encode(enc, mean, std, tiles_u8, bs=64):
    """(n, 224, 224, 3) uint8 -> (n, 4 views, D) features, CLS + mean of patch tokens."""
    out = []
    for i in range(0, len(tiles_u8), bs):
        x = (torch.from_numpy(tiles_u8[i:i + bs]).to(DEVICE).permute(0, 3, 1, 2).float() / 255 - mean) / std
        views = []
        with torch.autocast("cuda", enabled=DEVICE == "cuda"):
            for v in dihedral_views(x):
                t = enc.forward_features(v).float()
                views.append(torch.cat([t[:, 0], t[:, enc.num_prefix_tokens:].mean(1)], dim=-1))
        out.append(torch.stack(views, 1).cpu().numpy())
    return np.concatenate(out)


def fit_head(cfg, data_dir, cache, enc, mean, std):
    photos = build_photo_table(data_dir, cfg.data.train_img, cfg.data.test_img, cfg.data.ppm)
    labels = load_labels(data_dir, cfg.data.labels)
    idx = make_tiles(photos, (224,), Path(cache) / "tiles", cfg.tiling.px_per_mm)[224]
    tr = idx[(idx["split"] == "train") & idx["sample_id"].isin(labels.index)]
    from PIL import Image
    X = encode(enc, mean, std, np.stack([np.asarray(Image.open(p).convert("RGB")) for p in tr["path"]]))
    n, v, d = X.shape
    Y = np.repeat(labels.loc[tr["sample_id"], GRAIN_COLS].to_numpy(np.float64), v, axis=0)
    sc = StandardScaler().fit(X.reshape(n * v, d))
    ridge = Ridge(alpha=cfg.inference.default_alpha).fit(sc.transform(X.reshape(n * v, d)), Y)
    return sc, ridge


def predict_curve(photo_paths, ppm, enc, mean, std, sc, ridge, px_per_mm=4):
    """Resample to px_per_mm, cut 224-px tiles, encode, ridge, mean per photo, mean over photos, valid curve."""
    ppms = [ppm] * len(photo_paths) if np.isscalar(ppm) else list(ppm)
    per_photo = []
    for p, s in zip(photo_paths, ppms):
        img, _ = load_at_ppm(p, s, px_per_mm)
        tiles = tiles_of(img)
        if len(tiles) == 0:
            continue
        f_ = encode(enc, mean, std, tiles)
        n, v, d = f_.shape
        per_photo.append(ridge.predict(sc.transform(f_.reshape(n * v, d))).reshape(n, v, -1).mean(1).mean(0))
    if not per_photo:
        raise ValueError("no photo is large enough for one 56 mm tile")
    return enforce_monotone(np.mean(per_photo, 0))[0]


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", nargs="+", required=True, help="photos of ONE physical sample")
    ap.add_argument("--ppm", nargs="+", type=float, required=True, help="px/mm of each stored file (one value = all)")
    ap.add_argument("--data-dir", default=cfg.data.dir, help="competition data, used to fit the ridge head")
    ap.add_argument("--cache", default=cfg.data.output_dir)
    args = ap.parse_args()

    enc, mean, std = build_encoder(cfg.inference.backbone)
    head_path = Path(args.cache) / "inference_head.pkl"
    if head_path.exists():
        sc, ridge = pickle.loads(head_path.read_bytes())
    else:
        sc, ridge = fit_head(cfg, args.data_dir, args.cache, enc, mean, std)
        head_path.write_bytes(pickle.dumps((sc, ridge)))
    ppm = args.ppm[0] if len(args.ppm) == 1 else args.ppm
    curve = predict_curve(args.photos, ppm, enc, mean, std, sc, ridge, cfg.tiling.px_per_mm)
    for c, v in zip(GRAIN_COLS, curve):
        print(f"{c:>7} mm: {v:6.2f} % passing")


if __name__ == "__main__":
    main()
