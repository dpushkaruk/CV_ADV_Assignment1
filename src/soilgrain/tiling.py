"""Tiles of a fixed physical size, cached once.

Every photo is resampled to px_per_mm and cut into a hard, centred grid (no overlap: overlapping tiles are
near-duplicates). At 4 px/mm, 224 px = 56 mm and 384 px = 96 mm. 4 px/mm is the finest resolution reachable
without upsampling the training photos; fines (< 0.063 mm) are sub-pixel at every available resolution.
"""
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None


def load_at_ppm(path, ppm_eff, px_per_mm):
    """EXIF rotation + resample so that 1 mm = px_per_mm pixels. Returns (image, factor)."""
    f = px_per_mm / ppm_eff
    with Image.open(path) as im:
        w, h = im.size
        tw, th = max(1, round(w * f)), max(1, round(h * f))
        if f < 1:
            im.draft("RGB", (tw, th))   # fast JPEG DCT downscale, never below the target size
        img = ImageOps.exif_transpose(im).convert("RGB")
    if img.size not in {(tw, th), (th, tw)}:
        size = (tw, th) if (img.width >= img.height) == (tw >= th) else (th, tw)
        img = img.resize(size, Image.Resampling.LANCZOS)
    return img, f


def tile_positions(length, tile):
    """Hard grid along one axis, remainder split evenly between both borders."""
    if length < tile:
        return []
    n = length // tile
    offset = (length - n * tile) // 2
    return [offset + i * tile for i in range(n)]


def tiles_of(img, tile=224):
    """All tiles of an already resampled image as a uint8 array (n, tile, tile, 3)."""
    ys, xs = tile_positions(img.height, tile), tile_positions(img.width, tile)
    if not ys or not xs:
        return np.zeros((0, tile, tile, 3), np.uint8)
    return np.stack([np.asarray(img.crop((x, y, x + tile, y + tile))) for y in ys for x in xs])


def tile_dir(root, px_per_mm, tile_px):
    return Path(root) / f"ppm{px_per_mm:g}_t{tile_px}"


def _tile_photo(job, root, px_per_mm):
    photo, sizes = job
    img, factor = load_at_ppm(photo["path"], photo["ppm_eff"], px_per_mm)
    stem = Path(photo["file"]).stem
    rows = []
    for t in sizes:
        out = tile_dir(root, px_per_mm, t) / photo["split"] / photo["sample_id"]
        out.mkdir(parents=True, exist_ok=True)
        ys, xs = tile_positions(img.height, t), tile_positions(img.width, t)
        coverage = len(ys) * len(xs) * t * t / (img.width * img.height)
        i = 0
        for y in ys:
            for x in xs:
                tile = img.crop((x, y, x + t, y + t))
                rel = f"{photo['split']}/{photo['sample_id']}/{stem}_{i:03d}.png"
                tile.save(tile_dir(root, px_per_mm, t) / rel)
                rows.append({"tile_px": t, "split": photo["split"], "sample_id": photo["sample_id"],
                             "file": photo["file"], "device": photo["device"], "tile": rel, "x": x, "y": y,
                             "std": float(np.asarray(tile, dtype=np.float32).std()),
                             "factor": factor, "coverage": coverage})
                i += 1
    return rows


def make_tiles(photos, sizes, root, px_per_mm=4, workers=4):
    """Cut (or reuse) the tile cache; returns {tile_px: index DataFrame with absolute paths}."""
    meta = {"px_per_mm": px_per_mm, "format": "png", "grid": "hard-centred"}
    todo = [t for t in sizes
            if not ((tile_dir(root, px_per_mm, t) / "index.csv").exists()
                    and (tile_dir(root, px_per_mm, t) / "meta.json").exists()
                    and json.loads((tile_dir(root, px_per_mm, t) / "meta.json").read_text()) == {**meta, "tile_px": t})]
    if todo:
        usable = photos[photos["ppm_eff"].notna()]
        jobs = [(r, todo) for r in usable.to_dict("records")]
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for r in pool.map(partial(_tile_photo, root=root, px_per_mm=px_per_mm), jobs):
                rows.extend(r)
        rows = pd.DataFrame(rows)
        for t in todo:
            rows[rows["tile_px"] == t].reset_index(drop=True).to_csv(tile_dir(root, px_per_mm, t) / "index.csv", index=False)
            (tile_dir(root, px_per_mm, t) / "meta.json").write_text(json.dumps({**meta, "tile_px": t}))
    index = {}
    for t in sizes:
        idx = pd.read_csv(tile_dir(root, px_per_mm, t) / "index.csv", dtype={"sample_id": str})
        idx["path"] = [str(tile_dir(root, px_per_mm, t) / r) for r in idx["tile"]]
        index[t] = idx
    return index


def aggregate_tiles(preds, photo_ids, sample_ids, mono=True):
    """tiles -> mean per photo -> mean per sample (first-seen order), valid curves."""
    from .metric import enforce_monotone

    df = pd.DataFrame(np.asarray(preds))
    df["photo"], df["sid"] = np.asarray(photo_ids), np.asarray(sample_ids)
    per_photo = df.groupby(["sid", "photo"], sort=False).mean()
    per_sample = per_photo.groupby(level="sid", sort=False).mean()
    return per_sample.index.to_numpy(), (enforce_monotone(per_sample.to_numpy()) if mono else per_sample.to_numpy())
