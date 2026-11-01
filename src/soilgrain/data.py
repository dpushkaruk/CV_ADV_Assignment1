"""Photo table: sample id, device and effective px/mm of every photo.

ppm_updated.csv gives px/mm per device at native resolution. Training photos are stored downscaled and without
EXIF, so the scale of each photo is  ppm_eff = ppm_native * stored long side / native long side.
The device comes from EXIF Model (test photos) or the file-name prefix (train photos). The sample id is the token
before the photo number: Motorola_Edge_60_fusion_H374_01 -> H374.
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ExifTags, Image

from .metric import GRAIN_COLS

Image.MAX_IMAGE_PIXELS = None   # 48 MP+ photos

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
_UMLAUTS = {"ü": "ue", "Ü": "Ue", "ä": "ae", "Ä": "Ae", "ö": "oe", "Ö": "Oe"}
_EXIF = {v: k for k, v in ExifTags.TAGS.items()}


def norm_name(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def parse_train_name(stem):
    """'Motorola_Edge_60_fusion_H374_01' -> ('Motorola_Edge_60_fusion', 'H374')"""
    tokens = stem.split("_")
    if len(tokens) < 3 or not tokens[-1].isdigit():
        return None
    return "_".join(tokens[:-2]), tokens[-2]


def parse_test_name(stem):
    """'<prefix>_HPC_Münster,BS6,9_0-10m (1)' -> ('<prefix>', 'HPC_Muenster_BS6_9_0-10m')"""
    idx = stem.find("HPC_")
    if idx == -1:
        return None
    sid = re.sub(r"\s*\(\d+\)$", "", stem[idx:])
    for a, b in _UMLAUTS.items():
        sid = sid.replace(a, b)
    return stem[:idx].strip("_ -"), sid.replace(",", "_")


def read_photo_meta(path):
    """Stored size (before EXIF rotation) + EXIF make/model/orientation, without decoding pixels."""
    with Image.open(path) as im:
        exif = im.getexif()
        return {"w": im.width, "h": im.height, "exif_make": exif.get(_EXIF["Make"]),
                "exif_model": exif.get(_EXIF["Model"]), "orientation": exif.get(_EXIF["Orientation"])}


def match_device(ppm_df, exif_model, name_device):
    by_camera = {norm_name(c): i for i, c in zip(ppm_df.index, ppm_df["camera"])}
    by_phone = {norm_name(p): i for i, p in zip(ppm_df.index, ppm_df["phone"])}
    for key, table in ((exif_model, by_camera), (name_device, by_phone),
                       (exif_model, by_phone), (name_device, by_camera)):
        if key and norm_name(key) in table:
            return table[norm_name(key)]
    return None


def build_photo_table(data_dir, train_img, test_img, ppm_csv="ppm_updated.csv"):
    data_dir = Path(data_dir)
    ppm_df = pd.read_csv(data_dir / ppm_csv)
    rows = []
    for split, img_dir, parse in (("train", data_dir / train_img, parse_train_name),
                                  ("test", data_dir / test_img, parse_test_name)):
        for p in sorted(x for x in img_dir.rglob("*") if x.suffix.lower() in IMG_EXTS and x.is_file()):
            parsed = parse(p.stem)
            if parsed is None:
                warnings.warn(f"cannot parse sample id from {p.name}, skipped")
                continue
            name_device, sid = parsed
            meta = read_photo_meta(p)
            row = {"split": split, "sample_id": sid, "file": p.name, "path": str(p), "name_device": name_device, **meta}
            dev = match_device(ppm_df, meta["exif_model"], name_device)
            if dev is not None:
                d = ppm_df.loc[dev]
                row.update(device=d["phone"], native_w=int(d["width"]), native_h=int(d["height"]), ppm_native=float(d["ppm"]))
            rows.append(row)
    df = pd.DataFrame(rows)
    for col in ("device", "native_w", "native_h", "ppm_native"):
        if col not in df:
            df[col] = np.nan
    long_side, short_side = df[["w", "h"]].max(axis=1), df[["w", "h"]].min(axis=1)
    native_long, native_short = df[["native_w", "native_h"]].max(axis=1), df[["native_w", "native_h"]].min(axis=1)
    df["ppm_eff"] = df["ppm_native"] * long_side / native_long
    # a different aspect ratio means the stored photo was cropped, not only resized -> scale is uncertain
    df["aspect_ok"] = ((long_side / short_side) / (native_long / native_short) - 1).abs() < 0.02
    df["fov_long_mm"] = long_side / df["ppm_eff"]
    df["fov_short_mm"] = short_side / df["ppm_eff"]
    if df["device"].isna().any():
        warnings.warn(f"photos with unknown device (not tiled): {df.loc[df['device'].isna(), 'file'].tolist()[:5]}")
    if (df["device"].notna() & ~df["aspect_ok"]).any():
        warnings.warn("some photos have a non-native aspect ratio (cropped?): ppm_eff assumes a long-side resize")
    return df


def load_labels(data_dir, labels_csv="Training_labels_updated.csv"):
    return (pd.read_csv(Path(data_dir) / labels_csv, dtype={"sample_id": str})
            .set_index("sample_id")[GRAIN_COLS].astype(float))
