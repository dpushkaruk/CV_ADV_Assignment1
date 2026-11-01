"""Photo table, tile cache and folds file (Part 0). Run once; every experiment reuses the outputs.

    python scripts/prepare_data.py --data-dir /kaggle/input/competitions/soil-grain-size-from-photos --out /kaggle/working
"""
import argparse
from pathlib import Path

from soilgrain.config import load_config
from soilgrain.data import build_photo_table, load_labels
from soilgrain.folds import make_folds, stratified_subset
from soilgrain.tiling import make_tiles


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=cfg.data.dir)
    ap.add_argument("--out", default=cfg.data.output_dir)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    photos = build_photo_table(args.data_dir, cfg.data.train_img, cfg.data.test_img, cfg.data.ppm)
    photos.to_csv(out / "photos.csv", index=False)
    labels = load_labels(args.data_dir, cfg.data.labels)
    train_sids = set(photos.loc[photos["split"] == "train", "sample_id"])
    assert train_sids == set(labels.index), (sorted(train_sids ^ set(labels.index)))

    index = make_tiles(photos, cfg.tiling.tile_sizes, out / "tiles", cfg.tiling.px_per_mm, cfg.tiling.workers)
    for t, idx in index.items():
        print(f"[tiles] {t} px = {t / cfg.tiling.px_per_mm:.0f} mm: {len(idx)} tiles")

    folds = make_folds(labels, train_sids)
    folds.to_csv(out / "folds.csv", index=False)
    print(f"{len(folds)} LOSO folds | ablation subset:", stratified_subset(labels, folds, cfg.validation.ablation_folds))


if __name__ == "__main__":
    main()
