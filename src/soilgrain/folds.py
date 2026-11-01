"""Leave-one-sample-out folds over physical samples and the stratified subset used for ablations."""
import numpy as np
import pandas as pd

from .metric import d_value, soil_class


def make_folds(labels, sample_ids):
    """One fold per physical sample: all tiles of all photos of that sample are held out together."""
    sids = sorted(sample_ids)
    return pd.DataFrame({"sample_id": sids, "fold": np.arange(len(sids)),
                         "soil_class": [soil_class(labels.loc[s].to_numpy()) for s in sids]})


def stratified_subset(labels, folds, n):
    """n samples, proportional to the soil classes, evenly spread over D50 inside each class."""
    sids = folds["sample_id"].tolist()
    soil = folds.set_index("sample_id")["soil_class"]
    d50 = pd.Series({s: d_value(labels.loc[s].to_numpy(), 50) for s in sids})
    out = []
    for _, g in soil.groupby(soil):
        k = max(1, int(round(n * len(g) / len(sids))))
        order_ = d50.loc[g.index].sort_values().index.tolist()
        out += [order_[int(i)] for i in np.unique(np.linspace(0, len(order_) - 1, min(k, len(order_))).round().astype(int))]
    return sorted(out)
