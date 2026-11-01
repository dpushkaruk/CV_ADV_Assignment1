"""Competition metric (EMD on the log-diameter grid) and complementary metrics.

EMD on the log-diameter grid = L1 between cumulative curves weighted by the log-widths of the sieve bins.
All bins are ~0.5 decades wide, so EMD ~ 0.5 * sum |delta| over sieves 2..11 (the 0.002 mm sieve never enters
the metric). A pure scale error shifts the curve along log d: a factor-k error costs 100 * |log10 k| (x2 -> 30.1).
"""
import numpy as np

GRAIN_COLS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]
DIAMETERS = np.array([float(c) for c in GRAIN_COLS])
LOG_D = np.log10(DIAMETERS)
LOG_WIDTHS = np.diff(LOG_D)
UNIFORM_CURVE = np.array([9.09, 18.18, 27.27, 36.36, 45.45, 54.55, 63.64, 72.73, 81.82, 90.91, 100.0])


def _2d(a):
    a = np.asarray(a, dtype=np.float64)
    return a[np.newaxis] if a.ndim == 1 else a


def emd_per_sample(y_true, y_pred):
    y_true, y_pred = _2d(y_true), _2d(y_pred)
    return (np.abs(y_true[:, 1:] - y_pred[:, 1:]) * LOG_WIDTHS).sum(axis=1)


def emd_score(y_true, y_pred):
    return float(emd_per_sample(y_true, y_pred).mean())


def scale_error_cost(k):
    """EMD of a curve shifted by a pure scale factor k (no clipping at the grid ends)."""
    return 100 * abs(np.log10(k))


def enforce_monotone(arr):
    """Valid curve: clipped to [0, 100], non-decreasing, ends at 100."""
    arr = np.maximum.accumulate(np.clip(_2d(arr).copy(), 0, 100), axis=1)
    arr[:, -1] = 100.0
    return arr


def d_value(curve, p):
    """Diameter (mm) where p % passes; log-linear interpolation, censored at 0.002 / 200 mm."""
    curve = np.asarray(curve, dtype=np.float64)
    above = np.nonzero(curve >= p)[0]
    if len(above) == 0:
        return float(DIAMETERS[-1])
    i = above[0]
    if i == 0:
        return float(DIAMETERS[0])
    c0, c1 = curve[i - 1], curve[i]
    t = 0.0 if c1 == c0 else (p - c0) / (c1 - c0)
    return float(10 ** (LOG_D[i - 1] + t * (LOG_D[i] - LOG_D[i - 1])))


def d_value_errors(y_true, y_pred, ps=(10, 30, 50, 60)):
    """|log10 D_p(pred) - log10 D_p(true)|; 0.301 = factor-2 error in that characteristic grain size."""
    y_true, y_pred = _2d(y_true), _2d(y_pred)
    return {f"D{p}_log_err": float(np.mean([abs(np.log10(d_value(b, p)) - np.log10(d_value(a, p)))
                                            for a, b in zip(y_true, y_pred)])) for p in ps}


# alternative target: the quantile function p -> log10 D_p. W1 between two distributions on the log-diameter
# axis == L1 between their quantile functions == the EMD of the competition.
PS = np.arange(2.5, 100.0, 2.5)


def curves_to_quantiles(Y):
    return np.array([[np.log10(d_value(c, p)) for p in PS] for c in _2d(Y)])


def quantiles_to_curve(Q):
    out = []
    for q in _2d(Q):
        q = np.maximum.accumulate(np.clip(q, LOG_D[0], LOG_D[-1]))
        c = np.interp(LOG_D, np.concatenate([[LOG_D[0]], q, [LOG_D[-1]]]), np.concatenate([[0.0], PS, [100.0]]))
        c[-1] = 100.0
        out.append(c)
    return np.array(out)


def soil_class(curve):
    """fine if >= 50 % passes 0.063 mm, else sand vs gravel by which fraction is larger."""
    curve = np.asarray(curve, dtype=np.float64)
    fines, p2 = curve[GRAIN_COLS.index("0.063")], curve[GRAIN_COLS.index("2")]
    if fines >= 50:
        return "fine"
    return "sand" if p2 - fines >= 100 - p2 else "gravel"
