"""PHM2010 flank wear per cut.

Official split: c1, c4, c6 are labelled (train and validation), c2, c3, c5 are not (test).
315 cuts per tool, VB measured on the 3 edges in micrometres. The whole dataset runs one
cutting condition, so nothing here speaks about other speeds, feeds or depths.
"""
import csv
import json
import math
import os

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "phm2010", "raw")
TRAIN_CUTTERS = ("c1", "c4", "c6")
TEST_CUTTERS = ("c2", "c3", "c5")
VB_LIM_UM = 150.0
HEAD_CUTS = 30


def load_wear(cutter, raw=RAW):
    """[(cut, max VB across the 3 edges, um), ...] sorted by cut."""
    path = os.path.join(raw, cutter, f"{cutter}_wear.csv")
    with open(path, newline="") as f:
        rows = [(int(r["cut"]), max(float(r["flute_1"]), float(r["flute_2"]), float(r["flute_3"])))
                for r in csv.DictReader(f)]
    return sorted(rows)


def observed_life(series, vb_lim=VB_LIM_UM):
    return next(c for c, v in series if v >= vb_lim)


def ols(xs, ys):
    """y = a + b*x."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def pooled_slope(series_list):
    """Shared exponent p with a free intercept per tool."""
    xs, ys = [], []
    for s in series_list:
        x = [math.log(c) for c, _ in s]
        y = [math.log(v) for _, v in s]
        mx, my = sum(x) / len(x), sum(y) / len(y)
        xs += [xi - mx for xi in x]
        ys += [yi - my for yi in y]
    return sum(a * b for a, b in zip(xs, ys)) / sum(a * a for a in xs)


def life_from_head(p, head, vb_lim=VB_LIM_UM):
    b = sum(math.log(v) - p * math.log(c) for c, v in head) / len(head)
    return math.exp((math.log(vb_lim) - b) / p)


def _selfcheck():
    s = [(t, 100.0 * t ** 0.5) for t in range(1, 316)]
    assert abs(pooled_slope([s]) - 0.5) < 1e-9, pooled_slope([s])
    assert abs(life_from_head(0.5, s[:HEAD_CUTS], vb_lim=200.0) - (200.0 / 100.0) ** 2) < 1e-6
    assert observed_life(s, vb_lim=200.0) == 4
    print("selfcheck ok")


def main():
    _selfcheck()
    series = {c: load_wear(c) for c in TRAIN_CUTTERS}
    print(f"{'tool':6} {'cuts':>5} {'VB_first':>9} {'VB_last':>8} {'life(150um)':>12}")
    for c, s in series.items():
        print(f"{c:6} {len(s):5d} {s[0][1]:9.1f} {s[-1][1]:8.1f} {observed_life(s):12d}")

    # ponytail: the window start is the knob; cuts 1 to 30 are break-in and bias the intercept
    results = []
    for start in (1, 31, 61):
        print(f"\ncalibrate on cuts {start}..{start + HEAD_CUTS - 1}, exponent p from the training pair")
        print(f"{'train':9} {'val':4} {'p':>6} {'life_obs':>9} {'life_tx':>9} {'err%':>8} {'life_self':>10} {'err%':>9}")
        for val in TRAIN_CUTTERS:
            train = [series[c] for c in TRAIN_CUTTERS if c != val]
            p = pooled_slope(train)
            obs = observed_life(series[val])
            head = series[val][start - 1:start - 1 + HEAD_CUTS]
            pred_tx = life_from_head(p, head)
            a, b = ols([math.log(c) for c, _ in head], [math.log(v) for _, v in head])
            pred_self = math.exp((math.log(VB_LIM_UM) - a) / b)
            et, es = 100 * (pred_tx - obs) / obs, 100 * (pred_self - obs) / obs
            print(f"{'+'.join(c for c in TRAIN_CUTTERS if c != val):9} {val:4} {p:6.3f} {obs:9d} "
                  f"{pred_tx:9.0f} {et:8.1f} {pred_self:10.0f} {es:9.1f}")
            results.append(dict(start=start, val=val, p=p, life_obs=obs, life_pred_transfer=pred_tx,
                                err_pct_transfer=et, life_pred_selfit=pred_self, err_pct_selfit=es))

    err_by_val = {v: [abs(r["err_pct_transfer"]) for r in results if r["val"] == v] for v in TRAIN_CUTTERS}
    print("\nmean absolute error of transfer, per validation tool, over all windows:")
    for v, e in sorted(err_by_val.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"  val={v}  {sum(e) / len(e):6.1f}%   {[round(x, 1) for x in e]}")
    out = os.path.join(os.path.dirname(RAW), "..", "..", "processed", "split_study.json")
    with open(out, "w") as f:
        json.dump(dict(vb_lim_um=VB_LIM_UM, head_cuts=HEAD_CUTS, results=results), f, indent=2)
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
