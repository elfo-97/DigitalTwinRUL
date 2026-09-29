# DigitalTwinRUL

Remaining useful life (RUL) of a milling insert, guessed from the wear curves of other
tools. The level of the curve belongs to the individual tool. The shape looks like it
belongs to the class: same tool model, same material, same cutting condition. So we give a
new tool a short calibration window and answer with a life figure plus an equation you can
read and argue with.

## Data

PHM Society 2010 Data Challenge (PHM2010). High-speed CNC milling of stainless steel
(HRC52) with a 6 mm ball-nose cutter, at n = 10400 rpm, f = 1555 mm/min, radial ap 0.125 mm
and axial ap 0.2 mm. Every tool in the dataset runs that one condition, so nothing here says
anything about other speeds, feeds or depths. Six tools, 315 cuts each, 7 signal channels at
50 kHz per cut, flank wear (VB) measured on the 3 edges after every cut.

Mirror used: [Kaggle `rabahba/phm-data-challenge-2010`](https://www.kaggle.com/datasets/rabahba/phm-data-challenge-2010) (CC0).

```bash
uv run --no-project --with kagglehub python -c "import kagglehub; print(kagglehub.dataset_download('rabahba/phm-data-challenge-2010'))"
# move the result to  data/raw/phm2010/raw/   (18 GB, git-ignored)
```

The challenge split, kept as it is:

| tools | use | VB labels |
|---|---|---|
| `c1`, `c4`, `c6` | train / validation | yes, 315 cuts x 3 edges in micrometres |
| `c2`, `c3`, `c5` | test | not published, so they are only useful for eyeballing plots |

We train on c1 plus c4 and validate on c6. That was a choice, not a given: we ran the three
possible train/validation arrangements and this one had the smallest error.

## Run

```bash
python phm2010.py            # official split, life-error table (stdlib only)
python extract_features.py   # raw signals -> data/processed/cuts.csv (needs pandas)
python train.py              # column study, equations, leakage audit
```

## Reference

PHM Society 2010 Conference Data Challenge, CNC milling machine cutter RUL.
See also Agogino & Goebel, *Milling Data Set*, NASA Prognostics Data Repository.
