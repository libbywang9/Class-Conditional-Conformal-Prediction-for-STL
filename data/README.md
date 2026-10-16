# Naval data (not included)

This demo uses the naval surveillance dataset of

> Z. Kong, A. Jones, A. Medina Ayala, E. Aydin Gol, and C. Belta,
> "Temporal logic inference for classification and prediction from data,"
> in *Proc. 17th International Conference on Hybrid Systems: Computation and Control (HSCC)*, 2014.

Place the data file here as `data/naval_class.pkl` (or pass another location with
`python run_demo.py --data <file>`).

Expected format: a pickled list `[train_data, train_label, val_data, val_label]` with

| element | shape | content |
|---|---|---|
| `train_data` | (1600, 61, 2) | trajectories, 61 time steps, planar position (x, y) in raw units |
| `train_label` | (1600,) | class labels 1, 2, 3 |
| `val_data` | (400, 61, 2) | as above |
| `val_label` | (400,) | as above |

`demo/data.py` normalises x and y to [0, 1] with the minimum and maximum of `train_data`,
merges the 2000 trajectories and draws a stratified 60 / 15 / 12.5 / 12.5 split
(train 1200 / val 300 / cal 249 / test 251) with seed 42.
