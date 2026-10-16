# Multiclass STL with class conditional conformal prediction (demo)

Demo code for the paper "Learning Tight Multiclass Signal Temporal Logic Specifications with Class Conditional Conformal Guarantees", currently under
review. The full code will be released after publication.

The demo runs the method on the naval surveillance dataset, with one STL formula per class, in three steps:

1. learn the formulas with a margin objective (Stage 1),
2. refine their predicate thresholds while keeping the formula structure fixed (Stage 2),
3. calibrate prediction sets, once with a single threshold and once with a separate
   threshold for each class, using the decoding loss as the score and alpha = 0.1.

The demo is meant to show how the method works and does not reproduce the tables in the paper. Stage 2 uses one fixed setting instead of the sweep used in the paper. The printed formulas are not pruned, so they keep conditions that hold for all the data, and some bounds lie far outside the data range.

## Setup

Python 3.10.

```
pip install -r requirements.txt
```

Put the naval data file in `data/` as described in `data/README.md`.

## Run

```
python run_demo.py --stage1 pretrained
python run_demo.py --stage1 train
```

The first command loads the trained Stage 1 model from `pretrained/`. The second trains Stage 1 from scratch with seed 42. On our machine both give the same Stage 1 model and the same final results. Each run takes about two minutes on a multi core CPU and longer on a laptop. All hyperparameters are in `config.yaml`.

## Output

The report gives accuracy, the positive satisfaction rate (PSR), the codeword satisfaction rate (CSR), and the mean robustness of the formulas each class should satisfy, on the validation and test splits, before and after refinement. For both calibration methods it gives the marginal and per class test coverage, the average set size and the fraction of empty sets. Coverage is measured on a single test split of 251 trajectories, so values a little below 0.90 are expected from sampling variation alone.

Results are saved in `outputs/<mode>/`, including the report, both models, the validation history of Stage 2, and the formulas before and after refinement.

## Stage 2 checkpoint

Stage 2 runs for 2000 epochs and is evaluated on the validation split every 10 epochs. Among checkpoints that lose at most 0.02 validation accuracy relative to Stage 1, those whose mean robustness over satisfied required formulas is within 0.005 of the lowest value are kept, and the one with the highest validation PSR is used. If no checkpoint qualifies, the Stage 1 model is kept.

## Files

```
run_demo.py           entry point
config.yaml           hyperparameters
demo/data.py          normalization and data split
demo/stl_layers.py    STL network layers
demo/model.py         STL network and checkpoints
demo/stage1.py        Stage 1 training
demo/stage2.py        Stage 2 refinement and checkpoint choice
demo/metrics.py       accuracy, PSR, CSR, robustness statistics
demo/conformal.py     pooled and class conditional calibration
demo/formulas.py      formulas in the original coordinates
pretrained/           Stage 1 model
data/                 naval data goes here
```

## Data

The naval dataset comes from Kong et al., "Temporal logic inference for classification and prediction from data", HSCC 2014.

## Notice

All rights reserved until publication. No permission is granted to use, copy or redistribute this code at this time.