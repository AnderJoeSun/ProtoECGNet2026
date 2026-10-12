# ProtoECGNet2026

> **Status.** Only this README is public at present. Every other file in this
> repository — source code, the per-trial result tables and the environment
> record — will be uploaded when the paper is published, and this README will
> be replaced by its final version at the same time. Until then, the file
> names and commands below describe what will be released.

Code for

> **Cross-Dataset Augmentation for Improved External Atrial Flutter Detection in Multi-Label ECG with Heterogeneous Label Spaces**
> Joe Shen et al., 2026. Manuscript in preparation.

The study borrows atrial-flutter (AFLT) recordings from a donor dataset
(Chapman–Shaoxing) into the training split of a target dataset (PTB-XL or SPH),
trains with a masked binary cross-entropy that supervises the borrowed records
only on AFLT, and evaluates on two held-out external datasets (Georgia and
CPSC-2018-Extra). This repository releases the **pure_clf** pipeline with the
**resnet1d18** backbone, which produces every number in the paper.

*pure_clf* means "classifier only": the plain multi-label classifier — the 1D
ResNet backbone followed by a linear head, trained with (masked) binary
cross-entropy — without ProtoECGNet's prototype layer, prototype losses and
projection stage. It is `--training_stage pure_clf` in `src/main.py`. The paper
studies where the AFLT supervision comes from, so it holds the architecture at
this plain classifier; the prototype pipeline it is built on is retained in the
repository but is not used for any result in the paper.

It is built on [ProtoECGNet](https://github.com/bbj-lab/protoecgnet)
(Sethi et al., MLHC 2025, MIT License). The upstream prototype pipeline is kept
in full. Its original README is preserved as
[`README_ProtoECGNet.md`](README_ProtoECGNet.md), and [`NOTICE`](NOTICE) lists
which files come from upstream and what changed.

Some internal code comments are written in Chinese.

---

## 1. Environment

```bash
conda env create -f reproducibility/environment.yml -n protoecgnet2026
conda activate protoecgnet2026
```

`reproducibility/` records the environment of the paper's runs:

- `environment.yml`: the conda environment.
- `requirements-frozen.txt`: the pip freeze.
- `nvidia-smi.txt`: the single GPU used.
- `env.info.log.txt`: key versions (PyTorch 2.6.0, PyTorch Lightning 2.5.0.post0, Optuna 4.2.1, NeuroKit2 0.2.12, CUDA 12.4).

The root `environment.yml` is the upstream ProtoECGNet environment, kept
unchanged for reference.

## 2. Data

No ECG data is distributed with this repository. Download:

| Dataset | Role | Source |
|---|---|---|
| PTB-XL v1.0.3 | target | https://physionet.org/content/ptb-xl/1.0.3/ |
| SPH | target | https://doi.org/10.6084/m9.figshare.c.5779802.v1 |
| Chapman–Shaoxing | AFLT donor | PhysioNet/CinC Challenge 2021 v1.0.3, `training/chapman_shaoxing` |
| Georgia | external test | PhysioNet/CinC Challenge 2021 v1.0.3, `training/georgia` |
| CPSC-2018-Extra | external test | PhysioNet/CinC Challenge 2021 v1.0.3, `training/cpsc_2018_extra` |
| Ningbo | label audit only (Supplementary S2) | PhysioNet/CinC Challenge 2021 v1.0.3, `training/ningbo` |

Challenge 2021 data: https://physionet.org/content/challenge-2021/1.0.3/

**Paths.** The code keeps the directory layout of the machine the paper was
run on. Either reproduce this layout (symbolic links are enough) or edit the
constants in `src/my_config.py` and at the top of each `src/process_*.py`:

| What | Path used in the code |
|---|---|
| PTB-XL raw | `/data/datasets/ptb-xl-v1.0.3/1.0.3` |
| SPH raw | `/data/datasets/sph/sph_v2022` (records under `records/`) |
| Challenge 2021 raw | `/data/datasets/physionet.org/files/challenge-2021/1.0.3/training/<dataset>` |
| Preprocessed data, logs, checkpoints, test results | `/data/datasets/pp/<dataset>/...` |

The two target datasets are read through a copy of their own label file, so
that the file the preprocessing writes back never overwrites the one shipped by
the provider:

- PTB-XL: `my_config.PTBXL_V1_0_3_METADATA_PATH` points at
  `ptbxl_database.ori.csv`, which is simply PTB-XL's own `ptbxl_database.csv`
  (copy or rename it, or set the constant to `ptbxl_database.csv`).
- SPH: `my_config.SPH_V2022_METADATA_PATH` points at `metadata.ori.csv`, which
  is SPH's own `metadata.csv` (same choice).

### The `pp` tree

`pp` stands for *preprocessing*: `/data/datasets/pp/` holds the preprocessed
version of all five datasets, plus everything the runs write. One directory per
dataset, with the same layout:

```
/data/datasets/pp/
├── ptbxl/ptbxl_v1.0.3/                 # target   (my_config.PTBXL_V1_0_3_PREPROCESSING_DIR)
├── sph/sph_v2022/                      # target   (my_config.SPH_V2022_PREPROCESSING_DIR)
├── chapman_shaoxing/chapman_shaoxing_v2021/   # AFLT donor
├── georgia/georgia_v2021/              # external test
└── cpsc/cpsc_2018_extra/               # external test
```

Inside each dataset directory:

```
metadata.csv                 one row per record: labels, split, processing status
ecg_data_500hz/              raw signal at the native rate, as .npy
ecg_data_500hz_cleaned/      after NeuroKit2 cleaning, still at 500 Hz
ecg_data_100hz_cleaned/      after resampling to 100 Hz
ecg_data_100hz_cn/           cleaned + per-lead z-normalised; this is what training reads
*_plot/, comparison/         inspection plots written alongside each stage
```

The two target directories also carry the run outputs:

```
/data/datasets/pp/<ptbxl|sph>/
├── logs/           per-run TensorBoard logs and the per-class metrics the analysis scripts read
├── checkpoints/    per-run model weights
└── test_results/   per-sample predictions: internal test split and both external sets
```

The Optuna studies (`optuna_pureclf_*.db`) are created in the repository root
by the pipeline. They are not distributed.

## 3. Preprocessing

Each record is cleaned lead by lead at its native rate with NeuroKit2's default
ECG cleaning, resampled to 100 Hz and z-normalised per lead (paper §3.3). The
donor additionally passes the per-lead signal-quality filter
(Supplementary S1.1).

```bash
python3 src/process_ptbxl_1.0.3.py         # target: PTB-XL   (+ the donor)
python3 src/process_sph.py                 # target: SPH      (+ the donor)
python3 src/process_georgia.py             # external: Georgia
python3 src/process_cpsc_2018_extra.py     # external: CPSC-2018-Extra
```

`src/process_chapman_shaoxing.py` is not run on its own. The two target scripts
import `aug_aflt_chapman_shaoxing()` from it and call it while they build their
preprocessed metadata, so the donor's AFLT records are prepared and appended in
the same pass (`PP_PTBXL_V1_0_3` and `PP_SPH_V2022` respectively).

Five-dataset AFLT leakage audit (paper Table 2):

```bash
python3 src/compare_aflt_duplicates.py \
  --ptbxl_dir        /data/datasets/pp/ptbxl/ptbxl_v1.0.3 \
  --ptbxl_metadata   /data/datasets/pp/ptbxl/ptbxl_v1.0.3/metadata.csv \
  --sph_dir          /data/datasets/pp/sph/sph_v2022 \
  --sph_metadata     /data/datasets/pp/sph/sph_v2022/metadata.csv \
  --cs_dir           /data/datasets/pp/chapman_shaoxing/chapman_shaoxing_v2021 \
  --georgia_dir      /data/datasets/pp/georgia/georgia_v2021 \
  --georgia_metadata /data/datasets/pp/georgia/georgia_v2021/metadata.csv \
  --cpsc_dir         /data/datasets/pp/cpsc/cpsc_2018_extra \
  --include_corr     # waveform correlation >= 0.99 on the first 1,000 samples
```

## 4. Reproducing the experiments

`src/reproduce.py` runs the protocol of paper §3.6 for a target dataset, in
this order:

1. Floor calibration (N0).
2. The non-augmented and augmented validation-objective searches (N1, A1), with the plateau stopping rule.
3. Stable-baseline trial selection.
4. Five-seed evaluation of the eight selected trials of each arm (`noaug`, `aug_nomask`, `aug_mask`), including the two external test sets.
5. Aggregation into the multiseed CSV.

The script is idempotent: re-running it resumes from the first unfinished
stage.

Run from the repository root:

```bash
# SPH
python3 src/reproduce.py --project_root $PWD --quick \
    --datasets sph --backbones resnet1d18 \
    --baseline_select_mode stable --with_mask_ablation

# PTB-XL
python3 src/reproduce.py --project_root $PWD --quick \
    --datasets ptbxl --backbones resnet1d18 \
    --baseline_select_mode stable --with_mask_ablation
```

- Add `--print_only` to either command to print the status of every stage and the command it would run next, without running anything.
- `--stop_after <stage>` stops after the given stage.
- The paper's plateau rule (at least 150 completed trials, no improvement over the last 100) is the default.
- The model seeds are 42, 123, 456, 789 and 2024.
- Each search runs hundreds of trials, so a full run is long on a single GPU.

The outputs of the paper's runs are in `results/`:

| File | Content |
|---|---|
| `results/pureclf_ptbxl_multiseed_resnet1d18_20260915095813.csv` | PTB-XL: 3 arms × 8 trials; the five-seed mean and s.d. of every class on the internal test split and on both external sets |
| `results/pureclf_sph_multiseed_resnet1d18_20260915101448.csv` | SPH: the same |

## 5. Statistics and figures

The first two commands need only `results/`. The others read the run logs or
outputs under `/data/datasets/pp/<dataset>/`.

```bash
# Mann–Whitney U, exact p and bootstrap CI (paper §4.2)
python3 src/compute_significance.py \
    --ptbxl results/pureclf_ptbxl_multiseed_resnet1d18_20260915095813.csv \
    --sph   results/pureclf_sph_multiseed_resnet1d18_20260915101448.csv

# Figures 4 and 6 (written to ./figs/)
python3 src/make_figs.py

# Seed noise (paper §4.2, Supplementary Table S9)
python3 src/measure_seed_noise.py --log_dir /data/datasets/pp/ptbxl/logs \
    --backbone resnet1d18 --variant noaug \
    --trials 115,121,182,185,197,207,209,218 \
    --reliable AFIB,1AVB,IVCD,STACH,SARRH,SBRAD
python3 src/measure_seed_noise.py --log_dir /data/datasets/pp/sph/logs \
    --backbone resnet1d18 --variant noaug \
    --trials 83,113,122,161,166,188,198,202 \
    --reliable AFIB,STACH,SARRH,SBRAD

# Reliable-class effects of masking and of the full recipe (Tables 6 and 7)
python3 src/analyze_mask_reliable_classes.py \
    --log_dir /data/datasets/pp/ptbxl/logs --dataset ptbxl --backbone resnet1d18
python3 src/analyze_mask_reliable_classes.py \
    --log_dir /data/datasets/pp/sph/logs --dataset sph --backbone resnet1d18 \
    --reliable AFIB,STACH,SARRH,SBRAD

# Precision–recall curves (Figure 5, Supplementary Figures S7–S12)
PP=/data/datasets/pp OUTD=figs_out bash src/make_pr_figs.sh
```

Supporting scripts:

- `src/batch_eval_third_party.py` regenerates the per-sample external predictions.
- `src/summarize_thirdparty_ap.py` summarises them.
- `src/build_aflt_pr_curves.py` is called by `make_pr_figs.sh`.

**Ningbo label audit (Supplementary S2).**

```bash
python3 src/process_ningbo.py
python3 src/afl_vs_afib_spectral.py calibrate \
    --root /data/datasets/physionet.org/files/challenge-2021/1.0.3/training/chapman_shaoxing \
    --out chapman_cal.json --csv chapman_feat.csv
python3 src/afl_vs_afib_spectral.py apply \
    --root /data/datasets/physionet.org/files/challenge-2021/1.0.3/training/ningbo \
    --model chapman_cal.json --csv ningbo_scores.csv
```

## 6. Paper → code

| Paper | Code |
|---|---|
| §3.3 preprocessing, Supplementary S1.1 quality filter, Figure S3 | `src/process_*.py`, `src/ecg_utils.py` |
| Table 2 leakage audit | `src/compare_aflt_duplicates.py` |
| §3.4 model, masked BCE, class weighting | `src/main.py` (`--training_stage pure_clf`), `src/training_functions.py`, `src/proto_models1D.py` (`masked_bce_loss`), `src/backbones.py` (`resnet1d18`) |
| §3.6 searches, gate objective, floors, selection; Supplementary S1.2–S1.3 | `src/reproduce.py`, `src/tune_pure_clf_optuna.py`, `src/optuna_plateau_stopper.py` |
| Five-seed and external evaluation | `src/reproduce.py`, `src/run_exp_pure_clf_on_{ptbxl,sph}.py`, `src/eval_third_party.py` |
| Tables 4–5, Supplementary Tables S8 and S10 | `src/aggregate_multiseed.py` → `results/` |
| §4.2 significance and seed noise, Table S9 | `src/compute_significance.py`, `src/measure_seed_noise.py` |
| Tables 6–7 | `src/analyze_mask_reliable_classes.py` |
| Figures 4 and 6 | `src/make_figs.py` |
| Figure 5, Figures S7–S12 | `src/make_pr_figs.sh`, `src/build_aflt_pr_curves.py` |
| Supplementary S2 (Ningbo), Table S5, Figure S2 | `src/process_ningbo.py`, `src/afl_vs_afib_spectral.py` |

## 7. Per-sample predictions and checkpoints

The per-sample prediction tables and the trained checkpoints of all 240
reported runs are archived at @@ZENODO-DOI@@. `src/backup_paper_artifacts.py`
built that archive from the run directories and can verify it: it recomputes
AFLT AP from the archived predictions and compares every cell with `results/`.

## 8. Backbones

Only `resnet1d18` is part of this release. The other backbones added during the
study (Transformer1D, InceptionTime1D, XResNet1D, LSTM1D) are **TBD** stubs in
`src/backbones.py`: their names remain valid options, but constructing one
raises `NotImplementedError`. The upstream ProtoECGNet ResNet backbones are
unchanged.

## 9. Citation

Citation details will be added here once the paper is published. Please also
cite ProtoECGNet (see `README_ProtoECGNet.md`).

## 10. License

MIT (see `LICENSE`). Upstream ProtoECGNet: Copyright (c) 2025 Sahil Sethi.
This work: Copyright (c) 2026 Joe Shen.
