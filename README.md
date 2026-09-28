# test47 — test46 + multi-snapshot training data + random-start generation + live progress log

`test47.py` is `test46.py` (l≤5 NequIP, 192-atom unit cell — itself the supercomputer version of
test33/test45: DM2 NequIP denoising autoencoder, resumable training and generation) plus three
further changes:

1. **Multiple training references.** test33/test45/test46 train on a single ideal crystal
   duplicated `--duplicate` times, so the model only ever sees synthetic rattle noise, never real
   thermal displacement. test47 defaults to `--reference-frames data/reference_frames.npz`: 18 real
   thermal snapshots (9 each from `md/traj_0.lammpstrj`/`traj_1.lammpstrj`'s beta-cristobalite NPT
   MD, index `2000::1000`). Each frame is duplicated `--duplicate` times (dataset size becomes
   N × `--duplicate`). Pass `--single-reference` for the old single-structure behaviour.
2. **Random-start generation.** test33/test45/test46 always start generation from the ideal
   crystal itself, which only tests whether the model can denoise back to a known-good structure,
   not whether it can generate one from nothing. `--init random` (new; default stays `crystal` for
   backward compatibility) starts from atoms placed uniformly at random in the cell instead. Run
   both `--init crystal` and `--init random` (different `--output-dir`) to compare.
3. **Live progress file.** Every progress message (epoch/step lines, resume/checkpoint notices) is
   now also appended to `<output-dir>/progress.log` as it happens, not only printed to stdout — so
   progress is visible as a file inside the output directory while the job is still running,
   without needing the PBS job ID / stdout log filename.

Everything else, including the l≤5 irreps and `--replicate` defaulting to `1 1 1` (the raw
192-atom unit cell), is unchanged from test46:

| | test33 | test45 | test47 |
|---|---|---|---|
| `irreps_hidden` | `64x0e + 32x1e` | `+ 16x2e + 8x3e` | `+ 16x2e + 8x3e + 4x4e + 2x5e` |
| `irreps_edge` | `4x0e + 4x1e + 2x2e` | `+ 2x3e` | `+ 2x3e + 1x4e + 1x5e` |
| `--replicate` (atoms) | 2 2 2 (1536) | 2 2 2 (1536) | **1 1 1 (192)** |

`--irreps-hidden`, `--irreps-edge` and `--num-convs` are CLI-overridable (e.g. to shrink the l≥2
channel counts if l≤5 causes a CUDA OOM). Everything else is unchanged from test45.

At the default `--replicate 1 1 1`, the 13.57 Å unit cell's half-box (6.79 Å) is smaller than
test33/test45's `--large-cutoff` default (10.0 Å), which would reintroduce the duplicate-periodic-
image bug their module docstring's "fix #1" describes. test47 instead defaults `--large-cutoff` to
5.0 (equal to `--cutoff`), which stays under the half-box and avoids it — at the cost of leaving no
rattle margin above `--cutoff` (a pair just beyond 5.0 Å before noise can never appear as an edge
even if noise later brings it within 5.0 Å; see the flag's `--help` text).

## スパコンでの実行（test33 と同じ: Singularity + DM2 をバインド）

DM2 (graphite) はイメージに入れず、ソースを `PYTHONPATH` で渡します。

```bash
git clone https://github.com/haru2225/test47.git
git clone https://github.com/digital-synthesis-lab/DM2.git      # test47 と同じ階層に置く
cd test47

module load singularity     # サイトのモジュール名に合わせる
singularity build test47-pytorch-2.5.0-cu124.sif Singularity.def
# Apptainer: apptainer build test47-pytorch-2.5.0-cu124.sif Singularity.def

qsub -P PROJECT_ID run_test47.pbs
# DM2 が別の場所なら: qsub -P PROJECT_ID -v DM2_ROOT=/abs/path/DM2 run_test47.pbs
```

- 学習・生成とも途中保存されます(walltime 切れ = 時間予算 11.5 h で保存して終了)。同じ `qsub` を再投入すると続きから再開します。
- 最終 checkpoint `checkpoints/test47_sio2_crystal_2x2x2_nequip.pt` があれば学習をスキップします。
- 主な環境変数: `NUM_UPDATES`, `DUPLICATE`, `GEN_NOISY_STEPS`, `GEN_POLISH_STEPS`, `TIME_BUDGET_HOURS`,
  `SIF_IMAGE`, `OUTPUT_DIR`, `CHECKPOINT_PATH`, `INIT`(`crystal`/`random`), `SINGLE_REFERENCE`(`1` で
  test46 の単一構造学習に戻す)。`--reference-frames` の変更は今のところ `run_test47.pbs` からは
  未対応で、直接 `test47.py` を呼んでください。
- 出力: `test47-output/` (`metrics.json`, 結合長/角度/損失の図, `final_structure.extxyz`, Si のみの CG 軌道,
  `progress.log` — 実行中も追記されるログ)。
- ランダム開始と結晶開始を比較する例:
  ```bash
  python test47.py --init crystal --output-dir test47-output-crystal ...
  python test47.py --init random  --output-dir test47-output-random  ...
  ```

`DM2` は MIT (Tim Hsu; Digital Synthesis Lab @ UCLA)。
