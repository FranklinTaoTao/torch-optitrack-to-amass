# Torch OptiTrack to AMASS

This project converts labeled OptiTrack `.c3d` motion-capture files into
AMASS-style SMPL-X `.npz`/`.pkl` outputs. It is a PyTorch/GPU refactor of the
old chumpy-based MoSh++ conversion script in `../convert_optitrack_to_amass.py`.

The converter keeps the same two-stage idea:

1. **Stage I** fits subject shape and marker locations on the SMPL-X body.
2. **Stage II** fits frame-by-frame motion using a GPU dogleg solver with a fast
   finite-difference Jacobian.

The default settings are the current high-accuracy recipe: MoSh-style Stage I,
FD dogleg Stage II, block size 1, toe pose frozen, and a conservative wrist
stability prior.

## Setup

Use the existing conda environment:

```bash
/home/franklin/miniconda3/envs/smpl/bin/python \
  torch_optitrack_to_amass/convert_optitrack_to_amass_torch.py --help
```

On a new machine, install dependencies into an environment with PyTorch,
CUDA, `smplx`, and the SMPL-X model files available:

```bash
pip install -r torch_optitrack_to_amass/requirements.txt
```

Model/support files are expected under `support_files`, unless passed with
`--support-base-dir`, `--model-base-dir`, or `--model-file`.

## Basic Usage

Run a full conversion:

```bash
/home/franklin/miniconda3/envs/smpl/bin/python \
  torch_optitrack_to_amass/convert_optitrack_to_amass_torch.py \
  --mocap "c3dexamples/Take 2026-04-16 03.50.18 PM_cal.c3d" \
  --work-base-dir torch_optitrack_to_amass/experiments/my_fit \
  --support-base-dir support_files \
  --model-base-dir support_files \
  --marker-layout "mosh_output/take_2026-04-16-cal_runA/marker_layouts/Take 2026-04-16 03.50.18 PM_cal_smplx.json" \
  --labels-map-json mosh_output/optitrack_to_amass_label_map_suggested.json \
  --gender female \
  --device auto \
  --verbose \
  --log-every 2048
```

Outputs are written under:

```text
<work-base-dir>/workspace/<input-folder>/
```

The main AMASS-compatible file is:

```text
*_stageii.npz
```

It contains fields such as `poses`, `trans`, `betas`, `root_orient`,
`pose_body`, `pose_hand`, `markers_latent`, and `latent_labels`.

## Fast Test

For a tiny smoke test:

```bash
/home/franklin/miniconda3/envs/smpl/bin/python \
  torch_optitrack_to_amass/convert_optitrack_to_amass_torch.py \
  --mocap "c3dexamples/Take 2026-04-16 03.50.18 PM_cal.c3d" \
  --work-base-dir torch_optitrack_to_amass/experiments/smoke_test \
  --support-base-dir support_files \
  --model-base-dir support_files \
  --marker-layout "mosh_output/take_2026-04-16-cal_runA/marker_layouts/Take 2026-04-16 03.50.18 PM_cal_smplx.json" \
  --labels-map-json mosh_output/optitrack_to_amass_label_map_suggested.json \
  --gender female \
  --end-fidx 5 \
  --stagei-num-frames 2 \
  --stagei-epochs 2 \
  --verbose
```

## Reuse Stage I

If shape/marker fitting has already been done, reuse the Stage-I pickle and fit
only motion:

```bash
/home/franklin/miniconda3/envs/smpl/bin/python \
  torch_optitrack_to_amass/convert_optitrack_to_amass_torch.py \
  --mocap "c3dexamples/Take 2026-04-16 03.50.18 PM_cal.c3d" \
  --work-base-dir torch_optitrack_to_amass/experiments/my_stageii_only_fit \
  --support-base-dir support_files \
  --model-base-dir support_files \
  --marker-layout "mosh_output/take_2026-04-16-cal_runA/marker_layouts/Take 2026-04-16 03.50.18 PM_cal_smplx.json" \
  --labels-map-json mosh_output/optitrack_to_amass_label_map_suggested.json \
  --gender female \
  --stagei-pkl path/to/*_female_stagei.pkl \
  --verbose \
  --log-every 2048
```

## Useful Options

- `--end-fidx N`: fit only the first `N` frames.
- `--start-fidx N`: start from frame `N`.
- `--stagei-pkl FILE`: skip Stage I and reuse an existing shape/marker fit.
- `--stagei-only`: run only Stage I.
- `--device auto|cuda|cpu`: choose compute device.
- `--stageii-block-size 1`: default and recommended for accuracy.
- `--stageii-wrist-velocity-weight 0`: disable the wrist stability prior.

Block size 2 is available for experiments, but it can drift on long sequences,
so block size 1 remains the recommended default.

## License

MIT License. See `LICENSE`.
