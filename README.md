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

## Required Files

You need:

- A labeled OptiTrack `.c3d` file: passed with `--mocap`.
- SMPL-X model/support files: used for the body model and pose prior.
- Optionally, a marker label map JSON: passed with `--labels-map-json` when
  your C3D labels need to be renamed to the marker names used by this tool.
- Optionally, a marker layout JSON: passed with `--marker-layout` if you already
  have one.

`--marker-layout` is an **input** file when provided. It tells the converter
which SMPL-X surface vertex each marker starts from. If you do not provide it,
the converter creates a new marker layout JSON automatically and writes it into
the output directory.

An example OptiTrack label map is included at
`examples/optitrack_to_amass_label_map_suggested.json`. Use it directly if your
C3D marker labels match that convention, or copy/edit it for your own marker
names.

By default, `--support-base-dir` and `--model-base-dir` point to the
development layout's `support_files` folder. If your SMPL-X files and
`pose_body_prior.pkl` are already there, you can omit both options. If they are
somewhere else, pass:

```bash
--support-base-dir /path/to/support_files \
--model-base-dir /path/to/support_files
```

You can also bypass model lookup completely with:

```bash
--model-file /path/to/support_files/smplx/male/model.pkl
```

## Basic Usage

Run a full conversion:

```bash
/home/franklin/miniconda3/envs/smpl/bin/python \
  torch_optitrack_to_amass/convert_optitrack_to_amass_torch.py \
  --mocap "c3dexamples/Take 2026-04-16 03.50.18 PM_cal.c3d" \
  --output-dir torch_optitrack_to_amass/experiments/my_fit \
  --labels-map-json torch_optitrack_to_amass/examples/optitrack_to_amass_label_map_suggested.json \
  --gender female \
  --device auto \
  --verbose \
  --log-every 2048
```

With `--output-dir`, all generated files are written directly into that folder:

```text
my_fit/
  Take_..._smplx.json              # generated marker layout, if --marker-layout was omitted
  Take_..._female_stagei.pkl
  Take_..._female_stagei.npz
  Take_..._stageii.pkl
  Take_..._stageii.npz             # main AMASS-compatible output
```

The old `--work-base-dir` option is still available for compatibility with the
original MoSh++ folder convention. It writes to
`<work-base-dir>/workspace/<input-folder>/`. For new runs, prefer
`--output-dir`.

The main AMASS-compatible file is `*_stageii.npz`. It contains fields such as
`poses`, `trans`, `betas`, `root_orient`,
`pose_body`, `pose_hand`, `markers_latent`, and `latent_labels`.

## Fast Test

For a tiny smoke test:

```bash
/home/franklin/miniconda3/envs/smpl/bin/python \
  torch_optitrack_to_amass/convert_optitrack_to_amass_torch.py \
  --mocap "c3dexamples/Take 2026-04-16 03.50.18 PM_cal.c3d" \
  --output-dir torch_optitrack_to_amass/experiments/smoke_test \
  --labels-map-json torch_optitrack_to_amass/examples/optitrack_to_amass_label_map_suggested.json \
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
  --output-dir torch_optitrack_to_amass/experiments/my_stageii_only_fit \
  --labels-map-json torch_optitrack_to_amass/examples/optitrack_to_amass_label_map_suggested.json \
  --gender female \
  --stagei-pkl path/to/*_female_stagei.pkl \
  --verbose \
  --log-every 2048
```

## Useful Options

- `--end-fidx N`: fit only the first `N` frames.
- `--start-fidx N`: start from frame `N`.
- `--output-dir DIR`: write all outputs directly into `DIR`.
- `--work-base-dir DIR`: legacy output root using the MoSh++ workspace layout.
- `--marker-layout FILE`: optional existing marker layout input. If omitted,
  one is generated.
- `--labels-map-json FILE`: optional input label map from C3D labels to marker names.
- `--stagei-pkl FILE`: skip Stage I and reuse an existing shape/marker fit.
- `--stagei-only`: run only Stage I.
- `--device auto|cuda|cpu`: choose compute device.
- `--stageii-block-size 1`: default and recommended for accuracy.
- `--stageii-wrist-velocity-weight 0`: disable the wrist stability prior.

Block size 2 is available for experiments, but it can drift on long sequences,
so block size 1 remains the recommended default.

## License

MIT License. See `LICENSE`.
