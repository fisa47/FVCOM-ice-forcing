# Ice Forcing Workflow

Three-step workflow to create FVCOM ice forcing from DMI sea ice data.

## Requirements

```bash
conda install -n YOUR_ENV numpy xarray scipy pyproj dask netCDF4 matplotlib
pip install colormaps
```

FFmpeg required for animation:
```bash
brew install ffmpeg
```

## Workflow

### Step 1: Interpolate DMI to FVCOM Grid

```bash
python step1_interpolate.py --dmi DMI --fvcom Sval3_restart.nc --output daily_forcing
```

Options: `--kernel 9` (spatial smoothing), `--max-steps N` (test)

### Step 2: Animate and Review

```bash
python step2_animate.py --input daily_forcing --frames frames_qc --mesh Sval3_restart.nc
```

Options: `--output anim.mp4`, `--aice-max 1.0`, `--hice-max 1.0`, `--fps 10`, `--dpi 100`

**Manual QC:** Review frames, rename bad ones: `ice_20160115.png → ice_20160115_BAD.png`

### Step 3: Apply Temporal Smoothing

```bash
python step3_smooth.py --input daily_forcing --output ice_forcing_final.nc --frames frames_qc
```

Options: `--window 7`, `--ramp-days 30`, `--hice-min 0.1`, `--isalt 10.0`

## Default Processing (Step 3)

| Feature | Description | Default |
|---------|-------------|---------|
| Gap filling | Linear interpolation for missing days | ON |
| Weak ice removal | AICE < 0.05 → 0 | ON |
| Ramp-up | Gradual ice introduction | 30 days |
| Min HICE | HICE ≥ 0.1m where ice exists | ON |
| ISALT | Set to 10 PSU where ice exists | ON |
| BAD filter | Exclude dates marked in frames | ON |

---

## Add Ice to Input Files

Scripts in `add_ice_to_input/` prepare FVCOM input files with ice variables.

### Add Ice to Restart File

Initialize ice state from forcing file:

```bash
python add_ice_to_input/add_ice_to_restart.py \
    --forcing ice_forcing_final.nc \
    --restart Sval3_restart.nc \
    --output Sval3_restart_with_ice.nc
```

Options: `--time-index 0` (which timestep from forcing to use)

### Add Zero Ice to Ocean Forcing

For runs without ice - adds zero ice variables to ocean forcing:

```bash
python add_ice_to_input/add_zero_ice_to_ocn.py \
    --input add_ice_to_input/ocn \
    --output Sval3_ocn_with_ice.nc
```

Processes all `.nc` files in folder, concatenates along time dimension.

---

## Quick Example

```bash
conda activate YOUR_ENV

# Create ice forcing
python step1_interpolate.py --dmi DMI --fvcom Sval3_restart.nc --output daily_forcing
python step2_animate.py --input daily_forcing --frames frames_qc --mesh Sval3_restart.nc
# >>> Review frames, mark bad ones with _BAD suffix <<<
python step3_smooth.py --input daily_forcing --output ice_forcing_final.nc --frames frames_qc

# Prepare input files
python add_ice_to_input/add_ice_to_restart.py \
    --forcing ice_forcing_final.nc --restart Sval3_restart.nc --output Sval3_restart_ice.nc
python add_ice_to_input/add_zero_ice_to_ocn.py \
    --input add_ice_to_input/ocn --output Sval3_ocn_ice.nc
```
