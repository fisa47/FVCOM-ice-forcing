#!/usr/bin/env python3
"""
Step 3: Apply temporal smoothing to daily ice forcing.
Filters out BAD dates marked in frames directory, interpolates gaps,
applies smoothing, ramping, and enforces physical constraints.
"""

import numpy as np
import xarray as xr
from pathlib import Path
import re
import argparse
import sys


def get_bad_dates(frames_dir):
    """Extract dates marked as BAD from PNG filenames."""
    frames_dir = Path(frames_dir)
    if not frames_dir.exists():
        return set()
    
    bad_dates = set()
    for f in frames_dir.glob('ice_*_BAD.png'):
        match = re.search(r'ice_(\d{8})_BAD\.png', f.name)
        if match:
            bad_dates.add(match.group(1))
    return bad_dates


def main(input_dir, output_file, frames_dir, window=7, ramp_days=30,
         hice_min=0.1, aice_threshold=1e-5, isalt_value=10.0):
    """Apply temporal smoothing with all corrections."""
    print("="*60)
    print("Step 3: Apply temporal smoothing and physical constraints")
    print("="*60)
    
    input_dir = Path(input_dir)
    
    # Get bad dates from frames
    print(f"\n[1/6] Checking for BAD frames...")
    bad_dates = get_bad_dates(frames_dir) if frames_dir else set()
    if bad_dates:
        print(f"      Found {len(bad_dates)} BAD dates to exclude: {sorted(bad_dates)[:3]}{'...' if len(bad_dates) > 3 else ''}")
    else:
        print(f"      No BAD frames found.")
    
    # Find and filter files
    print(f"\n[2/6] Loading daily forcing files...")
    all_files = sorted(input_dir.glob('ice_forcing_*.nc'))
    files = [f for f in all_files if f.stem.split('_')[-1] not in bad_dates]
    
    if not files:
        raise FileNotFoundError(f"No valid ice forcing files in {input_dir}")
    
    print(f"      Total files: {len(all_files)}, Valid: {len(files)}, Excluded: {len(all_files) - len(files)}")
    
    # Load data
    print(f"      Loading into memory...")
    ds = xr.open_mfdataset(files, concat_dim='time', combine='nested', decode_times=False).load()
    print(f"      Loaded {len(ds.time)} time steps.")
    
    # Fill time gaps with interpolation
    print(f"\n[3/6] Checking for time gaps...")
    if len(ds.time) > 1:
        time_diff = np.median(np.diff(ds.time.values))
        gaps = np.where(np.diff(ds.time.values) > time_diff * 1.5)[0]
        
        if len(gaps) > 0:
            print(f"      Found {len(gaps)} gaps, interpolating...")
            t0, t1 = ds.time.values[0], ds.time.values[-1]
            complete_time = np.linspace(t0, t1, int(np.round((t1 - t0) / time_diff)) + 1)
            ds = ds.reindex(time=complete_time, method=None)
            for var in ['AICE', 'HICE']:
                if var in ds:
                    ds[var] = ds[var].interpolate_na(dim='time', method='linear')
            print(f"      Interpolated to {len(ds.time)} time steps.")
        else:
            print(f"      No gaps found.")
    
    # Temporal smoothing
    print(f"\n[4/6] Applying {window}-day rolling mean...")
    for var in ['AICE', 'HICE']:
        if var in ds:
            ds[var] = ds[var].rolling(time=window, center=True, min_periods=1).mean()
    print(f"      Smoothing applied to AICE" + (", HICE" if 'HICE' in ds else ""))
    
    # Clip weak ice (AICE < 0.05 -> 0)
    if 'AICE' in ds:
        aice = ds['AICE'].values
        aice[aice < 0.05] = 0.0
        ds['AICE'].values = aice
    
    # Apply ramping
    print(f"\n[5/6] Applying physical constraints...")
    if ramp_days > 0 and ramp_days <= len(ds.time):
        print(f"      Ramp-up: {ramp_days} days (quadratic+linear)")
        weights = np.zeros(ramp_days)
        half = ramp_days // 2
        # Quadratic then linear ramp
        for i in range(half):
            weights[i] = 0.2 * (i / half) ** 2
        for i in range(half, ramp_days):
            weights[i] = 0.2 + 0.8 * (i - half) / (ramp_days - half)
        
        for var in ['AICE', 'HICE']:
            if var in ds:
                data = ds[var].values
                for i in range(ramp_days):
                    data[i] *= weights[i]
                ds[var].values = data
        
        # Post-ramp: enforce AICE >= 0.05 or 0
        if 'AICE' in ds:
            aice = ds['AICE'].values
            mask = (aice > 0) & (aice < 0.05)
            aice[mask] = 0.0
            ds['AICE'].values = aice
    
    # Enforce minimum HICE where ice exists
    if 'HICE' in ds and 'AICE' in ds:
        hice = ds['HICE'].values
        aice = ds['AICE'].values
        thin_ice = (hice < hice_min) & (aice > aice_threshold)
        n_corrected = thin_ice.sum()
        hice[thin_ice] = hice_min
        ds['HICE'].values = hice
        print(f"      Min HICE: {hice_min}m where AICE > {aice_threshold} ({n_corrected} cells adjusted)")
    
    # Set ISALT where ice exists
    if 'AICE' in ds:
        aice = np.nan_to_num(ds['AICE'].values, nan=0.0)
        isalt = np.zeros_like(aice, dtype=np.float32)
        isalt[aice > aice_threshold] = isalt_value
        ds['ISALT'] = xr.DataArray(isalt, dims=ds['AICE'].dims, coords=ds['AICE'].coords)
        ds['ISALT'].attrs = {'long_name': 'Ice salinity', 'units': '1e-3'}
        print(f"      ISALT: {isalt_value} PSU where ice present")
    
    # Ensure UICE/VICE exist
    if 'nele' in ds.dims:
        n_times, n_elem = len(ds.time), ds.dims['nele']
        for var in ['UICE', 'VICE']:
            if var not in ds:
                ds[var] = xr.DataArray(
                    np.zeros((n_times, n_elem), dtype=np.float32),
                    dims=('time', 'nele'),
                    attrs={'long_name': f'Ice velocity {var[-1]}', 'units': 'm/s'}
                )
    
    # Save output
    print(f"\n[6/6] Saving output...")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    encoding = {v: {'zlib': True, 'complevel': 4} for v in ds.data_vars}
    print(f"      Writing: {output_file}")
    ds.to_netcdf(output_file, encoding=encoding, unlimited_dims=['time'])
    
    print("\n" + "="*60)
    print("Complete!")
    print("="*60)
    print(f"\nOutput: {output_file}")
    print(f"  Time steps: {len(ds.time)}")
    print(f"  Variables:  AICE, HICE, ISALT, UICE, VICE")
    print(f"  AICE range: [{ds['AICE'].min().values:.3f}, {ds['AICE'].max().values:.3f}]")
    if 'HICE' in ds:
        print(f"  HICE range: [{ds['HICE'].min().values:.3f}, {ds['HICE'].max().values:.3f}]")
    
    ds.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply temporal smoothing to ice forcing')
    parser.add_argument('--input', type=str, required=True, help='Directory with daily forcing files')
    parser.add_argument('--output', type=str, required=True, help='Output NetCDF file')
    parser.add_argument('--frames', type=str, default=None, help='Frames directory to check for BAD dates')
    parser.add_argument('--window', type=int, default=7, help='Smoothing window in days (default: 7)')
    parser.add_argument('--ramp-days', type=int, default=30, help='Ramp-up days (default: 30, 0 to disable)')
    parser.add_argument('--hice-min', type=float, default=0.1, help='Minimum HICE in meters (default: 0.1)')
    parser.add_argument('--aice-threshold', type=float, default=1e-5, help='AICE threshold (default: 1e-5)')
    parser.add_argument('--isalt', type=float, default=10.0, help='ISALT value in PSU (default: 10.0)')
    args = parser.parse_args()
    
    try:
        main(args.input, args.output, args.frames, args.window, args.ramp_days,
             args.hice_min, args.aice_threshold, args.isalt)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
