#!/usr/bin/env python3
"""
Step 1: Interpolate DMI sea ice data to FVCOM grid.
Produces daily NetCDF forcing files in output directory.
"""

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter
from pyproj import Transformer
from pathlib import Path
import argparse
import sys


def sod_to_thickness(sod):
    """Convert stage of development to ice thickness (meters)."""
    thickness = np.zeros_like(sod, dtype=np.float32)
    for cat, thick in {-1: 0.0, 0: 0.15, 1: 0.50, 2: 0.80, 3: 1.00, 4: 2.00}.items():
        thickness[sod == cat] = thick
    return thickness


def spatial_smooth(data, mask, kernel=9):
    """Apply spatial smoothing to valid data points."""
    data_masked = np.where(mask, data, 0.0)
    smoothed = uniform_filter(data_masked.astype(np.float64), size=kernel, mode='constant')
    mask_smooth = uniform_filter(mask.astype(np.float64), size=kernel, mode='constant')
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(mask_smooth > 0, smoothed / mask_smooth, 0.0)
    return np.where(mask, result, data)


def process_timestep(sic, sod, time_val, dmi_lon, dmi_lat, target_points,
                     fvcom_x, fvcom_y, fvcom_xc, fvcom_yc, output_dir, kernel=9):
    """Process single timestep: interpolate DMI to FVCOM grid using KDTree."""
    has_sod = sod is not None
    
    # Valid data mask
    valid = (sic >= 0) & (sic <= 100) & (~np.isnan(sic))
    
    n_nodes = len(target_points)
    if valid.sum() == 0:
        sic_interp = np.zeros(n_nodes, dtype=np.float32)
        hice_interp = np.zeros(n_nodes, dtype=np.float32) if has_sod else None
    else:
        # Smooth SIC
        sic_smooth = spatial_smooth(sic, valid, kernel)
        
        # Build KDTree from valid DMI points (fast nearest neighbor)
        pts = np.column_stack([dmi_lon[valid], dmi_lat[valid]])
        tree = cKDTree(pts)
        
        # Find nearest neighbors
        _, indices = tree.query(target_points, k=1)
        
        # Interpolate using nearest neighbor
        sic_interp = sic_smooth[valid].flatten()[indices]
        sic_interp = np.clip(sic_interp / 100.0, 0, 1).astype(np.float32)
        
        if has_sod:
            hice = sod_to_thickness(sod)
            hice_smooth = spatial_smooth(hice, valid, kernel)
            hice_interp = hice_smooth[valid].flatten()[indices]
            hice_interp = np.clip(hice_interp, 0, None).astype(np.float32)

    # Time conversion to MJD
    time_day = np.datetime64(time_val, 'D').astype('datetime64[s]')
    time_mjd = float(time_day.astype('datetime64[s]').astype(np.float64) / 86400.0 + 40587.0)
    
    # Build dataset
    n_elem = len(fvcom_xc)
    data_vars = {
        'x': (['node'], fvcom_x), 'y': (['node'], fvcom_y),
        'xc': (['nele'], fvcom_xc), 'yc': (['nele'], fvcom_yc),
        'time': (['time'], np.array([time_mjd], dtype=np.float32)),
        'Itime': (['time'], np.array([int(time_mjd)], dtype=np.int32)),
        'Itime2': (['time'], np.array([0], dtype=np.int32)),
        'AICE': (['time', 'node'], sic_interp[np.newaxis, :]),
        'ISALT': (['time', 'node'], np.zeros((1, n_nodes), dtype=np.float32)),
        'UICE': (['time', 'nele'], np.zeros((1, n_elem), dtype=np.float32)),
        'VICE': (['time', 'nele'], np.zeros((1, n_elem), dtype=np.float32)),
    }
    if has_sod:
        data_vars['HICE'] = (['time', 'node'], hice_interp[np.newaxis, :])
    
    out_ds = xr.Dataset(data_vars)
    out_ds['time'].attrs = {'units': 'days since 1858-11-17 00:00:00', 'format': 'MJD'}
    out_ds['AICE'].attrs = {'long_name': 'Ice concentration', 'units': '1'}
    if has_sod:
        out_ds['HICE'].attrs = {'long_name': 'Ice thickness', 'units': 'm'}
    
    # Save
    date_str = str(time_day).split('T')[0].replace('-', '')
    outfile = output_dir / f'ice_forcing_{date_str}.nc'
    encoding = {v: {'zlib': True, 'complevel': 4} for v in ['AICE', 'HICE', 'ISALT', 'UICE', 'VICE'] if v in data_vars}
    out_ds.to_netcdf(outfile, encoding=encoding, unlimited_dims=['time'])
    
    return date_str, float(sic_interp.mean())


def main(dmi_path, fvcom_file, output_dir, kernel=9, max_steps=None):
    """Main interpolation routine."""
    print("="*60)
    print("Step 1: Interpolate DMI sea ice data to FVCOM grid")
    print("="*60)
    
    dmi_path, output_dir = Path(dmi_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find DMI files
    print(f"\n[1/4] Scanning DMI files from: {dmi_path}")
    dmi_files = sorted(dmi_path.glob('dmi_asip_*.nc')) if dmi_path.is_dir() else [dmi_path]
    if not dmi_files:
        raise FileNotFoundError(f"No DMI files found in {dmi_path}")
    print(f"      Found {len(dmi_files)} DMI file(s)")
    
    # Load FVCOM grid
    print(f"\n[2/4] Loading FVCOM grid: {fvcom_file}")
    with xr.open_dataset(fvcom_file, decode_times=False) as fv:
        fvcom_x = fv['x'].values.astype(np.float32)
        fvcom_y = fv['y'].values.astype(np.float32)
        fvcom_xc = fv['xc'].values.astype(np.float32)
        fvcom_yc = fv['yc'].values.astype(np.float32)
    print(f"      Nodes: {len(fvcom_x)}, Elements: {len(fvcom_xc)}")
    
    # Transform UTM33N -> WGS84
    print(f"\n[3/4] Transforming coordinates (UTM33N -> WGS84)...")
    transformer = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)
    fvcom_lon, fvcom_lat = transformer.transform(fvcom_x, fvcom_y)
    target_points = np.column_stack([fvcom_lon, fvcom_lat])
    print(f"      Lon range: [{fvcom_lon.min():.2f}, {fvcom_lon.max():.2f}]")
    print(f"      Lat range: [{fvcom_lat.min():.2f}, {fvcom_lat.max():.2f}]")
    
    # Check existing files
    existing = {f.stem.split('_')[-1] for f in output_dir.glob('ice_forcing_*.nc')}
    if existing:
        print(f"      Found {len(existing)} existing output files (will skip)")
    
    print(f"\n[4/4] Processing timesteps...")
    print(f"      Output: {output_dir}")
    print(f"      Smoothing kernel: {kernel}")
    
    # Process each DMI file
    processed = 0
    skipped = 0
    total_processed = 0
    
    for file_idx, dmi_file in enumerate(dmi_files):
        # Open file and read static data once
        with xr.open_dataset(dmi_file) as ds:
            dmi_lon = ds['lon'].values
            dmi_lat = ds['lat'].values
            has_sod = 'sod' in ds.variables
            n_times = len(ds.time)
            
            if max_steps is not None and total_processed + n_times > max_steps:
                n_times = max_steps - total_processed
            
            print(f"\n      File {file_idx+1}/{len(dmi_files)}: {dmi_file.name} ({n_times} timesteps)")
            
            for t in range(n_times):
                time_val = ds['time'].isel(time=t).values
                date_str = str(np.datetime64(time_val, 'D')).replace('-', '')
                
                if date_str in existing:
                    skipped += 1
                    continue
                
                # Read data for this timestep
                sic = ds['sic'].isel(time=t).values
                sod = ds['sod'].isel(time=t).values if has_sod else None
                
                # Process
                process_timestep(sic, sod, time_val, dmi_lon, dmi_lat, target_points,
                                fvcom_x, fvcom_y, fvcom_xc, fvcom_yc, output_dir, kernel)
                processed += 1
                
                # Progress
                if processed % 5 == 0 or processed == 1:
                    print(f"        Processed: {processed}, Skipped: {skipped}")
            
            total_processed += n_times
            if max_steps is not None and total_processed >= max_steps:
                break
    
    print("\n" + "="*60)
    print(f"Complete! Processed: {processed}, Skipped: {skipped}")
    print(f"Output files in: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interpolate DMI ice data to FVCOM grid')
    parser.add_argument('--dmi', type=str, required=True, help='DMI data path (file or directory)')
    parser.add_argument('--fvcom', type=str, required=True, help='FVCOM grid file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--kernel', type=int, default=9, help='Spatial smoothing kernel (default: 9)')
    parser.add_argument('--max-steps', type=int, default=None, help='Max timesteps (for testing)')
    args = parser.parse_args()
    
    try:
        main(args.dmi, args.fvcom, args.output, args.kernel, args.max_steps)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
