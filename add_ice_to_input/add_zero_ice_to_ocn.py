#!/usr/bin/env python3
"""Add zero ice variables to FVCOM ocean forcing files."""

import numpy as np
import xarray as xr
from pathlib import Path
import argparse
import sys


def add_zero_ice(ds):
    """Add zero ice variables to dataset."""
    n_times = ds.dims.get('time', 1)
    n_nodes = ds.dims['node']
    n_elem = ds.dims['nele']
    
    ice_vars = {
        'AICE': (np.zeros((n_times, n_nodes), dtype=np.float32), ['time', 'node']),
        'HICE': (np.zeros((n_times, n_nodes), dtype=np.float32), ['time', 'node']),
        'UICE': (np.zeros((n_times, n_elem), dtype=np.float32), ['time', 'nele']),
        'VICE': (np.zeros((n_times, n_elem), dtype=np.float32), ['time', 'nele']),
        'zisf': (np.zeros((n_times, n_nodes), dtype=np.float32), ['time', 'node']),
        'isisfn': (np.zeros((n_times, n_nodes), dtype=np.int32), ['time', 'node']),
        'isisfc': (np.zeros((n_times, n_elem), dtype=np.int32), ['time', 'nele']),
        'meltrate': (np.zeros((n_times, n_nodes), dtype=np.float64), ['time', 'node']),
    }
    
    for name, (data, dims) in ice_vars.items():
        ds[name] = xr.DataArray(data, dims=dims)
    
    return ds


def main(input_folder, output_file):
    """Process all ocean files and add zero ice."""
    input_folder = Path(input_folder)
    files = sorted(input_folder.glob('*.nc'))
    
    if not files:
        raise FileNotFoundError(f"No NetCDF files in {input_folder}")
    
    print(f"Processing {len(files)} files from {input_folder}")
    
    # Load and process each file
    datasets = []
    for f in files:
        ds = xr.open_dataset(f, decode_times=False)
        ds = add_zero_ice(ds)
        datasets.append(ds)
        print(f"  {f.name}: {ds.dims.get('time', 1)} timesteps")
    
    # Separate time-independent variables
    static_vars = ['x', 'y', 'xc', 'yc', 'lon', 'lat', 'lonc', 'latc', 
                   'nv', 'h', 'h_center', 'siglay', 'siglay_center', 
                   'siglev', 'siglev_center']
    
    time_vars = [v for v in datasets[0].variables 
                 if 'time' in datasets[0][v].dims and v not in static_vars]
    time_vars += ['AICE', 'HICE', 'UICE', 'VICE', 'zisf', 'isisfn', 'isisfc', 'meltrate']
    
    # Concatenate time-dependent, keep static from first file
    combined_time = xr.concat([ds[time_vars] for ds in datasets], dim='time')
    out = datasets[0][[v for v in static_vars if v in datasets[0]]].copy()
    for var in combined_time.variables:
        out[var] = combined_time[var]
    out.attrs = datasets[0].attrs.copy()
    
    # Save
    encoding = {v: {'zlib': True, 'complevel': 4} 
                for v in ['AICE', 'HICE', 'UICE', 'VICE', 'zisf', 'isisfn', 'isisfc', 'meltrate']}
    out.to_netcdf(output_file, encoding=encoding, unlimited_dims=['time'])
    
    print(f"\nOutput: {output_file}")
    print(f"  Total timesteps: {out.dims['time']}")
    print(f"  Added zero ice variables: AICE, HICE, UICE, VICE, zisf, isisfn, isisfc, meltrate")
    
    for ds in datasets:
        ds.close()
    out.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add zero ice to ocean forcing files')
    parser.add_argument('--input', type=str, required=True, help='Folder with ocean forcing files')
    parser.add_argument('--output', type=str, required=True, help='Output combined file')
    args = parser.parse_args()
    
    try:
        main(args.input, args.output)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
