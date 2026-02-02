#!/usr/bin/env python3
"""Add ice variables from forcing file to FVCOM restart file."""

import numpy as np
import xarray as xr
from pathlib import Path
import argparse
import sys


def main(forcing_file, restart_file, output_file, time_index=0):
    """Add ice variables from forcing to restart file."""
    print(f"Adding ice from {Path(forcing_file).name} to {Path(restart_file).name}")
    
    forcing = xr.open_dataset(forcing_file, decode_times=False)
    restart = xr.open_dataset(restart_file, decode_times=False)
    
    n_nodes = restart.dims['node']
    n_elem = restart.dims['nele']
    n_times = restart.dims.get('time', 1)
    
    # Extract ice at specified time
    aice = forcing['AICE'].isel(time=time_index).values
    hice = forcing['HICE'].isel(time=time_index).values if 'HICE' in forcing else np.zeros(n_nodes, dtype=np.float32)
    uice = forcing['UICE'].isel(time=time_index).values if 'UICE' in forcing else np.zeros(n_elem, dtype=np.float32)
    vice = forcing['VICE'].isel(time=time_index).values if 'VICE' in forcing else np.zeros(n_elem, dtype=np.float32)
    
    # Copy restart and add ice variables (replicated for all restart times)
    out = restart.copy(deep=True)
    
    ice_vars = {
        'AICE': (np.tile(aice[None, :], (n_times, 1)), ['time', 'node'], 'Ice concentration'),
        'HICE': (np.tile(hice[None, :], (n_times, 1)), ['time', 'node'], 'Ice thickness'),
        'UICE': (np.tile(uice[None, :], (n_times, 1)), ['time', 'nele'], 'Eastward ice velocity'),
        'VICE': (np.tile(vice[None, :], (n_times, 1)), ['time', 'nele'], 'Northward ice velocity'),
        'zisf': (np.zeros((n_times, n_nodes), dtype=np.float32), ['time', 'node'], 'Ice draft'),
        'isisfn': (np.zeros((n_times, n_nodes), dtype=np.int32), ['time', 'node'], 'Ice shelf mask (nodes)'),
        'isisfc': (np.zeros((n_times, n_elem), dtype=np.int32), ['time', 'nele'], 'Ice shelf mask (cells)'),
        'meltrate': (np.zeros((n_times, n_nodes), dtype=np.float64), ['time', 'node'], 'Ice shelf melt rate'),
    }
    
    for name, (data, dims, desc) in ice_vars.items():
        out[name] = xr.DataArray(data, dims=dims, attrs={'long_name': desc})
    
    # Save
    encoding = {v: {'zlib': True, 'complevel': 4} for v in ice_vars}
    out.to_netcdf(output_file, encoding=encoding)
    
    print(f"Output: {output_file}")
    print(f"  AICE: [{aice.min():.3f}, {aice.max():.3f}], mean={aice.mean():.3f}")
    print(f"  HICE: [{hice.min():.3f}, {hice.max():.3f}] m")
    
    forcing.close()
    restart.close()
    out.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add ice to FVCOM restart file')
    parser.add_argument('--forcing', type=str, required=True, help='Ice forcing file')
    parser.add_argument('--restart', type=str, required=True, help='Input restart file')
    parser.add_argument('--output', type=str, required=True, help='Output restart file')
    parser.add_argument('--time-index', type=int, default=0, help='Time index from forcing (default: 0)')
    args = parser.parse_args()
    
    try:
        main(args.forcing, args.restart, args.output, args.time_index)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
