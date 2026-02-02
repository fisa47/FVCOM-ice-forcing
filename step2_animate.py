#!/usr/bin/env python3
"""
Step 2: Animate ice forcing and save PNG frames for QC.
User reviews frames and renames bad ones with _BAD suffix (e.g., ice_20160115_BAD.png).
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.animation import FuncAnimation, FFMpegWriter
import colormaps as cmaps
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import sys


def main(input_dir, output_mp4, frames_dir, mesh_file, aice_max=1.0, hice_max=1.0, fps=10, dpi=100):
    """Create animation and save frames for QC."""
    print("="*60)
    print("Step 2: Generate animation and QC frames")
    print("="*60)
    
    input_dir = Path(input_dir)
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all daily files
    print(f"\n[1/4] Loading daily forcing files from: {input_dir}")
    files = sorted(input_dir.glob('ice_forcing_*.nc'))
    if not files:
        raise FileNotFoundError(f"No ice forcing files in {input_dir}")
    
    print(f"      Found {len(files)} files, loading into memory...")
    
    # Load x, y from first file (they're the same in all files)
    with xr.open_dataset(files[0], decode_times=False) as ds0:
        x = ds0['x'].values
        y = ds0['y'].values
    
    # Load time-varying data, excluding static coordinates from concat
    ds = xr.open_mfdataset(files, concat_dim='time', combine='nested', decode_times=False,
                           data_vars='minimal', coords='minimal', compat='override').load()
    print(f"      Loaded successfully.")
    
    # Load mesh connectivity
    print(f"\n[2/4] Setting up triangulation...")
    if mesh_file and Path(mesh_file).exists():
        print(f"      Loading mesh from: {mesh_file}")
        with xr.open_dataset(mesh_file, decode_times=False) as mesh:
            if 'nv' in mesh.variables:
                nv = mesh['nv'].values
    else:
        raise FileNotFoundError(f"Mesh file required for triangulation: {mesh_file}")
    
    triangles = nv.T - 1  # Transpose and convert to 0-indexed
    triang = Triangulation(x, y, triangles)
    print(f"      Nodes: {len(x)}, Triangles: {len(triangles)}")
    
    n_times = len(ds.time)
    has_hice = 'HICE' in ds.variables
    print(f"      Time steps: {n_times}")
    print(f"      Variables: AICE" + (", HICE" if has_hice else ""))
    
    # Convert MJD to dates
    base = datetime(1858, 11, 17)
    dates = [base + timedelta(days=float(t)) for t in ds['time'].values]
    labels = [d.strftime('%Y-%m-%d') for d in dates]
    print(f"      Date range: {labels[0]} to {labels[-1]}")
    
    # Setup figure
    ncols = 2 if has_hice else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    if ncols == 1:
        axes = [axes]
    
    # Initial plots
    aice0 = ds['AICE'].isel(time=0).values
    im1 = axes[0].tripcolor(triang, aice0, cmap=cmaps.BlueYellowRed, shading='gouraud', vmin=0, vmax=aice_max)
    plt.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.1, label='AICE')
    title1 = axes[0].set_title(f'AICE - {labels[0]}')
    axes[0].set_aspect('equal')
    
    if has_hice:
        hice0 = ds['HICE'].isel(time=0).values
        im2 = axes[1].tripcolor(triang, hice0, cmap=cmaps.BlueYellowRed, shading='gouraud', vmin=0, vmax=hice_max)
        plt.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.1, label='HICE [m]')
        title2 = axes[1].set_title(f'HICE - {labels[0]}')
        axes[1].set_aspect('equal')
    
    plt.tight_layout()
    
    print(f"\n[3/4] Rendering {n_times} frames to: {frames_dir}")
    
    # Render and save frames
    for frame in range(n_times):
        aice = ds['AICE'].isel(time=frame).values
        im1.set_array(aice)
        title1.set_text(f'AICE - {labels[frame]}')
        
        if has_hice:
            hice = ds['HICE'].isel(time=frame).values
            im2.set_array(hice)
            title2.set_text(f'HICE - {labels[frame]}')
        
        # Save frame
        date_str = labels[frame].replace('-', '')
        fig.savefig(frames_dir / f'ice_{date_str}.png', dpi=dpi, bbox_inches='tight')
        
        # Progress every 10%
        if frame % max(1, n_times // 10) == 0 or frame == n_times - 1:
            pct = (frame + 1) / n_times * 100
            print(f"      {frame + 1}/{n_times} frames ({pct:.0f}%) - {labels[frame]}")
    
    print(f"\n[4/4] Saving outputs...")
    
    # Save MP4 if requested
    if output_mp4:
        output_mp4 = Path(output_mp4)
        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        print(f"      Writing MP4: {output_mp4}")
        
        def update(frame):
            aice = ds['AICE'].isel(time=frame).values
            im1.set_array(aice)
            title1.set_text(f'AICE - {labels[frame]}')
            if has_hice:
                hice = ds['HICE'].isel(time=frame).values
                im2.set_array(hice)
                title2.set_text(f'HICE - {labels[frame]}')
            return [im1, title1] + ([im2, title2] if has_hice else [])
        
        anim = FuncAnimation(fig, update, frames=n_times, interval=1000/fps, blit=False)
        anim.save(output_mp4, writer=FFMpegWriter(fps=fps, bitrate=2000), dpi=dpi)
        print(f"      MP4 saved.")
    
    ds.close()
    plt.close(fig)
    
    print("\n" + "="*60)
    print("Complete!")
    print("="*60)
    print(f"\nFrames saved to: {frames_dir}")
    if output_mp4:
        print(f"Animation saved to: {output_mp4}")
    print(f"\n>>> NEXT: Review frames and rename bad ones with _BAD suffix <<<")
    print(f"    Example: ice_20160115.png -> ice_20160115_BAD.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Animate ice forcing for QC')
    parser.add_argument('--input', type=str, required=True, help='Directory with daily forcing files')
    parser.add_argument('--output', type=str, default=None, help='Output MP4 file (optional)')
    parser.add_argument('--frames', type=str, required=True, help='Output directory for PNG frames')
    parser.add_argument('--mesh', type=str, default=None, help='FVCOM mesh file for triangulation')
    parser.add_argument('--aice-max', type=float, default=1.0, help='AICE colorbar max (default: 1.0)')
    parser.add_argument('--hice-max', type=float, default=1.0, help='HICE colorbar max (default: 1.0)')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second (default: 10)')
    parser.add_argument('--dpi', type=int, default=100, help='Resolution (default: 100)')
    args = parser.parse_args()
    
    try:
        main(args.input, args.output, args.frames, args.mesh, args.aice_max, args.hice_max, args.fps, args.dpi)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
