#!/usr/bin/env python3
"""
Script to create animations of DSSF spinon+photon data across magnetic field strengths
for pyrochlore classical spin simulations.

Usage:
    python animate_DSSF_pyrochlore.py /path/to/root_dir --fps 5
    python animate_DSSF_pyrochlore.py /path/to/root_dir --output /path/to/output --fps 10
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import re
import argparse


def extract_field_value(field_dir):
    """Extract numerical field value from directory name like 'field_0.1' or 'h=0.1'"""
    # Try different patterns
    patterns = [r'field[_=]([0-9.]+)', r'h[_=]([0-9.]+)', r'B[_=]([0-9.]+)']
    for pattern in patterns:
        match = re.search(pattern, field_dir)
        if match:
            return float(match.group(1))
    return None


def collect_DSSF_data_all_fields(root_dir, verbose=True):
    """
    Collect DSSF spinon+photon data (local and global) from all field directories.
    Returns: field_data list of dicts
    """
    base_dir = os.path.abspath(root_dir)
    if not os.path.isdir(base_dir):
        raise ValueError(f"Provided root_dir {root_dir} is not a directory.")

    # Directories to skip (output/metadata directories, not field data)
    skip_dirs = {'animations', 'comparison_plots', 'results', '__pycache__', '.git'}
    
    field_dirs = sorted(
        entry for entry in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, entry)) and entry not in skip_dirs
    )

    if not field_dirs:
        raise RuntimeError(f"No field directories found in {root_dir}.")

    # Collect data from all fields
    field_data = []
    
    for field_dir in field_dirs:
        field_path = os.path.join(base_dir, field_dir)
        results_dir = os.path.join(field_path, "results")
        
        # Extract field value
        field_val = extract_field_value(field_dir)
        if field_val is None:
            if verbose:
                print(f"[collect_DSSF_data] Could not extract field value from {field_dir}, skipping.")
            continue
        
        # Look for DSSF data files
        local_file = os.path.join(results_dir, "DSSF_local_xx.txt")
        local_zz_file = os.path.join(results_dir, "DSSF_local_zz.txt")
        global_file = os.path.join(results_dir, "DSSF_global_xx.txt")
        global_zz_file = os.path.join(results_dir, "DSSF_global_zz.txt")
        
        if not (os.path.exists(local_file) and os.path.exists(local_zz_file) and 
                os.path.exists(global_file) and os.path.exists(global_zz_file)):
            if verbose:
                print(f"[collect_DSSF_data] Missing DSSF files in {field_dir}, skipping.")
            continue
        
        try:
            # Read the data
            data_local_xx = np.loadtxt(local_file)
            data_local_zz = np.loadtxt(local_zz_file)
            data_global_xx = np.loadtxt(global_file)
            data_global_zz = np.loadtxt(global_zz_file)
            
            w = data_local_xx[:, 0]  # frequency/energy axis
            
            # Compute spinon+photon (S_xx + S_zz)
            dssf_local_spinon_photon = data_local_xx[:, 1] + data_local_zz[:, 1]
            dssf_global_spinon_photon = data_global_xx[:, 1] + data_global_zz[:, 1]
            
            field_data.append({
                'field': field_val,
                'w': w,
                'dssf_local_spinon_photon': dssf_local_spinon_photon,
                'dssf_global_spinon_photon': dssf_global_spinon_photon,
                'dssf_local_xx': data_local_xx[:, 1],
                'dssf_local_zz': data_local_zz[:, 1],
                'dssf_global_xx': data_global_xx[:, 1],
                'dssf_global_zz': data_global_zz[:, 1]
            })
            
            if verbose:
                print(f"[collect_DSSF_data] Loaded data for {field_dir} (field={field_val})")
                
        except Exception as e:
            if verbose:
                print(f"[collect_DSSF_data] Error reading data from {field_dir}: {e}")
            continue
    
    if not field_data:
        raise RuntimeError(f"No valid DSSF data found in {root_dir}")
    
    # Sort by field value
    field_data.sort(key=lambda x: x['field'])
    
    return field_data


def animate_DSSF_spinon_photon(root_dir, output_dir=None, fps=5, energy_conversion=0.063):
    """
    Create animations for DSSF spinon+photon (S_xx + S_zz) across all magnetic field strengths.
    Creates separate animations for local and global channels.
    
    Parameters:
    -----------
    root_dir : str
        Root directory containing field subdirectories
    output_dir : str, optional
        Directory to save animations (default: root_dir/animations)
    fps : int
        Frames per second for animation
    energy_conversion : float
        Conversion factor to real energy units (meV), default 0.063
    """
    # Collect all data
    print("[animate_DSSF_spinon_photon] Collecting data from all field directories...")
    field_data = collect_DSSF_data_all_fields(root_dir, verbose=True)
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(root_dir, "animations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data arrays
    field_values = np.array([d['field'] for d in field_data])
    n_fields = len(field_values)
    
    # Check if all w arrays have the same length; if not, interpolate to a common grid
    w_lengths = [len(d['w']) for d in field_data]
    if len(set(w_lengths)) > 1:
        print(f"[animate_DSSF_spinon_photon] Warning: Different frequency grid sizes detected: {set(w_lengths)}")
        print("[animate_DSSF_spinon_photon] Interpolating all data to common frequency grid...")
        
        # Use the longest w array as the reference
        max_idx = np.argmax(w_lengths)
        w_common = field_data[max_idx]['w']
        
        # Interpolate all data to common grid
        for i, data in enumerate(field_data):
            if len(data['w']) != len(w_common):
                # Create interpolation functions
                data['dssf_local_spinon_photon'] = np.interp(w_common, data['w'], data['dssf_local_spinon_photon'])
                data['dssf_global_spinon_photon'] = np.interp(w_common, data['w'], data['dssf_global_spinon_photon'])
                data['dssf_local_xx'] = np.interp(w_common, data['w'], data['dssf_local_xx'])
                data['dssf_local_zz'] = np.interp(w_common, data['w'], data['dssf_local_zz'])
                data['dssf_global_xx'] = np.interp(w_common, data['w'], data['dssf_global_xx'])
                data['dssf_global_zz'] = np.interp(w_common, data['w'], data['dssf_global_zz'])
                data['w'] = w_common
        
        w = w_common * energy_conversion  # Convert to meV
    else:
        # All have same w array
        w = field_data[0]['w'] * energy_conversion  # Convert to meV
    
    # Stack all DSSF data (now they all have the same length)
    dssf_local_stack = np.array([d['dssf_local_spinon_photon'] for d in field_data])
    dssf_global_stack = np.array([d['dssf_global_spinon_photon'] for d in field_data])
    dssf_local_xx_stack = np.array([d['dssf_local_xx'] for d in field_data])
    dssf_local_zz_stack = np.array([d['dssf_local_zz'] for d in field_data])
    dssf_global_xx_stack = np.array([d['dssf_global_xx'] for d in field_data])
    dssf_global_zz_stack = np.array([d['dssf_global_zz'] for d in field_data])
    
    # Determine global y-axis limits for consistency
    y_max_local = np.max(dssf_local_stack) * 1.1
    y_max_global = np.max(dssf_global_stack) * 1.1
    
    # Animation for LOCAL channel
    print("[animate_DSSF_spinon_photon] Creating animation for LOCAL channel...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    line_total, = ax.plot([], [], 'r-', linewidth=2, label='S_xx + S_zz (Total)')
    line_xx, = ax.plot([], [], 'b--', linewidth=1.5, label='S_xx')
    line_zz, = ax.plot([], [], 'g--', linewidth=1.5, label='S_zz')
    
    ax.set_xlabel('ω (meV)', fontsize=12)
    ax.set_ylabel('DSSF (Local)', fontsize=12)
    ax.set_xlim(w.min(), w.max())
    ax.set_ylim(0, y_max_local)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    title_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, 
                         ha='center', va='top', fontsize=14, fontweight='bold')
    
    def init_local():
        line_total.set_data([], [])
        line_xx.set_data([], [])
        line_zz.set_data([], [])
        title_text.set_text('')
        return line_total, line_xx, line_zz, title_text
    
    def animate_local(frame):
        line_total.set_data(w, dssf_local_stack[frame])
        line_xx.set_data(w, dssf_local_xx_stack[frame])
        line_zz.set_data(w, dssf_local_zz_stack[frame])
        title_text.set_text(f'DSSF Spinon+Photon (Local) | Field = {field_values[frame]:.3f}')
        return line_total, line_xx, line_zz, title_text
    
    anim_local = FuncAnimation(fig, animate_local, init_func=init_local,
                               frames=n_fields, interval=1000//fps, blit=True, repeat=True)
    
    output_file_local = os.path.join(output_dir, 'DSSF_spinon_photon_local.gif')
    anim_local.save(output_file_local, writer=PillowWriter(fps=fps))
    print(f"[animate_DSSF_spinon_photon] Saved: {output_file_local}")
    plt.close(fig)
    
    # Animation for GLOBAL channel
    print("[animate_DSSF_spinon_photon] Creating animation for GLOBAL channel...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    line_total, = ax.plot([], [], 'r-', linewidth=2, label='S_xx + S_zz (Total)')
    line_xx, = ax.plot([], [], 'b--', linewidth=1.5, label='S_xx')
    line_zz, = ax.plot([], [], 'g--', linewidth=1.5, label='S_zz')
    
    ax.set_xlabel('ω (meV)', fontsize=12)
    ax.set_ylabel('DSSF (Global)', fontsize=12)
    ax.set_xlim(w.min(), w.max())
    ax.set_ylim(0, y_max_global)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    title_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, 
                         ha='center', va='top', fontsize=14, fontweight='bold')
    
    def init_global():
        line_total.set_data([], [])
        line_xx.set_data([], [])
        line_zz.set_data([], [])
        title_text.set_text('')
        return line_total, line_xx, line_zz, title_text
    
    def animate_global(frame):
        line_total.set_data(w, dssf_global_stack[frame])
        line_xx.set_data(w, dssf_global_xx_stack[frame])
        line_zz.set_data(w, dssf_global_zz_stack[frame])
        title_text.set_text(f'DSSF Spinon+Photon (Global) | Field = {field_values[frame]:.3f}')
        return line_total, line_xx, line_zz, title_text
    
    anim_global = FuncAnimation(fig, animate_global, init_func=init_global,
                                frames=n_fields, interval=1000//fps, blit=True, repeat=True)
    
    output_file_global = os.path.join(output_dir, 'DSSF_spinon_photon_global.gif')
    anim_global.save(output_file_global, writer=PillowWriter(fps=fps))
    print(f"[animate_DSSF_spinon_photon] Saved: {output_file_global}")
    plt.close(fig)
    
    print("[animate_DSSF_spinon_photon] Animation creation complete!")
    return output_file_local, output_file_global


def main():
    parser = argparse.ArgumentParser(
        description="Create animations of DSSF spinon+photon data across magnetic field strengths."
    )
    parser.add_argument(
        "root_dir",
        help="Root directory containing field subdirectories with DSSF data."
    )
    parser.add_argument(
        "--output",
        help="Output directory for animations (default: root_dir/animations)."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second for animations (default: 5)."
    )
    parser.add_argument(
        "--energy-conversion",
        type=float,
        default=0.063,
        help="Energy conversion factor to meV (default: 0.063)."
    )
    
    args = parser.parse_args()
    
    try:
        animate_DSSF_spinon_photon(
            args.root_dir,
            output_dir=args.output,
            fps=args.fps,
            energy_conversion=args.energy_conversion
        )
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
