#!/usr/bin/env python3
"""
Create static comparison plots showing DSSF evolution across field strengths.
This provides a preview of what the animations show, in a single static figure.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.gridspec import GridSpec


def extract_field_value(field_dir):
    """Extract numerical field value from directory name."""
    import re
    patterns = [r'field[_=]([0-9.]+)', r'h[_=]([0-9.]+)', r'B[_=]([0-9.]+)']
    for pattern in patterns:
        match = re.search(pattern, field_dir)
        if match:
            return float(match.group(1))
    return None


def collect_DSSF_data_all_fields(root_dir, verbose=True):
    """Collect DSSF spinon+photon data from all field directories."""
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

    field_data = []
    
    for field_dir in field_dirs:
        field_path = os.path.join(base_dir, field_dir)
        results_dir = os.path.join(field_path, "results")
        
        field_val = extract_field_value(field_dir)
        if field_val is None:
            if verbose:
                print(f"Could not extract field value from {field_dir}, skipping.")
            continue
        
        local_file = os.path.join(results_dir, "DSSF_local_xx.txt")
        local_zz_file = os.path.join(results_dir, "DSSF_local_zz.txt")
        global_file = os.path.join(results_dir, "DSSF_global_xx.txt")
        global_zz_file = os.path.join(results_dir, "DSSF_global_zz.txt")
        
        if not all(os.path.exists(f) for f in [local_file, local_zz_file, global_file, global_zz_file]):
            if verbose:
                print(f"Missing DSSF files in {field_dir}, skipping.")
            continue
        
        try:
            data_local_xx = np.loadtxt(local_file)
            data_local_zz = np.loadtxt(local_zz_file)
            data_global_xx = np.loadtxt(global_file)
            data_global_zz = np.loadtxt(global_zz_file)
            
            w = data_local_xx[:, 0]
            
            field_data.append({
                'field': field_val,
                'w': w,
                'dssf_local_spinon_photon': data_local_xx[:, 1] + data_local_zz[:, 1],
                'dssf_global_spinon_photon': data_global_xx[:, 1] + data_global_zz[:, 1],
                'dssf_local_xx': data_local_xx[:, 1],
                'dssf_local_zz': data_local_zz[:, 1],
                'dssf_global_xx': data_global_xx[:, 1],
                'dssf_global_zz': data_global_zz[:, 1]
            })
            
            if verbose:
                print(f"Loaded data for {field_dir} (field={field_val})")
                
        except Exception as e:
            if verbose:
                print(f"Error reading data from {field_dir}: {e}")
            continue
    
    if not field_data:
        raise RuntimeError(f"No valid DSSF data found in {root_dir}")
    
    field_data.sort(key=lambda x: x['field'])
    return field_data


def create_comparison_plots(root_dir, output_dir=None, energy_conversion=0.063, max_fields=10):
    """
    Create static comparison plots showing multiple field strengths.
    
    Parameters:
    -----------
    root_dir : str
        Root directory containing field subdirectories
    output_dir : str, optional
        Directory to save plots (default: root_dir/comparison_plots)
    energy_conversion : float
        Conversion factor to meV
    max_fields : int
        Maximum number of fields to show (will subsample if more)
    """
    print(f"Collecting data from {root_dir}...")
    field_data = collect_DSSF_data_all_fields(root_dir, verbose=True)
    
    if output_dir is None:
        output_dir = os.path.join(root_dir, "comparison_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    field_values = np.array([d['field'] for d in field_data])
    n_fields = len(field_values)
    
    # Check if all w arrays have the same length; if not, interpolate to a common grid
    w_lengths = [len(d['w']) for d in field_data]
    if len(set(w_lengths)) > 1:
        print(f"Warning: Different frequency grid sizes detected: {set(w_lengths)}")
        print("Interpolating all data to common frequency grid...")
        
        # Use the longest w array as the reference
        max_idx = np.argmax(w_lengths)
        w_common = field_data[max_idx]['w']
        
        # Interpolate all data to common grid
        for i, data in enumerate(field_data):
            if len(data['w']) != len(w_common):
                data['dssf_local_spinon_photon'] = np.interp(w_common, data['w'], data['dssf_local_spinon_photon'])
                data['dssf_global_spinon_photon'] = np.interp(w_common, data['w'], data['dssf_global_spinon_photon'])
                data['dssf_local_xx'] = np.interp(w_common, data['w'], data['dssf_local_xx'])
                data['dssf_local_zz'] = np.interp(w_common, data['w'], data['dssf_local_zz'])
                data['dssf_global_xx'] = np.interp(w_common, data['w'], data['dssf_global_xx'])
                data['dssf_global_zz'] = np.interp(w_common, data['w'], data['dssf_global_zz'])
                data['w'] = w_common
    
    # Subsample if too many fields
    if n_fields > max_fields:
        indices = np.linspace(0, n_fields-1, max_fields, dtype=int)
        field_data_subset = [field_data[i] for i in indices]
        print(f"Subsampling {max_fields} from {n_fields} field strengths")
    else:
        field_data_subset = field_data
        indices = range(n_fields)
    
    w = field_data[0]['w'] * energy_conversion
    
    # Create comparison plot for LOCAL channel
    print("Creating comparison plot for LOCAL channel...")
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(len(field_data_subset), 1, figure=fig, hspace=0.3)
    
    for i, data in enumerate(field_data_subset):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(w, data['dssf_local_spinon_photon'], 'r-', linewidth=1.5, label='S_xx + S_zz')
        ax.plot(w, data['dssf_local_xx'], 'b--', linewidth=1, alpha=0.7, label='S_xx')
        ax.plot(w, data['dssf_local_zz'], 'g--', linewidth=1, alpha=0.7, label='S_zz')
        ax.set_ylabel('DSSF', fontsize=10)
        ax.text(0.02, 0.95, f"Field = {data['field']:.3f}", transform=ax.transAxes,
                va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
        if i == len(field_data_subset) - 1:
            ax.set_xlabel('ω (meV)', fontsize=11)
        else:
            ax.set_xticklabels([])
    
    fig.suptitle('DSSF Spinon+Photon Evolution (Local Channel)', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'DSSF_comparison_local.pdf'), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(output_dir, 'DSSF_comparison_local.png'), bbox_inches='tight', dpi=150)
    print(f"Saved: {os.path.join(output_dir, 'DSSF_comparison_local.pdf')}")
    plt.close()
    
    # Create comparison plot for GLOBAL channel
    print("Creating comparison plot for GLOBAL channel...")
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(len(field_data_subset), 1, figure=fig, hspace=0.3)
    
    for i, data in enumerate(field_data_subset):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(w, data['dssf_global_spinon_photon'], 'r-', linewidth=1.5, label='S_xx + S_zz')
        ax.plot(w, data['dssf_global_xx'], 'b--', linewidth=1, alpha=0.7, label='S_xx')
        ax.plot(w, data['dssf_global_zz'], 'g--', linewidth=1, alpha=0.7, label='S_zz')
        ax.set_ylabel('DSSF', fontsize=10)
        ax.text(0.02, 0.95, f"Field = {data['field']:.3f}", transform=ax.transAxes,
                va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
        if i == len(field_data_subset) - 1:
            ax.set_xlabel('ω (meV)', fontsize=11)
        else:
            ax.set_xticklabels([])
    
    fig.suptitle('DSSF Spinon+Photon Evolution (Global Channel)', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'DSSF_comparison_global.pdf'), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(output_dir, 'DSSF_comparison_global.png'), bbox_inches='tight', dpi=150)
    print(f"Saved: {os.path.join(output_dir, 'DSSF_comparison_global.pdf')}")
    plt.close()
    
    # Create 2D heatmap-style plot
    print("Creating 2D heatmap plots...")
    
    # Stack all data for heatmap
    dssf_local_stack = np.array([d['dssf_local_spinon_photon'] for d in field_data])
    dssf_global_stack = np.array([d['dssf_global_spinon_photon'] for d in field_data])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Local heatmap
    im1 = ax1.imshow(dssf_local_stack, aspect='auto', origin='lower',
                     extent=[w.min(), w.max(), field_values.min(), field_values.max()],
                     cmap='hot', interpolation='bilinear')
    ax1.set_xlabel('ω (meV)', fontsize=12)
    ax1.set_ylabel('Magnetic Field', fontsize=12)
    ax1.set_title('DSSF Spinon+Photon (Local)', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Intensity')
    
    # Global heatmap
    im2 = ax2.imshow(dssf_global_stack, aspect='auto', origin='lower',
                     extent=[w.min(), w.max(), field_values.min(), field_values.max()],
                     cmap='hot', interpolation='bilinear')
    ax2.set_xlabel('ω (meV)', fontsize=12)
    ax2.set_ylabel('Magnetic Field', fontsize=12)
    ax2.set_title('DSSF Spinon+Photon (Global)', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Intensity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'DSSF_heatmap.pdf'), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(output_dir, 'DSSF_heatmap.png'), bbox_inches='tight', dpi=150)
    print(f"Saved: {os.path.join(output_dir, 'DSSF_heatmap.pdf')}")
    plt.close()
    
    print("\nComparison plots created successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Create static comparison plots of DSSF data across field strengths."
    )
    parser.add_argument("root_dir", help="Root directory containing field subdirectories.")
    parser.add_argument("--output", help="Output directory for plots (default: root_dir/comparison_plots).")
    parser.add_argument("--max-fields", type=int, default=10,
                        help="Maximum number of fields to show in comparison (default: 10).")
    parser.add_argument("--energy-conversion", type=float, default=0.063,
                        help="Energy conversion factor to meV (default: 0.063).")
    
    args = parser.parse_args()
    
    try:
        create_comparison_plots(
            args.root_dir,
            output_dir=args.output,
            energy_conversion=args.energy_conversion,
            max_fields=args.max_fields
        )
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
