#!/usr/bin/env python3
"""
Example script to read and analyze parallel tempering HDF5 output
from ClassicalSpin_Cpp simulations.

Usage:
    python read_pt_hdf5.py <output_directory>
    
Example:
    python read_pt_hdf5.py output_ncto_strain_sweep/
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path


def read_replica_data(h5_file_path):
    """
    Read data from a single replica's HDF5 file.
    
    Returns:
        dict: Dictionary containing all data and metadata
    """
    with h5py.File(h5_file_path, 'r') as f:
        data = {}
        
        # Read metadata
        metadata = {}
        for key in f['/metadata'].attrs.keys():
            metadata[key] = f['/metadata'].attrs[key]
        data['metadata'] = metadata
        
        # Read time series
        timeseries = {}
        for key in f['/timeseries'].keys():
            timeseries[key] = f[f'/timeseries/{key}'][:]
        data['timeseries'] = timeseries
        
        # Read observables
        observables = {}
        for key in f['/observables'].keys():
            observables[key] = f[f'/observables/{key}'][()]
        data['observables'] = observables
        
    return data


def read_aggregated_data(h5_file_path):
    """
    Read aggregated temperature scan data.
    
    Returns:
        dict: Dictionary with temperature, specific_heat, and errors
    """
    with h5py.File(h5_file_path, 'r') as f:
        data = {
            'temperature': f['/temperature_scan/temperature'][:],
            'specific_heat': f['/temperature_scan/specific_heat'][:],
            'specific_heat_error': f['/temperature_scan/specific_heat_error'][:]
        }
    return data


def plot_timeseries(data, rank=0):
    """
    Plot time series data from a replica.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    T = data['metadata']['temperature']
    n_samples = len(data['timeseries']['energy'])
    steps = np.arange(n_samples)
    
    # Energy time series
    ax = axes[0, 0]
    ax.plot(steps, data['timeseries']['energy'], alpha=0.7)
    ax.axhline(data['observables']['energy_mean'], 
               color='r', linestyle='--', label='Mean')
    ax.fill_between(steps, 
                     data['observables']['energy_mean'] - data['observables']['energy_error'],
                     data['observables']['energy_mean'] + data['observables']['energy_error'],
                     alpha=0.3, color='r')
    ax.set_xlabel('Monte Carlo Step')
    ax.set_ylabel('Energy per Site')
    ax.set_title(f'Energy Time Series (T={T:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Magnetization magnitude
    ax = axes[0, 1]
    mag = data['timeseries']['magnetization']
    mag_magnitude = np.linalg.norm(mag, axis=1)
    ax.plot(steps, mag_magnitude, alpha=0.7)
    ax.set_xlabel('Monte Carlo Step')
    ax.set_ylabel('|M|')
    ax.set_title(f'Magnetization Magnitude (T={T:.4f})')
    ax.grid(True, alpha=0.3)
    
    # Energy histogram
    ax = axes[1, 0]
    ax.hist(data['timeseries']['energy'], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(data['observables']['energy_mean'], 
               color='r', linestyle='--', linewidth=2, label='Mean')
    ax.set_xlabel('Energy per Site')
    ax.set_ylabel('Counts')
    ax.set_title(f'Energy Distribution (T={T:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sublattice magnetizations
    ax = axes[1, 1]
    n_sublattices = data['metadata']['n_sublattices']
    for alpha in range(n_sublattices):
        key = f'sublattice_mag_{alpha}'
        if key in data['timeseries']:
            sub_mag = data['timeseries'][key]
            sub_mag_magnitude = np.linalg.norm(sub_mag, axis=1)
            ax.plot(steps, sub_mag_magnitude, alpha=0.7, label=f'Sublattice {alpha}')
    ax.set_xlabel('Monte Carlo Step')
    ax.set_ylabel('|M_sublattice|')
    ax.set_title(f'Sublattice Magnetizations (T={T:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_temperature_scan(agg_data):
    """
    Plot specific heat vs temperature.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    T = agg_data['temperature']
    C_V = agg_data['specific_heat']
    dC_V = agg_data['specific_heat_error']
    
    ax.errorbar(T, C_V, yerr=dC_V, fmt='o-', capsize=5, 
                linewidth=2, markersize=8, label='Specific Heat')
    
    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel('Specific Heat $C_V$', fontsize=12)
    ax.set_title('Parallel Tempering: Specific Heat vs Temperature', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    return fig


def print_summary(data):
    """
    Print a summary of the replica data.
    """
    metadata = data['metadata']
    obs = data['observables']
    
    print("\n" + "="*60)
    print("PARALLEL TEMPERING REPLICA SUMMARY")
    print("="*60)
    
    print("\nSimulation Parameters:")
    print(f"  Temperature:          {metadata['temperature']:.6f}")
    print(f"  Lattice Size:         {metadata['lattice_size']}")
    print(f"  Spin Dimension:       {metadata['spin_dim']}")
    print(f"  Number of Sublattices: {metadata['n_sublattices']}")
    print(f"  Equilibration Sweeps: {metadata['n_anneal']}")
    print(f"  Measurement Sweeps:   {metadata['n_measure']}")
    print(f"  Probe Rate:           {metadata['probe_rate']}")
    print(f"  Acceptance Rate:      {metadata['acceptance_rate']:.4f}")
    print(f"  Swap Accept Rate:     {metadata['swap_acceptance_rate']:.4f}")
    
    print("\nThermodynamic Observables:")
    print(f"  Energy per Site:      {obs['energy_mean']:.8f} ± {obs['energy_error']:.8f}")
    print(f"  Specific Heat:        {obs['specific_heat_mean']:.8f} ± {obs['specific_heat_error']:.8f}")
    
    print("\nSublattice Magnetizations:")
    for alpha in range(metadata['n_sublattices']):
        mean_key = f'sublattice_mag_{alpha}_mean'
        error_key = f'sublattice_mag_{alpha}_error'
        if mean_key in obs:
            mean = obs[mean_key]
            error = obs[error_key]
            mag = np.linalg.norm(mean)
            print(f"  Sublattice {alpha}: |M| = {mag:.6f}")
            print(f"    Components: {mean}")
    
    print("\nFile Information:")
    print(f"  Creation Time:        {metadata['creation_time']}")
    print(f"  Code Version:         {metadata['code_version']}")
    print(f"  File Format:          {metadata['file_format']}")
    print("="*60 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python read_pt_hdf5.py <output_directory>")
        print("\nExample: python read_pt_hdf5.py output_parallel_tempering/")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    
    if not output_dir.exists():
        print(f"Error: Directory {output_dir} does not exist")
        sys.exit(1)
    
    # Find all rank directories
    rank_dirs = sorted(output_dir.glob("rank_*"))
    
    if not rank_dirs:
        print(f"Error: No rank_* directories found in {output_dir}")
        sys.exit(1)
    
    print(f"Found {len(rank_dirs)} rank directories")
    
    # Read and plot data from first replica
    rank_0_file = rank_dirs[0] / "parallel_tempering_data.h5"
    if rank_0_file.exists():
        print(f"\nReading {rank_0_file}...")
        data = read_replica_data(rank_0_file)
        print_summary(data)
        
        # Plot time series
        fig = plot_timeseries(data, rank=0)
        plt.savefig(output_dir / "timeseries_rank0.png", dpi=150, bbox_inches='tight')
        print(f"Saved time series plot to {output_dir}/timeseries_rank0.png")
        plt.close()
    else:
        print(f"Warning: {rank_0_file} not found")
    
    # Read and plot aggregated data
    agg_file = output_dir / "parallel_tempering_aggregated.h5"
    if agg_file.exists():
        print(f"\nReading {agg_file}...")
        agg_data = read_aggregated_data(agg_file)
        
        print("\nTemperature Scan Summary:")
        print(f"  Number of temperatures: {len(agg_data['temperature'])}")
        print(f"  Temperature range: {agg_data['temperature'].min():.4f} - {agg_data['temperature'].max():.4f}")
        print(f"  Max specific heat: {agg_data['specific_heat'].max():.6f}")
        
        # Plot temperature scan
        fig = plot_temperature_scan(agg_data)
        plt.savefig(output_dir / "specific_heat_vs_T.png", dpi=150, bbox_inches='tight')
        print(f"Saved temperature scan plot to {output_dir}/specific_heat_vs_T.png")
        plt.close()
    else:
        print(f"Warning: {agg_file} not found")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
