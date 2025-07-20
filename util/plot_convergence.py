#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def plot_convergence_analysis(data_file, output_dir="./"):
    """
    Plot convergence analysis from the mixed lattice simulated annealing
    
    Args:
        data_file: Path to convergence_data.txt file
        output_dir: Directory to save plots
    """
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        return
    
    # Read data
    try:
        data = np.loadtxt(data_file, skiprows=1)  # Skip header
        steps = data[:, 0]
        energies = data[:, 1]
        acceptance_rates = data[:, 2]
        config_changes = data[:, 3]
    except Exception as e:
        print(f"Error reading data: {e}")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Simulated Annealing Convergence Analysis', fontsize=16)
    
    # Energy evolution
    axes[0, 0].plot(steps, energies, 'b-', alpha=0.7, linewidth=1)
    axes[0, 0].set_xlabel('MC Steps')
    axes[0, 0].set_ylabel('Energy')
    axes[0, 0].set_title('Energy Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add moving average
    if len(energies) > 100:
        window = min(100, len(energies) // 10)
        energy_smooth = np.convolve(energies, np.ones(window)/window, mode='same')
        axes[0, 0].plot(steps, energy_smooth, 'r-', linewidth=2, label='Moving Average')
        axes[0, 0].legend()
    
    # Acceptance rate
    axes[0, 1].plot(steps, acceptance_rates, 'g-', alpha=0.7, linewidth=1)
    axes[0, 1].axhline(y=0.44, color='red', linestyle='--', label='Target (0.44)')
    axes[0, 1].axhline(y=0.39, color='orange', linestyle=':', alpha=0.7, label='Tolerance')
    axes[0, 1].axhline(y=0.49, color='orange', linestyle=':', alpha=0.7)
    axes[0, 1].set_xlabel('MC Steps')
    axes[0, 1].set_ylabel('Acceptance Rate')
    axes[0, 1].set_title('Acceptance Rate Evolution')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Configuration changes
    axes[1, 0].semilogy(steps, config_changes, 'm-', alpha=0.7, linewidth=1)
    axes[1, 0].set_xlabel('MC Steps')
    axes[1, 0].set_ylabel('Configuration Change (log scale)')
    axes[1, 0].set_title('Configuration Stability')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Energy variance (rolling)
    if len(energies) > 100:
        window = min(100, len(energies) // 5)
        energy_variance = []
        variance_steps = []
        
        for i in range(window, len(energies)):
            local_energies = energies[i-window:i]
            variance = np.var(local_energies, ddof=1)
            energy_variance.append(variance)
            variance_steps.append(steps[i])
        
        axes[1, 1].semilogy(variance_steps, energy_variance, 'c-', alpha=0.7, linewidth=1)
        axes[1, 1].set_xlabel('MC Steps')
        axes[1, 1].set_ylabel('Energy Variance (log scale)')
        axes[1, 1].set_title(f'Energy Variance (window={window})')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, 'convergence_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.show()
    
    # Print convergence statistics
    print("\nConvergence Statistics:")
    print("=" * 40)
    print(f"Final energy: {energies[-1]:.6f}")
    print(f"Final acceptance rate: {acceptance_rates[-1]:.4f}")
    print(f"Final config change: {config_changes[-1]:.2e}")
    
    if len(energies) > 100:
        final_100_energies = energies[-100:]
        energy_std = np.std(final_100_energies, ddof=1)
        energy_mean = np.mean(final_100_energies)
        print(f"Final 100 steps - Energy mean: {energy_mean:.6f}, std: {energy_std:.2e}")
        
        # Check convergence criteria
        energy_variance_final = np.var(final_100_energies, ddof=1)
        acceptance_deviation = abs(np.mean(acceptance_rates[-100:]) - 0.44)
        config_change_final = np.mean(config_changes[-100:])
        
        print(f"\nConvergence Check (last 100 steps):")
        print(f"Energy variance: {energy_variance_final:.2e} (target: < 1e-8)")
        print(f"Acceptance deviation: {acceptance_deviation:.4f} (target: < 0.05)")
        print(f"Config change: {config_change_final:.2e} (target: < 1e-6)")
        
        # Overall convergence assessment
        energy_converged = energy_variance_final < 1e-8
        acceptance_converged = acceptance_deviation < 0.05
        config_converged = config_change_final < 1e-6
        
        print(f"\nConvergence Status:")
        print(f"Energy: {'✓' if energy_converged else '✗'}")
        print(f"Acceptance: {'✓' if acceptance_converged else '✗'}")
        print(f"Configuration: {'✓' if config_converged else '✗'}")
        print(f"Overall: {'✓ CONVERGED' if all([energy_converged, acceptance_converged, config_converged]) else '✗ NOT CONVERGED'}")

def plot_energy_only(data_file, output_dir="./"):
    """Simple energy-only plot for quick visualization"""
    data = np.loadtxt(data_file, skiprows=1)
    steps = data[:, 0]
    energies = data[:, 1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, energies, 'b-', alpha=0.7, linewidth=1, label='Energy')
    
    # Add moving average
    if len(energies) > 50:
        window = min(50, len(energies) // 10)
        energy_smooth = np.convolve(energies, np.ones(window)/window, mode='same')
        plt.plot(steps, energy_smooth, 'r-', linewidth=2, label=f'Moving Average (window={window})')
    
    plt.xlabel('MC Steps')
    plt.ylabel('Energy')
    plt.title('Energy Evolution During Simulated Annealing')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_file = os.path.join(output_dir, 'energy_evolution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Energy plot saved to: {output_file}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_convergence.py <convergence_data.txt> [output_dir]")
        print("Example: python plot_convergence.py enhanced_SA/convergence_data.txt ./plots/")
        sys.exit(1)
    
    data_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plot_convergence_analysis(data_file, output_dir)
