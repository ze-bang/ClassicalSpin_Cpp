import numpy as np
from matplotlib.patches import FancyArrowPatch
import argparse
import os

import matplotlib.pyplot as plt

def read_data_file(filename):
    """Read data from a file and return as numpy array."""
    with open(filename, 'r') as f:
        lines = f.readlines()
        # Skip comments and filter empty lines
        data_lines = [line.strip() for line in lines if not line.startswith('//') and line.strip()]
        data = np.array([list(map(float, line.split())) for line in data_lines])
    return data

kitaevLocal = np.array([[1,-1,-1],[-1,1,-1],[-1,-1,1]])/np.sqrt(3)

def plot_spin_configuration(pos_file, spin_file, filename, kitaev=False):
    # Read position and spin data
    positions = read_data_file(pos_file)
    spins = read_data_file(spin_file)
    
    # Extract x, y coordinates from positions and x, y components from spins
    x_pos = positions[:, 0]
    y_pos = positions[:, 1]
    
    # Spin components
    if kitaev:
        spins = np.einsum('ij, jk->ik', spins, kitaevLocal)
        spin_x = spins[:, 0]
        spin_y = spins[:, 1]
        spin_z = spins[:, 2]
    else:
        spin_x = spins[:, 0]
        spin_y = spins[:, 1]
        spin_z = spins[:, 2]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot spin arrows
    # Scale factor for arrow length
    scale = 0.3
    
    # Plot each spin as an arrow
    for i in range(len(x_pos)):
        ax.arrow(x_pos[i], y_pos[i], 
                 scale * spin_x[i], scale * spin_y[i],
                 head_width=0.1, head_length=0.15, 
                 fc='red', ec='black', 
                 length_includes_head=True)
    
    # Plot positions (lattice sites)
    ax.scatter(x_pos, y_pos, color='blue', s=10, alpha=0.5)
    
    # Set plot limits with some padding
    x_min, x_max = min(x_pos), max(x_pos)
    y_min, y_max = min(y_pos), max(y_pos)
    padding = 1.0
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    
    # Add title and labels
    ax.set_title('Spin Configuration in Real Space')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    
    # Add colorbar to represent spin z component
    scatter = ax.scatter(x_pos, y_pos, c=spin_z, cmap='coolwarm', s=0)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Spin z-component')
    
    # Equal aspect ratio to avoid distortion
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()



if __name__ == "__main__":
    # Example usage

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot spin configuration from files in a directory.')
    parser.add_argument('directory', type=str, help='Directory containing position and spin files')
    args = parser.parse_args()

    # Get directory from command line
    directory = args.directory

    # Update file paths to use the specified directory
    pos_file = os.path.join(directory, 'pos.txt')
    spin_file = os.path.join(directory, 'spin.txt')


    # Plot without Kitaev transformation
    plot_spin_configuration(pos_file, spin_file, args.directory+'spin_config.png', False)
