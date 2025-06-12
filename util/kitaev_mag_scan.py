import os
import numpy as np
from single_q import SingleQ
from mpi4py import MPI
from matplotlib.colors import Normalize
from single_q_fixed_q import SingleQFixedQ

# Base directory for output
base_dir = "/home/pc_linux/ClassicalSpin_Cpp/BCAO_mag_field_scan"
os.makedirs(base_dir, exist_ok=True)


# Default J parameters: [J1, Jpmpm, Jzp, Delta1, J2, J3, Delta3]
default_J = [-6.54, 0.0, -3.76, 0.36, -0.21, 1.9, 0.0]

# Use the same lattice size as in the example files
lattice_size = 24

def compute_spin_configuration_single_q(B_field):
    """Compute spin configuration using SingleQ with given magnetic field."""
    # Create SingleQ model with magnetic field
    model = SingleQ(lattice_size, default_J, B_field=B_field)
    
    # Extract spin configuration and energy
    spins = model.generate_spin_configuration()
    
    model_0 = SingleQFixedQ(0,0, lattice_size, default_J, B_field=B_field)
    model_zzy = SingleQFixedQ(0, 0.5, lattice_size, default_J, B_field=B_field)
    model_zzx = SingleQFixedQ(0.5, 0, lattice_size, default_J, B_field=B_field)
    model_zz = SingleQFixedQ(0.5, 0.5, lattice_size, default_J, B_field=B_field)

    # Generate spin configurations for fixed q models
    spins_0 = model_0.generate_spin_configuration()
    spins_zzy = model_zzy.generate_spin_configuration()
    spins_zzx = model_zzx.generate_spin_configuration()
    spins_zz = model_zz.generate_spin_configuration()

    # Compare energies of all models
    energies = {
        "SingleQ": model.opt_energy,
        "q=(0,0)": model_0.opt_energy,
        "q=(0,0.5)": model_zzy.opt_energy,
        "q=(0.5,0)": model_zzx.opt_energy,
        "q=(0.5,0.5)": model_zz.opt_energy
    }

    # Find the model with lowest energy
    lowest_energy_model = min(energies, key=energies.get)
    print(f"Lowest energy model: {lowest_energy_model} with energy {energies[lowest_energy_model]}")

    # Return results based on the model with lowest energy
    if lowest_energy_model == "SingleQ":
        return spins, model.opt_energy, model.opt_params[0], model.opt_params[1]
    elif lowest_energy_model == "q=(0,0)":
        return spins_0, model_0.opt_energy, 0, 0
    elif lowest_energy_model == "q=(0,0.5)":
        return spins_zzy, model_zzy.opt_energy, 0, 0.5
    elif lowest_energy_model == "q=(0.5,0)":
        return spins_zzx, model_zzx.opt_energy, 0.5, 0
    else:  # q=(0.5,0.5)
        return spins_zz, model_zz.opt_energy, 0.5, 0.5

def process_grid_1d(b_magnitude_min, b_magnitude_max, n_points, direction='z', output_dir=None):
    """
    Process a 1D scan of magnetic field magnitude along a specific direction.
    
    Args:
        b_magnitude_min: Minimum magnetic field magnitude
        b_magnitude_max: Maximum magnetic field magnitude
        n_points: Number of points to scan
        direction: Direction of field ('x', 'y', or 'z')
        output_dir: Directory to save results
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Create field values
    b_magnitudes = np.linspace(b_magnitude_min, b_magnitude_max, n_points)
    
    # Create direction vector
    if direction == 'x':
        dir_vec = np.array([1, 0, 0])
    elif direction == 'y':
        dir_vec = np.array([0, 1, 0])
    elif direction == 'z':
        dir_vec = np.array([0, 0, 1])
    else:
        raise ValueError("Direction must be 'x', 'y', or 'z'")
    
    # Distribute points among processes
    points_per_process = n_points // size
    remainder = n_points % size
    start_idx = rank * points_per_process + min(rank, remainder)
    end_idx = start_idx + points_per_process + (1 if rank < remainder else 0)
    my_magnitudes = b_magnitudes[start_idx:end_idx]
    
    # Set up output directory and files on root process
    if rank == 0:
        if output_dir is None:
            output_dir = os.path.join(base_dir, f"B_field_{direction}_scan")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        phase_diagram_file = os.path.join(output_dir, f"phase_diagram_B_{direction}.txt")
        with open(phase_diagram_file, 'w') as f:
            f.write(f"# B_{direction} Energy Q1 Q2 Phase\n")
        
        print(f"Processing 1D scan: B_{direction}=[{b_magnitude_min}, {b_magnitude_max}], {n_points} points")
        print(f"Distributing {n_points} points across {size} processes")
    
    # Ensure all processes wait for directory creation
    comm.Barrier()
    if output_dir is None:
        output_dir = os.path.join(base_dir, f"B_field_{direction}_scan")
    
    # Process assigned points
    local_results = []
    for b_mag in my_magnitudes:
        B_field = b_mag * dir_vec
        print(f"Process {rank}: Computing B_{direction}={b_mag:.4f}")
        
        point_dir = os.path.join(output_dir, f"B_{direction}_{b_mag:.4f}")
        if not os.path.exists(point_dir):
            os.makedirs(point_dir)
        
        try:
            spins, energy, q1, q2 = compute_spin_configuration_single_q(B_field)
            
            # Save spin configuration
            output_path = os.path.join(point_dir, "spins_single_q.txt")
            with open(output_path, 'w') as f:
                for i in range(spins.shape[0]):
                    f.write(f"{spins[i, 0]} {spins[i, 1]} {spins[i, 2]}\n")
            
            # Save parameters
            params_path = os.path.join(point_dir, "parameters.txt")
            with open(params_path, 'w') as f:
                f.write(f"B_field: [{B_field[0]}, {B_field[1]}, {B_field[2]}]\n")
                f.write(f"Energy: {energy}\n")
                f.write(f"Q_vector: ({q1}, {q2})\n")
            
            local_results.append((b_mag, energy, q1, q2))
            
        except Exception as e:
            print(f"Process {rank}: Error for B_{direction}={b_mag}: {e}")
    
    # Gather results from all processes
    all_results = comm.gather(local_results, root=0)
    
    # Root process writes results to file
    if rank == 0:
        all_results_flat = [r for results in all_results for r in results]
        all_results_flat.sort(key=lambda x: x[0])  # Sort by field magnitude
        
        with open(phase_diagram_file, 'a') as f:
            for b_mag, energy, q1, q2 in all_results_flat:
                f.write(f"{b_mag} {energy} {q1} {q2}\n")
        
        print(f"Completed 1D scan along {direction} direction")

def calculate_magnetization(spins):
    """Calculate the total magnetization vector from spin configuration."""
    return np.mean(spins, axis=0)

def plot_magnetization_1d(output_dir, direction='z'):
    """Plot magnetization vs field strength for 1D scan."""
    import matplotlib.pyplot as plt
    
    # Read phase diagram file
    phase_diagram_file = os.path.join(output_dir, f"phase_diagram_B_{direction}.txt")
    
    # Load data
    data = np.loadtxt(phase_diagram_file)
    b_values = data[:, 0]
    
    # Calculate magnetization for each point
    mag_x, mag_y, mag_z, mag_total = [], [], [], []
    
    for i, b_val in enumerate(b_values):
        # Load spin configuration v
        point_dir = os.path.join(output_dir, f"B_{direction}_{b_val:.4f}")
        spin_file = os.path.join(point_dir, "spins_single_q.txt")
        
        if os.path.exists(spin_file):
            spins = np.loadtxt(spin_file)
            magnetization = calculate_magnetization(spins)
            mag_x.append(magnetization[0])
            mag_y.append(magnetization[1])
            mag_z.append(magnetization[2])
            mag_total.append(np.linalg.norm(magnetization))
            
            # Save magnetization to file
            mag_file = os.path.join(point_dir, "magnetization.txt")
            with open(mag_file, 'w') as f:
                f.write(f"M_x: {magnetization[0]}\n")
                f.write(f"M_y: {magnetization[1]}\n")
                f.write(f"M_z: {magnetization[2]}\n")
                f.write(f"M_total: {np.linalg.norm(magnetization)}\n")
    
    # Create magnetization summary file
    mag_summary_file = os.path.join(output_dir, f"magnetization_vs_B_{direction}.txt")
    with open(mag_summary_file, 'w') as f:
        f.write(f"# B_{direction} M_x M_y M_z M_total\n")
        for i, b_val in enumerate(b_values):
            if i < len(mag_x):
                f.write(f"{b_val} {mag_x[i]} {mag_y[i]} {mag_z[i]} {mag_total[i]}\n")
    
    # Plot magnetization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    b_values = np.array(b_values)/6.54

    # Plot magnetization components
    ax1.plot(b_values[:len(mag_x)], mag_x, 'r-', label='M_x')
    ax1.plot(b_values[:len(mag_y)], mag_y, 'g-', label='M_y')
    ax1.plot(b_values[:len(mag_z)], mag_z, 'b-', label='M_z')
    ax1.set_xlabel(f'B_{direction}/ |J_1|')
    ax1.set_ylabel('Magnetization components')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title(f'Magnetization components vs B_{direction}')
    
    # Plot total magnetization
    ax2.plot(b_values[:len(mag_total)], mag_total, 'k-', linewidth=2)
    ax2.set_xlabel(f'B_{direction}/ |J_1|')
    ax2.set_ylabel('Total magnetization |M|')
    ax2.grid(True)
    ax2.set_title(f'Total magnetization vs B_{direction}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'magnetization_vs_B_{direction}.png'), dpi=300)
    plt.close()
    
    print(f"Magnetization plot saved to {output_dir}/magnetization_vs_B_{direction}.png")

def plot_magnetization_2d(output_dir, plane='xy'):
    """Plot magnetization heatmaps for 2D scan."""
    import matplotlib.pyplot as plt
    
    # Read phase diagram file
    phase_diagram_file = os.path.join(output_dir, f"phase_diagram_B_{plane}.txt")
    
    # Load data
    data = np.loadtxt(phase_diagram_file)
    
    # Extract unique values for grid
    if plane == 'xy':
        b1_values = np.unique(data[:, 0])
        b2_values = np.unique(data[:, 1])
        xlabel, ylabel = 'B_x', 'B_y'
    elif plane == 'xz':
        b1_values = np.unique(data[:, 0])
        b2_values = np.unique(data[:, 2])
        xlabel, ylabel = 'B_x', 'B_z'
    else:  # yz
        b1_values = np.unique(data[:, 1])
        b2_values = np.unique(data[:, 2])
        xlabel, ylabel = 'B_y', 'B_z'
    
    # Create grids for magnetization
    mag_total_grid = np.zeros((len(b2_values), len(b1_values)))
    mag_x_grid = np.zeros((len(b2_values), len(b1_values)))
    mag_y_grid = np.zeros((len(b2_values), len(b1_values)))
    mag_z_grid = np.zeros((len(b2_values), len(b1_values)))
    
    # Calculate magnetization for each point
    for i, row in enumerate(data):
        bx, by, bz = row[0], row[1], row[2]
        
        # Load spin configuration
        point_dir = os.path.join(output_dir, f"Bx_{bx:.4f}_By_{by:.4f}_Bz_{bz:.4f}")
        spin_file = os.path.join(point_dir, "spins_single_q.txt")
        
        if os.path.exists(spin_file):
            spins = np.loadtxt(spin_file)
            magnetization = calculate_magnetization(spins)
            
            # Find indices in grid
            if plane == 'xy':
                idx1 = np.argmin(np.abs(b1_values - bx))
                idx2 = np.argmin(np.abs(b2_values - by))
            elif plane == 'xz':
                idx1 = np.argmin(np.abs(b1_values - bx))
                idx2 = np.argmin(np.abs(b2_values - bz))
            else:  # yz
                idx1 = np.argmin(np.abs(b1_values - by))
                idx2 = np.argmin(np.abs(b2_values - bz))
            
            mag_total_grid[idx2, idx1] = np.linalg.norm(magnetization)
            mag_x_grid[idx2, idx1] = magnetization[0]
            mag_y_grid[idx2, idx1] = magnetization[1]
            mag_z_grid[idx2, idx1] = magnetization[2]
            
            # Save magnetization to file
            mag_file = os.path.join(point_dir, "magnetization.txt")
            with open(mag_file, 'w') as f:
                f.write(f"M_x: {magnetization[0]}\n")
                f.write(f"M_y: {magnetization[1]}\n")
                f.write(f"M_z: {magnetization[2]}\n")
                f.write(f"M_total: {np.linalg.norm(magnetization)}\n")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot total magnetization
    im1 = axes[0, 0].imshow(mag_total_grid, origin='lower', 
                            extent=[b1_values[0], b1_values[-1], b2_values[0], b2_values[-1]], 
                            aspect='auto', cmap='viridis')
    axes[0, 0].set_xlabel(xlabel)
    axes[0, 0].set_ylabel(ylabel)
    axes[0, 0].set_title('Total Magnetization |M|')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot M_x
    im2 = axes[0, 1].imshow(mag_x_grid, origin='lower',
                            extent=[b1_values[0], b1_values[-1], b2_values[0], b2_values[-1]],
                            aspect='auto', cmap='RdBu', norm=Normalize(vmin=-1, vmax=1))
    axes[0, 1].set_xlabel(xlabel)
    axes[0, 1].set_ylabel(ylabel)
    axes[0, 1].set_title('M_x')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot M_y
    im3 = axes[1, 0].imshow(mag_y_grid, origin='lower',
                            extent=[b1_values[0], b1_values[-1], b2_values[0], b2_values[-1]],
                            aspect='auto', cmap='RdBu', norm=Normalize(vmin=-1, vmax=1))
    axes[1, 0].set_xlabel(xlabel)
    axes[1, 0].set_ylabel(ylabel)
    axes[1, 0].set_title('M_y')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Plot M_z
    im4 = axes[1, 1].imshow(mag_z_grid, origin='lower',
                            extent=[b1_values[0], b1_values[-1], b2_values[0], b2_values[-1]],
                            aspect='auto', cmap='RdBu', norm=Normalize(vmin=-1, vmax=1))
    axes[1, 1].set_xlabel(xlabel)
    axes[1, 1].set_ylabel(ylabel)
    axes[1, 1].set_title('M_z')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'magnetization_heatmap_{plane}.png'), dpi=300)
    plt.close()
    
    print(f"Magnetization heatmap saved to {output_dir}/magnetization_heatmap_{plane}.png")

if __name__ == "__main__":
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Example 1: 1D scan along z direction
    # process_grid_1d(0.0, 5, 100, direction='x')
    plot_magnetization_1d(os.path.join(base_dir, "B_field_x_scan"), direction='x')
    # process_grid_1d(0.0, 5, 100, direction='y')
    plot_magnetization_1d(os.path.join(base_dir, "B_field_y_scan"), direction='y')
    
    # Example 2: 2D scan in xy plane with Bz=0
    # process_grid_2d((0.0, 5.0), (0.0, 5.0), 20, 20, plane='xy', b_fixed=0.0)
    
    # Example 3: 3D scan (use smaller grid for testing)
    # process_grid_3d((0.0, 2.0), (0.0, 2.0), (0.0, 2.0), 10, 10, 10)