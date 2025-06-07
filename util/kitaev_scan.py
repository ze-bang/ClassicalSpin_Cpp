import os
import numpy as np
import re
from single_q import SingleQ
from luttinger_tisza import luttinger_tisza_method
from single_q_fixed_q import SingleQFixedQ
from mpi4py import MPI
# Base directory containing all subdirectories
base_dir = "/home/pc_linux/ClassicalSpin_Cpp/BCAO_sasha_phase_zero"
# Import MPI

# Initialize MPI environment

# Default J parameters: [J1, Jpmpm, Jzp, Delta1, J2, J3, Delta3]
# default_J = [-6.54, 0.15, -3.76, 0.36, -0.21, 1.70, 0.03]
default_J = [-6.54, 0.0, -3.76, 0.36, -0.21, 1.70, 0.0]

# Use the same lattice size as in the example files
lattice_size = 24

def parse_directory_name(dir_name):
    """Extract J3 and Jzp values from directory name."""
    match = re.match(r'J3_(\d+\.\d+)_Jzp_(-?\d+\.\d+)', dir_name)
    if match:
        j3 = float(match.group(1))
        jzp = float(match.group(2))
        return j3, jzp
    return None, None

def compute_spin_configuration_single_q(J3, Jzp):
    """Compute spin configuration using SingleQAnsatz."""
    # Update J parameters with the provided J3 and Jzp values
    J = default_J.copy()
    J[5] = J3  # J3 is at index 5
    J[2] = Jzp  # Jzp is at index 2
    
    # Create SingleQAnsatz model
    model = SingleQ(lattice_size, J)
    
    # Extract magnetization and energy
    spins = model.generate_spin_configuration()    
    # Scale energy to match format in phase_diagram.txt

    model_0 = SingleQFixedQ(0,0, lattice_size, J)
    model_zzy = SingleQFixedQ(0, 0.5, lattice_size, J)
    model_zzx = SingleQFixedQ(0.5, 0, lattice_size, J)
    model_zz = SingleQFixedQ(0.5, 0.5, lattice_size, J)

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


def compute_spin_configuration_luttinger_tisza(J3, Jzp):
    """Compute spin configuration using Luttinger-Tisza method."""
    # Update J parameters with the provided J3 and Jzp values
    J = default_J.copy()
    J[5] = J3  # J3 is at index 5
    J[2] = Jzp  # Jzp is at index 2
    
    # Create Luttinger-Tisza model
    k_opt, energy, spins, positions, constraint = luttinger_tisza_method(lattice_size, J)
    k_opt_1, energy_1, spins_1, positions_1, constraint_1 = luttinger_tisza_method(lattice_size, J, True)
    
    # Compare energies of both methods
    if energy < energy_1:
        print(f"Luttinger-Tisza method: Using k_opt={k_opt} with energy={energy}")
    else:
        print(f"Luttinger-Tisza method: Using k_opt={k_opt_1} with energy={energy_1}")
        k_opt = k_opt_1
        spins = spins_1

    return spins, energy, k_opt[0], k_opt[1]

def process_directories():
    """Process all subdirectories in the base directory."""
    # Get all items in the base directory
    items = os.listdir(base_dir)
    
    # Filter for directories matching the J3_*_Jzp_* pattern
    j3_jzp_dirs = [item for item in items 
                   if os.path.isdir(os.path.join(base_dir, item)) 
                   and item.startswith("J3_") and "Jzp_" in item]
    
    phase_diagram_single_q = os.path.join(base_dir, "phase_diagram_single_q.txt")
    with open(phase_diagram_single_q, 'w') as f:
        f.write("# J3 Jzp Energy Q1 Q2\n")
    phase_diagram_luttinger_tisza = os.path.join(base_dir, "phase_diagram_luttinger_tisza.txt")
    with open(phase_diagram_luttinger_tisza, 'w') as f:
        f.write("# J3 Jzp Energy Q1 Q2\n")
    print(f"Found {len(j3_jzp_dirs)} directories to process")
    
    for dir_name in j3_jzp_dirs:
        j3, jzp = parse_directory_name(dir_name)
        if j3 is None or jzp is None:
            print(f"Skipping directory with invalid format: {dir_name}")
            continue
        
        print(f"Processing directory: {dir_name} (J3={j3}, Jzp={jzp})")
        
        try:
            spins, energy, q1, q2 = compute_spin_configuration_single_q(j3, jzp)            
            # Create output file path
            output_path = os.path.join(base_dir, dir_name, "spins_single_q.txt")
            
            # Save to file
            with open(output_path, 'w') as f:
                for i in range(spins.shape[0]):
                    f.write(f"{spins[i, 0]} {spins[i, 1]} {spins[i, 2]}\n")
            
            with open(phase_diagram_single_q, 'a') as f:
                f.write(f"{j3} {jzp} {energy} {q1} {q2}\n")

            print(f"Saved results to {output_path}")
        
        except Exception as e:
            print(f"Error processing {dir_name}: {e}")

        try:
            spins, energy, q1, q2 = compute_spin_configuration_luttinger_tisza(j3, jzp)            
            # Create output file path
            output_path = os.path.join(base_dir, dir_name, "spins_LT.txt")
            
            # Save to file
            with open(output_path, 'w') as f:
                for i in range(spins.shape[0]):
                    f.write(f"{spins[i, 0]} {spins[i, 1]} {spins[i, 2]}\n")
            
            with open(phase_diagram_luttinger_tisza, 'a') as f:
                f.write(f"{j3} {jzp} {energy} {q1} {q2}\n")

            print(f"Saved results to {output_path}")
        
        except Exception as e:
            print(f"Error processing {dir_name}: {e}")

def process_grid(j3_min, j3_max, jzp_min, jzp_max, n_j3, n_jzp, output_dir=None):
    """
    Process a grid of J3 and Jzp values.
    
    Args:
        j3_min: Minimum value for J3
        j3_max: Maximum value for J3
        jzp_min: Minimum value for Jzp
        jzp_max: Maximum value for Jzp
        n_j3: Number of points in J3 dimension
        n_jzp: Number of points in Jzp dimension
        output_dir: Directory to save results (defaults to base_dir)
    """
    # Calculate total grid points and distribute among processes
    total_points = n_j3 * n_jzp
    j3_values = np.linspace(j3_min, j3_max, n_j3)
    jzp_values = np.linspace(jzp_min, jzp_max, n_jzp)
    all_points = [(j3, jzp) for j3 in j3_values for jzp in jzp_values]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Print MPI information
    if rank == 0:
        print(f"Running with {size} MPI processes")


    # Determine which points each process handles
    points_per_process = total_points // size
    remainder = total_points % size
    start_idx = rank * points_per_process + min(rank, remainder)
    end_idx = start_idx + points_per_process + (1 if rank < remainder else 0)
    my_points = all_points[start_idx:end_idx]

    # Set up directories and output files on root process
    if rank == 0:
        if output_dir is None:
            output_dir = base_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create output files with headers
        phase_diagram_single_q = os.path.join(output_dir, "phase_diagram_single_q.txt")
        phase_diagram_luttinger_tisza = os.path.join(output_dir, "phase_diagram_luttinger_tisza.txt")
        for filename in [phase_diagram_single_q, phase_diagram_luttinger_tisza]:
            with open(filename, 'w') as f:
                f.write("# J3 Jzp Energy Q1 Q2\n")
        
        print(f"Processing grid: J3=[{j3_min}, {j3_max}], Jzp=[{jzp_min}, {jzp_max}], {n_j3}x{n_jzp} points")
        print(f"Distributing {total_points} points across {size} processes")

    # Ensure all processes have the output directory
    comm.Barrier()
    if output_dir is None:
        output_dir = base_dir
    if not os.path.exists(output_dir) and rank != 0:
        os.makedirs(output_dir)

    # Process assigned points
    local_results_sq = []
    local_results_lt = []
    for j3, jzp in my_points:
        print(f"Process {rank}: Computing J3={j3:.4f}, Jzp={jzp:.4f}")
        point_dir = os.path.join(output_dir, f"J3_{j3:.4f}_Jzp_{jzp:.4f}")
        if not os.path.exists(point_dir):
            os.makedirs(point_dir)
        
        try:
            # SingleQ method
            spins, energy, q1, q2 = compute_spin_configuration_single_q(j3, jzp)
            output_path = os.path.join(point_dir, "spins_single_q.txt")
            with open(output_path, 'w') as f:
                for i in range(spins.shape[0]):
                    f.write(f"{spins[i, 0]} {spins[i, 1]} {spins[i, 2]}\n")
            local_results_sq.append((j3, jzp, energy, q1, q2))
        except Exception as e:
            print(f"Process {rank}: Error in SingleQ for J3={j3}, Jzp={jzp}: {e}")
        
        try:
            # Luttinger-Tisza method
            spins, energy, q1, q2 = compute_spin_configuration_luttinger_tisza(j3, jzp)
            output_path = os.path.join(point_dir, "spins_LT.txt")
            with open(output_path, 'w') as f:
                for i in range(spins.shape[0]):
                    f.write(f"{spins[i, 0]} {spins[i, 1]} {spins[i, 2]}\n")
            local_results_lt.append((j3, jzp, energy, q1, q2))
        except Exception as e:
            print(f"Process {rank}: Error in LT for J3={j3}, Jzp={jzp}: {e}")

    # Gather results from all processes
    all_results_sq = comm.gather(local_results_sq, root=0)
    all_results_lt = comm.gather(local_results_lt, root=0)

    # Root process writes results to files
    if rank == 0:
        # Flatten nested lists
        all_results_sq_flat = [r for results in all_results_sq for r in results]
        all_results_lt_flat = [r for results in all_results_lt for r in results]
        
        # Sort by J3 and Jzp for consistent output
        all_results_sq_flat.sort(key=lambda x: (x[0], x[1]))
        all_results_lt_flat.sort(key=lambda x: (x[0], x[1]))
        
        # Write SingleQ results
        with open(phase_diagram_single_q, 'a') as f:
            for j3, jzp, energy, q1, q2 in all_results_sq_flat:
                f.write(f"{j3} {jzp} {energy} {q1} {q2}\n")
        
        # Write Luttinger-Tisza results
        with open(phase_diagram_luttinger_tisza, 'a') as f:
            for j3, jzp, energy, q1, q2 in all_results_lt_flat:
                f.write(f"{j3} {jzp} {energy} {q1} {q2}\n")
        
        print(f"Completed processing {total_points} grid points")
   
if __name__ == "__main__":
    process_grid(0.2*6.54, 0.4*6.54, 0.0, -0.8*6.54, 20, 20, output_dir=base_dir)