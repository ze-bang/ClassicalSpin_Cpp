import os
import numpy as np
import re
from single_q import SingleQAnsatz
from luttinger_tisza import luttinger_tisza_method

# Base directory containing all subdirectories
base_dir = "/home/pc_linux/ClassicalSpin_Cpp/BCAO_sasha_phase"

# Default J parameters: [J1, Jpmpm, Jzp, Delta1, J2, J3, Delta3]
default_J = [-6.54, 0.15, -3.76, 0.36, -0.21, 1.70, 0.03]

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
    model = SingleQAnsatz(lattice_size, J)
    
    # Extract magnetization and energy
    spins = model.generate_spin_configuration()    
    # Scale energy to match format in phase_diagram.txt
    # Based on observed values, we need to scale by approximately 2000
    
    return spins, model.opt_energy, model.opt_params[0], model.opt_params[1]

def compute_spin_configuration_luttinger_tisza(J3, Jzp):
    """Compute spin configuration using Luttinger-Tisza method."""
    # Update J parameters with the provided J3 and Jzp values
    J = default_J.copy()
    J[5] = J3  # J3 is at index 5
    J[2] = Jzp  # Jzp is at index 2
    
    # Create Luttinger-Tisza model
    k_opt, energy, spins, positions = luttinger_tisza_method(lattice_size, J)
    
    
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

if __name__ == "__main__":
    process_directories()