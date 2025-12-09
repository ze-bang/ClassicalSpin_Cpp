import os
import numpy as np
import re
from pathlib import Path

import matplotlib.pyplot as plt
import plot_spin_config_2d
from reader_honeycomb import SSSF2D, ordering_q_SSSF2D

def read_spin_configuration(filepath):
    """Read the spin configuration from a file."""
    spins = []
    with open(filepath, 'r') as f:
        for line in f:
            # Skip comment lines (those starting with //)
            if line.strip().startswith('//'):
                continue
            # Parse the line to get the spin components
            try:
                x, y, z = map(float, line.strip().split())
                spins.append([x, y, z])
            except ValueError:
                continue
    return np.array(spins)

def is_ferromagnetic(q1, q2, tolerance=0.02):
    return np.mod(q1, 1) < tolerance and np.mod(q2, 1) < tolerance


def is_zigzag(q1, q2, tolerance=0.02):
    zz_order = [0.0, 0.5]
    
    # Check if q1 is close to any rational fraction
    q1_is = any(abs(np.mod(q1, 1) - frac) < tolerance for frac in zz_order)
    
    # Check if q2 is close to any rational fraction
    q2_is = any(abs(np.mod(q2, 1) - frac) < tolerance for frac in zz_order)
    
    # Return True if neither q1 nor q2 are close to rational fractions
    return q1_is and q2_is


def is_doublezigzag(q1, q2, tolerance=0.02):
    # Common rational fractions to check against
    dzz_order = [0.25, 0.75]
    dzz_order1 = [0]
    
    # Check if q1 is close to any rational fraction
    q1_is = any(abs(np.mod(q1, 1) - frac) < tolerance for frac in dzz_order)
    
    # Check if q2 is close to any rational fraction
    q2_is = any(abs(np.mod(q2, 1) - frac) < tolerance for frac in dzz_order1)

    is_dzz_1 = q1_is and q2_is

    # Check if q1 is close to any rational fraction
    q1_is = any(abs(np.mod(q1, 1) - frac) < tolerance for frac in dzz_order1)
    
    # Check if q2 is close to any rational fraction
    q2_is = any(abs(np.mod(q2, 1) - frac) < tolerance for frac in dzz_order)

    is_dzz_2 = q1_is and q2_is
    
    # Return True if neither q1 nor q2 are close to rational fractions
    return is_dzz_1 or is_dzz_2


def is_noncommensurate(q1, q2, tolerance=0.02):
    """Check if the wave vectors represent a non-commensurate state.
    
    Returns True if q1 and q2 are not close to simple rational values.
    """
    # Common rational fractions to check against
    rational_fractions = [0, 0.25, 0.333, 0.5, 0.667, 0.75, 1.0]
    
    # Check if q1 is close to any rational fraction
    q1_is_rational = any(abs(np.mod(q1, 1) - frac) < tolerance for frac in rational_fractions)
    
    # Check if q2 is close to any rational fraction
    q2_is_rational = any(abs(np.mod(q2, 1) - frac) < tolerance for frac in rational_fractions)
    
    # Return True if neither q1 nor q2 are close to rational fractions
    return not (q1_is_rational or q2_is_rational)



def extract_parameters(dirname):
    """Extract the parameters A and B from directory name of format J3_A_Jzp_B."""
    pattern = r'J3_([0-9.-]+)_Jzp_([0-9.-]+)'
    match = re.search(pattern, dirname)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def scan_directories(root_dir, tolerance=0.02):
    """Scan the directories and check for ferromagnetic states."""
    fm_points = []
    zz_points = []
    dzz_points = []
    non_comm_points = []
    non_fm_points = []
    
    # Convert to Path object
    root_path = Path(root_dir)
    
    # Find all subdirectories matching the pattern
    matching_dirs = list(root_path.glob("J3_*_Jzp_*"))
    total_dirs = len(matching_dirs)
    
    print(f"Found {total_dirs} directories to scan.")
    
    for i, dirpath in enumerate(matching_dirs):
        if i % max(1, total_dirs//10) == 0:
            print(f"Progress: {i}/{total_dirs} directories")
            
        if dirpath.is_dir():
            # Extract parameters from directory name
            j3, jzp = extract_parameters(dirpath.name)
            if j3 is not None and jzp is not None:
                # Check if spins.txt exists in this directory
                # Parse the lowest energy methods from file
                def parse_lowest_energy_methods(file_path="/home/pc_linux/ClassicalSpin_Cpp/BCAO_sasha_phase/lowest_energy_methods.txt"):
                    methods = {}
                    try:
                        with open(file_path, 'r') as f:
                            for line in f:
                                # Skip comments and empty lines
                                line = line.strip()
                                if not line or line.startswith('#'):
                                    continue
                                
                                try:
                                    # Parse the line
                                    parts = line.split()
                                    if len(parts) >= 4:
                                        j3 = float(parts[0])
                                        jzp = float(parts[1])
                                        method = parts[2]
                                        energy = float(parts[3])
                                        
                                        # Store the method and energy
                                        methods[(j3, jzp)] = (method, energy)
                                except ValueError:
                                    continue
                    except FileNotFoundError:
                        print(f"Warning: Could not find {file_path}")
                    
                    return methods

                # Find the closest key in the dictionary
                def find_closest_key(j3, jzp, methods_dict):
                    min_distance = float('inf')
                    closest_key = None
                    
                    for key in methods_dict.keys():
                        dict_j3, dict_jzp = key
                        # Calculate Euclidean distance
                        distance = ((dict_j3 - j3)**2 + (dict_jzp - jzp)**2)**0.5
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_key = key
                    
                    return closest_key

                # Load the methods dictionary
                lowest_energy_methods = parse_lowest_energy_methods()
                
                # Find the method for the given J3 and Jzp
                closest_key = find_closest_key(j3, jzp, lowest_energy_methods)
                lowest_method = lowest_energy_methods.get(closest_key, (None, None))[0]
                
                print(f"Processing directory: {dirpath} (J3={j3}, Jzp={jzp}, Method={lowest_method})")


                if lowest_method == "Classical":
                    spin_file = dirpath / "spins.txt"
                elif lowest_method == "LT":
                    spin_file = dirpath / "spins_LT.txt"
                elif lowest_method == "Single-Q":
                    spin_file = dirpath / "spins_single_q.txt"
                else:
                    # Default to spins.txt if method is unknown or None
                    spin_file = dirpath / "spins.txt"
                # Compute SSSF



                spins = read_spin_configuration(spin_file)
                pos = read_spin_configuration(dirpath / "pos.txt")
                sssf, K = SSSF2D(spins, pos, 100, str(dirpath) + "/")
                tempQ = ordering_q_SSSF2D(sssf, K)
                Q1, Q2 = tempQ[0], tempQ[1]   
                if is_ferromagnetic(Q1, Q2, tolerance):
                    fm_points.append((np.abs(j3/6.54), np.abs(jzp/6.54)))
                    # Save the phase information to a file
                    with open(dirpath / "phase.txt", 'w') as f:
                        f.write("FM")
                    print(f"Ferromagnetic state found at J3={j3}, Jzp={jzp}")

                elif is_zigzag(Q1, Q2, tolerance):
                    zz_points.append((np.abs(j3/6.54), np.abs(jzp/6.54)))
                    with open(dirpath / "phase.txt", 'w') as f:
                        f.write("ZZ")
                    print(f"Zigzag state found at J3={j3}, Jzp={jzp}")
                elif is_doublezigzag(Q1, Q2, tolerance):
                    dzz_points.append((np.abs(j3/6.54), np.abs(jzp/6.54)))
                    with open(dirpath / "phase.txt", 'w') as f:
                        f.write("DZZ")
                    print(f"Double zigzag state found at J3={j3}, Jzp={jzp}")
                elif is_noncommensurate(Q1, Q2, tolerance):
                    non_comm_points.append((np.abs(j3/6.54), np.abs(jzp/6.54)))
                    with open(dirpath / "phase.txt", 'w') as f:
                        f.write("Non-commensurate")
                    print(f"Non-commensurate state found at J3={j3}, Jzp={jzp}")
                else:
                    non_fm_points.append((np.abs(j3/6.54), np.abs(jzp/6.54)))
                    with open(dirpath / "phase.txt", 'w') as f:
                        f.write("Non-FM")
                    print(f"Non-ferromagnetic state found at J3={j3}, Jzp={jzp}")


                
                #Write out the Q values to the phase file
                with open(dirpath / "phase.txt", 'a') as f:
                    f.write(f" Q1={Q1} Q2={Q2}")
                    # Also write the lowest method
                    f.write(f" Method={lowest_method}")

                # Plot the spin configuration
                plot_file = dirpath / "spin_config.png"
                if not plot_file.is_file():
                    plot_spin_config_2d.plot_spin_configuration(
                        pos_file=dirpath / "pos.txt",
                        spin_file=spin_file,
                        filename=plot_file,
                        kitaev=False
                    )
                    print(f"Spin configuration plotted for J3={j3}, Jzp={jzp}")
    
    return fm_points, zz_points, dzz_points, non_comm_points, non_fm_points

def plot_phase_diagram(fm_points, zz_points, dzz_points, non_comm_points, non_fm_points):
    """Plot the phase diagram based on ferromagnetic points."""
    plt.figure(figsize=(10, 8))
    
    # Plot ferromagnetic points
    if fm_points:
        j3_values, jzp_values = zip(*fm_points)
        plt.scatter(j3_values, jzp_values, color='red', marker='o', label='FM')
    
    # Plot zigzag points
    if zz_points:
        j3_values_zz, jzp_values_zz = zip(*zz_points)
        plt.scatter(j3_values_zz, jzp_values_zz, color='green', marker='^', alpha=0.5, label='ZZ')

    if dzz_points:
        j3_values_dzz, jzp_values_dzz = zip(*dzz_points)
        plt.scatter(j3_values_dzz, jzp_values_dzz, color='orange', marker='s', alpha=0.5, label='Double ZZ')

    if non_comm_points:
        j3_values_non_comm, jzp_values_non_comm = zip(*non_comm_points)
        plt.scatter(j3_values_non_comm, jzp_values_non_comm, color='purple', marker='*', alpha=0.5, label='Non-commensurate')

    # Plot non-ferromagnetic points
    if non_fm_points:
        j3_values_non, jzp_values_non = zip(*non_fm_points)
        plt.scatter(j3_values_non, jzp_values_non, color='blue', marker='x', alpha=0.5, label='Unknown')
    
    plt.xlabel('J3/Jz')
    plt.ylabel('|Jzp/Jz|')
    plt.title('Phase Diagram - Ferromagnetic States')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plt.savefig("phase_diagram.png")
    plt.show()

def main():
    # Get the root directory
    root_dir = input("Enter the root directory path (or press Enter for current directory): ").strip()
    if not root_dir:
        root_dir = "."
    
    # Get the tolerance value
    tolerance_str = input("Enter tolerance for FM detection (default 0.02): ").strip()
    tolerance = float(tolerance_str) if tolerance_str else 0.02
    
    print(f"Scanning directories in {root_dir}...")
    fm_points, zz_points, dzz_points, non_comm_points, non_fm_points = scan_directories(root_dir, tolerance)
    
    print(f"Found {len(fm_points)} ferromagnetic states out of {len(fm_points) + len(non_fm_points)} total.")
    plot_phase_diagram(fm_points, zz_points, dzz_points, non_comm_points, non_fm_points)
    print("Phase diagram saved as 'phase_diagram.png'.")

if __name__ == "__main__":
    main()