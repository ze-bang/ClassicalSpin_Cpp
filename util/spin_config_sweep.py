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

def is_ferromagnetic(spins, tolerance=1e-3):
    """Check if all spins are aligned (ferromagnetic state)."""
    if len(spins) < 2:
        return False
    
    # Get the first spin as reference
    reference = spins[0]
    
    # average the spins
    avg_spin = np.mean(spins, axis=0)
    # Check if the average spin is close to the reference spin
    if np.linalg.norm(avg_spin - reference) > tolerance:
        return False
    
    return True


    
def is_zigzag(spins, tolerance=0.1):
    """Check if the spins are in a zigzag pattern."""
    if len(spins) < 2:
        return False
    
    # Normalize spins and filter out near-zero spins
    normalized_spins = []
    for spin in spins:
        norm = np.linalg.norm(spin)
        if norm > tolerance:
            normalized_spins.append(spin / norm)
    
    if len(normalized_spins) < 2:
        return False
    
    normalized_spins = np.array(normalized_spins)
    
    # Calculate average magnitude of individual spins
    avg_spin_norm = np.mean(np.linalg.norm(spins, axis=1))
    
    # Calculate the average spin
    avg_spin = np.mean(spins, axis=0)
    avg_norm = np.linalg.norm(avg_spin)
    
    # For zigzag, average spin should be small but individual spins should have significant magnitude
    if avg_norm < avg_spin_norm * 0.2:  # Average is much smaller than individual spins
        # Find dominant axis using SVD
        u, s, vh = np.linalg.svd(normalized_spins, full_matrices=False)
        dominant_axis = vh[0]
        
        # Project spins onto the dominant axis
        projections = np.dot(normalized_spins, dominant_axis)
        
        # Count positive and negative projections
        positive = np.sum(projections > 0.5)
        negative = np.sum(projections < -0.5)
        
        total_significant = positive + negative
        
        # Ensure most spins are captured by the dominant axis
        if total_significant < len(normalized_spins) * 0.7:
            return False
        
        # For a zigzag pattern, we expect significant numbers of spins pointing in both directions
        threshold = 0.25
        return (positive / total_significant >= threshold and 
                negative / total_significant >= threshold)

def is_double_zigzag(spins, positions=None, tolerance=1e-3):
    """
    Check if the spins are in a double zigzag pattern typical of Kitaev models.
    
    Args:
        spins: numpy array of spin vectors
        positions: numpy array of lattice positions (optional)
        tolerance: tolerance for comparison
        
    Returns:
        bool: True if the spins form a double zigzag pattern, False otherwise
    """
    if len(spins) < 4:
        return False
    
    # Normalize spins and filter out near-zero spins
    normalized_spins = []
    valid_indices = []
    
    for i, spin in enumerate(spins):
        norm = np.linalg.norm(spin)
        if norm > tolerance:
            normalized_spins.append(spin / norm)
            valid_indices.append(i)
    
    if len(normalized_spins) < 4:
        return False
    
    normalized_spins = np.array(normalized_spins)
    
    # Use SVD to find the principal spin directions
    u, s, vh = np.linalg.svd(normalized_spins, full_matrices=False)
    
    # Examine the singular values - double zigzag should have two significant components
    if s[1] < 0.5 * s[0] or s[2] > 0.2 * s[0]:
        return False
    
    # For double zigzag, spins should form two groups with opposite orientations
    principal_axis = vh[0]
    projections = np.dot(normalized_spins, principal_axis)
    
    # Count positive and negative projections
    positive = np.sum(projections > 0.5)
    negative = np.sum(projections < -0.5)
    
    # Check if significant portions of spins point in opposite directions
    if positive < 0.25 * len(projections) or negative < 0.25 * len(projections):
        return False
    
    # If positions are provided, check the spatial pattern
    if positions is not None and len(positions) >= len(spins):
        valid_positions = positions[valid_indices]
        
        # Group by y-coordinate to identify rows
        y_coords = valid_positions[:, 1]
        unique_y = np.sort(np.unique(np.round(y_coords, decimals=3)))
        
        if len(unique_y) >= 4:  # Need several rows for clear pattern
            # Check if alternating rows have opposite spin orientations
            row_patterns = []
            for y in unique_y:
                row_indices = np.where(np.abs(y_coords - y) < tolerance)[0]
                if len(row_indices) >= 3:
                    row_projections = projections[row_indices]
                    avg_projection = np.mean(row_projections)
                    row_patterns.append(1 if avg_projection > 0 else -1)
            
            # Look for alternating pattern
            if len(row_patterns) >= 4:
                alternating = True
                for i in range(2, len(row_patterns)):
                    if row_patterns[i] != row_patterns[i-2]:
                        alternating = False
                        break
                
                if alternating:
                    return True
    
    # Without positions or if spatial check fails, use distribution of projections
    projection_histogram, _ = np.histogram(projections, bins=20, range=(-1, 1))
    peaks = 0
    for i in range(1, len(projection_histogram)-1):
        if (projection_histogram[i] > projection_histogram[i-1] and 
            projection_histogram[i] > projection_histogram[i+1] and
            projection_histogram[i] > len(projections)/20):
            peaks += 1
    
    # Double zigzag typically shows two distinct peaks in projection distribution
    return peaks == 2


def extract_parameters(dirname):
    """Extract the parameters A and B from directory name of format J3_A_Jzp_B."""
    pattern = r'J3_([0-9.-]+)_Jzp_([0-9.-]+)'
    match = re.search(pattern, dirname)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def scan_directories(root_dir, tolerance=1e-5):
    """Scan the directories and check for ferromagnetic states."""
    fm_points = []
    zz_points = []
    dzz_points = []
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

                if spin_file.is_file():
                    # Read and check if ferromagnetic
                    spins = read_spin_configuration(spin_file)
                    pos = read_spin_configuration(dirpath / "pos.txt")
                    if is_ferromagnetic(spins, tolerance):
                        fm_points.append((j3, jzp))
                        # Save the phase information to a file
                        with open(dirpath / "phase.txt", 'w') as f:
                            f.write("FM")
                        print(f"Ferromagnetic state found at J3={j3}, Jzp={jzp}")
                    # elif is_double_zigzag(spins, pos, tolerance):
                    #     dzz_points.append((j3, jzp))
                    #     with open(dirpath / "phase.txt", 'w') as f:
                    #         f.write("DZZ")
                    #     print(f"Double zigzag state found at J3={j3}, Jzp={jzp}")
                    elif is_zigzag(spins, tolerance):
                        zz_points.append((j3, jzp))
                        with open(dirpath / "phase.txt", 'w') as f:
                            f.write("ZZ")
                        print(f"Zigzag state found at J3={j3}, Jzp={jzp}")
                    else:
                        non_fm_points.append((j3, jzp))
                        with open(dirpath / "phase.txt", 'w') as f:
                            f.write("UNKNOWN")
                        print(f"Ferromagnetic state found at J3={j3}, Jzp={jzp}")
                
                # Compute SSSF
                sssf, K = SSSF2D(spins, pos, 100, str(dirpath) + "/")
                tempQ = ordering_q_SSSF2D(sssf, K)
                Q1, Q2 = tempQ[0], tempQ[1]   
                
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
    
    return fm_points, zz_points, dzz_points, non_fm_points

def plot_phase_diagram(fm_points, zz_points, dzz_points, non_fm_points):
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
        plt.scatter(j3_values_dzz, jzp_values_dzz, color='orange', marker='s', alpha=0.5, label='DZZ')

    # Plot non-ferromagnetic points
    if non_fm_points:
        j3_values_non, jzp_values_non = zip(*non_fm_points)
        plt.scatter(j3_values_non, jzp_values_non, color='blue', marker='x', alpha=0.5, label='Non-FM')
    
    plt.xlabel('J3')
    plt.ylabel('Jzp')
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
    tolerance_str = input("Enter tolerance for FM detection (default 1e-5): ").strip()
    tolerance = float(tolerance_str) if tolerance_str else 1e-5
    
    print(f"Scanning directories in {root_dir}...")
    fm_points, zz_points, dzz_points, non_fm_points = scan_directories(root_dir, tolerance)
    
    print(f"Found {len(fm_points)} ferromagnetic states out of {len(fm_points) + len(non_fm_points)} total.")
    plot_phase_diagram(fm_points, zz_points, dzz_points, non_fm_points)
    print("Phase diagram saved as 'phase_diagram.png'.")

if __name__ == "__main__":
    main()