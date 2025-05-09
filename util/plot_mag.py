import os
import numpy as np
import re
from pathlib import Path
import argparse
from opt_einsum import contract

import matplotlib.pyplot as plt

z = np.array([np.array([1,1,1])/np.sqrt(3), np.array([1,-1,-1])/np.sqrt(3), np.array([-1,1,-1])/np.sqrt(3), np.array([-1,-1,1])/np.sqrt(3)])


def read_spin_file(file_path):
    """Read a spin.txt file and return a numpy array of spin vectors."""
    spins = []
    with open(file_path, 'r') as f:
        for line in f:
            # Skip empty lines or comments
            if not line.strip() or line.strip().startswith('//'):
                continue
            # Parse the spin components
            components = [float(x) for x in line.strip().split()]
            if len(components) == 3:  # Make sure we have x, y, z components
                spins.append(components)
    return np.array(spins)

def calculate_magnetization(spins, field_dir):
    """Calculate the magnetization (average spin)."""
    factor = contract('a, ia->i', field_dir, z)
    mag = np.zeros(3)
    for i in range(4):
        mag += np.mean(spins[i::4], axis=0) * factor[i]
    mag = mag[2]

    # mag = np.zeros(3)
    # for i in range(4):
    #     mag += np.mean(spins[i::4],axis=0)[2] * z[i]
    # mag = np.dot(mag, field_dir)
    return mag

def extract_field_strength(dirname):
    """Extract the field strength from a directory name like 'h_29.473684_index_8'."""
    match = re.search(r'h_([0-9.]+)_index_', dirname)
    if match:
        return float(match.group(1))
    return None

def scan_and_plot_magnetization(base_dir, field_dir):
    """Scan for spin.txt files and plot magnetization vs field strength."""
    field_strengths = []
    mag_magnitudes = []    
    print(f"Scanning directory: {base_dir}")
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(base_dir):
        if 'spin.txt' in files:
            dirname = os.path.basename(root)
            if 'h_' in dirname and '_index_' in dirname:
                field_strength = extract_field_strength(dirname)
                if field_strength is not None:
                    spin_file = os.path.join(root, 'spin.txt')
                    print(f"Processing: {spin_file}")
                    spins = read_spin_file(spin_file)
                    mag_magnitude = calculate_magnetization(spins, field_dir)
                    
                    field_strengths.append(field_strength)
                    mag_magnitudes.append(mag_magnitude)
    
    if not field_strengths:
        print("No spin files found matching the expected pattern.")
        return
    
    # Sort by field strength
    sorted_indices = np.argsort(field_strengths)
    field_strengths = np.array(field_strengths)[sorted_indices]
    mag_magnitudes = np.array(mag_magnitudes)[sorted_indices]
    
    # Plot magnetization magnitude
    plt.plot(field_strengths, mag_magnitudes, 'o-')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot pyrochlore magnetization from spin configurations.')
    parser.add_argument('--base_dir', type=str, default="/home/pc_linux/ClassicalSpin_Cpp", 
                        help='Base directory to search for spin files')
    parser.add_argument('--mode', type=str, default="ALLDIR",)
    args = parser.parse_args()
    h100 = np.array([1, 0, 0])
    h110 = np.array([1, 1, 0])/np.sqrt(2)
    h111 = np.array([1, 1, 1])/np.sqrt(3)

    if args.mode == "ALLDIR":

        print(f"Base directory: {args.base_dir}")


        file_111 = args.base_dir + "_111"
        file_110 = args.base_dir +"_110"
        file_100 = args.base_dir + "_100"

        plt.figure(figsize=(12, 10))

        scan_and_plot_magnetization(file_111, h111)
        scan_and_plot_magnetization(file_110, h110)
        scan_and_plot_magnetization(file_100, h100)

        plt.xlabel('Field Strength (h)')
        plt.ylabel('Magnetization Magnitude')
        plt.title('Magnetization Magnitude vs Field Strength')
        plt.legend(['h_111', 'h_110', 'h_100'], loc='upper right')
        plt.tight_layout()
        # output_file = os.path.join(args.base_dir, 'magnetization_vs_field.png')
        # plt.savefig(output_file)
        # print(f"Plot saved to: {output_file}")
        plt.show()
    elif args.mode == "SINGLE":
        print(f"Base directory: {args.base_dir}")

        fiels_str = args.base_dir.split("_")[-1]

        field_dir = None

        if fiels_str == "111":
            field_dir = h111
        elif fiels_str == "110":
            field_dir = h110
        elif fiels_str == "100":
            field_dir = h100
        

        scan_and_plot_magnetization(args.base_dir, field_dir)

        plt.xlabel('Field Strength (h)')
        plt.ylabel('Magnetization Magnitude')
        plt.title('Magnetization Magnitude vs Field Strength')
        plt.legend(['h_111', 'h_110', 'h_100'], loc='upper right')
        plt.tight_layout()
        # output_file = os.path.join(args.base_dir, 'magnetization_vs_field.png')
        # plt.savefig(output_file)
        # print(f"Plot saved to: {output_file}")
        plt.show()