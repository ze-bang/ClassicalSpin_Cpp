import numpy as np
from collections import defaultdict

#!/usr/bin/env python3
import matplotlib.pyplot as plt

def read_file(filename):
    """Read a phase diagram file and return data as a dictionary."""
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            values = list(map(float, line.strip().split()))
            if len(values) >= 3:
                j3, jzp = values[0], values[1]
                energy = values[-1] if filename.endswith('phase_diagram.txt') else values[2]
                key = (j3, jzp)
                data[key] = energy
    return data

def main():
    # Read data from all three files
    directory = 'BCAO_sasha_phase/'
    pd_data = read_file(directory+'phase_diagram.txt')
    lt_data = read_file(directory+'phase_diagram_luttinger_tisza.txt')
    sq_data = read_file(directory+'phase_diagram_single_q.txt')
    
    # Get all unique (J3, Jzp) pairs
    all_keys = set(pd_data.keys()) | set(lt_data.keys()) | set(sq_data.keys())
    
    # Compare energies and determine the lowest
    results = {}
    for key in all_keys:
        energies = {
            'Classical': pd_data.get(key, float('inf')) / (24*24*2),
            'LT': lt_data.get(key, float('inf')),
            'Single-Q': sq_data.get(key, float('inf'))
        }
        lowest_method = min(energies, key=energies.get)
        results[key] = (lowest_method, energies[lowest_method])
    
    # Print results
    print("J3\tJzp\tLowest Method\tEnergy")
    for (j3, jzp), (method, energy) in sorted(results.items()):
        print(f"{j3:.4f}\t{jzp:.4f}\t{method}\t{energy:.6f}")
    
    # Count how many times each method has the lowest energy
    method_counts = {method: 0 for method in ['Classical', 'LT', 'Single-Q']}
    for method, _ in results.values():
        method_counts[method] += 1
    
    print("\nSummary:")
    for method, count in method_counts.items():
        print(f"{method}: {count} times lowest ({count/len(results)*100:.1f}%)")
    
    # Create a phase diagram plot
    j3_values = [key[0] for key in results.keys()]
    jzp_values = [key[1] for key in results.keys()]
    
    # Convert method names to numbers for plotting
    method_to_num = {'Classical': 0, 'LT': 1, 'Single-Q': 2}
    colors = [method_to_num[results[key][0]] for key in results.keys()]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(j3_values, jzp_values, c=colors, cmap='viridis', 
                          marker='s', s=100, alpha=0.7)
    
    cbar = plt.colorbar(scatter, ticks=[0, 1, 2])
    cbar.set_ticklabels(['Classical', 'LT', 'Single-Q'])
    
    plt.xlabel('J3')
    plt.ylabel('Jzp')
    plt.title('Phase Diagram: Method with Lowest Energy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('lowest_energy_methods.png')
    plt.show()

    # Save the results to a file in the same directory
    with open(directory+'lowest_energy_methods.txt', 'w') as f:
        f.write("# J3\tJzp\tLowest Method\tEnergy\n")
        for (j3, jzp), (method, energy) in sorted(results.items()):
            f.write(f"{j3:.4f}\t{jzp:.4f}\t{method}\t{energy:.6f}\n")
        
        f.write("\n# Summary:\n")
        for method, count in method_counts.items():
            f.write(f"# {method}: {count} times lowest ({count/len(results)*100:.1f}%)\n")

if __name__ == "__main__":
    main()