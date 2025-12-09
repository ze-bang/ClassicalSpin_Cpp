import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Base directory
base_dir = "/scratch/zhouzb79/Potential_Param_Scan"

# Find all x and y direction pairs
param_sets = {}
for item in os.listdir(base_dir):
    item_path = os.path.join(base_dir, item)
    if os.path.isdir(item_path):
        # Extract parameter set (e.g., "fitting_param_2_x" -> "fitting_param_2")
        if item.endswith('_x'):
            param_name = item[:-2]
            if param_name not in param_sets:
                param_sets[param_name] = {}
            param_sets[param_name]['x'] = item_path
        elif item.endswith('_y'):
            param_name = item[:-2]
            if param_name not in param_sets:
                param_sets[param_name] = {}
            param_sets[param_name]['y'] = item_path

# Create plots - one plot per parameter set
fig, axes = plt.subplots(len(param_sets), 1, figsize=(10, 6*len(param_sets)))
if len(param_sets) == 1:
    axes = [axes]

for idx, (param_name, dirs) in enumerate(sorted(param_sets.items())):
    if 'x' not in dirs or 'y' not in dirs:
        print(f"Warning: Missing x or y direction for {param_name}")
        continue
    
    # Read x direction data
    x_file = os.path.join(dirs['x'], 'magnetization_vs_field.txt')
    y_file = os.path.join(dirs['y'], 'magnetization_vs_field.txt')
    
    if not os.path.exists(x_file) or not os.path.exists(y_file):
        print(f"Warning: Data files not found for {param_name}")
        continue
    
    # Load data
    data_x = np.loadtxt(x_file)
    data_y = np.loadtxt(y_file)
    
    # Extract columns: h, Mx, My, Mz, dMx, dMy, dMz
    h_x = data_x[:, 0]
    Mx_x = data_x[:, 1]
    dMx_x = data_x[:, 4]
    
    h_y = data_y[:, 0]
    My_y = data_y[:, 2]
    dMy_y = data_y[:, 5]
    
    # Plot Mx (from x-direction) and My (from y-direction) on the same plot
    ax = axes[idx]
    ax.errorbar(h_x, Mx_x, yerr=dMx_x, marker='o', linestyle='-', 
                 capsize=3, label='Mx (Field along x)', alpha=0.7, color='blue')
    ax.errorbar(h_y, My_y, yerr=dMy_y, marker='s', linestyle='--', 
                 capsize=3, label='My (Field along y)', alpha=0.7, color='red')
    ax.set_xlabel('Magnetic Field h', fontsize=12)
    ax.set_ylabel('Magnetization', fontsize=12)
    ax.set_title(f'{param_name}: Magnetization vs Field', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = os.path.join(base_dir, 'combined_magnetization_plots.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_file}")
plt.show()

# Also create a combined data file for each parameter set
for param_name, dirs in sorted(param_sets.items()):
    if 'x' not in dirs or 'y' not in dirs:
        continue
    
    x_file = os.path.join(dirs['x'], 'magnetization_vs_field.txt')
    y_file = os.path.join(dirs['y'], 'magnetization_vs_field.txt')
    
    if not os.path.exists(x_file) or not os.path.exists(y_file):
        continue
    
    data_x = np.loadtxt(x_file)
    data_y = np.loadtxt(y_file)
    
    # Create combined output
    output_combined = os.path.join(base_dir, f'{param_name}_combined.txt')
    with open(output_combined, 'w') as f:
        f.write('# Combined magnetization data\n')
        f.write('# h  Mx_x My_x dMx_x dMy_x  Mx_y My_y dMx_y dMy_y\n')
        for i in range(len(data_x)):
            f.write(f'{data_x[i,0]:.6f}  ')
            f.write(f'{data_x[i,1]:.6f} {data_x[i,2]:.6f} {data_x[i,4]:.6f} {data_x[i,5]:.6f}  ')
            f.write(f'{data_y[i,1]:.6f} {data_y[i,2]:.6f} {data_y[i,4]:.6f} {data_y[i,5]:.6f}\n')
    
    print(f"Combined data saved to: {output_combined}")
