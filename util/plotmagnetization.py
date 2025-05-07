import matplotlib.pyplot as plt
import numpy as np

# Load the first magnetization file
mag_data1 = np.loadtxt('BCAO_sasha_field_sweep_0_0.5_X/magnetization.txt', comments=['#', '//'])
fieldx = mag_data1[:, 0]/0.086
mx1 = mag_data1[:, 1]*2
my1 = mag_data1[:, 2]
mz1 = mag_data1[:, 3]

# Load the second magnetization file
mag_data2 = np.loadtxt('BCAO_sasha_field_sweep_0_0.5_Y/magnetization.txt', comments=['#', '//'])
field = mag_data2[:, 0]/0.086
mx = mag_data2[:, 1]
my = mag_data2[:, 2]*2
mz = mag_data2[:, 3]

# Create a figure with two subplots
fig, ax = plt.subplots(figsize=(12, 5))

# Plot the first magnetization data
ax.plot(fieldx, np.abs(mx1), 'o-', label='B // X')

# Plot the second magnetization data components
ax.plot(field, np.abs(my), 'g-', label='B // Y')
ax.set_xlabel('Field')
ax.set_ylabel('Magnetization components')
ax.set_title('Magnetization vs Field (second file)')
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.savefig('magnetization_plot.png')
plt.show()