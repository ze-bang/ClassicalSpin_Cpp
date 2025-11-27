from single_q_BCAO import SingleQ
import numpy as np  
import matplotlib.pyplot as plt
import argparse
def read_Param(config_file):
    """Read the configuration file and return a dictionary of parameters."""
    params = {}
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Handle special cases
                if key == 'field_dir':
                    params[key] = [float(x) for x in value.split(',')]
                elif key == 'dir':
                    params[key] = value
                else:
                    # Try to convert to int first, then float
                    try:
                        if '.' in value:
                            params[key] = float(value)
                        else:
                            params[key] = int(value)
                    except ValueError:
                        params[key] = value
    return params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot pyrochlore magnetization from spin configurations.')
    parser.add_argument('--config', type=str, default="config.txt",
                        help='Configuration file for SingleQ')
    args = parser.parse_args()

    params = read_Param(args.config)
    mu_B = 0.05788  # meV/T
    magnetization = np.zeros((params["num_steps"], 3))
    h = np.linspace(params["h_start"], params["h_end"], params["num_steps"])
    count = 0
    for h_val in h:
        step_dir = f"{params['dir']}/h={h_val:.3f}"

        B_field = mu_B*np.array([5*h_val * params["field_dir"][0], 5*h_val * params["field_dir"][1], 2.5*h_val * params["field_dir"][2]])
        J = [params["J1xy"], params["J1z"], params["D"], params["E"], params["F"], params["G"], params["J3xy"], params["J3z"]]
        single_q = SingleQ(24, J, B_field)
        # Generate and analyze the spin configuration
        spins = single_q.generate_spin_configuration()

        # Calculate magnetization
        magnetization[count] = np.array(single_q.calculate_magnetization(spins)['total'])
        print(f"h = {h_val:.3f}, Magnetization = {magnetization[count]}")

        count += 1

    # Plot the magnetization
    plt.plot(h, magnetization[:, 0], label='Mx')
    plt.plot(h, magnetization[:, 1], label='My')
    plt.plot(h, magnetization[:, 2], label='Mz')
    plt.xlabel('Magnetic Field (T)')
    plt.ylabel('Magnetization')
    plt.title('Magnetization vs Magnetic Field')
    plt.legend()
    plt.grid()
    plt.savefig(f"{params['dir']}/magnetization_vs_field_single_Q.png")
    plt.show()
    plt.close()