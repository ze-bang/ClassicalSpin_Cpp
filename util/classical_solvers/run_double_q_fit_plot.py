import argparse
import numpy as np
import matplotlib.pyplot as plt

from double_q_meron_antimeron import DoubleQMeronAntimeron


def _parse_param_file(path):
    params = {}
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if "," in val:
                params[key] = [v.strip() for v in val.split(",")]
            else:
                params[key] = val
    return params


def _as_float(val):
    return float(val[0] if isinstance(val, list) else val)


def _as_float_list(val):
    seq = val if isinstance(val, list) else [val]
    return [float(v) for v in seq]


def load_inputs(param_path):
    params = _parse_param_file(param_path)
    j_keys = ["J1xy", "J1z", "D", "E", "F", "G", "J3xy", "J3z"]
    J = [_as_float(params[k]) for k in j_keys]

    lattice = _as_float_list(params.get("lattice_size", [4, 4, 1]))
    L = int(lattice[0])

    field_strength = _as_float(params.get("field_strength", 0.0))
    field_dir = _as_float_list(params.get("field_direction", [0.0, 0.0, 0.0]))
    B_field = field_strength * np.array(field_dir)
    return L, J, B_field


def plot_xy_projection(positions, spins, title="XY projection", save_path=None, show=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(
        positions[:, 0],
        positions[:, 1],
        spins[:, 0],
        spins[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.002,
        color="tab:blue",
    )
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Fit double-Q meron-antimeron ansatz and plot XY projection.")
    parser.add_argument("param_file", help="Path to BCAO parameter file (e.g., fitting_param_4_x.param)")
    parser.add_argument("--save", dest="save", default="double_q_xy.png", help="Output image path")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Disable interactive display")
    parser.add_argument("--L", dest="L", type=int, default=None, help="Override lattice size (default from param file)")
    parser.add_argument("--restarts", dest="restarts", type=int, default=3, help="Number of optimization restarts")
    args = parser.parse_args()

    L, J, B_field = load_inputs(args.param_file)
    if args.L is not None:
        L = args.L
        print(f"Using override lattice size L={L}")
    if args.L is not None:
        L = args.L
        print(f"Using override lattice size L={L}")
    
    print(f"Starting double-Q fit with L={L}, J={J}")
    model = DoubleQMeronAntimeron(L=L, J=J, B_field=B_field)
    model.opt_params, model.opt_energy = model.find_minimum_energy(n_restarts=args.restarts)

    spins = model.generate_spin_configuration()
    mags = model.calculate_magnetization(spins)

    print("Double-Q fit complete")
    print(f"Lattice L: {L}")
    print(f"J parameters: {J}")
    print(f"B_field: {B_field}")
    print(f"Energy per site: {model.opt_energy:.6f}")
    print(f"Total magnetization: {mags['total']}")

    plot_xy_projection(
        model.positions,
        spins,
        title=f"Double-Q XY projection (L={L})",
        save_path=args.save,
        show=args.show,
    )
    print(f"Saved XY projection to {args.save}")


if __name__ == "__main__":
    main()
