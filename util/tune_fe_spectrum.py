#!/usr/bin/env python3
"""Quick fine-tuning scan for qFM frequency."""
import numpy as np
import subprocess
import os
import re
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'util'))
from analyze_fe_spectrum import load_trajectory, compute_spectrum, find_peaks

BUILD_DIR = os.path.join(os.path.dirname(__file__), '..', 'build')
BUILD_DIR = os.path.abspath(BUILD_DIR)
TEMPLATE = os.path.join(os.path.dirname(__file__), '..', 'example_configs', 
                        'TmFeO3', 'pump_probe_fe_only_spectrum.param')
TEMPLATE = os.path.abspath(TEMPLATE)


def run_with_params(Ka, Kc, D1, tag):
    """Run sim and return peak frequencies."""
    with open(TEMPLATE) as f:
        content = f.read()
    
    content = re.sub(r'^Ka\s*=.*$', f'Ka = {Ka}', content, flags=re.MULTILINE)
    content = re.sub(r'^Kc\s*=.*$', f'Kc = {Kc}', content, flags=re.MULTILINE)
    content = re.sub(r'^D1\s*=.*$', f'D1 = {D1}', content, flags=re.MULTILINE)
    content = re.sub(r'^output_dir\s*=.*$', f'output_dir = tune_{tag}', content, flags=re.MULTILINE)
    # Full run for accurate frequency resolution
    content = re.sub(r'^md_time_end\s*=.*$', 'md_time_end = 400.0', content, flags=re.MULTILINE)
    
    param_file = os.path.join(BUILD_DIR, f'tune_{tag}.param')
    with open(param_file, 'w') as f:
        f.write(content)
    
    result = subprocess.run(
        ['mpirun', '-np', '1', './spin_solver', param_file],
        cwd=BUILD_DIR, capture_output=True, text=True, timeout=120
    )
    
    if result.returncode != 0:
        print(f"FAILED: {tag}")
        return None, None
    
    traj_file = os.path.join(BUILD_DIR, f'tune_{tag}', 'sample_0', 'pump_probe_trajectory.txt')
    if not os.path.exists(traj_file):
        print(f"No trajectory: {tag}")
        return None, None
    
    t, G, L, F = load_trajectory(traj_file)
    freq, power = compute_spectrum(t, np.hstack([G, F]), t_start=5.0)
    peaks, _ = find_peaks(freq, power, min_freq=0.1, max_freq=2.5)
    
    if len(peaks) >= 2:
        return sorted(peaks[:2])
    return peaks, None


# Current best: Ka=-0.026, Kc=-0.029, D1=0.048 → qFM=0.358, qAFM=1.119
# Need qFM up by ~6%. Try increasing |Ka-Kc| or D1

print("Fine-tuning qFM frequency")
print("Target: qFM=0.38 THz, qAFM=1.12 THz")
print("=" * 70)
print(f"{'Ka':>8s} {'Kc':>8s} {'D1':>8s} | {'qFM':>8s} {'qAFM':>8s} | {'err_FM':>8s} {'err_AFM':>8s}")
print("-" * 70)

# Targeted scan: Ka jumps from 0.358→0.460 between -0.026 and -0.024
# Need fine grid in Ka, keep Kc=-0.029, D1=0.048
scan_params = [
    # Fine Ka grid between -0.026 and -0.024
    (-0.0255, -0.029, 0.048),
    (-0.0250, -0.029, 0.048),
    (-0.0245, -0.029, 0.048),
    # Also try D1 variations at intermediate Ka
    (-0.0255, -0.029, 0.050),
    (-0.0250, -0.029, 0.050),
    (-0.0250, -0.029, 0.046),
]

results = []
for Ka, Kc, D1 in scan_params:
    tag = f"Ka{Ka:.4f}_Kc{Kc:.4f}_D1{D1:.4f}"
    tag = tag.replace('-', 'm')
    p = run_with_params(Ka, Kc, D1, tag)
    if p[0] is not None and p[1] is not None:
        f1, f2 = p
        err_fm = abs(f1 - 0.38)
        err_afm = abs(f2 - 1.12)
        err = np.sqrt(err_fm**2 + err_afm**2)
        results.append((Ka, Kc, D1, f1, f2, err))
        print(f"{Ka:>8.4f} {Kc:>8.4f} {D1:>8.4f} | {f1:>8.4f} {f2:>8.4f} | {err_fm:>8.4f} {err_afm:>8.4f}")

print("\n" + "=" * 70)
print("Best parameters (sorted by total error):")
results.sort(key=lambda x: x[5])
for r in results[:5]:
    Ka, Kc, D1, f1, f2, err = r
    print(f"  Ka={Ka:.4f}, Kc={Kc:.4f}, D1={D1:.4f} → qFM={f1:.4f}, qAFM={f2:.4f} THz (err={err:.4f})")
