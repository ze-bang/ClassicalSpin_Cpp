"""
Analysis and Visualization for BCAO Phase Exploration Results.

Generates infographics showing:
1. Phase distribution pie chart
2. Parameter space scatter plots (colored by phase)
3. Feature importance bar chart
4. Exploration history timeline
5. Phase boundary projections
"""

import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# Phase colors
PHASE_COLORS = {
    "Ferromagnetic": "#e41a1c",
    "Antiferromagnetic": "#377eb8",
    "Zigzag": "#4daf4a",
    "Double-Q Commensurate": "#984ea3",
    "Double-Q Zigzag": "#984ea3",  # Legacy alias
    "120° Order": "#ff7f00",
    "Incommensurate Γ→M": "#ffff33",
    "Incommensurate Γ→K": "#a65628",
    "Double-Q Incommensurate Γ→M": "#f781bf",
    "Double-Q Incommensurate Γ→K": "#999999",
    "Double-Q Meron-Antimeron": "#66c2a5",
    "Triple-Q Commensurate": "#8dd3c7",
    "Triple-Q Skyrmion": "#bebada",
    "Triple-Q Incommensurate": "#fb8072",
    "Incommensurate": "#fc8d62",
    "Double-Q Incommensurate": "#8da0cb",
    "Disordered": "#e78ac3",
    "Unknown": "#a6d854",
}

PARAM_NAMES = ['J1z', 'D', 'E', 'F', 'G', 'J3xy', 'J3z']


def load_exploration_results(results_dir: str) -> Tuple[Dict, List[Dict], Optional[object]]:
    """Load exploration history and model from results directory.
    
    Returns:
        Tuple of (metadata, points_list, model)
    """
    results_path = Path(results_dir)
    
    # Load history
    history_file = results_path / "exploration_history.json"
    if not history_file.exists():
        raise FileNotFoundError(f"History file not found: {history_file}")
    
    with open(history_file, 'r') as f:
        data = json.load(f)
    
    # Extract metadata and points
    metadata = data.get('metadata', {})
    phase_counts = data.get('phase_counts', {})
    points = data.get('points', [])
    
    # Combine metadata with phase_counts
    metadata['phase_counts'] = phase_counts
    
    # Load model if exists
    model_file = results_path / "surrogate_model.pkl"
    model = None
    if model_file.exists():
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    
    return metadata, points, model


def plot_phase_distribution(history: List[Dict], output_dir: str):
    """Create pie chart of phase distribution."""
    phases = [p['phase'] for p in history if p.get('phase')]
    phase_counts = {}
    for phase in phases:
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    # Sort by count
    sorted_phases = sorted(phase_counts.items(), key=lambda x: -x[1])
    labels = [p[0] for p in sorted_phases]
    sizes = [p[1] for p in sorted_phases]
    colors = [PHASE_COLORS.get(p, '#808080') for p in labels]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=None,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        pctdistance=0.75
    )
    
    # Legend
    ax.legend(wedges, [f"{l} ({s})" for l, s in zip(labels, sizes)],
              title="Phases",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))
    
    ax.set_title(f'Phase Distribution (n={len(phases)})', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/phase_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/phase_distribution.png")


def plot_parameter_scatter(history: List[Dict], output_dir: str):
    """Create scatter plots of parameter space colored by phase."""
    # Extract parameters and phases
    points = []
    for p in history:
        if p.get('phase') and p.get('params'):
            params = p['params']
            points.append({
                'J1z': params.get('J1z_norm', 0),
                'D': params.get('D_norm', 0),
                'E': params.get('E_norm', 0),
                'F': params.get('F_norm', 0),
                'G': params.get('G_norm', 0),
                'J3xy': params.get('J3xy_norm', 0),
                'J3z': params.get('J3z_norm', 0),
                'phase': p['phase']
            })
    
    if not points:
        print("  No valid points for scatter plot")
        return
    
    # Create phase-to-color mapping
    unique_phases = list(set(p['phase'] for p in points))
    phase_to_idx = {phase: i for i, phase in enumerate(unique_phases)}
    
    # Key parameter pairs to visualize
    param_pairs = [
        ('J1z', 'E'),
        ('J1z', 'F'),
        ('E', 'F'),
        ('J3xy', 'J1z'),
        ('D', 'G'),
        ('J3xy', 'J3z'),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (px, py) in enumerate(param_pairs):
        ax = axes[idx]
        
        for phase in unique_phases:
            phase_points = [p for p in points if p['phase'] == phase]
            if phase_points:
                x = [p[px] for p in phase_points]
                y = [p[py] for p in phase_points]
                color = PHASE_COLORS.get(phase, '#808080')
                ax.scatter(x, y, c=color, label=phase, alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel(f'{px} (normalized)', fontsize=10)
        ax.set_ylabel(f'{py} (normalized)', fontsize=10)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax.set_title(f'{px} vs {py}', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # Create legend
    handles = [mpatches.Patch(color=PHASE_COLORS.get(p, '#808080'), label=p) 
               for p in unique_phases]
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.15, 0.5),
               fontsize=9, title='Phases')
    
    plt.suptitle('Parameter Space Exploration by Phase', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/parameter_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/parameter_scatter.png")


def plot_feature_importance(model_data, output_dir: str):
    """Plot feature importance from the surrogate model."""
    if model_data is None:
        print("  No model available for feature importance")
        return
    
    # Handle dictionary format (model + label_encoder)
    if isinstance(model_data, dict):
        model = model_data.get('model')
    else:
        model = model_data
    
    if model is None:
        print("  No model found in model_data")
        return
    
    # Try to get feature importance
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            print("  Model doesn't have feature importance attribute")
            return
    except Exception as e:
        print(f"  Could not extract feature importance: {e}")
        return
    
    # Ensure we have the right number of features
    if len(importances) != len(PARAM_NAMES):
        print(f"  Feature count mismatch: {len(importances)} vs {len(PARAM_NAMES)}")
        # Truncate or pad
        if len(importances) > len(PARAM_NAMES):
            importances = importances[:len(PARAM_NAMES)]
        else:
            importances = np.pad(importances, (0, len(PARAM_NAMES) - len(importances)))
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    sorted_names = [PARAM_NAMES[i] for i in indices]
    sorted_imp = importances[indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_names)))
    bars = ax.barh(range(len(sorted_names)), sorted_imp, color=colors)
    
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance for Phase Classification', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sorted_imp)):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=10)
    
    ax.set_xlim(0, max(sorted_imp) * 1.2)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/feature_importance.png")


def plot_exploration_timeline(history: List[Dict], output_dir: str):
    """Plot exploration timeline showing phase discoveries."""
    if not history:
        return
    
    # Sort by point_id
    sorted_history = sorted(history, key=lambda x: x.get('point_id', 0))
    
    phases = [p.get('phase', 'Unknown') for p in sorted_history]
    point_ids = [p.get('point_id', i) for i, p in enumerate(sorted_history)]
    
    # Cumulative phase counts
    unique_phases = list(set(phases))
    cumulative = {phase: [] for phase in unique_phases}
    
    for i, phase in enumerate(phases):
        for p in unique_phases:
            prev = cumulative[p][-1] if cumulative[p] else 0
            cumulative[p].append(prev + (1 if phase == p else 0))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top: Phase at each point
    colors = [PHASE_COLORS.get(p, '#808080') for p in phases]
    ax1.scatter(point_ids, [1]*len(point_ids), c=colors, s=50, alpha=0.8)
    ax1.set_xlim(min(point_ids)-1, max(point_ids)+1)
    ax1.set_ylim(0.5, 1.5)
    ax1.set_yticks([])
    ax1.set_xlabel('Exploration Point', fontsize=11)
    ax1.set_title('Phase Discovery Timeline', fontsize=12, fontweight='bold')
    
    # Add legend
    handles = [mpatches.Patch(color=PHASE_COLORS.get(p, '#808080'), label=p) 
               for p in unique_phases]
    ax1.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1),
               fontsize=8, title='Phases')
    
    # Bottom: Cumulative counts
    for phase in unique_phases:
        color = PHASE_COLORS.get(phase, '#808080')
        ax2.plot(point_ids, cumulative[phase], label=phase, color=color, linewidth=2)
    
    ax2.set_xlabel('Exploration Point', fontsize=11)
    ax2.set_ylabel('Cumulative Count', fontsize=11)
    ax2.set_title('Cumulative Phase Counts', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/exploration_timeline.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/exploration_timeline.png")


def plot_q_vector_distribution(history: List[Dict], output_dir: str):
    """Plot distribution of Q-vectors in reciprocal space.
    
    Creates separate subplots for single-Q, double-Q, and triple-Q states.
    Includes coordinate annotations and connections between Q-vectors in multi-Q states.
    """
    # Collect all Q-vectors with their metadata
    q_data = []  # List of (q_vector, phase, point_id, q_type)
    
    for p in history:
        if p.get('decision_flags') and p.get('phase'):
            flags = p['decision_flags']
            phase = p['phase']
            point_id = p.get('point_id', 0)
            
            # Collect q1, q2, q3 if they exist
            q_vectors = []
            for q_name in ['q1', 'q2', 'q3']:
                if q_name in flags and flags[q_name].get('position'):
                    q_vec = flags[q_name]['position']
                    if q_vec and len(q_vec) >= 2:
                        q_vectors.append((q_vec, q_name))
            
            # Store all Q-vectors for this point
            for q_vec, q_name in q_vectors:
                q_data.append({
                    'q': q_vec,
                    'phase': phase,
                    'point_id': point_id,
                    'q_type': q_name,
                    'n_q': len(q_vectors)  # Number of Q-vectors for this state
                })
    
    if not q_data:
        print("  No Q-vector data available")
        return
    
    # Separate by number of Q-vectors
    single_q = [d for d in q_data if d['n_q'] == 1]
    double_q = [d for d in q_data if d['n_q'] == 2]
    triple_q = [d for d in q_data if d['n_q'] == 3]
    
    n_1q = len(set(d['point_id'] for d in single_q))
    n_2q = len(set(d['point_id'] for d in double_q))
    n_3q = len(set(d['point_id'] for d in triple_q))
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Helper function to add symmetry points and lines
    def add_symmetry_elements(ax):
        # Mark high-symmetry points
        ax.plot(0, 0, 'k*', markersize=15, label='Γ (0,0)', zorder=100)
        ax.plot(0.5, 0, 'kD', markersize=10, label='M (0.5,0)', zorder=100)
        ax.plot(0, 0.5, 'kD', markersize=10, alpha=0.6, zorder=100)
        ax.plot(1/3, 1/3, 'kp', markersize=10, label='K (0.333,0.333)', zorder=100)
        
        # Add symmetry lines
        ax.plot([0, 0.5], [0, 0], 'k--', alpha=0.2, linewidth=0.5)
        ax.plot([0, 1/3], [0, 1/3], 'k--', alpha=0.2, linewidth=0.5)
        ax.plot([0.5, 1/3], [0, 1/3], 'k--', alpha=0.2, linewidth=0.5)
        
        ax.set_xlabel('Qₓ (r.l.u.)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Qᵧ (r.l.u.)', fontsize=11, fontweight='bold')
        ax.set_xlim(-0.05, 0.6)
        ax.set_ylim(-0.05, 0.6)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # ========== SUBPLOT 1: Single-Q states ==========
    ax1 = axes[0]
    if single_q:
        unique_phases = list(set(d['phase'] for d in single_q))
        for phase in unique_phases:
            phase_qs = [d for d in single_q if d['phase'] == phase]
            x = [d['q'][0] for d in phase_qs]
            y = [d['q'][1] for d in phase_qs]
            color = PHASE_COLORS.get(phase, '#808080')
            ax1.scatter(x, y, c=color, label=phase, alpha=0.8, 
                       s=80, edgecolors='black', linewidth=1.2, marker='o')
            
            # Annotate a few representative points
            if len(phase_qs) <= 10:
                for i, d in enumerate(phase_qs[:5]):  # Show up to 5
                    ax1.annotate(f"({d['q'][0]:.3f},{d['q'][1]:.3f})", 
                               (d['q'][0], d['q'][1]), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=7, alpha=0.7)
    
    add_symmetry_elements(ax1)
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax1.set_title(f'Single-Q States (n={n_1q})', fontsize=12, fontweight='bold')
    
    # ========== SUBPLOT 2: Double-Q states ==========
    ax2 = axes[1]
    if double_q:
        # Group by point_id to connect q1 and q2
        point_ids = list(set(d['point_id'] for d in double_q))
        unique_phases = list(set(d['phase'] for d in double_q))
        
        # Plot each phase
        for phase in unique_phases:
            phase_points = [pid for pid in point_ids 
                          if any(d['point_id'] == pid and d['phase'] == phase for d in double_q)]
            
            color = PHASE_COLORS.get(phase, '#808080')
            
            for pid in phase_points:
                point_qs = [d for d in double_q if d['point_id'] == pid]
                if len(point_qs) >= 2:
                    # Plot Q-vectors
                    x = [d['q'][0] for d in point_qs[:2]]
                    y = [d['q'][1] for d in point_qs[:2]]
                    ax2.scatter(x, y, c=color, alpha=0.7, 
                              s=70, edgecolors='black', linewidth=1, marker='s')
                    
                    # Draw line connecting the Q-vectors
                    ax2.plot(x, y, color=color, alpha=0.3, linewidth=1.5, linestyle='--')
                    
                    # Annotate with q1, q2 labels
                    if len(phase_points) <= 20:  # Only annotate if not too crowded
                        for d in point_qs[:2]:
                            label = f"{d['q_type']}"
                            ax2.annotate(label, (d['q'][0], d['q'][1]), 
                                       xytext=(3, 3), textcoords='offset points',
                                       fontsize=6, alpha=0.8)
            
            # Add to legend
            ax2.scatter([], [], c=color, s=70, marker='s', 
                       edgecolors='black', linewidth=1, label=phase, alpha=0.7)
    
    add_symmetry_elements(ax2)
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax2.set_title(f'Double-Q States (n={n_2q})', fontsize=12, fontweight='bold')
    
    # ========== SUBPLOT 3: Triple-Q states ==========
    ax3 = axes[2]
    if triple_q:
        # Group by point_id to connect q1, q2, q3
        point_ids = list(set(d['point_id'] for d in triple_q))
        unique_phases = list(set(d['phase'] for d in triple_q))
        
        for phase in unique_phases:
            phase_points = [pid for pid in point_ids 
                          if any(d['point_id'] == pid and d['phase'] == phase for d in triple_q)]
            
            color = PHASE_COLORS.get(phase, '#808080')
            
            for pid in phase_points:
                point_qs = [d for d in triple_q if d['point_id'] == pid]
                if len(point_qs) >= 3:
                    # Plot Q-vectors
                    x = [d['q'][0] for d in point_qs[:3]]
                    y = [d['q'][1] for d in point_qs[:3]]
                    ax3.scatter(x, y, c=color, alpha=0.7, 
                              s=100, edgecolors='black', linewidth=1.5, marker='^')
                    
                    # Draw triangle connecting the Q-vectors
                    x_closed = x + [x[0]]
                    y_closed = y + [y[0]]
                    ax3.plot(x_closed, y_closed, color=color, alpha=0.4, 
                           linewidth=1.5, linestyle='-')
                    
                    # Annotate coordinates for triple-Q
                    for d in point_qs[:3]:
                        ax3.annotate(f"{d['q_type']}\n({d['q'][0]:.3f},{d['q'][1]:.3f})", 
                                   (d['q'][0], d['q'][1]), 
                                   xytext=(8, 8), textcoords='offset points',
                                   fontsize=6, alpha=0.8, 
                                   bbox=dict(boxstyle='round,pad=0.3', 
                                           facecolor=color, alpha=0.3))
            
            # Add to legend
            ax3.scatter([], [], c=color, s=100, marker='^', 
                       edgecolors='black', linewidth=1.5, label=phase, alpha=0.7)
    else:
        ax3.text(0.3, 0.3, 'No Triple-Q States Found', 
                fontsize=12, ha='center', va='center', style='italic', alpha=0.5)
    
    add_symmetry_elements(ax3)
    if triple_q:
        ax3.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax3.set_title(f'Triple-Q States (n={n_3q})', fontsize=12, fontweight='bold')
    
    # Overall title
    plt.suptitle('Q-vector Distribution in Reciprocal Space', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/q_vector_distribution.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_dir}/q_vector_distribution.png")
    print(f"    Single-Q states: {n_1q}")
    print(f"    Double-Q states: {n_2q}")
    print(f"    Triple-Q states: {n_3q}")


def plot_energy_landscape(history: List[Dict], output_dir: str):
    """Plot energy vs phase and parameters."""
    points = []
    for p in history:
        if p.get('phase') and p.get('params') and p.get('energy', 0) != 0:
            params = p['params']
            points.append({
                'energy': p['energy'],
                'phase': p['phase'],
                'J1z': params.get('J1z_norm', 0),
                'J3xy': params.get('J3xy_norm', 0),
            })
    
    if not points:
        print("  No energy data available")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Energy distribution by phase
    unique_phases = list(set(p['phase'] for p in points))
    phase_energies = {phase: [p['energy'] for p in points if p['phase'] == phase] 
                      for phase in unique_phases}
    
    positions = range(len(unique_phases))
    colors = [PHASE_COLORS.get(p, '#808080') for p in unique_phases]
    
    bp = ax1.boxplot([phase_energies[p] for p in unique_phases], 
                      positions=positions, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels([p[:15] + '...' if len(p) > 15 else p for p in unique_phases], 
                        rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Energy per site', fontsize=11)
    ax1.set_title('Energy Distribution by Phase', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: Energy vs J3xy (key parameter)
    for phase in unique_phases:
        phase_pts = [p for p in points if p['phase'] == phase]
        if phase_pts:
            x = [p['J3xy'] for p in phase_pts]
            y = [p['energy'] for p in phase_pts]
            color = PHASE_COLORS.get(phase, '#808080')
            ax2.scatter(x, y, c=color, label=phase, alpha=0.6, s=40)
    
    ax2.set_xlabel('J3xy (normalized)', fontsize=11)
    ax2.set_ylabel('Energy per site', fontsize=11)
    ax2.set_title('Energy vs Third-Neighbor Coupling', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/energy_landscape.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/energy_landscape.png")


def generate_summary_report(history: List[Dict], model, output_dir: str, metadata: Dict = None):
    """Generate a text summary report."""
    phases = [p['phase'] for p in history if p.get('phase')]
    phase_counts = {}
    for phase in phases:
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    report = []
    report.append("=" * 70)
    report.append("BCAO PHASE EXPLORATION SUMMARY REPORT")
    report.append("=" * 70)
    
    if metadata:
        report.append(f"\nExploration timestamp: {metadata.get('timestamp', 'N/A')}")
        report.append(f"Lattice size (L): {metadata.get('L', 'N/A')}")
        report.append(f"Target phase: {metadata.get('target_phase', 'N/A')}")
    
    report.append(f"\nTotal points explored: {len(history)}")
    report.append(f"Classified points: {len(phases)}")
    report.append(f"Unique phases found: {len(phase_counts)}")
    
    report.append("\n" + "-" * 40)
    report.append("PHASE DISTRIBUTION")
    report.append("-" * 40)
    for phase, count in sorted(phase_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(phases)
        report.append(f"  {phase}: {count} ({pct:.1f}%)")
    
    # Target phase analysis
    target_phases = ["Double-Q Meron-Antimeron", "Double-Q Incommensurate Γ→M", 
                     "Double-Q Incommensurate Γ→K", "Double-Q Incommensurate"]
    target_points = [p for p in history if p.get('phase') in target_phases]
    
    if target_points:
        report.append("\n" + "-" * 40)
        report.append("DOUBLE-Q PHASE ANALYSIS")
        report.append("-" * 40)
        report.append(f"Total double-Q points: {len(target_points)}")
        
        for phase in target_phases:
            pts = [p for p in target_points if p['phase'] == phase]
            if pts:
                report.append(f"\n  {phase}: {len(pts)} points")
                # Show parameter ranges
                params_list = [p['params'] for p in pts if p.get('params')]
                if params_list:
                    for param in PARAM_NAMES:
                        key = f"{param}_norm"
                        vals = [p.get(key, 0) for p in params_list]
                        if vals:
                            report.append(f"    {param}: [{min(vals):.3f}, {max(vals):.3f}]")
    
    # Feature importance
    if model and hasattr(model, 'feature_importances_'):
        report.append("\n" + "-" * 40)
        report.append("FEATURE IMPORTANCE")
        report.append("-" * 40)
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        for i in sorted_idx:
            if i < len(PARAM_NAMES):
                report.append(f"  {PARAM_NAMES[i]}: {importances[i]:.4f}")
    
    report.append("\n" + "=" * 70)
    
    # Save report
    report_text = "\n".join(report)
    with open(f"{output_dir}/summary_report.txt", 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n  Saved: {output_dir}/summary_report.txt")


def main():
    parser = argparse.ArgumentParser(description="Analyze BCAO exploration results")
    parser.add_argument('results_dir', type=str, help='Directory containing exploration results')
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='Output directory for plots (default: same as results_dir)')
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.results_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("BCAO PHASE EXPLORATION ANALYSIS")
    print("=" * 70)
    
    # Load results
    print("\nLoading results...")
    metadata, points, model = load_exploration_results(args.results_dir)
    print(f"  Loaded {len(points)} exploration points")
    print(f"  Total phases found: {len(metadata.get('phase_counts', {}))}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    print("\n1. Phase distribution pie chart...")
    plot_phase_distribution(points, output_dir)
    
    print("\n2. Parameter space scatter plots...")
    plot_parameter_scatter(points, output_dir)
    
    print("\n3. Feature importance...")
    plot_feature_importance(model, output_dir)
    
    print("\n4. Exploration timeline...")
    plot_exploration_timeline(points, output_dir)
    
    print("\n5. Q-vector distribution...")
    plot_q_vector_distribution(points, output_dir)
    
    print("\n6. Energy landscape...")
    plot_energy_landscape(points, output_dir)
    
    print("\n7. Summary report...")
    generate_summary_report(points, model, output_dir, metadata)
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
