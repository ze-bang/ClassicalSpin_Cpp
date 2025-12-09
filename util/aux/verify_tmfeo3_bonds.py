#!/usr/bin/env python3
"""
Verify TmFeO3 Fe-Fe bond connectivity.

This script computes all Fe-Fe distances and categorizes bonds into:
- Nearest Neighbors (NN / J1)
- Next-Nearest Neighbors (NNN / J2)

It also compares against the bonds defined in the simulation code.
"""

import numpy as np
from itertools import product
from collections import defaultdict

# Fe atomic positions in fractional coordinates (Pbnm structure)
FE_POSITIONS = {
    0: np.array([0.0, 0.5, 0.5]),
    1: np.array([0.5, 0.0, 0.5]),
    2: np.array([0.5, 0.0, 0.0]),
    3: np.array([0.0, 0.5, 0.0]),
}

# Lattice vectors (orthorhombic, normalized to 1 for simplicity)
# In real TmFeO3: a ≈ 5.25 Å, b ≈ 5.60 Å, c ≈ 7.60 Å
LATTICE_VECTORS = np.eye(3)  # Using normalized lattice


def compute_distance(site_i, site_j, offset, lattice=LATTICE_VECTORS):
    """Compute distance between site_i and site_j with unit cell offset."""
    pos_i = FE_POSITIONS[site_i]
    pos_j = FE_POSITIONS[site_j] + np.array(offset)
    
    # Convert to Cartesian coordinates
    cart_i = lattice @ pos_i
    cart_j = lattice @ pos_j
    
    return np.linalg.norm(cart_j - cart_i)


def find_all_bonds(max_distance=1.5, offset_range=2):
    """Find all Fe-Fe bonds up to max_distance."""
    bonds = []
    
    for site_i in range(4):
        for site_j in range(4):
            for offset in product(range(-offset_range, offset_range + 1), repeat=3):
                # Skip self-interaction at origin
                if site_i == site_j and offset == (0, 0, 0):
                    continue
                
                dist = compute_distance(site_i, site_j, offset)
                
                if dist < max_distance:
                    bonds.append({
                        'site_i': site_i,
                        'site_j': site_j,
                        'offset': offset,
                        'distance': dist
                    })
    
    return bonds


def categorize_bonds(bonds, tolerance=0.01):
    """Categorize bonds by distance into NN and NNN groups."""
    # Get unique distances
    distances = sorted(set(round(b['distance'], 3) for b in bonds))
    
    print("=" * 80)
    print("UNIQUE Fe-Fe DISTANCES (normalized lattice)")
    print("=" * 80)
    for i, d in enumerate(distances):
        count = sum(1 for b in bonds if abs(b['distance'] - d) < tolerance)
        print(f"  Distance {i+1}: {d:.4f}  ({count} bonds total)")
    print()
    
    # Categorize
    categories = defaultdict(list)
    
    for bond in bonds:
        d = bond['distance']
        if abs(d - 0.5) < tolerance:
            categories['J1c (NN out-of-plane, d=0.5)'].append(bond)
        elif abs(d - 0.707) < tolerance:
            categories['J1ab (NN in-plane, d=0.707)'].append(bond)
        elif abs(d - 0.866) < tolerance:
            categories['J2_diag (NNN diagonal, d=0.866)'].append(bond)
        elif abs(d - 1.0) < tolerance:
            categories['J2_same (NNN same-sublattice, d=1.0)'].append(bond)
        elif abs(d - 1.118) < tolerance:
            categories['3rd NN (d=1.118)'].append(bond)
        else:
            categories[f'Other (d={d:.3f})'].append(bond)
    
    return categories


def format_bond(bond):
    """Format a bond for printing."""
    return f"({bond['site_i']}→{bond['site_j']}, {{{bond['offset'][0]},{bond['offset'][1]},{bond['offset'][2]}}})"


def print_category_details(categories):
    """Print detailed bond information for each category."""
    
    for cat_name in sorted(categories.keys()):
        bonds = categories[cat_name]
        print("=" * 80)
        print(f"{cat_name}")
        print(f"Total bonds: {len(bonds)}")
        print("=" * 80)
        
        # Group by (site_i, site_j) pair
        by_pair = defaultdict(list)
        for b in bonds:
            by_pair[(b['site_i'], b['site_j'])].append(b['offset'])
        
        for (si, sj), offsets in sorted(by_pair.items()):
            offset_str = ", ".join(f"{{{o[0]},{o[1]},{o[2]}}}" for o in sorted(offsets))
            print(f"  ({si}→{sj}): [{len(offsets)} bonds] {offset_str}")
        print()


def get_implemented_bonds():
    """Return the bonds as implemented in the simulation code."""
    implemented = {
        'J1c': [
            (0, 3, (0, 0, 0)), (0, 3, (0, 0, 1)),
            (1, 2, (0, 0, 0)), (1, 2, (0, 0, 1)),
        ],
        'J1ab_Ja': [
            (1, 0, (0, 0, 0)), (1, 0, (1, -1, 0)),
            (2, 3, (0, 0, 0)), (2, 3, (1, -1, 0)),
        ],
        'J1ab_Jb': [
            (1, 0, (0, -1, 0)), (1, 0, (1, 0, 0)),
            (2, 3, (0, -1, 0)), (2, 3, (1, 0, 0)),
        ],
        'J2_same_sublattice': [
            # Along a-axis
            (0, 0, (1, 0, 0)), (1, 1, (1, 0, 0)), (2, 2, (1, 0, 0)), (3, 3, (1, 0, 0)),
            # Along b-axis
            (0, 0, (0, 1, 0)), (1, 1, (0, 1, 0)), (2, 2, (0, 1, 0)), (3, 3, (0, 1, 0)),
            # Along c-axis (newly added)
            (0, 0, (0, 0, 1)), (1, 1, (0, 0, 1)), (2, 2, (0, 0, 1)), (3, 3, (0, 0, 1)),
        ],
        'J2c_0_2': [
            (0, 2, (0, 0, 0)), (0, 2, (0, 1, 0)), (0, 2, (-1, 0, 0)), (0, 2, (-1, 1, 0)),
            (0, 2, (0, 0, 1)), (0, 2, (0, 1, 1)), (0, 2, (-1, 0, 1)), (0, 2, (-1, 1, 1)),
        ],
        'J2c_1_3': [
            (1, 3, (0, 0, 0)), (1, 3, (0, -1, 0)), (1, 3, (1, 0, 0)), (1, 3, (1, -1, 0)),
            (1, 3, (0, 0, 1)), (1, 3, (0, -1, 1)), (1, 3, (1, 0, 1)), (1, 3, (1, -1, 1)),
        ],
    }
    return implemented


def get_canonical_bond(site_i, site_j, offset):
    """
    Return the canonical form of a bond (i→j, offset) or its reverse (j→i, -offset).
    Convention: smaller site index first, or if equal, positive offset direction.
    """
    reverse = (site_j, site_i, tuple(-np.array(offset)))
    forward = (site_i, site_j, tuple(offset))
    
    if site_i < site_j:
        return forward
    elif site_j < site_i:
        return reverse
    else:
        # Same site, choose positive offset direction
        if offset[0] > 0 or (offset[0] == 0 and offset[1] > 0) or (offset[0] == 0 and offset[1] == 0 and offset[2] > 0):
            return forward
        else:
            return reverse


def symmetrize_bonds(bond_set):
    """Convert a set of bonds to canonical form (removing duplicates from reverse bonds)."""
    canonical = set()
    for b in bond_set:
        canonical.add(get_canonical_bond(b[0], b[1], b[2]))
    return canonical


def verify_implementation(categories):
    """Compare computed bonds with implemented bonds."""
    implemented = get_implemented_bonds()
    
    print("=" * 80)
    print("VERIFICATION: Comparing computed bonds vs implementation")
    print("=" * 80)
    print("Note: Bonds are symmetrized (i→j ≡ j→i with reversed offset)")
    print()
    
    # Check J1c
    print("[J1c - NN out-of-plane (d=0.5)]")
    j1c_computed = set()
    for b in categories.get('J1c (NN out-of-plane, d=0.5)', []):
        j1c_computed.add((b['site_i'], b['site_j'], tuple(b['offset'])))
    
    j1c_impl = set(implemented['J1c'])
    
    j1c_computed_sym = symmetrize_bonds(j1c_computed)
    j1c_impl_sym = symmetrize_bonds(j1c_impl)
    
    if j1c_computed_sym == j1c_impl_sym:
        print(f"  ✓ All {len(j1c_computed_sym)} unique bonds implemented correctly")
    else:
        missing = j1c_computed_sym - j1c_impl_sym
        extra = j1c_impl_sym - j1c_computed_sym
        if missing:
            print(f"  ⚠ Missing bonds: {missing}")
        if extra:
            print(f"  ⚠ Extra bonds: {extra}")
    
    # Check J1ab
    print("\n[J1ab - NN in-plane (d=0.707)]")
    j1ab_computed = set()
    for b in categories.get('J1ab (NN in-plane, d=0.707)', []):
        j1ab_computed.add((b['site_i'], b['site_j'], tuple(b['offset'])))
    
    j1ab_impl = set(implemented['J1ab_Ja'] + implemented['J1ab_Jb'])
    
    j1ab_computed_sym = symmetrize_bonds(j1ab_computed)
    j1ab_impl_sym = symmetrize_bonds(j1ab_impl)
    
    if j1ab_computed_sym == j1ab_impl_sym:
        print(f"  ✓ All {len(j1ab_computed_sym)} unique bonds implemented correctly")
    else:
        missing = j1ab_computed_sym - j1ab_impl_sym
        extra = j1ab_impl_sym - j1ab_computed_sym
        if missing:
            print(f"  ⚠ Missing bonds: {missing}")
        if extra:
            print(f"  ⚠ Extra bonds: {extra}")
    
    # Check J2 diagonal
    print("\n[J2_diag - NNN diagonal (d=0.866)]")
    j2diag_computed = set()
    for b in categories.get('J2_diag (NNN diagonal, d=0.866)', []):
        j2diag_computed.add((b['site_i'], b['site_j'], tuple(b['offset'])))
    
    j2diag_impl = set(implemented['J2c_0_2'] + implemented['J2c_1_3'])
    
    j2diag_computed_sym = symmetrize_bonds(j2diag_computed)
    j2diag_impl_sym = symmetrize_bonds(j2diag_impl)
    
    if j2diag_computed_sym == j2diag_impl_sym:
        print(f"  ✓ All {len(j2diag_computed_sym)} unique bonds implemented correctly")
    else:
        missing = j2diag_computed_sym - j2diag_impl_sym
        extra = j2diag_impl_sym - j2diag_computed_sym
        if missing:
            print(f"  ⚠ Missing bonds: {missing}")
        if extra:
            print(f"  ⚠ Extra bonds: {extra}")
    
    # Check J2 same-sublattice
    print("\n[J2_same - NNN same-sublattice (d=1.0)]")
    j2same_computed = set()
    for b in categories.get('J2_same (NNN same-sublattice, d=1.0)', []):
        j2same_computed.add((b['site_i'], b['site_j'], tuple(b['offset'])))
    
    j2same_impl = set(implemented['J2_same_sublattice'])
    
    j2same_computed_sym = symmetrize_bonds(j2same_computed)
    j2same_impl_sym = symmetrize_bonds(j2same_impl)
    
    if j2same_computed_sym == j2same_impl_sym:
        print(f"  ✓ All {len(j2same_computed_sym)} unique bonds implemented correctly")
    else:
        missing = j2same_computed_sym - j2same_impl_sym
        extra = j2same_impl_sym - j2same_computed_sym
        print(f"  Implemented: {len(j2same_impl_sym)} unique bonds")
        print(f"  Computed:    {len(j2same_computed_sym)} unique bonds")
        if missing:
            print(f"  ⚠ Missing bonds ({len(missing)}):")
            for m in sorted(missing):
                print(f"      {m}")
        if extra:
            print(f"  ⚠ Extra bonds ({len(extra)}):")
            for e in sorted(extra):
                print(f"      {e}")


def print_coordination_summary(categories):
    """Print coordination number summary for each site."""
    print("=" * 80)
    print("COORDINATION SUMMARY (bonds per site)")
    print("=" * 80)
    
    coord = {i: defaultdict(int) for i in range(4)}
    
    for cat_name, bonds in categories.items():
        for b in bonds:
            coord[b['site_i']][cat_name] += 1
    
    for site in range(4):
        print(f"\nSite {site} at {FE_POSITIONS[site]}:")
        total = 0
        for cat_name in sorted(coord[site].keys()):
            count = coord[site][cat_name]
            print(f"  {cat_name}: {count}")
            total += count
        print(f"  TOTAL: {total}")


def print_bond_table():
    """Print a summary table of all bond types."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TmFeO3 Fe-Fe BOND CONNECTIVITY SUMMARY                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Fe Sites (fractional coords):                                                 ║
║   Site 0: (0.0, 0.5, 0.5)  ─┐                                                 ║
║   Site 1: (0.5, 0.0, 0.5)  ─┴─ z=0.5 layer                                    ║
║   Site 2: (0.5, 0.0, 0.0)  ─┐                                                 ║
║   Site 3: (0.0, 0.5, 0.0)  ─┴─ z=0.0 layer                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  NEAREST NEIGHBORS (J1):                                                     ║
║  ───────────────────────                                                     ║
║  • J1c  (d=0.5):   Out-of-plane bonds along c-axis                           ║
║                    (0↔3) and (1↔2) pairs                                     ║
║                                                                              ║
║  • J1ab (d=0.707): In-plane diagonal bonds                                   ║
║                    (1↔0) and (2↔3) pairs                                     ║
║                    Split into Ja and Jb for DM interaction symmetry          ║
║                                                                              ║
║  NEXT-NEAREST NEIGHBORS (J2):                                                ║
║  ────────────────────────────                                                ║
║  • J2_diag (d=0.866): Cross-sublattice diagonal bonds                        ║
║                       (0↔2) and (1↔3) pairs, 8 bonds each                    ║
║                                                                              ║
║  • J2_same (d=1.0):   Same-sublattice bonds                                  ║
║                       Along a-axis: {1,0,0} (J2ab coupling)                  ║
║                       Along b-axis: {0,1,0} (J2ab coupling)                  ║
║                       Along c-axis: {0,0,1} (J2c_same coupling)              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def main():
    print_bond_table()
    
    # Find all bonds
    bonds = find_all_bonds(max_distance=1.2)
    
    # Categorize
    categories = categorize_bonds(bonds)
    
    # Print details
    print_category_details(categories)
    
    # Print coordination summary
    print_coordination_summary(categories)
    
    # Verify against implementation
    verify_implementation(categories)
    
    print("\n" + "=" * 80)
    print("NOTES:")
    print("=" * 80)
    print("""
1. The J1 bonds (NN) are fully implemented:
   - J1c connects vertically stacked pairs (0↔3, 1↔2)
   - J1ab connects in-plane diagonal pairs (1↔0, 2↔3)

2. The J2 bonds (NNN) are fully implemented:
   - J2_diag (cross-sublattice) uses J2c coupling
   - J2_same (same-sublattice) along a,b uses J2ab coupling
   - J2_same (same-sublattice) along c uses J2c_same coupling (default=0)

3. The c-axis same-sublattice J2 coupling (J2c_same) defaults to 0 because:
   - In real TmFeO3, c ≈ 7.6 Å while a,b ≈ 5.3-5.6 Å
   - The c-axis same-sublattice distance is physically much larger
   - This coupling is expected to be much weaker
   - Set J2c_same > 0 in config to enable if needed

4. For the simulation code, bonds are defined one-way (i→j with offset).
   The code should automatically add the reverse bond (j→i with -offset)
   or the Hamiltonian should be symmetric.
""")


if __name__ == "__main__":
    main()
