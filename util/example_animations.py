#!/usr/bin/env python3
"""
Example script demonstrating how to use the DSSF animation functionality.
"""

import sys
import os

# Add the util directory to the path if needed
sys.path.insert(0, os.path.dirname(__file__))

from reader_pyrochlore import (
    animate_DSSF_spinon_photon,
    collect_DSSF_data_all_fields,
    extract_field_value
)

def example_usage():
    """Example showing different ways to use the animation functions."""
    
    # Example 1: Create animations with default settings
    print("Example 1: Create animations with default settings")
    print("-" * 50)
    root_dir = "/path/to/your/MD_pi_flux_data"
    
    try:
        # This will create animations in root_dir/animations/
        animate_DSSF_spinon_photon(root_dir)
        print("✓ Animations created successfully!")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()
    
    # Example 2: Create animations with custom settings
    print("Example 2: Create animations with custom settings")
    print("-" * 50)
    
    try:
        animate_DSSF_spinon_photon(
            root_dir=root_dir,
            output_dir="/path/to/custom/output",
            fps=10,  # Higher frame rate
            energy_conversion=0.063  # Convert to meV
        )
        print("✓ Custom animations created successfully!")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()
    
    # Example 3: Collect and inspect data before animating
    print("Example 3: Collect and inspect data")
    print("-" * 50)
    
    try:
        field_data = collect_DSSF_data_all_fields(root_dir, verbose=True)
        print(f"\nCollected data from {len(field_data)} field directories:")
        for data in field_data:
            print(f"  Field = {data['field']:.3f}, "
                  f"ω points = {len(data['w'])}, "
                  f"max DSSF (local) = {data['dssf_local_spinon_photon'].max():.4e}")
        print("✓ Data collection successful!")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()
    
    # Example 4: Test field value extraction
    print("Example 4: Test field value extraction")
    print("-" * 50)
    
    test_names = [
        "field_0.0",
        "field=0.5",
        "h_1.0",
        "h=2.5",
        "B_3.0",
        "B=4.5",
        "invalid_name"
    ]
    
    for name in test_names:
        field_val = extract_field_value(name)
        status = "✓" if field_val is not None else "✗"
        print(f"  {status} '{name}' -> {field_val}")


def check_requirements():
    """Check if required packages are installed."""
    print("Checking requirements...")
    print("-" * 50)
    
    required_packages = [
        ('numpy', 'np'),
        ('matplotlib', 'plt'),
        ('scipy', None),
        ('opt_einsum', None)
    ]
    
    all_ok = True
    for package, alias in required_packages:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            all_ok = False
    
    print()
    if all_ok:
        print("✓ All required packages are installed!")
    else:
        print("✗ Some packages are missing. Install them with:")
        print("  pip install numpy matplotlib scipy opt_einsum Pillow")
    
    return all_ok


if __name__ == "__main__":
    print("=" * 60)
    print("DSSF Animation Examples")
    print("=" * 60)
    print()
    
    # First check requirements
    if check_requirements():
        print()
        print("=" * 60)
        print("Usage Examples (with placeholder paths)")
        print("=" * 60)
        print()
        print("NOTE: Replace '/path/to/your/MD_pi_flux_data' with actual path")
        print()
        
        # Show examples (won't actually run without real data)
        # Uncomment the line below if you have real data to test with:
        # example_usage()
    else:
        print()
        print("Please install missing packages before running animations.")
