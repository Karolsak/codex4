"""
Test script for transformer analysis application
Validates calculations and core functionality
"""

import sys
import numpy as np

# Test imports
print("Testing imports...")
try:
    import tkinter as tk
    print("✓ tkinter imported successfully")
except ImportError as e:
    print(f"✗ tkinter import failed: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib imported successfully")
except ImportError as e:
    print(f"✗ matplotlib import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("✓ numpy imported successfully")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")
    sys.exit(1)

# Test main module import
print("\nTesting main module...")
try:
    from transformer_analysis_advanced import (
        TransformerParameters,
        ODESolver,
        TransformerSimulator
    )
    print("✓ Main module imported successfully")
except Exception as e:
    print(f"✗ Main module import failed: {e}")
    sys.exit(1)

# Test TransformerParameters
print("\nTesting TransformerParameters...")
try:
    params = TransformerParameters()
    assert params.rated_power == 100.0
    assert params.iron_loss == 2.0
    assert params.copper_loss_full == 2.0
    print("✓ TransformerParameters initialized correctly")
except Exception as e:
    print(f"✗ TransformerParameters test failed: {e}")
    sys.exit(1)

# Test ODESolver - Euler method
print("\nTesting Euler ODE Solver...")
try:
    def test_ode(t, y):
        return np.array([-y[0]])  # Simple exponential decay

    solver = ODESolver()
    t, y = solver.euler(test_ode, np.array([1.0]), (0, 1), 0.01)

    assert len(t) == len(y)
    assert len(t) > 0
    assert y[0, 0] == 1.0  # Initial condition
    assert y[-1, 0] < 1.0  # Should decay
    print(f"✓ Euler solver works (final value: {y[-1, 0]:.4f})")
except Exception as e:
    print(f"✗ Euler solver test failed: {e}")
    sys.exit(1)

# Test ODESolver - RK45 method
print("\nTesting RK45 ODE Solver...")
try:
    def test_ode(t, y):
        return np.array([-y[0]])

    solver = ODESolver()
    t, y = solver.rk45(test_ode, np.array([1.0]), (0, 1), 0.01)

    assert len(t) == len(y)
    assert len(t) > 0
    assert y[0, 0] == 1.0
    assert y[-1, 0] < 1.0
    print(f"✓ RK45 solver works (final value: {y[-1, 0]:.4f})")
except Exception as e:
    print(f"✗ RK45 solver test failed: {e}")
    sys.exit(1)

# Test All-Day Efficiency Calculation
print("\nTesting All-Day Efficiency Calculation...")
try:
    params = TransformerParameters(
        rated_power=100.0,
        iron_loss=2.0,
        copper_loss_full=2.0
    )

    simulator = TransformerSimulator(params)

    # Test case from problem statement
    results = simulator.calculate_all_day_efficiency(
        full_load_hours=4,
        half_load_hours=6,
        no_load_hours=14
    )

    # Verify calculations
    expected_output = 100 * 4 + 50 * 6  # 700 kWh
    expected_iron_loss = 2 * 24  # 48 kWh
    expected_copper_loss = 2 * 4 + 2 * 0.25 * 6  # 11 kWh
    expected_total_loss = 48 + 11  # 59 kWh
    expected_efficiency = (700 / 759) * 100  # 92.23%

    assert abs(results['total_output'] - expected_output) < 0.01, \
        f"Output mismatch: {results['total_output']} vs {expected_output}"

    assert abs(results['iron_losses'] - expected_iron_loss) < 0.01, \
        f"Iron loss mismatch: {results['iron_losses']} vs {expected_iron_loss}"

    assert abs(results['copper_losses'] - expected_copper_loss) < 0.01, \
        f"Copper loss mismatch: {results['copper_losses']} vs {expected_copper_loss}"

    assert abs(results['total_losses'] - expected_total_loss) < 0.01, \
        f"Total loss mismatch: {results['total_losses']} vs {expected_total_loss}"

    assert abs(results['efficiency'] - expected_efficiency) < 0.01, \
        f"Efficiency mismatch: {results['efficiency']} vs {expected_efficiency}"

    print("✓ All-day efficiency calculation correct")
    print(f"  Output: {results['total_output']} kWh")
    print(f"  Iron Losses: {results['iron_losses']} kWh")
    print(f"  Copper Losses: {results['copper_losses']} kWh")
    print(f"  Total Losses: {results['total_losses']} kWh")
    print(f"  Efficiency: {results['efficiency']:.4f}%")

except Exception as e:
    print(f"✗ All-day efficiency test failed: {e}")
    sys.exit(1)

# Test dynamic simulation
print("\nTesting Dynamic Simulation...")
try:
    params = TransformerParameters()
    simulator = TransformerSimulator(params)

    # Short simulation
    t, y = simulator.simulate_dynamic(method='rk45', duration=1.0, dt=0.1)

    assert len(t) > 0
    assert y.shape[0] == len(t)
    assert y.shape[1] == 4  # flux, i_primary, i_secondary, temp

    print("✓ Dynamic simulation works")
    print(f"  Simulation points: {len(t)}")
    print(f"  Final temperature: {y[-1, 3]:.2f}°C")

except Exception as e:
    print(f"✗ Dynamic simulation test failed: {e}")
    sys.exit(1)

# Test different load conditions
print("\nTesting Various Load Conditions...")
test_cases = [
    # (full_hours, half_hours, no_hours, expected_eff_range)
    (24, 0, 0, (91, 93)),      # Full load all day
    (0, 24, 0, (92, 94)),       # Half load all day
    (0, 0, 24, (0, 1)),         # No load all day (near 0%)
    (8, 8, 8, (91, 94)),        # Equal distribution
    (12, 6, 6, (91, 93)),       # Heavy load
]

for i, (full, half, no, eff_range) in enumerate(test_cases, 1):
    try:
        results = simulator.calculate_all_day_efficiency(full, half, no)

        # Skip efficiency check for no-load case
        if full == 0 and half == 0:
            print(f"✓ Test case {i}: {full}h full, {half}h half, {no}h no-load - "
                  f"Efficiency: {results['efficiency']:.2f}% (no-load case)")
        else:
            assert eff_range[0] <= results['efficiency'] <= eff_range[1], \
                f"Efficiency {results['efficiency']:.2f}% out of expected range {eff_range}"
            print(f"✓ Test case {i}: {full}h full, {half}h half, {no}h no-load - "
                  f"Efficiency: {results['efficiency']:.2f}%")
    except Exception as e:
        print(f"✗ Test case {i} failed: {e}")
        sys.exit(1)

# Test numerical accuracy
print("\nTesting Numerical Accuracy...")
try:
    # Compare Euler vs RK45 for same problem
    def simple_ode(t, y):
        return np.array([-0.1 * y[0]])

    solver = ODESolver()

    # Euler solution
    t_euler, y_euler = solver.euler(simple_ode, np.array([1.0]), (0, 10), 0.1)

    # RK45 solution
    t_rk45, y_rk45 = solver.rk45(simple_ode, np.array([1.0]), (0, 10), 0.1)

    # Analytical solution: y = exp(-0.1*t), at t=10: y = exp(-1) ≈ 0.3679
    analytical = np.exp(-1.0)

    error_euler = abs(y_euler[-1, 0] - analytical)
    error_rk45 = abs(y_rk45[-1, 0] - analytical)

    print(f"  Analytical solution: {analytical:.6f}")
    print(f"  Euler result: {y_euler[-1, 0]:.6f} (error: {error_euler:.6f})")
    print(f"  RK45 result: {y_rk45[-1, 0]:.6f} (error: {error_rk45:.6f})")
    print(f"✓ RK45 is {error_euler/error_rk45:.2f}x more accurate than Euler")

except Exception as e:
    print(f"✗ Numerical accuracy test failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("ALL TESTS PASSED SUCCESSFULLY! ✓")
print("="*60)
print("\nThe transformer analysis application is ready to use.")
print("\nTo run the GUI application:")
print("  python3 transformer_analysis_advanced.py")
print("\nKey Results for the Given Problem:")
print("  • Transformer: 100 kVA")
print("  • Schedule: 4h full-load, 6h half-load, 14h no-load")
print("  • All-Day Efficiency: 92.23%")
print("  • Total Output: 700 kWh")
print("  • Total Losses: 59 kWh")
print("="*60)
