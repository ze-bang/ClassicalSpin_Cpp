#!/usr/bin/env python3
"""
Debug the energy formula by manually computing for a simple case
"""
import numpy as np
from single_q_BCAO import SingleQ
from multi_q_BCAO import MultiQ

# Test with parameters where alpha should be zero (pure spiral)
L = 4
J = [-7.6, -1.2, 0.1, -0.1, 0, 0, 2.5, -0.85]

print("Testing Single-Q with pure spiral (alpha = 0)...")
single_q = SingleQ(L, J)

print(f"\nSingle-Q Results:")
print(f"  Energy: {single_q.opt_energy:.8f}")
print(f"  Q: ({single_q.opt_params[0]:.6f}, {single_q.opt_params[1]:.6f})")
print(f"  Alpha A: {single_q.opt_params[2]:.6f}")
print(f"  Alpha B: {single_q.opt_params[3]:.6f}")

# Now manually test the energy formula
Q1, Q2, alphaA, alphaB = single_q.opt_params[0], single_q.opt_params[1], single_q.opt_params[2], single_q.opt_params[3]
phiA, thetaA, psiA = single_q.opt_params[4], single_q.opt_params[5], single_q.opt_params[6]
phiB, thetaB, psiB = single_q.opt_params[7], single_q.opt_params[8], single_q.opt_params[9]

Qvec = Q1 * single_q.b1 + Q2 * single_q.b2

exA, eyA, ezA = SingleQ.RotatedBasis(phiA, thetaA, psiA)
exB, eyB, ezB = SingleQ.RotatedBasis(phiB, thetaB, psiB)

print(f"\n  sqrt(1-alphaA^2) = {np.sqrt(1-alphaA**2):.6f}")
print(f"  sqrt(1-alphaB^2) = {np.sqrt(1-alphaB**2):.6f}")

# Compute energy components manually
HAB_Q = single_q.HAB(Qvec)
HAB_mQ = single_q.HAB(-Qvec)
HBA_Q = single_q.HBA(Qvec)
HBA_mQ = single_q.HBA(-Qvec)

spinA_plus = exA + 1j * eyA
spinA_minus = exA - 1j * eyA
spinB_plus = exB + 1j * eyB
spinB_minus = exB - 1j * eyB

amp_A = np.sqrt(1 - alphaA**2)
amp_B = np.sqrt(1 - alphaB**2)

# A-B interaction energy
E_AB = amp_A * amp_B / 4 * (
    spinA_minus.dot(HAB_Q).dot(spinB_plus) +
    spinB_minus.dot(HBA_Q).dot(spinA_plus) +
    spinA_plus.dot(HAB_mQ).dot(spinB_minus) +
    spinB_plus.dot(HBA_mQ).dot(spinA_minus)
)

print(f"\nManual calculation:")
print(f"  E_AB = {np.real(E_AB):.8f}")
print(f"  (This should match part of the total energy)")

# Now try Multi-Q with the SAME parameters
print(f"\n\nNow testing Multi-Q with num_Q=1 using SAME Q-vector...")

# Manually set up multi-Q parameters to match single-Q
multi_q = MultiQ(L, 1, J)

# Set parameters manually: [Q1, Q2, amp_A, phi_A, theta_A, psi_A, amp_B, phi_B, theta_B, psi_B]
manual_params = np.array([
    Q1, Q2,  # Q-vector
    amp_A, phiA, thetaA, psiA,  # A sublattice
    amp_B, phiB, thetaB, psiB   # B sublattice
])

print(f"Manual params: Q=({Q1:.6f}, {Q2:.6f}), amp_A={amp_A:.6f}, amp_B={amp_B:.6f}")

E_manual = multi_q.E_per_UC(manual_params)
print(f"Multi-Q energy with manual params: {E_manual:.8f}")
print(f"Single-Q energy: {single_q.opt_energy:.8f}")
print(f"Difference: {abs(E_manual - single_q.opt_energy):.8f}")

if abs(E_manual - single_q.opt_energy) < 0.01:
    print("\n✓ Energies MATCH when using same parameters!")
else:
    print("\n✗ Energies DON'T MATCH even with same parameters!")
    print("   This indicates a bug in the multi-Q energy formula.")
