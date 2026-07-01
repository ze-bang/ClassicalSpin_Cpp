#!/usr/bin/env python3
"""
cef_rotation_subspace_test.py
Run the Stevens operator CEF calculation and test whether R_z(π) and R_x(π)
stay within the 3-level subspace spanned by E1,E2,E3.
"""
import numpy as np
from functools import lru_cache

class StevensOperators:
    def __init__(self, J):
        self.J = J
        self.dim = int(2*J + 1)
        self.m_values = np.arange(-J, J+1)
        self.X = J*(J+1)
        self._Jz = None
        self._Jplus = None
        self._Jminus = None

    @property
    def Jz(self):
        if self._Jz is None:
            self._Jz = np.diag(self.m_values)
        return self._Jz

    @property
    def Jplus(self):
        if self._Jplus is None:
            self._Jplus = np.zeros((self.dim, self.dim), dtype=complex)
            for i in range(self.dim-1):
                m = self.m_values[i]
                self._Jplus[i+1, i] = np.sqrt(self.X - m*(m+1))
        return self._Jplus

    @property
    def Jminus(self):
        if self._Jminus is None:
            self._Jminus = self.Jplus.T.conj()
        return self._Jminus

    def _matrix_power(self, M, n):
        if n == 0: return np.eye(self.dim, dtype=complex)
        elif n == 1: return M
        else: return np.linalg.matrix_power(M, n)

    def _anticommutator(self, A, B):
        return A @ B + B @ A

    def get_operator(self, l, m):
        ops = {
            (2,0): self._O20, (2,2): self._O22, (2,-2): self._O2m2,
            (4,0): self._O40, (4,2): self._O42, (4,-2): self._O4m2,
            (4,4): self._O44, (4,-4): self._O4m4,
            (6,0): self._O60, (6,2): self._O62, (6,-2): self._O6m2,
            (6,4): self._O64, (6,-4): self._O6m4,
            (6,6): self._O66, (6,-6): self._O6m6,
        }
        return ops[(l, m)]()

    def _O20(self):
        return 3*self._matrix_power(self.Jz,2) - self.X*np.eye(self.dim)
    def _O22(self):
        return (self._matrix_power(self.Jplus,2) + self._matrix_power(self.Jminus,2))/2
    def _O2m2(self):
        return (self._matrix_power(self.Jplus,2) - self._matrix_power(self.Jminus,2))/(2j)
    def _O40(self):
        Jz2=self._matrix_power(self.Jz,2); Jz4=self._matrix_power(self.Jz,4)
        I=np.eye(self.dim)
        return 35*Jz4-(30*self.X-25)*Jz2+(3*self.X**2-6*self.X)*I
    def _O42(self):
        Jz2=self._matrix_power(self.Jz,2)
        Jp2=self._matrix_power(self.Jplus,2); Jm2=self._matrix_power(self.Jminus,2)
        return self._anticommutator(Jp2+Jm2, 7*Jz2-(self.X+5)*np.eye(self.dim))/4
    def _O4m2(self):
        Jz2=self._matrix_power(self.Jz,2)
        Jp2=self._matrix_power(self.Jplus,2); Jm2=self._matrix_power(self.Jminus,2)
        return self._anticommutator(Jp2-Jm2, 7*Jz2-(self.X+5)*np.eye(self.dim))/(4j)
    def _O44(self):
        return (self._matrix_power(self.Jplus,4)+self._matrix_power(self.Jminus,4))/2
    def _O4m4(self):
        return (self._matrix_power(self.Jplus,4)-self._matrix_power(self.Jminus,4))/(2j)
    def _O60(self):
        Jz2=self._matrix_power(self.Jz,2); Jz4=self._matrix_power(self.Jz,4); Jz6=self._matrix_power(self.Jz,6)
        I=np.eye(self.dim)
        return (231*Jz6-(315*self.X-735)*Jz4+(105*self.X**2-525*self.X+294)*Jz2+(-5*self.X**3+40*self.X**2-60*self.X)*I)
    def _O62(self):
        Jz2=self._matrix_power(self.Jz,2); Jz4=self._matrix_power(self.Jz,4)
        Jp2=self._matrix_power(self.Jplus,2); Jm2=self._matrix_power(self.Jminus,2)
        I=np.eye(self.dim)
        B=33*Jz4-(18*self.X+123)*Jz2+(self.X**2+10*self.X+102)*I
        return self._anticommutator(Jp2+Jm2, B)/4
    def _O6m2(self):
        Jz2=self._matrix_power(self.Jz,2); Jz4=self._matrix_power(self.Jz,4)
        Jp2=self._matrix_power(self.Jplus,2); Jm2=self._matrix_power(self.Jminus,2)
        I=np.eye(self.dim)
        B=33*Jz4-(18*self.X+123)*Jz2+(self.X**2+10*self.X+102)*I
        return self._anticommutator(Jp2-Jm2, B)/(4j)
    def _O64(self):
        Jz2=self._matrix_power(self.Jz,2)
        Jp4=self._matrix_power(self.Jplus,4); Jm4=self._matrix_power(self.Jminus,4)
        return self._anticommutator(Jp4+Jm4, 11*Jz2-(self.X+38)*np.eye(self.dim))/4
    def _O6m4(self):
        Jz2=self._matrix_power(self.Jz,2)
        Jp4=self._matrix_power(self.Jplus,4); Jm4=self._matrix_power(self.Jminus,4)
        return self._anticommutator(Jp4-Jm4, 11*Jz2-(self.X+38)*np.eye(self.dim))/(4j)
    def _O66(self):
        return (self._matrix_power(self.Jplus,6)+self._matrix_power(self.Jminus,6))/2
    def _O6m6(self):
        return (self._matrix_power(self.Jplus,6)-self._matrix_power(self.Jminus,6))/(2j)

    def build_hamiltonian(self, coefficients):
        H = np.zeros((self.dim, self.dim), dtype=complex)
        for (l, m), B in coefficients.items():
            if B != 0:
                H += B * self.get_operator(l, m)
        return H


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # --- Build CEF Hamiltonian ---
    stevens = StevensOperators(J=6)
    coefficients = {
        (2, 0): -5.29e-1,  (2, 2): -1.35e-1,  (2,-2):  12.79e-1,
        (4, 0): -0.13e-3,  (4, 2): -1.7e-3,    (4,-2):   3.29e-3,
        (4, 4): -1.22e-3,  (4,-4): -9.57e-3,
        (6, 0):  0.2e-5,   (6, 2): -1.1e-5,    (6,-2):  -0.9e-5,
        (6, 4):  6.1e-5,   (6,-4):  0.3e-5,
        (6, 6): -0.9e-5,   (6,-6):  0.0,
    }
    H = stevens.build_hamiltonian(coefficients)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    eigenvalues -= eigenvalues.min()

    print("All 13 eigenvalues (meV, shifted so E1=0):")
    print(eigenvalues)

    # Take the 3 lowest: E1, E2, E3
    subspace_basis = eigenvectors[:, :3]   # columns: |E1>, |E2>, |E3>  in the |J,m> basis

    gap_12 = eigenvalues[1] - eigenvalues[0]
    gap_13 = eigenvalues[2] - eigenvalues[0]
    gap_14 = eigenvalues[3] - eigenvalues[0]   # first excluded level
    print(f"\nGap E2-E1 = {gap_12:.4f} meV")
    print(f"Gap E3-E1 = {gap_13:.4f} meV")
    print(f"Gap E4-E1 = {gap_14:.4f} meV  (first level outside subspace)")

    # --- Full-space rotation operators ---
    m_values = stevens.m_values
    Jx = (stevens.Jplus + stevens.Jminus) / 2
    Jy = (stevens.Jplus - stevens.Jminus) / (2j)
    Jz = stevens.Jz

    # R_z(pi) = exp(-i pi J_z) — diagonal in |m> basis
    Rz_full = np.diag(np.exp(-1j * np.pi * m_values))

    # R_x(pi) = exp(-i pi J_x) — use spectral decomposition of J_x
    evals_x, evecs_x = np.linalg.eigh(Jx)
    Rx_full = evecs_x @ np.diag(np.exp(-1j * np.pi * evals_x)) @ evecs_x.conj().T

    # --- Project into the 3-level subspace ---
    # P_sub = subspace_basis @ subspace_basis†  is the projector
    P_sub = subspace_basis @ subspace_basis.conj().T

    # The projected (3×3) matrix is V†RV where V = subspace_basis
    Rz_proj = subspace_basis.conj().T @ Rz_full @ subspace_basis
    Rx_proj = subspace_basis.conj().T @ Rx_full @ subspace_basis

    print("\n" + "="*65)
    print("SECTION A: Does R_z(π) stay within the 3-level subspace?")
    print("="*65)

    # The key test: how much of R|E_i> leaks into the 4th+ levels?
    # If R maps subspace → subspace, then P_sub R P_sub ψ = R P_sub ψ for all ψ in subspace.
    # Equivalently, the 3×3 projected matrix should be unitary.

    # Unitarity of the projected matrix (measures leakage)
    I3 = np.eye(3)
    err_Rz = np.linalg.norm(Rz_proj.conj().T @ Rz_proj - I3, ord='fro')
    err_Rx = np.linalg.norm(Rx_proj.conj().T @ Rx_proj - I3, ord='fro')

    print(f"\n  Unitarity error of proj Rz(π) [||U†U - I||_F]: {err_Rz:.6e}")
    print(f"  Unitarity error of proj Rx(π) [||U†U - I||_F]: {err_Rx:.6e}")
    print()
    print("  Interpretation:")
    print("    0.000 → rotation stays perfectly within subspace")
    print("    O(1)  → significant leakage into higher levels")
    print()

    # Quantify leakage per basis state
    print("  Leakage per basis state (||Q R |E_i>||^2, Q = complement projector):")
    Q_sub = np.eye(stevens.dim) - P_sub
    for i, label in enumerate(["E1", "E2", "E3"]):
        for R, Rname in [(Rz_full, "Rz(π)"), (Rx_full, "Rx(π)")]:
            rotated = R @ subspace_basis[:, i]
            leakage = np.real(rotated.conj() @ Q_sub @ rotated)
            print(f"    {Rname} |{label}>: leakage = {leakage:.6e}")

    # --- Projected matrix structure ---
    print("\n" + "="*65)
    print("SECTION B: Structure of projections Rz(π) and Rx(π) in the subspace")
    print("="*65)
    print("\n  Rz_proj (3×3):")
    print(Rz_proj)
    print("\n  Rx_proj (3×3):")
    print(Rx_proj)

    # --- Compare to the P_z = diag(+1,+1,-1) claim ---
    print("\n" + "="*65)
    print("SECTION C: Is Rz_proj proportional to P_z = diag(+1,+1,-1)?")
    print("="*65)
    # P_z in the (E1,E2,E3) ordering: diag(+1,+1,-1)
    Pz_expected = np.diag([1., 1., -1.])
    # But the actual eigenstates may have a global phase each, so compare |entries|
    print("\n  |Rz_proj|  (absolute values):")
    print(np.abs(Rz_proj))
    print("\n  Rz_proj diagonal (should be ±1 if it stays in subspace):")
    print(np.diag(Rz_proj))
    print("\n  Rz_proj off-diagonal norms:")
    off_diag_Rz = np.abs(Rz_proj.copy())
    np.fill_diagonal(off_diag_Rz, 0)
    print(f"    max |off-diag| = {off_diag_Rz.max():.6e}")

    print("\n  |Rx_proj|  (absolute values):")
    print(np.abs(Rx_proj))
    print("\n  Rx_proj diagonal:")
    print(np.diag(Rx_proj))
    off_diag_Rx = np.abs(Rx_proj.copy())
    np.fill_diagonal(off_diag_Rx, 0)
    print(f"    max |off-diag| = {off_diag_Rx.max():.6e}")

    # --- Check parity of each CEF eigenstate under Rz ---
    print("\n" + "="*65)
    print("SECTION D: Parity of each CEF level under Rz(π) and Rx(π)")
    print("="*65)
    print()
    for i, label in enumerate(["E1", "E2", "E3"]):
        psi = subspace_basis[:, i]
        rz_psi = Rz_full @ psi
        rx_psi = Rx_full @ psi

        # Project back onto the subspace
        rz_in  = subspace_basis.conj().T @ rz_psi   # 3-vector
        rx_in  = subspace_basis.conj().T @ rx_psi

        # Leakage
        rz_leak = 1.0 - np.real(np.dot(rz_in.conj(), rz_in))
        rx_leak = 1.0 - np.real(np.dot(rx_in.conj(), rx_in))

        # Overlap with each subspace state
        rz_overlaps = [f"{rz_in[j]:.4f}" for j in range(3)]
        rx_overlaps = [f"{rx_in[j]:.4f}" for j in range(3)]

        print(f"  |{label}> → Rz(π)|{label}>:")
        print(f"    overlap with (E1,E2,E3): {rz_overlaps}")
        print(f"    leakage to E4+: {rz_leak:.4e}")
        print(f"  |{label}> → Rx(π)|{label}>:")
        print(f"    overlap with (E1,E2,E3): {rx_overlaps}")
        print(f"    leakage to E4+: {rx_leak:.4e}")
        print()

    # --- Verdict ---
    print("="*65)
    print("VERDICT")
    print("="*65)
    THRESHOLD = 1e-3
    Rz_ok = err_Rz < THRESHOLD
    Rx_ok = err_Rx < THRESHOLD
    print(f"\n  Rz(π) unitarity error = {err_Rz:.4e}  → {'stays in subspace' if Rz_ok else 'LEAKS out of subspace'}")
    print(f"  Rx(π) unitarity error = {err_Rx:.4e}  → {'stays in subspace' if Rx_ok else 'LEAKS out of subspace'}")
    print()
    if not (Rz_ok and Rx_ok):
        print("  CONCLUSION: The original claim is CORRECT.")
        print("  R_z(π) and/or R_x(π) do NOT map the 3-level subspace")
        print("  onto itself — they mix E1,E2,E3 with higher CEF levels.")
        print("  The local frame construction in the code absorbs this mixing")
        print("  at the COUPLING level, but the rotation itself is not closed.")
    else:
        print("  CONCLUSION: The rebuttal holds.")
        print("  Both rotations stay within the 3-level subspace to machine precision.")
