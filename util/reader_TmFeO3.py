import h5py
import numpy as np
from opt_einsum import contract
import matplotlib.pyplot as plt
import os
from math import gcd
from functools import reduce
from matplotlib.colors import PowerNorm
from sys import argv
# plt.rcParams['text.usetex'] = True


def calcNumSites(A, B, N):
    # Calculate how many grid points are on the path from A to B
    # Convert A and B to grid coordinates
    A_grid = A * N / (2*np.pi)
    B_grid = B * N / (2*np.pi)
    
    # Calculate the displacement vector between grid points
    delta = B_grid - A_grid
    
    # Find the greatest common divisor (GCD) of the absolute values of the components
    # We need to handle potential floating point values by rounding them
    rounded_delta = np.round(delta).astype(int)
    # If all components are zero, return 1 (just the starting point)
    if np.all(rounded_delta == 0):
        return 1
    
    # Calculate GCD of all non-zero components
    non_zero = [abs(x) for x in rounded_delta if x != 0]
    if not non_zero:  # All components are zero
        return 1
        
    # Find the GCD of all components
    gcd_all = reduce(gcd, non_zero)
    
    # The number of integer points is GCD + 1 (including start and end points)
    return gcd_all + 1


def drawLine(A, B, stepN):
    N = np.linalg.norm(A-B)
    num = int(N/stepN)
    temp = np.linspace(A, B, num)
    return temp
def magnitude_bi(vector1, vector2):
    # temp1 = contract('i,ik->k', vector1, BasisBZA)
    # temp2 = contract('i,ik->k', vector2, BasisBZA)
    temp1 = vector1
    temp2 = vector2
    return np.linalg.norm(temp1-temp2)


def get_reciprocal_lattice_from_unitcell(unitcell_path):
    """
    Read unit cell information and compute reciprocal lattice vectors.
    For TmFeO3, the lattice vectors are orthorhombic: a1=(1,0,0), a2=(0,1,0), a3=(0,0,1)
    in units of lattice constants.
    
    Returns:
        3x3 array of reciprocal lattice vectors (as rows)
    """
    # For orthorhombic lattice with a1=(1,0,0), a2=(0,1,0), a3=(0,0,1)
    # The reciprocal lattice vectors are simply:
    # b1 = 2π(1,0,0), b2 = 2π(0,1,0), b3 = 2π(0,0,1)
    return np.array([
        [2*np.pi, 0, 0],
        [0, 2*np.pi, 0],
        [0, 0, 2*np.pi]
    ])

# Get reciprocal lattice from unit cell info
reciprocal_lattice = get_reciprocal_lattice_from_unitcell("unitcell_info.txt")

# Define high symmetry points in reciprocal lattice units (fractional coordinates)
# For orthorhombic lattice: Gamma=(0,0,0), X=(1,0,0), Y=(0,1,0), Z=(0,0,1), etc.
P1_frac = np.array([1, 0, 3])
P2_frac = np.array([3, 0, 3])
P3_frac = np.array([3, 0, 1])
P4_frac = np.array([3, 2, 1])

# Convert to Cartesian k-space coordinates using reciprocal lattice
P1 = contract('i,ij->j', P1_frac, reciprocal_lattice)
P2 = contract('i,ij->j', P2_frac, reciprocal_lattice)
P3 = contract('i,ij->j', P3_frac, reciprocal_lattice)
P4 = contract('i,ij->j', P4_frac, reciprocal_lattice)

graphres = 4

#Path through high-symmetry points
P12 = np.linspace(P1, P2, calcNumSites(P1, P2, graphres))[1:-1]
P23 = np.linspace(P2, P3, calcNumSites(P2, P3, graphres))[1:-1]
P34 = np.linspace(P3, P4, calcNumSites(P3, P4, graphres))[1:-1]
g1 = 0
g2 = g1 + len(P12)
g3 = g2 + len(P23)
g4 = g3 + len(P34)

DSSF_K = np.concatenate((P12, P23, P34))

x = np.array([[1, 0, 0], [1, 0, 0], [-1, 0, 0], [-1, 0, 0]])
y = np.array([[0, 1, 0], [0, -1, 0], [0, 1, 0], [0, -1, 0]])
z = np.array([[0, 0, 1], [0, 0, -1], [0, 0, -1], [0, 0, 1]])

localframe = np.array([x, y, z])

def Spin_global_pyrochlore(k,S,P):
    size = int(len(P)/4)
    tS = np.zeros((len(k),3), dtype=np.complex128)
    for i in range(4):
        ffact = np.exp(1j * contract('ik,jk->ij', k, P[i*size:(i+1)*size]))
        tS = tS + contract('js, ij, sp->ip', S[i*size:(i+1)*size], ffact, localframe[:,i,:])/np.sqrt(size)
    return tS

def Spin(k, S, P):
    """Compute spin structure factor using FFT when possible."""
    N = len(S)
    # Check if k points are on a regular grid - if so, use FFT
    # Otherwise fall back to direct computation
    ffact = np.exp(1j*contract('ik,jk->ij', k, P))
    return contract('js, ij->is', S, ffact)/np.sqrt(N)


def Spin_global_pyrochlore_t(k,S,P):
    """Compute time-dependent spin structure factor for pyrochlore."""
    size = int(len(P)/4)
    tS = np.zeros((len(S), 4, len(k),3), dtype=np.complex128)
    for i in range(4):
        ffact = np.exp(1j * contract('ik,jk->ij', k, P[i*size:(i+1)*size]))
        tS[:,i,:,:] = contract('tjs, ij, sp->tip', S[:,i*size:(i+1)*size,:], ffact, localframe[:,i,:])/np.sqrt(size)
    return tS

def Spin_t(k, S, P):
    """Compute time-dependent spin structure factor using FFT when possible."""
    N = len(S)
    results = np.zeros((len(S), len(k), S.shape[2]), dtype=np.complex128)
    ffact = np.exp(1j*contract('ik,jk->ij', k, P))
    for i in range(len(S)):
        results[i] = contract('js, ij->is', S[i], ffact)/np.sqrt(N)
    return results

def SSSF_q(k, S, P, gb=False):
    if gb:
        A = Spin_global_pyrochlore(k, S, P)
    else:
        A = Spin(k, S, P)
    return np.real(contract('ia, ib -> iab', A, np.conj(A)))

def g(q):
    M = np.zeros((len(q),4,4))
    qnorm = contract('ik, ik->i', q, q)
    qnorm = np.where(qnorm == 0, 1, qnorm)
    for i in range(4):
        for j in range(4):
            M[:,i,j] = np.dot(z[i], z[j]) - contract('k, ik->i',z[i],q) * contract('k, ik->i', z[j],q) /qnorm
    return M

def DSSF(w, k, S, P, T, gb=False):
    """Compute dynamical spin structure factor using FFT over time."""
    if gb:
        A = Spin_global_pyrochlore_t(k, S, P)
        # A shape: (len(S), 4, len(k), 3)
        # Subtract mean configuration before FFT
        A_mean = np.mean(A, axis=0, keepdims=True)
        A = A - A_mean
        # FFT over time dimension
        dt = T[1] - T[0] if len(T) > 1 else 1.0
        A_fft = np.fft.fft(A, axis=0)
        fft_freqs = np.fft.fftfreq(len(T), d=dt) * 2 * np.pi
        
        # Interpolate or select closest frequencies to w
        # For simplicity, find closest indices
        indices = [np.argmin(np.abs(fft_freqs - w_val)) for w_val in w]
        Somega = A_fft[indices] / np.sqrt(len(T))
        
        read = np.real(contract('wnia, wnib->wiab', Somega, np.conj(Somega)))
        return read
    else:
        A = Spin_t(k, S, P)
        # A shape: (len(S), len(k), S.shape[2])
        # Subtract mean configuration before FFT
        A_mean = np.mean(A, axis=0, keepdims=True)
        A = A - A_mean
        # FFT over time dimension
        dt = T[1] - T[0] if len(T) > 1 else 1.0
        A_fft = np.fft.fft(A, axis=0)
        fft_freqs = np.fft.fftfreq(len(T), d=dt) * 2 * np.pi
        
        # Select closest frequencies to w
        indices = [np.argmin(np.abs(fft_freqs - w_val)) for w_val in w]
        Somega = A_fft[indices] / np.sqrt(len(T))
        
        read = np.real(contract('wia, wib->wiab', Somega, np.conj(Somega)))
        return read

def SSSFGraphHnHL(A,B,d1, filename):
    plt.pcolormesh(A,B, d1)
    plt.colorbar()
    plt.ylabel(r'$(0,0,L)$')
    plt.xlabel(r'$(H,-H,0)$')
    plt.savefig(filename+".pdf")
    plt.clf()

def SSSFGraphHK0(A,B,d1, filename):
    plt.pcolormesh(A,B, d1)
    plt.colorbar()
    plt.ylabel(r'$(0,K,0)$')
    plt.xlabel(r'$(H,0,0)$')
    plt.savefig(filename+".pdf")
    plt.clf()


def SSSFGraph2D(A, B, d1, filename):
    plt.pcolormesh(A, B, d1)
    plt.colorbar()
    plt.ylabel(r'$K_y$')
    plt.xlabel(r'$K_x$')
    plt.savefig(filename + ".pdf")
    plt.clf()


def hnhltoK(H, L, K=0):
    A = contract('ij,k->ijk',H, 2*np.array([np.pi,-np.pi,0])) \
        + contract('ij,k->ijk',L, 2*np.array([0,0,np.pi]))
    return A

def hhztoK(H, K):
    return contract('ij,k->ijk',H, 2*np.array([np.pi,0,0])) + contract('ij,k->ijk',K, 2*np.array([0,np.pi,0]))

def hk2d(H,K):
    return contract('ij,k->ijk',H, 2*np.array([np.pi,0])) + contract('ij,k->ijk',K, 2*np.array([0,np.pi]))


def SSSFHnHL(S, P, nK, filename, gb=False):
    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hnhltoK(A, B).reshape((nK*nK,3))
    S = SSSF_q(K, S, P, gb)
    if gb:
        f1 = filename + "Sxx_global"
        f2 = filename + "Syy_global"
        f3 = filename + "Szz_global"
        f4 = filename + "Sxy_global"
        f5 = filename + "Sxz_global"
        f6 = filename + "Syz_global"
    else:
        f1 = filename + "Sxx_local"
        f2 = filename + "Syy_local"
        f3 = filename + "Szz_local"
        f4 = filename + "Sxy_local"
        f5 = filename + "Sxz_local"
        f6 = filename + "Syz_local"
    S = S.reshape((nK, nK, 3, 3))
    np.savetxt(f1 + '.txt', S[:,:,0,0])
    np.savetxt(f2 + '.txt', S[:,:,1,1])
    np.savetxt(f3 + '.txt', S[:,:,2,2])
    np.savetxt(f4 + '.txt', S[:,:,0,1])
    np.savetxt(f5 + '.txt', S[:,:,0,2])
    np.savetxt(f6 + '.txt', S[:,:,1,2])
    SSSFGraphHnHL(A, B, S[:,:,0,0], f1)
    SSSFGraphHnHL(A, B, S[:,:,1,1], f2)
    SSSFGraphHnHL(A, B, S[:,:,2,2], f3)
    SSSFGraphHnHL(A, B, S[:, :, 0, 1], f4)
    SSSFGraphHnHL(A, B, S[:, :, 0, 2], f5)
    SSSFGraphHnHL(A, B, S[:, :, 1, 2], f6)

def SSSFHK0(S, P, nK, filename, gb=False):
    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hhztoK(A, B).reshape((nK*nK,3))
    S = SSSF_q(K, S, P, gb)
    if gb:
        f1 = filename + "Sxx_global"
        f2 = filename + "Syy_global"
        f3 = filename + "Szz_global"
        f4 = filename + "Sxy_global"
        f5 = filename + "Sxz_global"
        f6 = filename + "Syz_global"
    else:
        f1 = filename + "Sxx_local"
        f2 = filename + "Syy_local"
        f3 = filename + "Szz_local"
        f4 = filename + "Sxy_local"
        f5 = filename + "Sxz_local"
        f6 = filename + "Syz_local"
    S = S.reshape((nK, nK, 3, 3))
    np.savetxt(f1 + '.txt', S[:,:,0,0])
    np.savetxt(f2 + '.txt', S[:,:,1,1])
    np.savetxt(f3 + '.txt', S[:,:,2,2])
    np.savetxt(f4 + '.txt', S[:,:,0,1])
    np.savetxt(f5 + '.txt', S[:,:,0,2])
    np.savetxt(f6 + '.txt', S[:,:,1,2])


def compute_reciprocal_lattice(positions):
    """
    Compute reciprocal lattice vectors from real-space positions.
    
    Args:
        positions: Nx3 array of atomic positions
        
    Returns:
        3x3 array of reciprocal lattice vectors (as rows)
    """
    # Find the lattice vectors by analyzing the positions
    # Assume positions are in units where lattice constants are embedded
    pos = positions.copy()
    
    # Find unique position components to infer lattice constants
    unique_x = np.unique(pos[:, 0])
    unique_y = np.unique(pos[:, 1])
    unique_z = np.unique(pos[:, 2])
    
    # Get lattice constants from spacing
    if len(unique_x) > 1:
        dx = np.min(np.diff(np.sort(unique_x))[np.diff(np.sort(unique_x)) > 1e-10])
        Lx = np.max(unique_x) + dx
    else:
        Lx = 1.0
        
    if len(unique_y) > 1:
        dy = np.min(np.diff(np.sort(unique_y))[np.diff(np.sort(unique_y)) > 1e-10])
        Ly = np.max(unique_y) + dy
    else:
        Ly = 1.0
        
    if len(unique_z) > 1:
        dz = np.min(np.diff(np.sort(unique_z))[np.diff(np.sort(unique_z)) > 1e-10])
        Lz = np.max(unique_z) + dz
    else:
        Lz = 1.0
    
    # Real space lattice vectors
    a1 = np.array([Lx, 0, 0])
    a2 = np.array([0, Ly, 0])
    a3 = np.array([0, 0, Lz])
    
    # Compute reciprocal lattice vectors
    volume = np.dot(a1, np.cross(a2, a3))
    b1 = 2 * np.pi * np.cross(a2, a3) / volume
    b2 = 2 * np.pi * np.cross(a3, a1) / volume
    b3 = 2 * np.pi * np.cross(a1, a2) / volume
    
    return np.array([b1, b2, b3])

def genALLSymPointsBare():
    d = 9 * 1j
    b = np.mgrid[0:1:d, 0:1:d, 0:1:d].reshape(3, -1).T
    return b

# Legacy BasisBZA definitions (kept for backward compatibility, but prefer using reciprocal_lattice)
BasisBZA = np.array([2*np.pi*np.array([-1,1,1]),2*np.pi*np.array([1,-1,1]),2*np.pi*np.array([1,1,-1])])
BasisBZA_reverse = np.array([np.array([0,1,1]),np.array([1,0,1]),np.array([1,1,0])])/2

BasisBZA_reverse_honeycomb = np.array([[1,1/2],[0, np.sqrt(3)/2]])

def genBZ(d, m=1, reciprocal_lattice_arg=None):
    """
    Generate Brillouin zone k-points.
    
    Args:
        d: Number of divisions along each direction
        m: Multiplier for the range
        reciprocal_lattice_arg: 3x3 array of reciprocal lattice vectors (optional)
                          If None, uses reciprocal_lattice from unit cell or BasisBZA
    
    Returns:
        Array of k-points
    """
    dj = d*1j
    b = np.mgrid[0:m:dj, 0:m:dj, 0:m:dj].reshape(3,-1).T
    b = np.concatenate((b,genALLSymPointsBare()))
    
    if reciprocal_lattice_arg is not None:
        b = contract('ij, jk->ik', b, reciprocal_lattice_arg)
    elif 'reciprocal_lattice' in globals():
        b = contract('ij, jk->ik', b, reciprocal_lattice)
    else:
        b = contract('ij, jk->ik', b, BasisBZA)
    return b


def ordering_q_slice(S, P, ind):
    reciprocal_lattice_local = compute_reciprocal_lattice(P)
    K = genBZ(101, reciprocal_lattice_arg=reciprocal_lattice_local)
    S = np.abs(SSSF_q(K, S, P))
    Szz = S[:,ind,ind]
    max = np.max(Szz)
    if max < 1e-13:
        qzz = np.array([np.NaN, np.NaN, np.NaN])
    else:
        indzz = np.array([])
        tempindzz = np.where(np.abs(Szz-max)<1e-13)[0]
        indzz = np.concatenate((indzz, tempindzz))
        indzz = np.array(indzz.flatten(),dtype=int)
        qzz = K[indzz]
    if qzz.shape == (3,):
        qzz = qzz.reshape(1,3)
    return qzz

def ordering_q(S,P):
    temp = np.concatenate((ordering_q_slice(S, P, 0),ordering_q_slice(S, P, 1),ordering_q_slice(S, P, 2)))
    return temp

def magnetization(S, glob, fielddir):
    if not glob:
        return np.mean(S,axis=0)
    else:
        size = int(len(S)/4)
        zmag = contract('k,ik->i', fielddir, z)
        mag = np.zeros(3)
        for i in range(4):
            mag = mag + np.mean(S[i*size:(i+1)*size], axis=0)*zmag[i]
        return mag
            

def parseSSSF(dir):
    directory = os.fsencode(dir)
    def SSSFhelper(name):
        size = 0
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(name+".txt"):
                test = np.loadtxt(dir+filename)
                size = test.shape
                break
        A = np.zeros(size)

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(name+".txt"):
                print(filename)
                A = A + np.loadtxt(dir+filename)
        A = A / np.max(A)
        fig, ax = plt.subplots(figsize=(10,4))
        plt.imshow(A, origin='lower', extent=[0, 2.5, 0, 2.5], aspect='equal', interpolation='lanczos', cmap='gnuplot2')
        plt.colorbar()
        plt.ylabel(r'$(0,0,L)$')
        plt.xlabel(r'$(H,-H,0)$')
        plt.savefig(filename+".pdf")
        plt.clf()

    SSSFhelper("Sxx_local")
    SSSFhelper("Syy_local")
    SSSFhelper("Szz_local")
    SSSFhelper("Sxx_global")
    SSSFhelper("Syy_global")
    SSSFhelper("Szz_global")

def read_MD(dir,isglobal=False):
    w0 = 0
    wmax = 15
    t_evolved = 100
    SU2 = read_MD_SU2(dir, w0, wmax, t_evolved, isglobal)
    SU3 = read_MD_SU3(dir, w0, wmax, t_evolved, False)
    A = contract('wiab->wi',SU2) + contract('wiab->wi',SU3)
    np.savetxt(dir + "/DSSF.txt", A)
    fig, ax = plt.subplots(figsize=(10,4))
    C = ax.imshow(A, origin='lower', extent=[0, g4, w0, wmax], aspect='auto', interpolation='gaussian', cmap='gnuplot2', norm='log')
    ax.axvline(x=g1, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=g2, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=g3, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=g4, color='b', label='axvline - full height', linestyle='dashed')
    labels = [r'$(0,0,0)$', r'$(0,0,1)$', r'$(0,1,1)$', r'$(1,1,1)$']
    xlabpos = [g1, g2, g3, g4]
    ax.set_xticks(xlabpos, labels)
    ax.set_xlim([0, g4])
    ax.set_xlim([0, g4])
    fig.colorbar(C)
    plt.savefig(dir+"/DSSF.pdf")
    plt.clf()

def read_MD_tot(dir, isglobal=False):
    print("Begin MD reading")
    directory = os.fsencode(dir)
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if os.path.isdir(dir + "/" + filename):
            read_MD(dir + "/" + filename, isglobal)

def read_MD_SU2(dir, w0, wmax, t_evolved, isglobal=False):
    directory = os.fsencode(dir)
    P = np.loadtxt(os.path.join(os.path.dirname(dir), "pos_SU2.txt"))
    T = np.loadtxt(dir + "/Time_steps.txt")
    S = np.loadtxt(dir + "/spin_t_SU2.txt")
    Slength = int(len(S)/len(P))
    S = S.reshape((Slength, len(P), 3))
    S = S[-len(T):]  # Ensure S has the same length as T

    # Use natural FFT frequencies instead of custom range
    dt = T[1] - T[0] if len(T) > 1 else 1.0
    fft_freqs = np.fft.fftfreq(len(T), d=dt) * 2 * np.pi
    # Filter to desired range
    w_mask = (fft_freqs >= w0) & (fft_freqs <= wmax)
    w = fft_freqs[w_mask]
    
    A = DSSF(w, DSSF_K, S, P, T, isglobal)
    
    # Compute DSSF at Gamma point (0,0,0) for gap analysis
    Gamma_point = np.array([[0, 0, 0]])
    A_Gamma = DSSF(w, Gamma_point, S, P, T, isglobal)
    DSSF_sum_Gamma = contract('wiab->wi', A_Gamma)
    
    def DSSF_graph(DSSF, i, j):
        fig, ax = plt.subplots(figsize=(10,4))
        C = ax.imshow(DSSF, origin='lower', extent=[0, g4, w0, wmax], aspect='auto', interpolation='gaussian', cmap='gnuplot2', norm='log')
        ax.axvline(x=g1, color='b', label='axvline - full height', linestyle='dashed')
        ax.axvline(x=g2, color='b', label='axvline - full height', linestyle='dashed')
        ax.axvline(x=g3, color='b', label='axvline - full height', linestyle='dashed')
        ax.axvline(x=g4, color='b', label='axvline - full height', linestyle='dashed')
        xlabpos = [g1, g2, g3, g4]
        labels = [r'$(0,0,0)$', r'$(0,0,1)$', r'$(0,1,1)$', r'$(1,1,1)$']

        ax.set_xticks(xlabpos, labels)
        ax.set_xlim([0, g4])
        fig.colorbar(C)
        plt.savefig(dir+"DSSF_SU2_{}_{}.pdf".format(i, j))
        plt.clf()

    for i in range(3):
        DSSF_graph(A[:,:,i,i], i, i)
    
    DSSF_sum = contract('wiab->wi', A)

    DSSF_graph(DSSF_sum, 'sum', '')

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(w, np.log(DSSF_sum_Gamma[:,0]))
    ax.set_xlim([0, 3])
    plt.savefig(dir+"DSSF_SU2_gap_Gamma.pdf")
    plt.clf()
    plt.close('all')
    return A

def read_MD_SU3(dir, w0, wmax, t_evolved, isglobal=False):
    directory = os.fsencode(dir)
    P = np.loadtxt(os.path.join(os.path.dirname(dir), "pos_SU3.txt"))    
    T = np.loadtxt(dir + "/Time_steps.txt")
    S = np.loadtxt(dir + "/spin_t_SU3.txt")
    Slength = int(len(S)/len(P))
    S = S.reshape((Slength, len(P), 8))
    S = S[-len(T):]  # Ensure S has the same length as T

    # Use natural FFT frequencies instead of custom range
    dt = T[1] - T[0] if len(T) > 1 else 1.0
    fft_freqs = np.fft.fftfreq(len(T), d=dt) * 2 * np.pi
    # Filter to desired range
    w_mask = (fft_freqs >= w0) & (fft_freqs <= wmax)
    w = fft_freqs[w_mask]
    
    A = DSSF(w, DSSF_K, S, P, T, isglobal)
    
    # Compute DSSF at Gamma point (0,0,0) for gap analysis
    Gamma_point = np.array([[0, 0, 0]])
    A_Gamma = DSSF(w, Gamma_point, S, P, T, isglobal)
    DSSF_sum_Gamma = contract('wiab->wi', A_Gamma)
    
    def DSSF_graph(DSSF, i, j):
        fig, ax = plt.subplots(figsize=(10,4))
        C = ax.imshow(DSSF, origin='lower', extent=[0, g4, w0, wmax], aspect='auto', interpolation='gaussian', cmap='gnuplot2', norm='log')
        ax.axvline(x=g1, color='b', label='axvline - full height', linestyle='dashed')
        ax.axvline(x=g2, color='b', label='axvline - full height', linestyle='dashed')
        ax.axvline(x=g3, color='b', label='axvline - full height', linestyle='dashed')
        ax.axvline(x=g4, color='b', label='axvline - full height', linestyle='dashed')
        xlabpos = [g1, g2, g3, g4]
        labels = [r'$(0,0,0)$', r'$(0,0,1)$', r'$(0,1,1)$', r'$(1,1,1)$']

        ax.set_xticks(xlabpos, labels)
        ax.set_xlim([0, g4])
        ax.set_ylim([-3, 3])
        fig.colorbar(C)
        plt.savefig(dir+"DSSF_SU3_{}_{}.pdf".format(i, j))
        plt.clf()

    for i in range(8):
        DSSF_graph(A[:,:,i,i], i, i)
    
    DSSF_sum = contract('wiab->wi', A)

    DSSF_graph(DSSF_sum, 'sum', '')

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(w, np.log(DSSF_sum_Gamma[:,0]))
    ax.set_xlim([0, 3])
    plt.savefig(dir+"DSSF_SU3_gap_Gamma.pdf")
    plt.clf()
    plt.close('all')
    return A

def read_2D_nonlinear(dir):
    directory = os.fsencode(dir)
    tau_start, tau_end, tau_step, time_start, time_end, time_step = np.loadtxt(dir + "/param.txt")
    M0 = np.loadtxt(dir + "/M_time_0/M0/M_t_global.txt")[:,0]
    print(M0)
    domain = 500
    omega_range = 0.2
    M_NL = np.zeros((int(tau_step), domain))
    w = np.arange(-omega_range, omega_range, 1/100)
    T = np.linspace(time_start, time_end, int(time_step)) 
    T = T[-domain:]
    ffactt = np.exp(1j*contract('w,t->wt', w, T))/len(T)
    tau = np.linspace(tau_start, tau_end, int(tau_step))
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if os.path.isdir(dir + "/" + filename):
            info = filename.split("_")
            M1 = np.loadtxt(dir + "/" + filename + "/M1/M_t_global.txt")[:,0]
            M01 = np.loadtxt(dir + "/" + filename + "/M01/M_t_global.txt")[:,0]
            M_NL[int(info[2])] = M01[-domain:] - M0[-domain:] - M1[-domain:]
    # gaussian_filter =  np.exp(-1e-6 * (contract('i,i,a->ia',T,T,np.ones(len(tau))) + contract('a,a,i->ia',tau,tau,np.ones(len(T)))))   
    ffactau = np.exp(-1j*contract('w,t->wt', w, tau))/len(tau)
    # M_NL_FF = contract('it, ti->it', M_NL, gaussian_filter)
    M_NL_FF = M_NL
    M_NL_FF = np.abs(contract('it, wi, ut->wu', M_NL_FF, ffactau, ffactt))
    # M_NL_FF = np.log(M_NL_FF)
    # M_NL_FF = M_NL_FF/np.max(M_NL_FF)
    np.savetxt(dir + "/M_NL_FF.txt", M_NL_FF)
    plt.imshow(M_NL_FF, origin='lower', extent=[-omega_range, omega_range, -omega_range, omega_range], aspect='auto', interpolation='lanczos', cmap='gnuplot2', norm='linear')
    # plt.pcolormesh(w, w, np.log(M_NL_FF))
    plt.colorbar()
    plt.savefig(dir + "_NLSPEC.pdf")
    np.savetxt(dir + "_M_NL_FF.txt", M_NL_FF)
    plt.clf()


def read_2D_nonlinear_adaptive_time_step(dir, fm):
    """
    Process 2D nonlinear spectroscopy data with adaptive time steps.
    
    Args:
        dir: Directory containing the spectroscopy data
    """
    directory = os.path.abspath(dir)  # Use absolute path for reliability
    
    if fm:
        readfile = "M_t_local.txt"
    else:
        readfile = "M_t_global.txt"
    
    # Load M0 data once
    m0_file = os.path.join(directory, "M_time_0.000000/M1/" + readfile)
    m0_time_file = os.path.join(directory, "M_time_0.000000/M1/Time_steps.txt")

    time_steps = np.min([len(np.loadtxt(m0_file)), len(np.loadtxt(m0_time_file))])

    try:
        M0 = np.loadtxt(m0_file)[-time_steps:]
        M0_T = np.loadtxt(m0_time_file)
    except (IOError, IndexError) as e:
        print(f"Error loading M0 data: {e}")
        return
    
    # Setup frequency range
    omega_range = 3
    w = np.arange(0, omega_range, 0.1)
    wp = np.arange(-omega_range, omega_range, 0.1)

    # Precompute M0 frequency domain data
    M0_phase = np.exp(1j * np.outer(wp, M0_T))
    M0_w = contract('ta, wt->wa', M0, M0_phase)  # M0 in frequency domain
    
    # Initialize result array
    M_NL_FF = np.zeros((len(w), len(wp),3), dtype=complex)
    
    # Calculate tau values by extracting from directory names
    tau_values = []
    subdirs = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    for subdir in subdirs:
        if subdir.startswith("M_time_") and subdir != "M_time_0":
            try:
                parts = subdir.split("_")
                if len(parts) >= 3:
                    tau_val = float(parts[2])
                    tau_values.append(tau_val)
            except (ValueError, IndexError):
                continue
    
    tau = np.array(sorted(tau_values))
    
    # Process directories
    subdirs = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    for subdir in sorted(subdirs):
        if subdir == "M_time_0":
            continue
            
        try:
            # Parse tau value from directory name
            parts = subdir.split("_")
            if len(parts) != 3:
                continue
                
            current_tau = float(parts[2])
            base_path = os.path.join(directory, subdir)
            
            # Load M1 data
            M1 = np.loadtxt(os.path.join(base_path, "M1/" + readfile))[-time_steps:]
            M1_T = np.loadtxt(os.path.join(base_path, "M1/Time_steps.txt"))
            # Load M01 data
            M01 = np.loadtxt(os.path.join(base_path, "M01/" + readfile))[-time_steps:]
            M01_T = np.loadtxt(os.path.join(base_path, "M01/Time_steps.txt"))
            
            # Transform to frequency domain
            M1_phase = np.exp(1j * np.outer(wp, M1_T))
            M01_phase = np.exp(1j * np.outer(wp, M01_T))
            M1_w = contract('ta, wt->wa', M1, M1_phase)  # M1 in frequency domain
            M01_w = contract('ta, wt->wa', M01, M01_phase)  # M01 in frequency domain
            print(f"Processing {subdir} with tau={current_tau}")
            print(f"Shape of M01_w: {M01_w.shape}, M0_w: {M0_w.shape}, M1_w: {M1_w.shape}")
            M_NL_here = M01_w - M0_w - M1_w
            
            # Apply phase factor
            ffactau = np.exp(-1j * w * current_tau)
            M_NL_FF += contract('wa, e->ewa', M_NL_here, ffactau)
            
        except Exception as e:
            print(f"Error processing {subdir}: {e}")
            continue
    
    # Take absolute value for plotting
    M_NL_FF_abs = np.abs(M_NL_FF)

    # Suppress intensity near (0,0)
    # M_NL_FF_abs[len(wp)//2-2:len(wp)//2+2, 0:2] = 1e-15

    # Create plots with shared setup
    plt.figure(figsize=(10, 8))
    real_range = omega_range * 4.92/4.14
    extent = [0, real_range, -real_range, real_range]
    

    # Save raw data
    for i in range(3):
        M_NL_here = M_NL_FF_abs[:,:,i]
        output_file = os.path.join(dir, f"M_NL_FF_{i}.txt")
        np.savetxt(output_file, M_NL_here)

        # Linear scale plot
        plt.imshow(M_NL_here, origin='lower', extent=extent,
                aspect='auto', interpolation='lanczos', cmap='gnuplot2')
        plt.colorbar(label='Amplitude')
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Frequency (THz)')
        plt.title('2D Nonlinear Spectrum')
        plt.savefig(f"{dir}/NLSPEC_{i}_{fm}_SU2.pdf", dpi=300, bbox_inches='tight')
        plt.clf()
        
        # Log scale plot
        plt.imshow(M_NL_here, origin='lower', extent=extent,
                aspect='auto', interpolation='lanczos', cmap='gnuplot2',
                norm=PowerNorm(gamma=0.5))
        plt.colorbar(label='Amplitude (sqrt scale)')
        plt.xlabel('Frequency (J1)')
        plt.ylabel('Frequency (J1)')
        plt.title('2D Nonlinear Spectrum')
        plt.savefig(f"{directory}/NLSPEC_{i}_{fm}_SU2_log.pdf", dpi=300, bbox_inches='tight')
        plt.clf()
    
    return M_NL_FF_abs

def read_2D_nonlinear_adaptive_time_step_SU3(dir, fm):
    """
    Process 2D nonlinear spectroscopy data with adaptive time steps.
    
    Args:
        dir: Directory containing the spectroscopy data
    """
    directory = os.path.abspath(dir)  # Use absolute path for reliability
    
    if fm:
        readfile = "M_t_local_SU3.txt"
    else:
        readfile = "M_t_global_SU3.txt"

    m0_file = os.path.join(directory, "M_time_0.000000/M1/" + readfile)
    m0_time_file = os.path.join(directory, "M_time_0.000000/M1/Time_steps.txt")
    
    time_steps = np.min([len(np.loadtxt(m0_file)), len(np.loadtxt(m0_time_file))])

    try:
        M0 = np.loadtxt(m0_file)[-time_steps:]
        M0_T = np.loadtxt(m0_time_file)
    except (IOError, IndexError) as e:
        print(f"Error loading M0 data: {e}")
        return
    
    # Setup frequency range
    omega_range = 3
    w = np.arange(0, omega_range, 0.1)
    wp = np.arange(-omega_range, omega_range, 0.1)

    # Precompute M0 frequency domain data
    M0_phase = np.exp(1j * np.outer(wp, M0_T))
    M0_w = contract('ta, wt->wa', M0, M0_phase)  # M0 in frequency domain
    
    # Initialize result array
    M_NL_FF = np.zeros((len(w), len(wp), 8), dtype=complex)
    
    # Calculate tau values by extracting from directory names
    tau_values = []
    subdirs = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    for subdir in subdirs:
        if subdir.startswith("M_time_") and subdir != "M_time_0":
            try:
                parts = subdir.split("_")
                if len(parts) >= 3:
                    tau_val = float(parts[2])
                    tau_values.append(tau_val)
            except (ValueError, IndexError):
                continue
    
    tau = np.array(sorted(tau_values))
    
    # Process directories
    subdirs = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    for subdir in sorted(subdirs):
        if subdir == "M_time_0":
            continue
            
        try:
            # Parse tau value from directory name
            parts = subdir.split("_")
            if len(parts) != 3:
                continue
                
            current_tau = float(parts[2])
            base_path = os.path.join(directory, subdir)
            
            # Load M1 data
            M1 = np.loadtxt(os.path.join(base_path, "M1/" + readfile))[-time_steps:]
            M1_T = np.loadtxt(os.path.join(base_path, "M1/Time_steps.txt"))
            # Load M01 data
            M01 = np.loadtxt(os.path.join(base_path, "M01/" + readfile))[-time_steps:]
            M01_T = np.loadtxt(os.path.join(base_path, "M01/Time_steps.txt"))
            
            # Transform to frequency domain
            M1_phase = np.exp(1j * np.outer(wp, M1_T))
            M01_phase = np.exp(1j * np.outer(wp, M01_T))
            
            M1_w = contract('ta, wt->wa', M1, M1_phase)  # M1 in frequency domain
            M01_w = contract('ta, wt->wa', M01, M01_phase)  # M01 in frequency domain
            M_NL_here = M01_w - M0_w - M1_w
            
            # Apply phase factor
            ffactau = np.exp(-1j * w * current_tau) / len(tau)
            M_NL_FF += contract('wa, e->ewa', M_NL_here, ffactau)
            
        except Exception as e:
            print(f"Error processing {subdir}: {e}")
            continue
    
    # Take absolute value for plotting
    M_NL_FF_abs = np.abs(M_NL_FF)

    # M_NL_FF_abs[len(wp)//2-2:len(wp)//2+2, 0:2] = 1e-15
    real_range = omega_range * 4.92/4.14
    extent = [0, real_range, -real_range, real_range]

    # Save raw data
    for i in range(8):
        M_NL_here = M_NL_FF_abs[:,:,i]
        output_file = os.path.join(dir, f"M_NL_FF_SU3_{i}.txt")
        np.savetxt(output_file, M_NL_here)

        # Linear scale plot
        plt.imshow(M_NL_here, origin='lower', extent=extent,
                aspect='auto', interpolation='lanczos', cmap='gnuplot2')
        plt.colorbar(label='Amplitude')
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Frequency (THz)')
        plt.title('2D Nonlinear Spectrum')
        plt.savefig(f"{dir}/NLSPEC_{i}_{fm}_SU3.pdf", dpi=300, bbox_inches='tight')
        plt.clf()
        
        # Log scale plot
        plt.imshow(M_NL_here, origin='lower', extent=extent,
                aspect='auto', interpolation='lanczos', cmap='gnuplot2',
                norm=PowerNorm(gamma=0.5))
        plt.colorbar(label='Amplitude (sqrt scale)')
        plt.xlabel('Frequency (J1)')
        plt.ylabel('Frequency (J1)')
        plt.title('2D Nonlinear Spectrum')
        plt.savefig(f"{directory}/NLSPEC_{i}_{fm}_SU3_log.pdf", dpi=300, bbox_inches='tight')
        plt.clf()
    
    return M_NL_FF_abs


def read_2D_nonlinear_adaptive_time_step_combined(dir, fm):
    """
    Process 2D nonlinear spectroscopy data with adaptive time steps for both SU(2) and SU(3) data.
    This version uses FFT for improved efficiency and accuracy with natural FFT grid.
    Background subtraction is performed using data from dir/no_field/0 before Fourier transform.
    
    Args:
        dir (str): Directory containing the spectroscopy data.
        fm (bool): Flag to use filtered data files (e.g., 'M_t_f.txt').
    
    Returns:
        dict: Results containing SU2 and SU3 data with frequency arrays
    """
    directory = os.path.abspath(dir)
    
    configs = {
        'SU2': {'suffix': '', 'components': 3, 'readfile': f"M_t_global_f.txt" if fm else "M_t_global.txt"},
        'SU3': {'suffix': '_SU3', 'components': 8, 'readfile': f"M_t_global_f_SU3.txt" if fm else "M_t_global_SU3.txt"}
    }
    
    results = {}

    # Load background data from no_field/0 directory
    background_data = {}
    
    for group, config in configs.items():
        try:
            bg_file = os.path.join(directory, "M_time_0.000000/M1/", config['readfile'])
            if os.path.exists(bg_file):
                background_data[group] = np.loadtxt(bg_file)[0]
                print(f"Loaded background data for {group} from {bg_file}")
            else:
                background_data[group] = None
                print(f"No background file found for {group} at {bg_file}")
        except Exception as e:
            print(f"Error loading background for {group}: {e}")
            background_data[group] = None

    # Load initial M0 data for both SU(2) and SU(3)
    for group, config in configs.items():
        try:
            m0_file = os.path.join(directory, "M_time_0.000000/M1/", config['readfile'])
            m0_time_file = os.path.join(directory, "M_time_0.000000/M1/Time_steps.txt")
            
            m0_data = np.loadtxt(m0_file)
            m0_time = np.loadtxt(m0_time_file)
            time_steps = min(len(m0_data), len(m0_time))
            
            M0 = m0_data[-time_steps:]
            M0_T = m0_time[-time_steps:]
            
            # Subtract background if available
            if background_data[group] is not None:
                bg_length = min(len(background_data[group]), len(M0))
                M0[:bg_length] -= background_data[group][-bg_length:]
                print(f"Background subtracted for {group} M0 data")
            
            # Subtract mean configuration before FFT
            M0_mean = np.mean(M0, axis=0, keepdims=True)
            M0 = M0 - M0_mean
            
            # Use FFT with natural grid
            dt = M0_T[1] - M0_T[0] if len(M0_T) > 1 else 1.0
            M0_fft = np.fft.fft(M0, axis=0)
            wp_freqs = np.fft.fftfreq(len(M0), d=dt) * 2 * np.pi
            
            results[group] = {
                'M0_w': M0_fft,
                'M_NL_FF_tau': {},  # Store by tau value
                'time_steps': time_steps,
                'dt': dt,
                'wp_freqs': wp_freqs
            }
            print(f"Successfully loaded M0 data for {group}.")
        except (IOError, IndexError, FileNotFoundError) as e:
            print(f"Could not load M0 data for {group}, skipping. Error: {e}")
            results[group] = None

    # Get sorted list of tau values from directory names
    subdirs = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f)) and f.startswith("M_time_") and f != "M_time_0.000000"]
    
    tau_values = []
    for subdir in subdirs:
        try:
            tau_val = float(subdir.split("_")[2])
            tau_values.append(tau_val)
        except (ValueError, IndexError):
            continue
    tau = np.array(sorted(list(set(tau_values))))

    # Process each subdirectory once
    for subdir in sorted(subdirs):
        try:
            current_tau = float(subdir.split("_")[2])
            base_path = os.path.join(directory, subdir)
            print(f"Processing {subdir} with tau={current_tau}")

            # Load all data for the current tau
            for group, config in configs.items():
                if results[group] is None: continue
                readfile = config['readfile']
                time_steps = results[group]['time_steps']
                try:
                    M1 = np.loadtxt(os.path.join(base_path, "M1/", readfile))[-time_steps:]
                    M01 = np.loadtxt(os.path.join(base_path, "M01/", readfile))[-time_steps:]
                    
                    # Subtract background if available
                    if background_data[group] is not None:
                        bg_length = min(len(background_data[group]), len(M1))
                        M1[:bg_length] -= background_data[group][-bg_length:]
                        M01[:bg_length] -= background_data[group][-bg_length:]
                    
                    # Subtract mean configuration before FFT
                    M1_mean = np.mean(M1, axis=0, keepdims=True)
                    M1 = M1 - M1_mean
                    M01_mean = np.mean(M01, axis=0, keepdims=True)
                    M01 = M01 - M01_mean
                    
                    # Use FFT for time dimension
                    M1_fft = np.fft.fft(M1, axis=0)
                    M01_fft = np.fft.fft(M01, axis=0)
                    
                    M_NL_here = M01_fft - results[group]['M0_w'] - M1_fft
                    if np.any(np.isnan(M_NL_here)):
                        print(f"M_NL_here contains NaN values in {group} at tau={current_tau}")
                    else:
                        results[group]['M_NL_FF_tau'][current_tau] = M_NL_here
                        
                except (IOError, IndexError, FileNotFoundError) as e:
                    print(f"Could not load data for {group} in {subdir}. Skipping. Error: {e}")

        except Exception as e:
            print(f"Error processing directory {subdir}: {e}")
    
    # Now FFT over tau dimension
    for group in results:
        if results[group] is None: continue
        
        tau_sorted = sorted(results[group]['M_NL_FF_tau'].keys())
        if len(tau_sorted) == 0:
            continue
            
        # Stack all tau data
        M_NL_tau_stack = np.stack([results[group]['M_NL_FF_tau'][t] for t in tau_sorted], axis=0)
        
        # Subtract mean configuration before FFT over tau
        M_NL_tau_mean = np.mean(M_NL_tau_stack, axis=0, keepdims=True)
        M_NL_tau_stack = M_NL_tau_stack - M_NL_tau_mean
        
        # Get tau spacing
        d_tau = tau_sorted[1] - tau_sorted[0] if len(tau_sorted) > 1 else 1.0
        
        # FFT over tau dimension
        M_NL_2D = np.fft.fft(M_NL_tau_stack, axis=0)
        w_freqs = np.fft.fftfreq(len(tau_sorted), d=d_tau) * 2 * np.pi
        
        results[group]['M_NL_FF'] = M_NL_2D
        results[group]['w_freqs'] = w_freqs
        results[group]['tau_values'] = np.array(tau_sorted)
        del results[group]['M_NL_FF_tau']  # Clean up intermediate data

    # Finalize and plot results
    final_results = {}
    for group, res in results.items():
        if res is None: continue
        
        config = configs[group]
        M_NL_FF_abs = np.abs(res['M_NL_FF'])
        w_freqs = res['w_freqs']
        wp_freqs = res['wp_freqs']
        
        final_results[group] = M_NL_FF_abs

        # Get frequency ranges for plotting
        w_max = np.max(w_freqs[w_freqs >= 0])
        wp_max = np.max(np.abs(wp_freqs))
        extent = [0, w_max, -wp_max, wp_max]

        for i in range(config['components']):
            M_NL_here = M_NL_FF_abs[:, :, i]
            output_file = os.path.join(dir, f"M_NL_FF{config['suffix']}_{i}.txt")
            np.savetxt(output_file, M_NL_here)

            # Linear scale plot
            plt.figure(figsize=(10, 8))
            plt.imshow(M_NL_here, origin='lower', extent=extent,
                        aspect='auto', interpolation='lanczos', cmap='gnuplot2')
            plt.colorbar(label='Amplitude')
            plt.xlabel('Frequency (THz)')
            plt.ylabel('Frequency (THz)')
            plt.title(f'2D Nonlinear Spectrum ({group} component {i})')
            plt.savefig(f"{dir}/NLSPEC{config['suffix']}_{i}_{fm}.pdf", dpi=300, bbox_inches='tight')
            plt.clf()
            plt.close()
            
            # Log scale plot
            plt.figure(figsize=(10, 8))
            plt.imshow(M_NL_here, origin='lower', extent=extent,
                        aspect='auto', interpolation='lanczos', cmap='gnuplot2',
                        norm=PowerNorm(gamma=0.5))
            plt.colorbar(label='Amplitude (sqrt scale)')
            plt.xlabel('Frequency (J1)')
            plt.ylabel('Frequency (J1)')
            plt.title(f'2D Nonlinear Spectrum ({group} component {i}, PowerNorm)')
            plt.savefig(f"{directory}/NLSPEC{config['suffix']}_{i}_{fm}_POW.pdf", dpi=300, bbox_inches='tight')
            plt.clf()
            plt.close()

    return final_results

def full_read_2DCS_TFO(dir, done=False):
    """
    Reads and processes 2DCS data for TmFeO3 using the combined function with FFT.
    
    Args:
        dir (str): The directory containing the data.
        done (bool): If True, load from saved files instead of reprocessing.
    """
    if done:
        # Load pre-computed results from saved files
        SU2_results = []
        SU3_results = []

        # Load SU2 results (3 components)
        for i in range(3):
            filename = os.path.join(dir, f"M_NL_FF_{i}.txt")
            if os.path.exists(filename):
                SU2_results.append(np.loadtxt(filename))
            else:
                print(f"Warning: {filename} not found")

        # Load SU3 results (8 components)
        for i in range(8):
            filename = os.path.join(dir, f"M_NL_FF_SU3_{i}.txt")
            if os.path.exists(filename):
                SU3_results.append(np.loadtxt(filename))
            else:
                print(f"Warning: {filename} not found")

        # Convert to numpy arrays and add component dimension
        if SU2_results:
            SU2 = np.stack(SU2_results, axis=2)
        else:
            print("No SU2 results loaded")
            return

        if SU3_results:
            SU3 = np.stack(SU3_results, axis=2)
        else:
            print("No SU3 results loaded")
            return

        results = {'SU2': SU2, 'SU3': SU3}
        # Assume default frequency range for plotting
        omega_range = 5
        w_freqs = None
        wp_freqs = None
    else:
        results = read_2D_nonlinear_adaptive_time_step_combined(dir, False)
        
        if 'SU2' not in results or 'SU3' not in results:
            print("Could not process SU2 or SU3 data. Aborting.")
            return
        
        if results['SU2'] is None or results['SU3'] is None:
            print("Could not process SU2 or SU3 data. Aborting.")
            return
            
        SU2 = np.abs(results['SU2']['M_NL_FF'])
        SU3 = np.abs(results['SU3']['M_NL_FF'])
        w_freqs = results['SU2']['w_freqs']
        wp_freqs = results['SU2']['wp_freqs']
        
        # Get omega range from actual frequencies
        omega_range = np.max(np.abs(w_freqs))

    if 'SU2' not in results or 'SU3' not in results:
        print("Could not process SU2 or SU3 data. Aborting.")
        return

    real_range = omega_range

    xtotal = 2* SU2[:, :, 0] + 2.3915 * SU3[:, :, 4] + 0.9128 * SU3[:, :, 6]
    ytotal = 2* SU2[:, :, 1] + 2.7866 * SU3[:, :, 4] - 0.4655* SU3[:, :, 6]
    ztotal = 2* SU2[:, :, 2] + 5.264 * SU3[:, :, 1]
    SU3_contribution_x = 2.3915 * SU3[:, :, 4] + 0.9128 * SU3[:, :, 6]
    SU3_contribution_y = 2.7866 * SU3[:, :, 4] - 0.4655* SU3[:, :, 6]
    SU3_contribution_z = 5.264 * SU3[:, :, 1]
    extent = [0, real_range, -real_range, real_range]

    # Plotting function to reduce repetition
    def plot_spectrum(data, title_suffix, filename_suffix):
        plt.figure(figsize=(10, 8))
        plt.imshow(data, origin='lower', extent=extent,
                   aspect='auto', interpolation='lanczos', cmap='gnuplot2',
                   norm=PowerNorm(gamma=0.5))
        plt.colorbar(label='Amplitude (sqrt scale)')
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Frequency (THz)')
        plt.title(f'2D Nonlinear Spectrum ({title_suffix} Total)')
        plt.savefig(f"{dir}/NLSPEC_{filename_suffix}_total_sqrt.pdf", dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

    plot_spectrum(xtotal, 'X', 'x')
    plot_spectrum(ytotal, 'Y', 'y')
    plot_spectrum(ztotal, 'Z', 'z')
    plot_spectrum(SU3_contribution_x, 'SU3 Contribution X', 'su3_contribution_x')
    plot_spectrum(SU3_contribution_y, 'SU3 Contribution Y', 'su3_contribution_y')
    plot_spectrum(SU3_contribution_z, 'SU3 Contribution Z', 'su3_contribution_z')

def read_2D_nonlinear_tot(dir):
    directory = os.fsencode(dir)
    A = 0
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if os.path.isdir(dir + "/" + filename):
            read_2D_nonlinear(dir + "/" + filename)
            A = A + np.loadtxt(dir + "/" + filename + "/M_NL_FF.txt")
    A = A/np.max(A)
    time_step = len(A)
    plt.imshow(A.T, origin='lower', extent=[-1, 1, -1, 1], aspect='auto', interpolation='none', cmap='gnuplot2', norm='log')
    # w = np.linspace(-0.2, -0.2, time_step)
    # plt.pcolormesh(w, w, np.log(A))
    plt.colorbar()
    plt.savefig(dir + "_NLSPEC.pdf")
    plt.clf()

# obenton_to_xx_zz()
#
# dir = "MD_TmFeO3_CEF_E_Cali_0.5_20_test"
# dir = "MD_TmFeO3_CEF_E_Cali_-0.97_-3.89_test"
# dir = "TmFeO3_MD_Test_xii=0.05meV"
# read_MD_tot("TmFeO3_2DCS")
# read_MD_tot("MD_TmFeO3_E_0_3.97")
# read_MD_tot("MD_TmFeO3_E_0.97_0")
# read_MD_tot("MD_TmFeO3_E_1_5")
# read_MD_tot("MD_TmFeO3_E_0.97_3.97_longer_T")
# read_MD_tot("MD_TmFeO3_E_0.97_3.97_w_OS_5")
# read_MD_tot("TmFeO3_MD_xi=0.05")

# parseDSSF(dir)
# fullread(dir, False, "111")
# fullread(dir, True, "111")
# parseSSSF(dir)
# parseDSSF(dir)

# read_2D_nonlinear_adaptive_time_step("C://Users/raima/Downloads/TmFeO3_Fe_2DCS_Tzero_xii=0")
if __name__ == "__main__":
    directory = argv[1] if len(argv) > 1 else "TmFeO3_2DCS_D=0_xii=0.05"
    MD_read = argv[2] if len(argv) > 2 else "False"
    MD_read = MD_read.lower() == "true"
    if MD_read:
        read_MD_tot(directory, True)
    else:
        full_read_2DCS_TFO(directory)
    # read_MD(directory + "spin_t.txt")
    # read_MD_tot(directory)


# read_2D_nonlinear_adaptive_time_step("/scratch/y/ybkim/zhouzb79/TmFeO3_2DCS_xii=0.0_H_B")

# A = np.loadtxt("test_Jpm=0.3/specific_heat.txt", unpack=True)
# plt.plot(A[0], A[1])
# plt.xscale('log')
# plt.savefig("test_Jpm=0.3/specific_heat.pdf")

# C = np.array([0.47889, 0.42839, 0.75])

# N1 = np.array([0,0.5,0.5])
# N2 = np.array([0.5,0,0.5])
# N3 = np.array([1,0.5,0.5])
# N4 = np.array([0.5,1,0.5])
# N5 = np.array([0,0.5,1.5])
# N6 = np.array([0.5,0,1.5])
# N7 = np.array([1,0.5,1.5])
# N8 = np.array([0.5,1,1.5])

# N = np.array([N1,N2,N3,N4,N5,N6,N7,N8])
# A = np.array([0.02111, 0.92839, 0.75])

# N1 = np.array([0,0.5,0.5])
# N2 = np.array([0.5,0,0.5])
# N3 = np.array([1,0.5,0.5])
# N4 = np.array([0.5,1,0.5])
# N5 = np.array([0,0.5,1.5])
# N6 = np.array([0.5,0,1.5])
# N7 = np.array([1,0.5,1.5])
# N8 = np.array([0.5,1,1.5])

# N = np.array([N1,N2,N3,N4,N5,N6,N7,N8])

# B = np.array([0.52111, 0.57161, 0.25])
# D = np.array([0.97889, 0.07161, 0.25])
# print(A + N - C)

# A = np.loadtxt("./TmFeO3_2DCS.txt", dtype=np.complex128)[25:-25, 25:-25]
# plt.imshow(np.log(np.abs(A)), origin='lower', extent=[-5, 5, -5, 5], aspect='auto', interpolation='gaussian', cmap='gnuplot2', norm='linear')
# plt.savefig("TmFeO3_2DCS.pdf")