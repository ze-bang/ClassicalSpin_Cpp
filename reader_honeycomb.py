import h5py
import numpy as np
from opt_einsum import contract
import matplotlib.pyplot as plt
import os
from math import gcd
from functools import reduce
# plt.rcParams['text.usetex'] = True

def honeycomb_reciprocal_basis():
    """
    Calculate reciprocal lattice vectors for a honeycomb lattice.
    
    Real space basis vectors:
    a1 = (1, 0, 0)
    a2 = (1/2, sqrt(3)/2, 0)
    
    Returns:
        numpy.ndarray: Reciprocal lattice vectors b1 and b2
    """
    a1 = np.array([1, 0, 0])
    a2 = np.array([1/2, np.sqrt(3)/2, 0])
    a3 = np.array([0, 0, 1])  # Third basis vector (perpendicular to plane)
    
    # Calculate reciprocal lattice vectors using the formula:
    # b_i = 2π * (a_j × a_k) / (a_i · (a_j × a_k))
    # where i,j,k are cyclic
    
    # Calculate cross products
    a2_cross_a3 = np.cross(a2, a3)
    a3_cross_a1 = np.cross(a3, a1)
    a1_cross_a2 = np.cross(a1, a2)
    
    # Calculate dot products for normalization
    vol = np.dot(a1, np.cross(a2, a3))
    
    # Calculate reciprocal lattice vectors
    b1 = 2 * np.pi * a2_cross_a3 / vol
    b2 = 2 * np.pi * a3_cross_a1 / vol
    b3 = 2 * np.pi * a1_cross_a2 / vol
    
    # Return the in-plane reciprocal lattice vectors
    return np.array([b1, b2, b3])

KBasis = honeycomb_reciprocal_basis()
print(KBasis)
def calcNumSites(A, B, N):
    # Convert A and B to the reciprocal lattice basis
    # KBasis contains the reciprocal lattice vectors
    
    # First we need to get the coordinates in terms of reciprocal lattice vectors
    # We are solving the equation: A = a*b1 + b*b2 + c*b3 where b1,b2,b3 are the reciprocal lattice vectors
    A_basis = np.linalg.solve(KBasis.T, A)
    B_basis = np.linalg.solve(KBasis.T, B)
    
    # Scale the coordinates based on the resolution N
    A_grid = A_basis * N
    B_grid = B_basis * N

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


def drawLine(A, B, N):
    temp = np.linspace(A, B, N, endpoint=False)[1:]
    return temp

def magnitude_bi(vector1, vector2):
    # temp1 = contract('i,ik->k', vector1, BasisBZA)
    # temp2 = contract('i,ik->k', vector2, BasisBZA)
    temp1 = vector1
    temp2 = vector2
    return np.linalg.norm(temp1-temp2)


stepN = 8


kitaevBasis = 4*np.pi/np.sqrt(3)*np.array([[np.sqrt(3)/2,-1/2,0],[0,1,0], [0,0,1]])

kitaevLocal = np.array([[1,-1,-1],[-1,1,-1],[-1,-1,1]])/np.sqrt(3)

Gamma = np.array([0, 0, 0])
K1 = np.array([1/3, 2/3, 0])
K2 = -np.array([2/3, 1/3, 0])
M1 = np.array([0, 1/2, 0])
M2 = np.array([1/2, 1/2, 0])
Gamma2 = np.array([0,1,0])
# LKitaev = 20

Gamma = contract('ij, i->j', kitaevBasis, Gamma)
K1 = contract('ij, i->j', kitaevBasis, K1)
K2 = contract('ij, i->j', kitaevBasis, K2)
M1 = contract('ij, i->j', kitaevBasis, M1)
M2 = contract('ij, i->j', kitaevBasis, M2)
Gamma2 = contract('ij, i->j', kitaevBasis, Gamma2)


# print(K2D)

P1_2D = drawLine(Gamma, M1, stepN)
P2_2D = drawLine(M2, Gamma, stepN)
P3_2D = drawLine(Gamma, K1, stepN)
P4_2D = drawLine(K2, Gamma, stepN)
P5_2D = drawLine(Gamma2, M1, stepN)
P6_2D = drawLine(M1, K1, int(stepN/2))

gGamma = 0
gM1M2 = len(P1_2D)
gGamma1 = gM1M2 + len(P2_2D)
gK1K2 = gGamma1 + len(P3_2D)
gGamma1Gamma2 = gK1K2 + len(P4_2D)
gM1 = gGamma1Gamma2 + len(P5_2D)
gK1 = gM1 + len(P6_2D)
DSSF_K = np.concatenate((P1_2D, P2_2D, P3_2D, P4_2D, P5_2D, P6_2D))


z = np.array([np.array([1,1,1])/np.sqrt(3), np.array([1,-1,-1])/np.sqrt(3), np.array([-1,1,-1])/np.sqrt(3), np.array([-1,-1,1])/np.sqrt(3)])
x = np.array([[-2,1,1],[-2,-1,-1],[2,1,-1], [2,-1,1]])/np.sqrt(6)
y = np.array([[0,-1,1],[0,1,-1],[0,-1,-1], [0,1,1]])/np.sqrt(2)
localframe = np.array([x,y,z])

def Spin(k, S, P):
    ffact = np.exp(1j*contract('ik,jk->ij', k, P))
    N = len(S)
    return contract('js, ij->is', S, ffact)/np.sqrt(N)

def Spin_t(k, S, P):
    ffact = np.exp(1j*contract('ik,jk->ij', k, P))
    N = S.shape[1]
    return contract('tjs, ij->tis', S, ffact)/np.sqrt(N)

def Spin_global(k, S, P):
    ffact = np.exp(1j*contract('ik,jk->ij', k, P))
    N = len(S)
    return contract('js, ij, sr->ir', S, ffact, kitaevLocal)/np.sqrt(N)

def Spin_t_global(k, S, P):
    ffact = np.exp(1j*contract('ik,jk->ij', k, P))
    N = S.shape[1]
    return contract('tjs, ij, sr->tir', S, ffact, kitaevLocal)/np.sqrt(N)

def SSSF_q(k, S, P, gb=False):
    A = Spin(k, S, P)
    zq = contract('ar, ir->ia', kitaevLocal, k)
    if gb:
        proj = contract('ar,br,i->iab', kitaevLocal,kitaevLocal, np.ones(len(k))) - contract('ia,ib,i->iab', zq,zq, 1/contract('ik,ik->i', k, k))    
        return np.real(contract('ia, ib, iab -> iab', A, np.conj(A),proj))
    else:
        return np.real(contract('ia, ib -> iab', A, np.conj(A)))

def DSSF(w, k, S, P, T, gb=False):
    A = Spin_t(k, S, P)
    ffactt = np.exp(1j*contract('w,t->wt', w, T))
    Somega = contract('tis, wt->wis', A, ffactt)/np.sqrt(len(T))
    zq = contract('ar, ir->ia', kitaevLocal, k)
    proj = contract('ar,br,i->iab', kitaevLocal,kitaevLocal, np.ones(len(k))) - contract('ia,ib,i->iab', zq,zq, 1/contract('ik,ik->i', k, k))
    read = np.real(contract('wia, wib, iab-> wi', Somega, np.conj(Somega), proj))
    return np.log(read)

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
    return contract('ij,k->ijk',H, 2*np.array([np.pi,0, 0])) + contract('ij,k->ijk',K, 2*np.array([0,np.pi, 0]))

def hhknk(H,K):
    return contract('ij,k->ijk',H, 2*np.array([np.pi,np.pi, 0])) + contract('ij,k->ijk',K, 2*np.array([np.pi,-np.pi, 0]))

def SSSF2D(S, P, nK, filename, gb=False):
    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hk2d(A, B).reshape((nK*nK,3))
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
    SSSFGraph2D(A, B, S[:,:,0,0], f1)
    SSSFGraph2D(A, B, S[:,:,1,1], f2)
    SSSFGraph2D(A, B, S[:,:,2,2], f3)
    SSSFGraph2D(A, B, S[:, :, 0, 1], f4)
    SSSFGraph2D(A, B, S[:, :, 0, 2], f5)
    SSSFGraph2D(A, B, S[:, :, 1, 2], f6)
    SSSFGraph2D(A, B, contract('ijab->ij', S), filename + "S_total")


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
    SSSFGraphHK0(A, B, S[:,:,0,0], f1)
    SSSFGraphHK0(A, B, S[:,:,1,1], f2)
    SSSFGraphHK0(A, B, S[:,:,2,2], f3)
    SSSFGraphHK0(A, B, S[:, :, 0, 1], f4)
    SSSFGraphHK0(A, B, S[:, :, 0, 2], f5)
    SSSFGraphHK0(A, B, S[:, :, 1, 2], f6)

def genALLSymPointsBare():
    d = 9 * 1j
    b = np.mgrid[0:1:d, 0:1:d, 0:1:d].reshape(3, -1).T
    return b
BasisBZA = np.array([2*np.pi*np.array([-1,1,1]),2*np.pi*np.array([1,-1,1]),2*np.pi*np.array([1,1,-1])])
BasisBZA_reverse = np.array([np.array([0,1,1]),np.array([1,0,1]),np.array([1,1,0])])/2

BasisBZA_reverse_honeycomb = np.array([[1,1/2, 0],[0, np.sqrt(3)/2,0],[0,0,1]])

def genBZ(d, m=1):
    dj = d*1j
    b = np.mgrid[0:m:dj, 0:m:dj, 0:m:dj].reshape(3,-1).T
    b = np.concatenate((b,genALLSymPointsBare()))
    b = contract('ij, jk->ik', b, BasisBZA)
    return b


def ordering_q_slice(S, P, ind):
    K = genBZ(101)
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

def magnetization(S):
    return np.mean(S,axis=0)

r = np.array([[0,1/2,1/2],[1/2,0,1/2],[1/2,1/2,0]])
NN = -np.array([[-1/4,-1/4,-1/4],[-1/4,1/4,1/4],[1/4,-1/4,1/4],[1/4,1/4,-1/4]])/2
z = np.array([[1,1,1],[1,-1,-1],[-1,1,-1], [-1,-1,1]])/np.sqrt(3)
y = np.array([[0,-1,1],[0,1,-1],[0,-1,-1], [0,1,1]])/np.sqrt(2)
x = np.array([[-2,1,1],[-2,-1,-1],[2,1,-1], [2,-1,1]])/np.sqrt(6)

def plottetrahedron(x,y,z, ax):
    center = x*r[0]+y*r[1]+z*r[2]
    coords = center + NN
    start = np.zeros((6,3))
    start[0] = start[1] = start[2] = coords[0]
    start[3] = start[4] = coords[1]
    start[5] = coords[2]
    end = np.zeros((6,3))
    end[0] = coords[1]
    end[1] = end[3] = coords[2]
    end[2] = end[4] = end[5] = coords[3]
    for i in range(6):
        ax.plot([start[i,0], end[i,0]], [start[i,1], end[i,1]], zs=[start[i,2], end[i,2]], color='blue')

def strip(k):
    temp = np.copy(k)
    while (temp>0.3).any():
        for i in range(3):
            if temp[i] > 0.3:
                temp[i] = temp[i]-0.5
    return temp

def findindex(k):
    if (k==np.array([1,1,1])/8).all():
        return 0
    elif (k == np.array([1, -1, -1]) / 8).all():
        return 1
    elif (k == np.array([-1, 1, -1]) / 8).all():
        return 2
    else:
        return 3
def graphconfig(S, P, filename):
    ax = plt.axes(projection='3d')
    for i in range(len(S)):
        A = strip(P[i])
        index = findindex(A)
        S[i] = S[i,0] * x[index] + S[i,1] * y[index] + S[i,2] * z[index]
        # print(P[i], A, index, S[i])
    for i in range(2):
        for j in range(2):
            for k in range(2):
                plottetrahedron(i,j,k,ax)
    S = S/2

    ax.scatter(P[:,0], P[:,1], P[:,2])
    ax.quiver(P[:,0], P[:,1], P[:,2],S[:,0], S[:,1], S[:,2], color='red', length=0.3)
    plt.savefig(filename)
    plt.clf()

def fullread(dir, gb=False, magi=""):
    directory = os.fsencode(dir)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if magi:
            mag = magi
        else:
            mag = filename[2:5]
        if filename.endswith(".h5") and not filename.endswith("time_evolved.h5"):
            print(filename)
            f = h5py.File(dir + filename, 'r')
            S = f['spins'][:]
            P = f['site_positions'][:]
            newdir = dir + filename[:-3] + "/"
            if not os.path.isdir(newdir):
                os.mkdir(newdir)
            if P.shape[1] == 2:
                SSSF2D(S,P, 100, newdir, gb)
            elif mag == "001":
                SSSFHK0(S, P, 50, newdir, gb)
            else:
                SSSFHnHL(S, P, 50, newdir, gb)
            # graphconfig(S, P, newdir+"plot.pdf")
            # A = ordering_q(S, P)
            # np.savetxt(newdir + "ordering_wave.txt", A)
            # M = magnetization(S)
            # np.savetxt(newdir + "magnetization.txt", M)
        if filename.endswith(".h5") and filename.endswith("time_evolved.h5"):
            print(filename)
            f = h5py.File(dir + filename, 'r')
            S = f['spins'][:]
            P = f['site_positions'][:]
            T = f['t'][:]
            w0 = 0
            wmax = 2.5
            w = np.linspace(w0, wmax, 1000)[1:]
            A = DSSF(w, DSSF_K, S, P, T, gb)
            A = A/np.max(A)
            np.savetxt(dir+filename[:-3]+".txt", A)

def obenton_phase_diagram(Jpm, Jpmpm):
    if Jpmpm>4.4*np.log10(-(Jpm-1.05)):
        return 5
    elif Jpmpm>8.7*np.log10(-(Jpm-1)):
        return 0
    else:
        return 1
def obenton_to_xx_zz():
    N = 200
    phase = np.zeros((N,N))
    Jpm = np.zeros((N,N))
    Jpmpm = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            Jxx = -1 + (2 / N * (i + 1))
            Jyy = -1 + (2 / N * (j + 1))
            Jpm[i, j] = -(Jxx + Jyy) / 4
            Jpmpm[i, j] = np.abs(Jxx - Jyy) / 4
            phase[i, j] = obenton_phase_diagram(Jpm[i,j], Jpmpm[i,j])
    plt.pcolormesh(Jpm, Jpmpm, phase)
    plt.ylim([0, 0.5])
    # plt.colorbar()
    plt.xlabel(r"$J_{\pm}/J_{yy}$")
    plt.ylabel(r"$J_{\pm\pm}/J_{yy}$")
    plt.savefig("o_benton_Jpm_Jpmpm.pdf")
    plt.clf()
    plt.imshow(phase.T, origin='lower', interpolation='lanczos', extent=[-1, 1, -1, 1], aspect='equal', cmap='Pastel1')
    # plt.colorbar()
    plt.xlabel(r"$J_{xx}/J_{yy}$")
    plt.ylabel(r"$J_{zz}/J_{yy}$")
    plt.savefig("o_benton_Jxx_Jzz.pdf")
    plt.clf()


def parseDSSF(dir):
    size = 0
    directory = os.fsencode(dir)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith("DSSF.txt"):
            test = np.loadtxt(dir+"/"+filename)
            size = test.shape
            break
    A = np.zeros(size)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith("DSSF.txt"):
            print(filename)
            A = A + np.loadtxt(dir+"/"+filename)

    A = A / np.max(A)
    fig, ax = plt.subplots(figsize=(10,4))


    C = ax.imshow(A, origin='lower', extent=[0, gK1, 0, 2.5], aspect='auto', interpolation='lanczos', cmap='gnuplot2')
    ax.axvline(x=gGamma, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gM1M2, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gGamma1, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gK1K2, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gM1, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gK1, color='b', label='axvline - full height', linestyle='dashed')

    xlabpos = [gGamma, gM1M2, gGamma1, gK1K2, gGamma1Gamma2, gM1, gK1]
    labels = [r'$K$', r'$\Gamma_0$', r'$M$', r'$\Gamma_1$', r'$K$', r'$M$']
    ax.set_xticks(xlabpos, labels)
    ax.set_xlim([0, gK1])
    fig.colorbar(C)
    plt.savefig(dir+"DSSF.pdf")
    plt.clf()

def read_MD_tot(dir):
    directory = os.fsencode(dir)
    nK = 50
    A = np.zeros((8, nK, nK))
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if os.path.isdir(dir + "/" + filename):
            # read_MD(dir + "/" + filename)
            A += read_MD_slice(dir + "/" + filename, nK)
    for i in range(2):
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(5,5))
        C = ax[0,0].imshow(A[4*i][0:int(nK/2), int(nK/2):nK], origin='lower', extent=[-1, 0, 0, 2], aspect='auto', cmap='gnuplot2')
        C = ax[0,1].imshow(A[4*i+1][int(nK/2):nK, int(nK/2):nK], origin='lower', extent=[0, 1, 0, 2], aspect='auto', cmap='gnuplot2')
        C = ax[1,1].imshow(A[4*i+2][int(nK/2):nK, 0:int(nK/2)], origin='lower', extent=[0, 1, -2, 0], aspect='auto', cmap='gnuplot2')
        C = ax[1,0].imshow(A[4*i+3][0:int(nK/2), 0:int(nK/2)], origin='lower', extent=[-1, 0, -2, 0], aspect='auto', cmap='gnuplot2')

        plt.savefig(dir + "/DSSF_" + str(i) + ".pdf")
        plt.clf()


def read_MD(dir):
    directory = os.fsencode(dir)
    P = np.loadtxt(dir + "/pos.txt")
    T = np.loadtxt(dir + "/Time_steps.txt")

    S = np.loadtxt(dir + "/spin_t.txt").reshape((len(T), len(P), 3))

    w0 = 0
    wmax = 15
    w = np.arange(w0, wmax, 1/600)[1:]

    A = DSSF(w, DSSF_K, S, P, T, True)
    A = A / np.max(A)
    np.savetxt(dir + "_DSSF.txt", A)
    fig, ax = plt.subplots(figsize=(10,4))

    C = ax.imshow(A, origin='lower', extent=[0, gK1, 0, 2.5], aspect='auto', interpolation='lanczos', cmap='gnuplot2')
    ax.axvline(x=gGamma, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gM1M2, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gGamma1, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gK1K2, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gM1, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gK1, color='b', label='axvline - full height', linestyle='dashed')

    xlabpos = [gGamma, gM1M2, gGamma1, gK1K2, gGamma1Gamma2, gM1, gK1]
    labels = [r'$\Gamma_1$', r'$M_1\quad M_2$', r'$\Gamma_1$', r'$K_1\quad K_2$', r'$\Gamma_1\quad\Gamma_2$', r'$M_1$', r'$K_1$']
    ax.set_xticks(xlabpos, labels)
    ax.set_xlim([0, gK1])
    fig.colorbar(C)
    plt.savefig(dir+"DSSF.pdf")
    plt.clf()

    SSSF2D(S[0], P, 100, dir, True)


def read_MD_slice(dir, nK):
    directory = os.fsencode(dir)
    P = np.loadtxt(dir + "/pos.txt")
    T = np.loadtxt(dir + "/Time_steps.txt")

    S = np.loadtxt(dir + "/spin_t.txt").reshape((len(T), len(P), 3))

    w = np.array([0.1, 1, 2, 3, 4, 5, 6, 7])

    nK = 50
    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hk2d(A, B).reshape((nK*nK,3))

    A = DSSF(w, K, S, P, T, True)
    np.savetxt(dir + "_DSSF.txt", A)
    A = A.reshape((len(w), nK, nK))
    for i in range(len(w)):
        fig, ax = plt.subplots(figsize=(5,5))
        C = ax.imshow(A[i], origin='lower', extent=[-1.5, 1.5, -2.5, 2.5], aspect='auto', cmap='gnuplot2')

        plt.savefig(dir + "/DSSF_" + str(w[i]) + ".pdf")
        plt.clf()
    return A


def plot_spin_config(P, S, field_dir, filename):
    from matplotlib.animation import FuncAnimation
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_axis_off()
    ax.scatter(P[:,0],P[:,1],P[:,2], color='w', edgecolors='b', s=60,alpha=1)
    ax.quiver(P[:,0], P[:,1], P[:,2],S[:,0], S[:,1], S[:,2], color='red', length=0.3)
    ax.plot_trisurf(P[:,0], P[:,1], P[:,2],triangles=[[0,1,2],[0,1,3],[0,2,3],[1,2,3]], edgecolor=[[0.1,0.1,0.1]], linewidth=1, shade=False, alpha=0)
    
    field_point = np.array([0.3, 0, 0])
    ax.quiver(field_point[0], field_point[1], field_point[2], field_dir[0], field_dir[1], field_dir[2], color='black', length=0.3)
    for i in range(4):
        ax.text(P[i,0], P[i,1], P[i,2], str(i), color='black', fontsize=12)
    ax.margins(0.2)
    ax.set_aspect('equal')
    def animate(i):
        ax.view_init(elev=20., azim=i*0.05)

    rot_animation = FuncAnimation(fig, animate, frames=np.arange(0, 362, 2), interval=100)
    rot_animation.save(filename+'.gif', dpi=80, writer='imagemagick')
    plt.clf()

def read_2D_nonlinear(dir):
    directory = os.fsencode(dir)
    tau_start, tau_end, tau_step, time_start, time_end, time_step, K, h = np.loadtxt(dir + "/param.txt")
    M0 = np.loadtxt(dir + "/M_time_0/M0/M_t.txt")[:,2]
    domain = 2401
    omega_range = 0.2
    M_NL = np.zeros((int(tau_step), domain))
    w = np.arange(-omega_range, omega_range, 1/600)
    T = np.linspace(time_start, time_end, int(time_step)) 
    T = T[-domain:]
    ffactt = np.exp(1j*contract('w,t->wt', w, T))/len(T)
    tau = np.linspace(tau_start, tau_end, int(tau_step))
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if os.path.isdir(dir + "/" + filename):
            info = filename.split("_")
            M1 = np.loadtxt(dir + "/" + filename + "/M1/M_t.txt")[:,2]
            M01 = np.loadtxt(dir + "/" + filename + "/M01/M_t.txt")[:,2]
            M_NL[int(info[2])] = M01[-domain:] - M0[-domain:] - M1[-domain:] + 0.57735
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
dir = "BCAO_zero_field_1.7K"
# dir = "Kitaev_BCAO"
read_MD_tot(dir)
# parseDSSF(dir)

# dir = "kitaev_honeycomb_nonlinear_Gamma=0.25_Gammap=-0.02_h=0.7"
# dir = "test_long_h=0.0"
# read_2D_nonlinear_tot(dir)
# dir = "pure_kitaev_2DCS_h=0.7_weak_pulse"
# read_2D_nonlinear_tot(dir)
# P = np.loadtxt("pos.txt")
# S = np.loadtxt("spin.txt")
# S_global = np.zeros(S.shape)
# for i in range(4):
#     S_global[i::4] = contract('js, sp->jp', S[i::4], localframe[:,i,:])
# plot_spin_config(P[0:4], S_global[0:4],  -np.array([1,1,1])/np.sqrt(3), "spin_config.pdf")

# dir = "./kitaev/"
# fullread(dir, True, "110")
# #
# dir = "./Jxx_-0.2_Jyy_1.0_Jzz_-0.2_gxx_0_gyy_0_gzz_1/"
# fullread(dir, True)
#
# dir = "./Jxx_0.2_Jyy_1.0_Jzz_0.2_gxx_0_gyy_0_gzz_1/"
# fullread(dir, True)
#
# dir = "./Jxx_0.6_Jyy_1.0_Jzz_0.6_gxx_0_gyy_0_gzz_1/"
# fullread(dir, True)