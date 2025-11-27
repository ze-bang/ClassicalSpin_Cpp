import argparse
import h5py
import numpy as np
from opt_einsum import contract
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation, PillowWriter
import re
# plt.rcParams['text.usetex'] = True


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


graphres = 2

Gamma = np.array([0, 0, 0])
K = 2 * np.pi * np.array([3/4, -3/4, 0])
W = 2 * np.pi * np.array([1, -1/2, 0])
X = 2 * np.pi * np.array([1, 0, 0])

L = np.pi * np.array([1, 1, 1])
U = 2 * np.pi * np.array([1/4, 1/4, 1])
W1 = 2 * np.pi * np.array([0, 1/2, 1])
X1 = 2 * np.pi * np.array([0, 0, 1])



stepN = np.linalg.norm(U-W1)/graphres


#Path to 1-10
GammaX = drawLine(Gamma, X, stepN)[1:]
XW = drawLine(X, W, stepN)
WK = drawLine(W, K, stepN)
KGamma = drawLine(K, Gamma, stepN)[:-1]

#Path to 111 and then 001
GammaL = drawLine(Gamma, L, stepN)[1:]
LU = drawLine(L, U, stepN)
UW1 = drawLine(U, W1, stepN)
W1X1 = drawLine(W1, X1, stepN)
X1Gamma = drawLine(X1, Gamma, stepN)[:-1]

gGamma1 = 0
gX = magnitude_bi(Gamma, X)
gW = gX + magnitude_bi(X, W)
gK = gW + magnitude_bi(W, K)

gGamma2 = gK + magnitude_bi(K, Gamma)
gL = gGamma2 + magnitude_bi(Gamma, L)
gU = gL + magnitude_bi(L, U)
gW1 = gU + magnitude_bi(U, W1)
gX1 = gW1 + magnitude_bi(W1, X1)
gGamma3 = gX1 + magnitude_bi(X1, Gamma)


# Gamma = np.array([0, 0, 0])
# P1 = 2 * np.pi * np.array([1, 0, 0])
# P2 = 2 * np.pi * np.array([2, 0, 0])
# P3 = 2 * np.pi * np.array([2, 1, 0])
# P4 = 2 * np.pi * np.array([2, 2, 0])
# P5 = 2 * np.pi * np.array([1, 1, 0])

P1 =  2 * np.pi * np.array([1, 1, 0])
P2 =  2 * np.pi * np.array([2, 2, 0])
P3 =  2 * np.pi * np.array([2, 2, 1])
P4 =  2 * np.pi * np.array([2, 2, 2])
P5 =  2 * np.pi * np.array([1, 1, 1])

# stepN = np.linalg.norm(Gamma-P1)/graphres
# stepN = 8

#Path to 1-10
# GammaP1 = drawLine(Gamma, P1, stepN)[1:]
# P12 = drawLine(P1, P2, stepN)[:-1]
# P23 = drawLine(P2, P3, stepN)[1:]
# P34 = drawLine(P3, P4, stepN)[:-1]
# P45 = drawLine(P4, P5, stepN)[1:-1]
# P5Gamma = drawLine(P5, Gamma, stepN)[1:-1]



# gGamma1 = 0
# g1 = magnitude_bi(Gamma, P1)
# g2 = g1 + magnitude_bi(P1, P2)
# g3 = g2 + magnitude_bi(P2, P3)
# g4 = g3 + magnitude_bi(P3, P4)
# g5 = g4 + magnitude_bi(P4, P5)
# gGamma4 = g5 + magnitude_bi(P5, Gamma)


DSSF_K = np.concatenate((GammaX, XW, WK, KGamma, GammaL, LU, UW1, W1X1, X1Gamma))
# DSSF_K = np.concatenate((GammaP1, P12, P23, P34, P45, P5Gamma))


z = np.array([np.array([1,1,1])/np.sqrt(3), np.array([1,-1,-1])/np.sqrt(3), np.array([-1,1,-1])/np.sqrt(3), np.array([-1,-1,1])/np.sqrt(3)])
x = np.array([[-2,1,1],[-2,-1,-1],[2,1,-1], [2,-1,1]])/np.sqrt(6)
y = np.array([[0,-1,1],[0,1,-1],[0,-1,-1], [0,1,1]])/np.sqrt(2)
localframe = np.array([x,y,z])

# def Spin_global_pyrochlore(k,S,P):
#     size = int(len(P)/4)
#     tS = np.zeros((len(k),3), dtype=np.complex128)
#     for i in range(4):
#         ffact = np.exp(1j * contract('ik,jk->ij', k, P[i*size:(i+1)*size]))
#         tS = tS + contract('js, ij, sp->ip', S[i*size:(i+1)*size], ffact, localframe[:,i,:])/np.sqrt(size)
#     return tS

def Spin(k, S, P):
    # Compute k·P more efficiently using numpy's matrix multiplication
    kP = np.dot(k, P.T)  # shape: (len(k), len(P))
    ffact = np.exp(1j * kP)
    N = len(S)
    return contract('js, ij->is', S, ffact)/np.sqrt(N)


def Spin_global_pyrochlore_t(k,S,P):
    size = int(len(P))
    tS = np.zeros((len(S), 4, len(k),3), dtype=np.complex128)
    for i in range(4):
        # Compute k·P more efficiently
        P_sub = P[i::4]
        kP = np.dot(k, P_sub.T)  # shape: (len(k), len(P_sub))
        ffact = np.exp(1j * kP)
        tS[:,i,:,:] = contract('tjs, ij->tis', S[:,i::4], ffact)/np.sqrt(size)
    return tS

def Spin_global_pyrochlore(k,S,P):
    size = int(len(P))
    tS = np.zeros((4, len(k),3), dtype=np.complex128)
    for i in range(4):
        # Compute k·P more efficiently
        P_sub = P[i::4]
        kP = np.dot(k, P_sub.T)  # shape: (len(k), len(P_sub))
        ffact = np.exp(1j * kP)
        tS[i,:,:] = contract('js, ij->is', S[i::4], ffact)/np.sqrt(size)
    return tS


def Spin_t(k, S, P):
    # Compute k·P more efficiently using numpy's matrix multiplication
    kP = np.dot(k, P.T)  # shape: (len(k), len(P))
    ffact = np.exp(1j * kP)
    N = len(S)
    return contract('tjs, ij->tis', S, ffact)/np.sqrt(N)


def g(q):
    M = np.zeros((len(q),4,4,3,3))
    qnorm = contract('ik, ik->i', q, q)
    qnorm = np.where(qnorm == 0, np.inf, qnorm)
    for i in range(4):
        for j in range(4):
            for a in range(3):
                for b in range(3):
                    M[:,i,j,a,b] = np.dot(localframe[a][i], localframe[b][j]) - contract('k, ik->i',localframe[a][i],q) * contract('k, ik->i', localframe[b][j],q) /qnorm
    return M

def projector(q):
    M = np.zeros((len(q),3,3))
    qnorm = contract('ik, ik->i', q, q)
    qnorm = np.where(qnorm == 0, np.inf, qnorm)
    for a in range(3):
        for b in range(3):
            M[:,a,b] = a == b - q[:,a]*q[:,b]/qnorm
    return M

def gg(q):
    M = np.zeros((len(q),4,4, 3, 3))
    for k in range(len(q)):
        for i in range(4):
            for j in range(4):
                for a in range(3):
                    for b in range(3):
                        if not np.dot(q[k],q[k]) == 0:
                            M[k, i,j,a,b] = np.dot(localframe[a][i], localframe[b][j]) - np.dot(localframe[a][i],q[k]) * np.dot(localframe[b][j],q[k])/ np.dot(q[k],q[k])
                        else:
                            M[k, i, j,a,b] = np.dot(localframe[a][i], localframe[b][j])
    return M

def ggz(q):
    M = np.zeros((len(q),4,4))
    for k in range(len(q)):
        for a in range(4):
            for b in range(4):
                if not np.dot(q[k],q[k]) == 0:
                    M[k, a, b] = np.dot(localframe[2][a], localframe[2][b]) - np.dot(localframe[2][a],q[k]) * np.dot(localframe[2][b],q[k])/ np.dot(q[k],q[k])
                else:
                    M[k, a, b] = np.dot(localframe[2][a], localframe[2][b])
    return M

def SSSF_q(k, S, P, gb=False):
    if gb:
        A = Spin_global_pyrochlore(k, S, P)
        read = np.abs(contract('nia, mib, inmab->inmab', A, np.conj(A), gg(k)))
        read = np.where(read <= 1e-8, 1e-8, read)
        return read
    else:
        A = Spin_global_pyrochlore(k, S, P)
        read = np.abs(contract('nia, mib->inmab', A, np.conj(A)))
        read = np.where(read <= 1e-8, 1e-8, read)
        return read

def DSSF(w, k, S, P, T, gb=False):
    dT = T[1]-T[0]
    A = Spin_global_pyrochlore_t(k, S, P)  # shape: (time, 4, k_points, 3)
    
    # Use FFT for time transform
    # If T spans from -t to +t, need to shift data for proper FFT
    nT = len(T)
    nT_padded = 2**int(np.ceil(np.log2(nT)))
    
    # Check if time array includes negative times (spans from -t to +t)
    # If so, shift the data so t=0 is at the beginning for FFT
    if T[0] < 0 and T[-1] > 0:
        # Data spans from -t to +t, use fftshift/ifftshift
        A_shifted = np.fft.ifftshift(A, axes=0)
        A_fft = np.fft.fft(A_shifted, n=nT_padded, axis=0)
        A_fft = np.fft.fftshift(A_fft, axes=0)
        freq = np.fft.fftshift(np.fft.fftfreq(nT_padded, d=dT))
    else:
        # Data starts from t=0 or is all positive, use standard FFT
        A_fft = np.fft.fft(A, n=nT_padded, axis=0)
        freq = np.fft.fftfreq(nT_padded, d=dT)
    
    # If w is None, return the FFT result with corresponding frequencies (no interpolation)
    if w is None:
        Somega = dT / (2*np.pi) * A_fft / np.sqrt(nT)
        if gb:
            read = np.abs(contract('wnia, wmib, inm->winmab', Somega, np.conj(Somega), ggz(k)))
            read = np.where(read <= 1e-8, 1e-8, read)
        else:
            read = np.abs(contract('wnia, wmib->winmab', Somega, np.conj(Somega)))
            read = np.where(read <= 1e-8, 1e-8, read)
        return freq, read
    
    # Otherwise, interpolate to the desired frequency grid
    Somega = np.zeros((len(w), A.shape[1], A.shape[2], A.shape[3]), dtype=np.complex128)
    for i in range(A.shape[1]):
        for j in range(A.shape[2]):
            for l in range(A.shape[3]):
                # Sort frequencies for interpolation
                idx = np.argsort(freq)
                interp_real = interp1d(freq[idx], A_fft[idx, i, j, l].real, kind='cubic', fill_value=0, bounds_error=False)
                interp_imag = interp1d(freq[idx], A_fft[idx, i, j, l].imag, kind='cubic', fill_value=0, bounds_error=False)
                Somega[:, i, j, l] = interp_real(w) + 1j * interp_imag(w)
    
    Somega = dT / (2*np.pi) * Somega / np.sqrt(nT) 
    
    if gb:
        read = np.abs(contract('wnia, wmib, inm, w->winmab', Somega, np.conj(Somega), ggz(k), w))
        read = np.where(read <= 1e-8, 1e-8, read)
        return read
    else:
        read = np.abs(contract('wnia, wmib, w->winmab', Somega, np.conj(Somega), w))
        read = np.where(read <= 1e-8, 1e-8, read)
        return read


def SSSFGraphHHL(A,B,d1, filename):
    fig, ax = plt.subplots(figsize=(10,4))
    C = ax.pcolormesh(A,B, d1)
    fig.colorbar(C)
    ax.set_ylabel(r'$(0,0,L)$')
    ax.set_xlabel(r'$(H,H,0)$')
    fig.savefig(filename+".pdf")
    fig.clf()
    plt.close()

def SSSFGraphHnHL(A,B,d1, filename):
    fig, ax = plt.subplots(figsize=(10,4))
    C = ax.pcolormesh(A,B, d1)
    fig.colorbar(C)
    ax.set_ylabel(r'$(0,0,L)$')
    ax.set_xlabel(r'$(H,-H,0)$')
    fig.savefig(filename+".pdf")
    fig.clf()
    plt.close()

def SSSFGraphHK0(A,B,d1, filename):
    fig, ax = plt.subplots(figsize=(10,4))
    C = ax.pcolormesh(A,B, d1)
    fig.colorbar(C)
    ax.set_ylabel(r'$(0,0,L)$')
    ax.set_xlabel(r'$(H,-H,0)$')
    fig.savefig(filename+".pdf")
    fig.clf()
    plt.close()

def SSSFGraphHnHn(A,B,d1, filename):
    fig, ax = plt.subplots(figsize=(10,4))
    C = ax.pcolormesh(A,B, d1)
    fig.colorbar(C)
    ax.set_ylabel(r'$(K,K,2K)$')
    ax.set_xlabel(r'$(H,-H,0)$')
    fig.savefig(filename+".pdf")
    fig.clf()
    plt.close()

def SSSFGraph2D(A, B, d1, filename):
    plt.pcolormesh(A, B, d1)
    plt.colorbar()
    plt.ylabel(r'$K_y$')
    plt.xlabel(r'$K_x$')
    plt.savefig(filename + ".pdf")
    plt.clf()


def hhltoK(H, L, K=0):
    A = contract('ij,k->ijk',H, 2*np.array([np.pi,np.pi,0])) \
        + contract('ij,k->ijk',L, 2*np.array([0,0,np.pi]))
    return A

def hnhltoK(H, L, K=0):
    A = contract('ij,k->ijk',H, 2*np.array([np.pi,-np.pi,0])) \
        + contract('ij,k->ijk',L, 2*np.array([0,0,np.pi]))
    return A

def hhztoK(H, K):
    return contract('ij,k->ijk',H, 2*np.array([np.pi,0,0])) + contract('ij,k->ijk',K, 2*np.array([0,np.pi,0]))

def hnhztoK(H, K):
    return contract('ij,k->ijk',H, 2*np.pi*np.array([1,-1,0])) + contract('ij,k->ijk',K, 2*np.pi*np.array([1,1,-2]))

def hk2d(H,K):
    return contract('ij,k->ijk',H, 2*np.array([np.pi,0])) + contract('ij,k->ijk',K, 2*np.array([0,np.pi]))

def SSSF2D(S, P, nK, filename, gb=False):
    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hk2d(A, B).reshape((nK*nK,2))
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

def SSSFHHL(S, P, nK, filename, gb=False):
    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hhltoK(A, B).reshape((nK*nK,3))
    S = SSSF_q(K, S, P, gb)
    S = S.reshape((nK, nK, 4,4, 3, 3))
    return S

def SSSFHnHL(S, P, nK, filename, gb=False):
    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hnhltoK(A, B).reshape((nK*nK,3))
    S = SSSF_q(K, S, P, gb)
    S = S.reshape((nK, nK, 4,4, 3, 3))
    return S

def SSSFHK0(S, P, nK, filename, gb=False):
    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hhztoK(A, B).reshape((nK*nK,3))
    S = SSSF_q(K, S, P, gb)
    S = S.reshape((nK, nK, 4,4, 3, 3))
    return S

def SSSFHnHn(S, P, nK, filename, gb=False):
    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hnhztoK(A, B).reshape((nK*nK,3))
    S = SSSF_q(K, S, P, gb)
    S = S.reshape((nK, nK, 4,4, 3, 3))
    return S

def genALLSymPointsBare():
    d = 9 * 1j
    b = np.mgrid[0:1:d, 0:1:d, 0:1:d].reshape(3, -1).T
    return b
BasisBZA = np.array([2*np.pi*np.array([-1,1,1]),2*np.pi*np.array([1,-1,1]),2*np.pi*np.array([1,1,-1])])
BasisBZA_reverse = np.array([np.array([0,1,1]),np.array([1,0,1]),np.array([1,1,0])])/2

BasisBZA_reverse_honeycomb = np.array([[1,1/2],[0, np.sqrt(3)/2]])

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

def magnetization(S, glob, fielddir, theta=0):
    if not glob:
        return np.mean(S,axis=0)
    else:
        size = int(len(S)/4)
        zmag = contract('k,ik->i', fielddir, z)
        mag = np.zeros(3)
        for i in range(4):
            mag = mag + np.mean(S[i*size:(i+1)*size, 2]*np.cos(theta)+S[i*size:(i+1)*size, 0]*np.cos(theta), axis=0)*zmag[i]
        return mag
            


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
            newdir = dir + filename[:-3] 
            # if not os.path.isdir(newdir):
            #     os.mkdir(newdir)
            if P.shape[1] == 2:
                SSSF2D(S,P, 100, newdir, gb)
            elif mag == "001":
                SSSFHK0(S, P, 50, newdir, gb)
            else:
                SSSFHHL(S, P, 50, newdir, gb)
        if filename.endswith(".h5") and filename.endswith("time_evolved.h5"):
            print(filename)
            f = h5py.File(dir + filename, 'r')
            S = f['spins'][:]
            P = f['site_positions'][:]
            T = f['t'][:]
            w0 = 0
            wmax = 2.5
            w = np.linspace(w0, wmax, 200)
            A = DSSF(w, DSSF_K, S, P, T, gb)
            A = A/np.max(A)
            if not gb:
                np.savetxt(dir+filename[:-3]+"_Sxx_local.txt", A[:,0,0])
                np.savetxt(dir+filename[:-3]+"_Syy_local.txt", A[:,1,1])
                np.savetxt(dir+filename[:-3]+"_Szz_local.txt", A[:,2,2])
            else:
                np.savetxt(dir+filename[:-3]+"_Sxx_global.txt", A[:,0,0])
                np.savetxt(dir+filename[:-3]+"_Syy_global.txt", A[:,1,1])
                np.savetxt(dir+filename[:-3]+"_Szz_global.txt", A[:,2,2])

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

# def parseDSSF(dir):
#     directory = os.fsencode(dir)
    
#     def DSSFHelper(name):
#         size = 0
#         for file in os.listdir(directory):
#             filename = os.fsdecode(file)
#             if filename.endswith("time_evolved_"+name+".txt"):
#                 test = np.loadtxt(dir+filename)
#                 size = test.shape
#                 break
#         A = np.zeros(size)

#         for file in os.listdir(directory):
#             filename = os.fsdecode(file)
#             if filename.endswith("time_evolved_"+name+".txt"):
#                 print(filename)
#                 A = A + np.loadtxt(dir+filename)
#         A = A / np.max(A)
#         fig, ax = plt.subplots(figsize=(10,4))

#         C = ax.imshow(A, origin='lower', extent=[0, gGamma3, 0, 2.5], aspect='auto', interpolation='lanczos', cmap='gnuplot2')
#         ax.axvline(x=gGamma1, color='b', label='axvline - full height', linestyle='dashed')
#         ax.axvline(x=gX, color='b', label='axvline - full height', linestyle='dashed')
#         ax.axvline(x=gW, color='b', label='axvline - full height', linestyle='dashed')
#         ax.axvline(x=gK, color='b', label='axvline - full height', linestyle='dashed')
#         ax.axvline(x=gGamma2, color='b', label='axvline - full height', linestyle='dashed')
#         ax.axvline(x=gL, color='b', label='axvline - full height', linestyle='dashed')
#         ax.axvline(x=gU, color='b', label='axvline - full height', linestyle='dashed')
#         ax.axvline(x=gW1, color='b', label='axvline - full height', linestyle='dashed')
#         ax.axvline(x=gX1, color='b', label='axvline - full height', linestyle='dashed')
#         ax.axvline(x=gGamma3, color='b', label='axvline - full height', linestyle='dashed')
#         xlabpos = [gGamma1, gX, gW, gK, gGamma2, gL, gU, gW1, gX1, gGamma3]
#         labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W^\prime$', r'$X^\prime$',
#                     r'$\Gamma$']
#         ax.set_xticks(xlabpos, labels)
#         ax.set_xlim([0, gGamma3])
#         fig.colorbar(C)
#         plt.savefig(dir+"DSSF.pdf")
#         plt.clf()
    
#     DSSFHelper("Sxx_local")
#     DSSFHelper("Syy_local")
#     DSSFHelper("Szz_local")
#     DSSFHelper("Sxx_global")
#     DSSFHelper("Syy_global")
#     DSSFHelper("Szz_global")

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

    C = ax.imshow(A, origin='lower', extent=[0, gGamma3, 0, 2.5], aspect='auto', interpolation='lanczos', cmap='gnuplot2')
    ax.axvline(x=gGamma1, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gX, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gW, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gK, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gGamma2, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gL, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gU, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gW1, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gX1, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gGamma3, color='b', label='axvline - full height', linestyle='dashed')
    xlabpos = [gGamma1, gX, gW, gK, gGamma2, gL, gU, gW1, gX1, gGamma3]
    labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W^\prime$', r'$X^\prime$',
                r'$\Gamma$']
    ax.set_xticks(xlabpos, labels)
    ax.set_xlim([0, gGamma3])
    fig.colorbar(C)
    plt.savefig(dir+"DSSF.pdf")
    plt.clf()


def read_MD_tot(dir, mag, SSSFGraph):
    directory = os.fsencode(dir)
    nK = 50
    S_local = np.zeros((nK,nK,4,4,3,3))
    S_global = np.zeros((nK,nK,4,4,3,3))
    w0 = 1e-2
    wmax = 10
    w = np.arange(w0, wmax, 1/600)
    DSSF_local = np.zeros((len(w), len(DSSF_K),4,4,3,3))
    DSSF_global = np.zeros((len(w), len(DSSF_K),4,4,3,3))
    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if os.path.isdir(dir + "/" + filename) and filename != "results":
            Sl, Sg, Dl, Dg = read_MD(dir + "/" + filename, mag, w)
            S_local = S_local + Sl
            S_global = S_global + Sg
            DSSF_local = DSSF_local + Dl
            DSSF_global = DSSF_global + Dg
    S_local = S_local / len(os.listdir(directory))
    S_global = S_global / len(os.listdir(directory))
    DSSF_local = DSSF_local / len(os.listdir(directory))
    DSSF_global = DSSF_global / len(os.listdir(directory))

    def SSSF_helper(S_local, S_global, filename):
        SSSFGraph(A, B, S_local[:,:,0,0], filename + "/Sxx_local")
        SSSFGraph(A, B, S_local[:,:,1,1], filename + "/Syy_local")
        SSSFGraph(A, B, S_local[:,:,2,2], filename + "/Szz_local")
        SSSFGraph(A, B, S_local[:, :, 0, 1], filename + "/Sxy_local")
        SSSFGraph(A, B, S_local[:, :, 0, 2], filename + "/Sxz_local")
        SSSFGraph(A, B, S_local[:, :, 1, 2], filename + "/Syz_local")
        SSSFGraph(A, B, S_global[:,:,0,0], filename + "/Sxx_global")
        SSSFGraph(A, B, S_global[:,:,1,1], filename + "/Syy_global")
        SSSFGraph(A, B, S_global[:,:,2,2], filename + "/Szz_global")
        SSSFGraph(A, B, S_global[:, :, 0, 1], filename + "/Sxy_global")
        SSSFGraph(A, B, S_global[:, :, 0, 2], filename + "/Sxz_global")
        SSSFGraph(A, B, S_global[:, :, 1, 2], filename + "/Syz_global")

    def DSSF_helper(DSSF, filename):
        # DSSF = DSSF / np.max(DSSF)
        fig, ax = plt.subplots(figsize=(10,4))
        C = ax.imshow(DSSF, origin='lower', extent=[0, gGamma3, w0, wmax], aspect='auto', interpolation='lanczos', cmap='gnuplot2')
        ax.axvline(x=gGamma1, color='b', label='axvline - full height', linestyle='dashed')
        ax.axvline(x=gX, color='b', label='axvline - full height', linestyle='dashed')
        ax.axvline(x=gW, color='b', label='axvline - full height', linestyle='dashed')
        ax.axvline(x=gK, color='b', label='axvline - full height', linestyle='dashed')
        ax.axvline(x=gGamma2, color='b', label='axvline - full height', linestyle='dashed')
        ax.axvline(x=gL, color='b', label='axvline - full height', linestyle='dashed')
        ax.axvline(x=gU, color='b', label='axvline - full height', linestyle='dashed')
        ax.axvline(x=gW1, color='b', label='axvline - full height', linestyle='dashed')
        ax.axvline(x=gX1, color='b', label='axvline - full height', linestyle='dashed')
        ax.axvline(x=gGamma3, color='b', label='axvline - full height', linestyle='dashed')
        xlabpos = [gGamma1, gX, gW, gK, gGamma2, gL, gU, gW1, gX1, gGamma3]
        labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W^\prime$', r'$X^\prime$',
                    r'$\Gamma$']
        ax.set_xticks(xlabpos, labels)
        ax.set_xlim([0, gGamma3])
        fig.colorbar(C)
        plt.savefig(filename+".pdf")
        plt.clf()
        plt.close()

    def DSSF_all_spin_components(DSSF, filename):
        com_string = np.array(["x", "y", "z"])
        for i in range(3):
            for j in range(3):
                DSSF_helper(DSSF[:,:,i,j], filename + "_"+ com_string[i]+com_string[j])
        DSSF_helper(np.sum(DSSF, axis=(2,3)), filename + "_sum")

    dir_to_save = dir + "/results"
    if not os.path.isdir(dir_to_save):
        os.mkdir(dir_to_save)
    if not os.path.isdir(dir_to_save + "/SSSF"):
        os.mkdir(dir_to_save + "/SSSF")
    if not os.path.isdir(dir_to_save + "/DSSF_local"):
        os.mkdir(dir_to_save + "/DSSF_local")
    if not os.path.isdir(dir_to_save + "/DSSF_global"):
        os.mkdir(dir_to_save + "/DSSF_global")
    S_global = np.log(S_global)
    S_local = np.log(S_local)
    DSSF_local = np.log(DSSF_local)
    DSSF_global = np.log(DSSF_global)
    SSSF_helper(contract('ijxyab->ijab', S_local), contract('ijxyab->ijab', S_global), dir_to_save + "/SSSF")
    DSSF_all_spin_components(contract('ijxyab->ijab',DSSF_local), dir_to_save + "/DSSF_local/")
    DSSF_all_spin_components(contract('ijxyab->ijab',DSSF_global), dir_to_save + "/DSSF_global/")

def generate_K_points_pengcheng_dai(H_range_min, H_range_max, nH, K_range_min, K_range_max, nK, L_range_min, L_range_max, nL):
    # H_vector = 2*np.pi*np.array([1, 1, -2])
    # K_vector = 2*np.pi*np.array([1, -1, 0])
    # L_vector = 2*np.pi*np.array([1, 1, 1])

    H_vector = 2*np.pi*np.array([1, 0, 0])
    K_vector = 2*np.pi*np.array([0, 1, 0])
    L_vector = 2*np.pi*np.array([0, 0, 1])

    # Create coefficient ranges
    h_values = np.linspace(H_range_min, H_range_max, nH)
    k_values = np.linspace(K_range_min, K_range_max, nK)
    l_values = np.linspace(L_range_min, L_range_max, nL)

    print(f"Generating K points with H from {H_range_min} to {H_range_max} ({nH} points), "
          f"K from {K_range_min} to {K_range_max} ({nK} points), "
          f"L from {L_range_min} to {L_range_max} ({nL} points).")
    
    # Calculate the actual k-space range covered
    # Get corner points of the box in HKL space
    corners_hkl = [
        [H_range_min, K_range_min, L_range_min],
        [H_range_min, K_range_min, L_range_max],
        [H_range_min, K_range_max, L_range_min],
        [H_range_min, K_range_max, L_range_max],
        [H_range_max, K_range_min, L_range_min],
        [H_range_max, K_range_min, L_range_max],
        [H_range_max, K_range_max, L_range_min],
        [H_range_max, K_range_max, L_range_max]
    ]
    
    # Transform corners to k-space
    k_corners = []
    for h, k, l in corners_hkl:
        k_point = h * H_vector + k * K_vector + l * L_vector
        k_corners.append(k_point)
    
    k_corners = np.array(k_corners)
    
    # Find min and max in each k-space direction
    k_min = np.min(k_corners, axis=0)
    k_max = np.max(k_corners, axis=0)
    
    print(f"Actual k-space range:")
    print(f"  kx: [{k_min[0]:.4f}, {k_max[0]:.4f}]")
    print(f"  ky: [{k_min[1]:.4f}, {k_max[1]:.4f}]")
    print(f"  kz: [{k_min[2]:.4f}, {k_max[2]:.4f}]")

    # Create a grid of all possible combinations
    h_grid, k_grid, l_grid = np.meshgrid(h_values, k_values, l_values, indexing='ij')
    h_grid = h_grid.flatten()
    k_grid = k_grid.flatten()
    l_grid = l_grid.flatten()
    
    # Calculate K points using linear combinations
    K_points = np.zeros((len(h_grid), 3))
    for i in range(len(h_grid)):
        K_points[i] = h_grid[i] * H_vector + k_grid[i] * K_vector + l_grid[i] * L_vector
    # Calculate the volume element dV
    dV = np.abs(np.linalg.det(np.array([H_vector, K_vector, L_vector]))) / (nH * nK * nL)

    return K_points, dV

def gaussian_resolution(energy, intensity, resolution_fwhm):
    """
    Convolve intensity with Gaussian experimental resolution.
    
    Parameters:
    - energy: array of energy values
    - intensity: array of intensity values
    - resolution_fwhm: Full Width at Half Maximum of the Gaussian resolution function
    """
    # Calculate energy spacing
    energy_spacing = np.mean(np.diff(energy))
    
    # Convert FWHM to standard deviation
    sigma = resolution_fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    # Convert sigma from energy units to array index units
    sigma_indices = sigma / energy_spacing
    
    # Apply Gaussian convolution
    convolved_intensity = gaussian_filter1d(intensity, sigma_indices)
    
    return convolved_intensity



def read_MD_int(dir, mag, SSSFGraph=None, run_ids=None):
    directory = os.fsencode(dir)
    w0 = 0.03
    wmax = 15
    K_, dV = generate_K_points_pengcheng_dai(0, 0, 1, 0, 0, 1, 1, 1, 1)

    DSSF_local = []
    DSSF_global = []
    nK = 50


    SSSF_local = np.zeros((nK,nK,4,4,3,3))
    SSSF_global = np.zeros((nK,nK,4,4,3,3))

    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)

    processed_runs = 0
    run_filter = None
    if run_ids is not None:
        run_filter = {str(r) for r in run_ids}

    for entry in sorted(os.listdir(directory)):
        run_dir = os.fsdecode(entry)
        run_path = os.path.join(dir, run_dir)
        if run_dir == "results" or not os.path.isdir(run_path):
            continue
        if run_filter is not None and run_dir not in run_filter:
            continue
        try:
            P = np.loadtxt(os.path.join(run_path, "pos.txt"))
            T = np.loadtxt(os.path.join(run_path, "Time_steps.txt"))
            S = np.loadtxt(os.path.join(run_path, "spin_t.txt")).reshape((len(T), len(P), 3))
            S0 = np.loadtxt(os.path.join(run_path, "spin_0.txt")).reshape((len(P), 3))
        except OSError as exc:
            print(f"[read_MD_int] Failed to read data for run {run_dir} in {dir}: {exc}")
            continue

        # Compute DSSF for this run (returns data with frequency from FFT)
        dssf_local_run = DSSF(None, K_, S, P, T, False)  # Returns (freq, K, 4, 4, 3, 3) with freq
        dssf_global_run = DSSF(None, K_, S, P, T, True)
        
        DSSF_local.append(dssf_local_run)
        DSSF_global.append(dssf_global_run)
        SSSF_local = SSSF_local + SSSFHnHn(S0, P, 50, run_path, False)
        SSSF_global = SSSF_global + SSSFHnHn(S0, P, 50, run_path, True)
        processed_runs += 1

    if processed_runs == 0:
        raise RuntimeError(f"No simulation data could be processed in {dir}.")

    SSSF_local = SSSF_local / processed_runs
    SSSF_global = SSSF_global / processed_runs
    
    # Average DSSF data and filter to [w0, wmax] window
    # Each element in DSSF_local/global is (freq, K, 4, 4, 3, 3)
    # Average across all runs
    freq_arrays = []
    dssf_local_arrays = []
    dssf_global_arrays = []
    
    for freq, dssf_data in DSSF_local:
        freq_arrays.append(freq)
        dssf_local_arrays.append(dssf_data)
    
    for freq, dssf_data in DSSF_global:
        dssf_global_arrays.append(dssf_data)
    
    # Create a common frequency grid for interpolation
    # Find the union of all frequency points in the [w0, wmax] window
    all_freqs = []
    for freq in freq_arrays:
        freq_mask = (freq >= w0) & (freq <= wmax)
        all_freqs.extend(freq[freq_mask])
    
    # Create a uniform frequency grid spanning [w0, wmax]
    # Use the median spacing to determine the number of points
    all_freqs = np.sort(np.unique(all_freqs))
    if len(all_freqs) > 1:
        median_spacing = np.median(np.diff(all_freqs))
        n_points = int((wmax - w0) / median_spacing) + 1
    else:
        n_points = 1024  # default fallback
    
    w = np.linspace(w0, wmax, n_points)
    
    # Interpolate each run's data to the common frequency grid
    dssf_local_filtered_list = []
    dssf_global_filtered_list = []
    
    for freq, dssf_data in DSSF_local:
        freq_mask = (freq >= w0) & (freq <= wmax)
        freq_filtered = freq[freq_mask]
        dssf_filtered = dssf_data[freq_mask]
        
        # Interpolate to common grid
        # dssf_data shape: (freq, 4, k_points, 4, 3, 3)
        dssf_interp = np.zeros((len(w),) + dssf_filtered.shape[1:])
        for i in range(dssf_filtered.shape[1]):
            for j in range(dssf_filtered.shape[2]):
                for k in range(dssf_filtered.shape[3]):
                    for a in range(dssf_filtered.shape[4]):
                        for b in range(dssf_filtered.shape[5]):
                            interp_func = interp1d(freq_filtered, dssf_filtered[:, i, j, k, a, b], 
                                                   kind='linear', fill_value='extrapolate', bounds_error=False)
                            dssf_interp[:, i, j, k, a, b] = interp_func(w)
        dssf_local_filtered_list.append(dssf_interp)
    
    for freq, dssf_data in DSSF_global:
        freq_mask = (freq >= w0) & (freq <= wmax)
        freq_filtered = freq[freq_mask]
        dssf_filtered = dssf_data[freq_mask]
        
        # Interpolate to common grid
        dssf_interp = np.zeros((len(w),) + dssf_filtered.shape[1:])
        for i in range(dssf_filtered.shape[1]):
            for j in range(dssf_filtered.shape[2]):
                for k in range(dssf_filtered.shape[3]):
                    for a in range(dssf_filtered.shape[4]):
                        for b in range(dssf_filtered.shape[5]):
                            interp_func = interp1d(freq_filtered, dssf_filtered[:, i, j, k, a, b], 
                                                   kind='linear', fill_value='extrapolate', bounds_error=False)
                            dssf_interp[:, i, j, k, a, b] = interp_func(w)
        dssf_global_filtered_list.append(dssf_interp)
    
    # Average the interpolated DSSF data
    DSSF_local_filtered = np.mean(dssf_local_filtered_list, axis=0)
    DSSF_global_filtered = np.mean(dssf_global_filtered_list, axis=0)
    
    DSSF_global = np.sum(contract('ijxyab->ijab', DSSF_global_filtered), axis=1) * dV
    DSSF_local = np.sum(contract('ijxyab->ijab', DSSF_local_filtered), axis=1) * dV

    # Plot all components of DSSF_global, DSSF_local and their sum
    def plot_DSSF_components(DSSF, w, dir, name):
        # Extend w to negative values and DSSF with zeros
        w_negative = -np.flip(w)
        w_extended = np.concatenate((w_negative, w))
        
        DSSF_extended = np.zeros((len(w_extended), 3, 3))
        DSSF_extended[len(w):, :, :] = DSSF

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        com_string = ['x', 'y', 'z']
        
        for i in range(3):
            for j in range(3):
                # Apply Gaussian resolution on the extended data
                convolved_DSSF = gaussian_resolution(w_extended, DSSF_extended[:, i, j], 0.04)
                
                # Plot the convolved data
                axes[i, j].plot(w_extended, convolved_DSSF, 'b-', label=f'S_{com_string[i]}{com_string[j]}')
                axes[i, j].set_xlabel('ω')
                axes[i, j].set_ylabel('DSSF')
                axes[i, j].legend()
                axes[i, j].grid(True)
                axes[i, j].set_title(f'S_{com_string[i]}{com_string[j]}')
        
        plt.tight_layout()
        plt.savefig(dir + "/DSSF_all_components_{}.pdf".format(name))
        plt.close()
        
        # Also save the data
        dir_to_save = dir + "/results"
        if not os.path.isdir(dir_to_save):
            os.mkdir(dir_to_save)
        
        for i in range(3):
            for j in range(3):
                convolved_DSSF_component = gaussian_resolution(w_extended, DSSF_extended[:, i, j], 0.04)
                np.savetxt(f"{dir_to_save}/DSSF_{name}_{com_string[i]}{com_string[j]}.txt", np.column_stack((w_extended, convolved_DSSF_component)))

        DSSF_spinon_photon = DSSF_extended[:, 0, 0] + DSSF_extended[:, 2, 2]
        convolved_DSSF_spinon_photon = gaussian_resolution(w_extended, DSSF_spinon_photon, 0.04)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(w_extended, convolved_DSSF_spinon_photon, 'r-', label='S_xx + S_zz')
        ax.plot(w_extended, gaussian_resolution(w_extended, DSSF_extended[:, 0, 0], 0.04), 'b--', label='S_xx')
        ax.plot(w_extended, gaussian_resolution(w_extended, DSSF_extended[:, 2, 2], 0.04), 'g--', label='S_zz')
        ax.set_xlabel('ω')
        ax.set_ylabel('DSSF Spinon + Photon')
        ax.legend()
        ax.grid(True)
        plt.savefig(dir + "/DSSF_spinon_photon_{}.pdf".format(name))
        plt.close()

        DSSF_sum = np.sum(DSSF_extended, axis=(1, 2))
        convolved_DSSF_sum = gaussian_resolution(w_extended, DSSF_sum, 0.04)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(w_extended, convolved_DSSF_sum, 'r-', label='Sum of all components')
        ax.set_xlabel('ω')
        ax.set_ylabel('DSSF Sum')
        ax.legend()
        ax.grid(True)
        plt.savefig(dir + "/DSSF_sum_{}.pdf".format(name))
        plt.close()

        np.savetxt(f"{dir_to_save}/DSSF_{name}_sum.txt", np.column_stack((w_extended, convolved_DSSF_sum)))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(w_extended, np.log(np.maximum(convolved_DSSF_sum, 1e-9)), 'r-', label='Sum of all components (log scale)')
        ax.set_xlabel('ω')
        ax.set_ylabel('log(DSSF Sum)')
        ax.legend()
        ax.grid(True)
        plt.savefig(dir + "/DSSF_sum_{}_log.pdf".format(name))
        plt.close()
    w = w * 0.063
    plot_DSSF_components(DSSF_global, w, dir, "global")
    plot_DSSF_components(DSSF_local, w, dir, "local")
    # SSSF_helper(contract('ijxyab->ijab', SSSF_local), contract('ijxyab->ijab', SSSF_global), dir + "/results/SSSF")


def read_MD_pi_flux_all(root_dir, mag="HnHn", SSSFGraph=None, verbose=True, run_ids=None):
    """Run read_MD_int for every field directory, averaging over all available trials."""
    base_dir = os.path.abspath(root_dir)
    if not os.path.isdir(base_dir):
        raise ValueError(f"Provided root_dir {root_dir} is not a directory.")

    # Directories to skip (output/metadata directories, not field data)
    skip_dirs = {'animations', 'comparison_plots', 'results', '__pycache__', '.git'}
    
    field_dirs = sorted(
        entry for entry in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, entry)) and entry not in skip_dirs
    )

    if not field_dirs:
        raise RuntimeError(f"No field directories found in {root_dir}.")

    for field_dir in field_dirs:
        field_path = os.path.join(base_dir, field_dir)
        if verbose:
            print(f"[read_MD_pi_flux_all] Processing {field_dir} with average over trials.")
        try:
            read_MD_int(field_path, mag, SSSFGraph, run_ids=run_ids)
        except RuntimeError as e:
            if verbose:
                print(f"[read_MD_pi_flux_all] Warning: Skipping {field_dir}: {e}")
            continue


def extract_field_value(field_dir):
    """Extract numerical field value from directory name like 'field_0.1' or 'h=0.1'"""
    # Try different patterns
    patterns = [r'field[_=]([0-9.]+)', r'h[_=]([0-9.]+)', r'B[_=]([0-9.]+)']
    for pattern in patterns:
        match = re.search(pattern, field_dir)
        if match:
            return float(match.group(1))
    return None


def collect_DSSF_data_all_fields(root_dir, verbose=True):
    """
    Collect DSSF spinon+photon data (local and global) from all field directories.
    Returns: field_values, w_data, dssf_spinon_photon_local_data, dssf_spinon_photon_global_data
    """
    base_dir = os.path.abspath(root_dir)
    if not os.path.isdir(base_dir):
        raise ValueError(f"Provided root_dir {root_dir} is not a directory.")

    # Directories to skip (output/metadata directories, not field data)
    skip_dirs = {'animations', 'comparison_plots', 'results', '__pycache__', '.git'}
    
    field_dirs = sorted(
        entry for entry in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, entry)) and entry not in skip_dirs
    )

    if not field_dirs:
        raise RuntimeError(f"No field directories found in {root_dir}.")

    # Collect data from all fields
    field_data = []
    
    for field_dir in field_dirs:
        field_path = os.path.join(base_dir, field_dir)
        results_dir = os.path.join(field_path, "results")
        
        # Extract field value
        field_val = extract_field_value(field_dir)
        if field_val is None:
            if verbose:
                print(f"[collect_DSSF_data] Could not extract field value from {field_dir}, skipping.")
            continue
        
        # Look for DSSF data files
        local_file = os.path.join(results_dir, "DSSF_local_xx.txt")
        local_zz_file = os.path.join(results_dir, "DSSF_local_zz.txt")
        global_file = os.path.join(results_dir, "DSSF_global_xx.txt")
        global_zz_file = os.path.join(results_dir, "DSSF_global_zz.txt")
        
        if not (os.path.exists(local_file) and os.path.exists(local_zz_file) and 
                os.path.exists(global_file) and os.path.exists(global_zz_file)):
            if verbose:
                print(f"[collect_DSSF_data] Missing DSSF files in {field_dir}, skipping.")
            continue
        
        try:
            # Read the data
            data_local_xx = np.loadtxt(local_file)
            data_local_zz = np.loadtxt(local_zz_file)
            data_global_xx = np.loadtxt(global_file)
            data_global_zz = np.loadtxt(global_zz_file)
            
            w = data_local_xx[:, 0]  # frequency/energy axis
            
            # Compute spinon+photon (S_xx + S_zz)
            dssf_local_spinon_photon = data_local_xx[:, 1] + data_local_zz[:, 1]
            dssf_global_spinon_photon = data_global_xx[:, 1] + data_global_zz[:, 1]
            
            field_data.append({
                'field': field_val,
                'w': w,
                'dssf_local_spinon_photon': dssf_local_spinon_photon,
                'dssf_global_spinon_photon': dssf_global_spinon_photon,
                'dssf_local_xx': data_local_xx[:, 1],
                'dssf_local_zz': data_local_zz[:, 1],
                'dssf_global_xx': data_global_xx[:, 1],
                'dssf_global_zz': data_global_zz[:, 1]
            })
            
            if verbose:
                print(f"[collect_DSSF_data] Loaded data for {field_dir} (field={field_val})")
                
        except Exception as e:
            if verbose:
                print(f"[collect_DSSF_data] Error reading data from {field_dir}: {e}")
            continue
    
    if not field_data:
        raise RuntimeError(f"No valid DSSF data found in {root_dir}")
    
    # Sort by field value
    field_data.sort(key=lambda x: x['field'])
    
    return field_data


def animate_DSSF_spinon_photon(root_dir, output_dir=None, fps=0.5, energy_conversion=0.063):
    """
    Create animations for DSSF spinon+photon (S_xx + S_zz) across all magnetic field strengths.
    Creates separate animations for local and global channels.
    
    Parameters:
    -----------
    root_dir : str
        Root directory containing field subdirectories
    output_dir : str, optional
        Directory to save animations (default: root_dir/animations)
    fps : int
        Frames per second for animation
    energy_conversion : float
        Conversion factor to real energy units (meV), default 0.063
    """
    # Collect all data
    print("[animate_DSSF_spinon_photon] Collecting data from all field directories...")
    field_data = collect_DSSF_data_all_fields(root_dir, verbose=True)
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(root_dir, "animations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data arrays
    field_values = np.array([d['field'] for d in field_data]) * energy_conversion / (2.5 * 0.0578) # Convert to meV
    n_fields = len(field_values)
    
    # Check if all w arrays have the same length; if not, interpolate to a common grid
    w_lengths = [len(d['w']) for d in field_data]
    if len(set(w_lengths)) > 1:
        print(f"[animate_DSSF_spinon_photon] Warning: Different frequency grid sizes detected: {set(w_lengths)}")
        print("[animate_DSSF_spinon_photon] Interpolating all data to common frequency grid...")
        
        # Use the longest w array as the reference
        max_idx = np.argmax(w_lengths)
        w_common = field_data[max_idx]['w']
        
        # Interpolate all data to common grid
        for i, data in enumerate(field_data):
            if len(data['w']) != len(w_common):
                # Create interpolation functions
                data['dssf_local_spinon_photon'] = np.interp(w_common, data['w'], data['dssf_local_spinon_photon'])
                data['dssf_global_spinon_photon'] = np.interp(w_common, data['w'], data['dssf_global_spinon_photon'])
                data['dssf_local_xx'] = np.interp(w_common, data['w'], data['dssf_local_xx'])
                data['dssf_local_zz'] = np.interp(w_common, data['w'], data['dssf_local_zz'])
                data['dssf_global_xx'] = np.interp(w_common, data['w'], data['dssf_global_xx'])
                data['dssf_global_zz'] = np.interp(w_common, data['w'], data['dssf_global_zz'])
                data['w'] = w_common
        
        w = w_common * energy_conversion  # Convert to meV
    else:
        # All have same w array
        w = field_data[0]['w'] * energy_conversion  # Convert to meV
    
    # Stack all DSSF data (now they all have the same length)
    dssf_local_stack = np.array([d['dssf_local_spinon_photon'] for d in field_data])
    dssf_global_stack = np.array([d['dssf_global_spinon_photon'] for d in field_data])
    dssf_local_xx_stack = np.array([d['dssf_local_xx'] for d in field_data])
    dssf_local_zz_stack = np.array([d['dssf_local_zz'] for d in field_data])
    dssf_global_xx_stack = np.array([d['dssf_global_xx'] for d in field_data])
    dssf_global_zz_stack = np.array([d['dssf_global_zz'] for d in field_data])
    
    # Determine global y-axis limits for consistency
    y_max_local = np.max(dssf_local_stack) * 1.1
    y_max_global = np.max(dssf_global_stack) * 1.1
    
    # Animation for LOCAL channel
    print("[animate_DSSF_spinon_photon] Creating animation for LOCAL channel...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    line_total, = ax.plot([], [], 'r-', linewidth=2, label='S_xx + S_zz (Total)')
    line_xx, = ax.plot([], [], 'b--', linewidth=1.5, label='S_xx')
    line_zz, = ax.plot([], [], 'g--', linewidth=1.5, label='S_zz')
    
    ax.set_xlabel('ω (meV)', fontsize=12)
    ax.set_ylabel('DSSF (Local)', fontsize=12)
    ax.set_xlim(w.min(), w.max())
    ax.set_ylim(0, y_max_local)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    title_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, 
                         ha='center', va='top', fontsize=14, fontweight='bold')
    
    def init_local():
        line_total.set_data([], [])
        line_xx.set_data([], [])
        line_zz.set_data([], [])
        title_text.set_text('')
        return line_total, line_xx, line_zz, title_text
    
    def animate_local(frame):
        line_total.set_data(w, dssf_local_stack[frame])
        line_xx.set_data(w, dssf_local_xx_stack[frame])
        line_zz.set_data(w, dssf_local_zz_stack[frame])
        title_text.set_text(f'DSSF Spinon+Photon (Local) | Field = {field_values[frame]:.3f}')
        return line_total, line_xx, line_zz, title_text
    
    anim_local = FuncAnimation(fig, animate_local, init_func=init_local,
                               frames=n_fields, interval=1000//fps, blit=True, repeat=True)
    
    output_file_local = os.path.join(output_dir, 'DSSF_spinon_photon_local.gif')
    anim_local.save(output_file_local, writer=PillowWriter(fps=fps))
    print(f"[animate_DSSF_spinon_photon] Saved: {output_file_local}")
    plt.close(fig)
    
    # Animation for GLOBAL channel
    print("[animate_DSSF_spinon_photon] Creating animation for GLOBAL channel...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    line_total, = ax.plot([], [], 'r-', linewidth=2, label='S_xx + S_zz (Total)')
    line_xx, = ax.plot([], [], 'b--', linewidth=1.5, label='S_xx')
    line_zz, = ax.plot([], [], 'g--', linewidth=1.5, label='S_zz')
    
    ax.set_xlabel('ω (meV)', fontsize=12)
    ax.set_ylabel('DSSF (Global)', fontsize=12)
    ax.set_xlim(w.min(), w.max())
    ax.set_ylim(0, y_max_global)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    title_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, 
                         ha='center', va='top', fontsize=14, fontweight='bold')
    
    def init_global():
        line_total.set_data([], [])
        line_xx.set_data([], [])
        line_zz.set_data([], [])
        title_text.set_text('')
        return line_total, line_xx, line_zz, title_text
    
    def animate_global(frame):
        line_total.set_data(w, dssf_global_stack[frame])
        line_xx.set_data(w, dssf_global_xx_stack[frame])
        line_zz.set_data(w, dssf_global_zz_stack[frame])
        title_text.set_text(f'DSSF Spinon+Photon (Global) | Field = {field_values[frame]:.3f}')
        return line_total, line_xx, line_zz, title_text
    
    anim_global = FuncAnimation(fig, animate_global, init_func=init_global,
                                frames=n_fields, interval=1000//fps, blit=True, repeat=True)
    
    output_file_global = os.path.join(output_dir, 'DSSF_spinon_photon_global.gif')
    anim_global.save(output_file_global, writer=PillowWriter(fps=fps))
    print(f"[animate_DSSF_spinon_photon] Saved: {output_file_global}")
    plt.close(fig)
    
    print("[animate_DSSF_spinon_photon] Animation creation complete!")
    return output_file_local, output_file_global


def read_MD(dir, mag, w):
    directory = os.fsencode(dir)
    P = np.loadtxt(dir + "/pos.txt")
    T = np.loadtxt(dir + "/Time_steps.txt")

    S = np.loadtxt(dir + "/spin_t.txt").reshape((len(T), len(P), 3))
    S0 = np.loadtxt(dir + "/spin_0.txt").reshape((len(P), 3))
    S_local = np.zeros((50,50,4,4,3,3))
    S_global = np.zeros((50,50,4,4,3,3))
    if P.shape[1] == 2:
        SSSF2D(S0,P, 100, dir, True)
    elif mag == "001":
        S_global = SSSFHK0(S0, P, 50, dir, True)
        S_local = SSSFHK0(S0, P, 50, dir, False)
    elif mag == "1-10":
        S_global = SSSFHHL(S0, P, 50, dir, True)
        S_local = SSSFHHL(S0, P, 50, dir, False)
    elif mag == "110":
        S_global = SSSFHnHL(S0, P, 50, dir, True)
        S_local = SSSFHnHL(S0, P, 50, dir, False)
    elif mag == "111":
        S_global = SSSFHnHn(S0, P, 50, dir, True)
        S_local = SSSFHnHn(S0, P, 50, dir, False)
    DSSF_local = DSSF(w, DSSF_K, S, P, T, False)
    DSSF_global = DSSF(w, DSSF_K, S, P, T, True)
    return S_local, S_global, DSSF_local, DSSF_global


def read_0_field(numJpm, dir):
    directory = os.fsencode(dir)
    phase_diagram = np.zeros((numJpm, numJpm))
    Jpms = np.zeros(numJpm)
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if os.path.isdir(dir + "/" + filename):
            info = filename.split("_")
            S = np.loadtxt(dir + "/" + filename + "/spin.txt")
            phase_diagram[int(info[5]), int(info[6])] = np.linalg.norm(magnetization(S, False, np.array([0,0,1])))
            Jpm = -(float(info[1]) + float(info[3]))/4
            Jpmpm = abs(float(info[1]) - float(info[3]))/4
            plt.scatter(Jpm, Jpmpm, c=np.linalg.norm(phase_diagram[int(info[5]), int(info[6])]), vmin=0, vmax=0.5)
    plt.savefig(dir+"phase_diagram_Jpmpm.pdf")
    plt.clf()
    fig, ax = plt.subplots(figsize=(10,4))
    C = ax.imshow(phase_diagram, origin='lower', extent=[-1, 1, -1, 1], aspect='auto', interpolation='lanczos', cmap='gnuplot2')
    fig.colorbar(C)
    plt.savefig(dir+"_phase_diagram.pdf")
    plt.clf()

def read_2D_nonlinear(dir):
    directory = os.fsencode(dir)
    tau_start, tau_end, tau_step, time_start, time_end, time_step = np.loadtxt(dir + "/param.txt")
    M0 = np.loadtxt(dir + "/M_time_0/M0/M_t.txt")[:,2]
    domain = int(len(M0)/2)
    omega_range = 15
    M_NL = np.zeros((int(tau_step), domain))
    dT = (time_end - time_start) / int(time_step)
    T = np.linspace(time_start, time_end, int(time_step)) 
    T = T[-domain:]
    dtau = (tau_end - tau_start) / int(tau_step)
    tau = np.linspace(tau_start, tau_end, int(tau_step))
    
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if os.path.isdir(dir + "/" + filename):
            info = filename.split("_")
            M1 = np.loadtxt(dir + "/" + filename + "/M1/M_t.txt")[:,2]
            M01 = np.loadtxt(dir + "/" + filename + "/M01/M_t.txt")[:,2]
            M_NL[int(info[2])] = M01[-domain:] - M0[-domain:] - M1[-domain:] 
    
    # Use 2D FFT for both tau and time transforms
    # Pad to next power of 2 for efficiency
    n_tau_padded = 2**int(np.ceil(np.log2(len(tau))))
    n_T_padded = 2**int(np.ceil(np.log2(len(T))))
    
    # 2D FFT
    M_NL_fft = np.fft.fft2(M_NL, s=(n_tau_padded, n_T_padded))
    M_NL_fft = np.fft.fftshift(M_NL_fft)
    
    # Generate frequency grids
    freq_tau = np.fft.fftshift(np.fft.fftfreq(n_tau_padded, d=dtau))
    freq_T = np.fft.fftshift(np.fft.fftfreq(n_T_padded, d=dT))
    
    # Find indices corresponding to the desired omega_range
    idx_tau = (freq_tau >= -omega_range) & (freq_tau <= omega_range)
    idx_T = (freq_T >= -omega_range) & (freq_T <= omega_range)
    
    M_NL_FF = np.abs(M_NL_fft[np.ix_(idx_tau, idx_T)])
    M_NL_FF = np.log(np.maximum(M_NL_FF, 1e-10))
    
    np.savetxt(dir + "/M_NL_FF.txt", M_NL_FF)
    plt.imshow(M_NL_FF, origin='lower', extent=[-omega_range, omega_range, -omega_range, omega_range], aspect='auto', interpolation='lanczos', cmap='gnuplot2', norm='linear')
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


def QFI_calculate(beta, K, S, P, T, dir, glob=False):
    omega = np.linspace(0, 8, 1000)
    toint = DSSF(omega, K, S, P, T, glob)
    toint = contract('wixyab, w->wixyab' ,toint,4*(1-np.exp(-omega*beta)) * np.tanh(omega/2*beta))
    toint = np.trapz(toint, x=omega, axis=0)
    return toint

def QFI(dir, K, beta):
    directory = os.fsencode(dir)
    QFI_local = np.zeros((len(K), 4, 4, 3, 3))
    QFI_global = np.zeros((len(K), 4, 4, 3, 3))
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if os.path.isdir(dir + "/" + filename) and filename != "results":
            P = np.loadtxt(dir + "/" + filename + "/pos.txt")
            T = np.loadtxt(dir + "/" + filename + "/Time_steps.txt")
            S = np.loadtxt(dir + "/" + filename + "/spin_t.txt").reshape((len(T), len(P), 3))

            QFI_local += QFI_calculate(beta, K, S, P, T, dir + "/" + filename, False)
            QFI_global += QFI_calculate(beta, K, S, P, T, dir + "/" + filename, True)
    QFI_local = QFI_local / len(os.listdir(directory))
    QFI_global = QFI_global / len(os.listdir(directory))
    QFI_local = contract('kxyab->kab', QFI_local)
    QFI_global = contract('kxyab->kab', QFI_global)
    print("QFI_local:", QFI_local)
    print("QFI_global:", QFI_global)

def plot_heat_capacity(dir):
    specific_heat = np.loadtxt(dir + "/heat_capacity.txt")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(specific_heat[:, 0], specific_heat[:, 1], label='Specific Heat')
    ax.errorbar(specific_heat[:, 0], specific_heat[:, 1], yerr=specific_heat[:, 2], fmt='.', capsize=5, ecolor='red', alpha=0.7, label='Error')
    ax.set_xlabel('Temperature (T)')
    ax.set_ylabel('Specific Heat (C)')
    ax.set_title('Specific Heat vs Temperature')
    ax.set_xscale('log')
    ax.legend()
    plt.savefig(dir + "/heat_capacity.pdf")
    plt.clf()


def plot_all_specific_heat(root_dir, make_combined_plot=False):
    """
    Find all subdirectories in the root directory and plot the specific heat for each one.
    Optionally create a combined plot with all curves.
    
    Parameters:
    - root_dir: String path to the root directory
    - make_combined_plot: Boolean, whether to make a combined plot of all heat capacities
    """
    # Keep track of whether any heat capacity files were found
    found_any = False
    
    # Store data for combined plot
    all_data = []
    all_labels = []
    
    # Walk through all subdirectories
    for subdir, dirs, files in os.walk(root_dir):
        # Skip the root directory itself
        if subdir == root_dir:
            continue
        
        # Check if heat_capacity.txt exists in this directory
        heat_capacity_file = os.path.join(subdir, "heat_capacity.txt")
        if os.path.isfile(heat_capacity_file):
            print(f"Processing {subdir}")
            found_any = True
            
            # Use the existing function to plot the heat capacity
            plot_heat_capacity(subdir)
            
            # Store data for combined plot
            if make_combined_plot:
                try:
                    specific_heat = np.loadtxt(heat_capacity_file)
                    all_data.append(specific_heat)
                    # Use directory name as label
                    all_labels.append(os.path.basename(subdir))
                except Exception as e:
                    print(f"Error loading data from {heat_capacity_file}: {e}")
    
    if not found_any:
        print(f"No heat_capacity.txt files found in any subdirectory of {root_dir}")
        return
        
    # Create combined plot if requested and data was found
    if make_combined_plot and all_data:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for data, label in zip(all_data, all_labels):
            ax.plot(data[:, 0], data[:, 1], label=label, marker='o', markersize=4, linestyle='-', alpha=0.7)
            # Add error bars if available (column 2 typically contains the error)
            if data.shape[1] > 2:
                ax.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], fmt='.', capsize=3, alpha=0.5)
        
        ax.set_xlabel('Temperature (T)')
        ax.set_ylabel('Specific Heat (C)')
        ax.set_title('Specific Heat vs Temperature - Combined Plot')
        
        # Set legend with appropriate size and columns based on number of entries
        if len(all_labels) > 12:
            ax.legend(loc='best', ncol=3, fontsize='small')
        elif len(all_labels) > 6:
            ax.legend(loc='best', ncol=2, fontsize='small')
        else:
            ax.legend(loc='best')
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Try to use log scale if the temperature range is wide
        if any(data[:, 0].max() / data[:, 0].min() > 10 for data in all_data):
            ax.set_xscale('log')
            plt.savefig(os.path.join(root_dir, "combined_heat_capacity_log.pdf"))
            # Also save a version with linear scale
            ax.set_xscale('linear')
            plt.savefig(os.path.join(root_dir, "combined_heat_capacity.pdf"))
        else:
            plt.savefig(os.path.join(root_dir, "combined_heat_capacity.pdf"))
        
        plt.close()
        print(f"Combined plot saved to {os.path.join(root_dir, 'combined_heat_capacity.pdf')}")


def _select_sssf_graph(mag):
    """Return the SSSF graphing helper that matches the chosen magnetization label."""
    mapping = {
        "001": SSSFGraphHK0,
        "1-10": SSSFGraphHHL,
        "110": SSSFGraphHnHL,
        "111": SSSFGraphHnHn,
        "HnHn": SSSFGraphHnHn,
        "HK0": SSSFGraphHK0,
        "HHL": SSSFGraphHHL,
        "HnHL": SSSFGraphHnHL,
    }
    return mapping.get(mag)


def main():
    parser = argparse.ArgumentParser(description="Process MD pi-flux simulation outputs.")
    parser.add_argument("root", help="Path to the MD_pi_flux directory to process.")
    parser.add_argument("--mag", default="HnHn", help="Magnetization direction label (e.g. HnHn, 001).")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress information.")
    parser.add_argument(
        "--run-ids",
        nargs="+",
        help="Optional list of specific run indices to include (default uses all).",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Create animations for DSSF spinon+photon data across all field strengths.",
    )
    parser.add_argument(
        "--animation-fps",
        type=int,
        default=0.5,
        help="Frames per second for animations (default: 5).",
    )
    parser.add_argument(
        "--animation-output",
        help="Output directory for animations (default: root/animations).",
    )

    args = parser.parse_args()

    run_ids = None
    if args.run_ids is not None:
        try:
            run_ids = [str(int(run_id)) for run_id in args.run_ids]
        except ValueError as exc:
            raise SystemExit(f"Invalid run id provided: {exc}") from exc

    graph_fn = _select_sssf_graph(args.mag)

    try:
        if args.animate:
            # Create animations
            print("Creating animations for DSSF spinon+photon data...")
            output_dir = args.animation_output if args.animation_output else None
            animate_DSSF_spinon_photon(
                args.root,
                output_dir=output_dir,
                fps=args.animation_fps
            )
        else:
            # Regular processing
            read_MD_pi_flux_all(
                args.root,
                mag=args.mag,
                SSSFGraph=graph_fn,
                verbose=not args.quiet,
                run_ids=run_ids,
            )
    except Exception as exc:
        if args.quiet:
            raise SystemExit(str(exc)) from exc
        raise


if __name__ == "__main__":
    main()

