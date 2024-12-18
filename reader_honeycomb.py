import h5py
import numpy as np
from opt_einsum import contract
import matplotlib.pyplot as plt
import os
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


graphres = 50

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
GammaX = drawLine(Gamma, X, stepN)
XW = drawLine(X, W, stepN)
WK = drawLine(W, K, stepN)
KGamma = drawLine(K, Gamma, stepN)

#Path to 111 and then 001
GammaL = drawLine(Gamma, L, stepN)
LU = drawLine(L, U, stepN)
UW1 = drawLine(U, W1, stepN)
W1X1 = drawLine(W1, X1, stepN)
X1Gamma = drawLine(X1, Gamma, stepN)

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


Gamma = np.array([0, 0, 0])
P1 = 2 * np.pi * np.array([1, 0, 0])
P2 = 2 * np.pi * np.array([2, 0, 0])
P3 = 2 * np.pi * np.array([2, -1, 0])
P4 = 2 * np.pi * np.array([2, -2, 0])
P5 = np.pi * np.array([1, -1, 0])

stepN = np.linalg.norm(Gamma-P1)/graphres


#Path to 1-10
GammaP1 = drawLine(Gamma, P1, stepN)
P12 = drawLine(P1, P2, stepN)
P23 = drawLine(P2, P3, stepN)
P34 = drawLine(P3, P4, stepN)
P45 = drawLine(P4, P5, stepN)
P5Gamma = drawLine(P5, Gamma, stepN)



gGamma1 = 0
g1 = magnitude_bi(Gamma, P1)
g2 = g1 + magnitude_bi(P1, P2)
g3 = g2 + magnitude_bi(P2, P3)
g4 = g3 + magnitude_bi(P3, P4)
g5 = g4 + magnitude_bi(P4, P5)
gGamma4 = g5 + magnitude_bi(P5, Gamma)


# DSSF_K = np.concatenate((GammaX, XW, WK, KGamma, GammaL, LU, UW1, W1X1, X1Gamma))

# DSSF_K = np.concatenate((GammaP1, P12, P23, P34, P45, P5Gamma))


kitaevBasis = 4*np.pi/np.sqrt(3)*np.array([[np.sqrt(3)/2,-1/2,0],[0,1,0], [0,0,1]])

kitaevLocal = np.array([[1,-1,-1],[-1,1,-1],[-1,-1,1]])/np.sqrt(3)

Gamma2D = np.array([0, 0, 0])
K2D = np.array([2/3, 1/3, 0])
M2D = np.array([1/2, 0, 0])
Gamma12D = 2*M2D

LKitaev = 20

K2D = contract('a, ak->k', K2D, kitaevBasis)
M2D = contract('a, ak->k', M2D, kitaevBasis)
Gamma12D = contract('a, ak->k', Gamma12D, kitaevBasis)


P1_2D = np.linspace(K2D, Gamma2D, int(LKitaev/(2*np.sqrt(3))+1), endpoint=False)[1:]
P2_2D = np.linspace(Gamma2D, M2D, int(LKitaev/2), endpoint=False)[1:]
P3_2D = np.linspace(M2D, Gamma12D, int(LKitaev/2), endpoint=False)[1:]
P4_2D = np.linspace(Gamma12D, K2D, int(LKitaev/(2*np.sqrt(3))+1), endpoint=False)[1:]
P5_2D = np.linspace(K2D, M2D, int(LKitaev/(4*np.sqrt(3))+1), endpoint=False)[1:]

gK2D = 0
gGamma12D = len(P1_2D)
gM2D = gGamma12D + len(P2_2D)
gGamma22D = gM2D + len(P3_2D)
gK12D = gGamma22D + len(P4_2D)
gM22D = gK12D + len(P5_2D)

DSSF_K = np.concatenate((P1_2D, P2_2D, P3_2D, P4_2D, P5_2D))


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

    C = ax.imshow(A, origin='lower', extent=[0, gM22D, 0, 2.5], aspect='auto', interpolation='lanczos', cmap='gnuplot2')
    ax.axvline(x=gK2D, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gGamma12D, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gM2D, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gGamma22D, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gK12D, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gM22D, color='b', label='axvline - full height', linestyle='dashed')

    xlabpos = [gK2D, gGamma12D, gM2D, gGamma22D, gK12D, gM22D]
    labels = [r'$K$', r'$\Gamma_0$', r'$M$', r'$\Gamma_1$', r'$K$', r'$M$']
    ax.set_xticks(xlabpos, labels)
    ax.set_xlim([0, gM22D])
    fig.colorbar(C)
    plt.savefig(dir+"DSSF.pdf")
    plt.clf()

def read_MD_tot(dir):
    directory = os.fsencode(dir)
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if os.path.isdir(dir + "/" + filename):
            read_MD(dir + "/" + filename)

def read_MD(dir):
    directory = os.fsencode(dir)
    P = np.loadtxt(dir + "/pos.txt")
    T = np.loadtxt(dir + "/Time_steps.txt")

    S = np.loadtxt(dir + "/spin_t.txt").reshape((len(T), len(P), 3))

    w0 = 0
    wmax = 3
    w = np.linspace(w0, wmax, 1000)[1:]
    A = DSSF(w, DSSF_K, S, P, T, True)
    A = A / np.max(A)
    np.savetxt(dir + "_DSSF.txt", A)
    fig, ax = plt.subplots(figsize=(10,4))

    C = ax.imshow(A, origin='lower', extent=[0, gM22D, 0, 3], aspect='auto', interpolation='lanczos', cmap='gnuplot2')
    ax.axvline(x=gK2D, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gGamma12D, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gM2D, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gGamma22D, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gK12D, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gM22D, color='b', label='axvline - full height', linestyle='dashed')

    xlabpos = [gK2D, gGamma12D, gM2D, gGamma22D, gK12D, gM22D]
    labels = [r'$K$', r'$\Gamma_0$', r'$M$', r'$\Gamma_1$', r'$K$', r'$M$']
    ax.set_xticks(xlabpos, labels)
    ax.set_xlim([0, gM22D])
    fig.colorbar(C)
    plt.savefig(dir+"DSSF.pdf")
    plt.clf()

    SSSF2D(S[0], P, 100, dir, True)


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
    omega_range = 5
    M_NL = np.zeros((int(tau_step), domain))
    w = np.linspace(-omega_range, omega_range, domain)
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
    M_NL_FF = np.log(M_NL_FF)
    M_NL_FF = M_NL_FF/np.max(M_NL_FF)
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
# dir = "Pure_Kitaev_h=0.06"
# read_MD_tot(dir)
# parseDSSF(dir)

# dir = "kitaev_honeycomb_nonlinear_Gamma=0.25_Gammap=-0.02_h=0.7"
# dir = "test_long_h=0.0"
# read_2D_nonlinear_tot(dir)
dir = "pure_kitaev_2DCS_h=0.7"
read_2D_nonlinear_tot(dir)
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