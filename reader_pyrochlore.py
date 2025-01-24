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


graphres = 12

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

stepN = np.linalg.norm(Gamma-P1)/graphres


#Path to 1-10
GammaP1 = drawLine(Gamma, P1, stepN)[1:]
P12 = drawLine(P1, P2, stepN)[:-1]
P23 = drawLine(P2, P3, stepN)[1:]
P34 = drawLine(P3, P4, stepN)[:-1]
P45 = drawLine(P4, P5, stepN)[1:-1]
P5Gamma = drawLine(P5, Gamma, stepN)[1:-1]



gGamma1 = 0
g1 = magnitude_bi(Gamma, P1)
g2 = g1 + magnitude_bi(P1, P2)
g3 = g2 + magnitude_bi(P2, P3)
g4 = g3 + magnitude_bi(P3, P4)
g5 = g4 + magnitude_bi(P4, P5)
gGamma4 = g5 + magnitude_bi(P5, Gamma)


# DSSF_K = np.concatenate((GammaX, XW, WK, KGamma, GammaL, LU, UW1, W1X1, X1Gamma))

DSSF_K = np.concatenate((GammaP1, P12, P23, P34, P45, P5Gamma))


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
    ffact = np.exp(1j*contract('ik,jk->ij', k, P))
    N = len(S)
    return contract('js, ij->is', S, ffact)/np.sqrt(N)


def Spin_global_pyrochlore_t(k,S,P):
    size = int(len(P)/4)
    tS = np.zeros((len(S), 4, len(k),3), dtype=np.complex128)
    # tS = np.zeros((len(S), len(k),3), dtype=np.complex128)
    for i in range(4):
        ffact = np.exp(1j * contract('ik,jk->ij', k, P[i::4]))
        # tS = tS + contract('tjs, ij, sp->tip', S[:,i::4], ffact, localframe[:,i,:])/np.sqrt(size)
        tS[:,i,:,:] = contract('tjs, ij->tis', S[:,i::4], ffact)/np.sqrt(size)
    return tS

def Spin_global_pyrochlore(k,S,P):
    size = int(len(P)/4)
    tS = np.zeros((4, len(k),3), dtype=np.complex128)
    for i in range(4):
        ffact = np.exp(1j * contract('ik,jk->ij', k, P[i::4]))
        tS[i,:,:] = contract('js, ij->is', S[i::4], ffact)/np.sqrt(size)
    return tS


def Spin_t(k, S, P):
    ffact = np.exp(1j*contract('ik,jk->ij', k, P))
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

def SSSF_q(k, S, P, gb=False):
    if gb:
        A = Spin_global_pyrochlore(k, S, P)
        read = np.abs(contract('nia, mib, inmab->inmab', A, np.conj(A), gg(k)))
        read = np.where(read <= 1e-8, 1e-8, read)
        return np.log(read)
    else:
        A = Spin_global_pyrochlore(k, S, P)
        read = np.abs(contract('nia, mib->inmab', A, np.conj(A)))
        read = np.where(read == 0, np.min(read), read)
        return np.log(read)

def DSSF(w, k, S, P, T, gb=False):
    ffactt = np.exp(1j*contract('w,t->wt', w, T))
    if gb:
        A = Spin_global_pyrochlore_t(k, S, P)
        Somega = contract('tnis, wt->wnis', A, ffactt)/np.sqrt(len(T))
        read = np.abs(contract('wnia, wmib, inmab->winmab', Somega, np.conj(Somega), gg(k)))
        read = np.where(read <= 1e-8, 1e-8, read)
        return np.log(read)

    else:
        # A = Spin_t(k, S, P)
        # Somega = contract('tis, wt->wis', A, ffactt)/np.sqrt(len(T))
        # read = np.real(contract('wia, wib->wiab', Somega, np.conj(Somega)))
        # # read = contract('wiab, ab->wi', read, g(k))
        # return np.log(read[:,:,2,2])
        A = Spin_global_pyrochlore_t(k, S, P)
        Somega = contract('tnis, wt->wnis', A, ffactt)/np.sqrt(len(T))
        read = np.abs(contract('wnia, wmib->winmab', Somega, np.conj(Somega)))
        read = np.where(read == 0, np.min(read), read)
        return np.log(read)


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
            w = np.linspace(w0, wmax, 1000)
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
    w0 = 0
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
            Sl, Sg, Dl, Dg = read_MD(dir + "/" + filename, mag)
            S_local = S_local + Sl
            S_global = S_global + Sg
            DSSF_local = DSSF_local + Dl
            DSSF_global = DSSF_global + Dg
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
        DSSF = DSSF / np.max(DSSF)
        # np.savetxt(filename+".txt", DSSF)
        fig, ax = plt.subplots(figsize=(10,4))
        C = ax.imshow(DSSF, origin='lower', extent=[0, gGamma4, w0, wmax], aspect='auto', interpolation='lanczos', cmap='gnuplot2')
        ax.axvline(x=gGamma1, color='b', label='axvline - full height', linestyle='dashed')
        ax.axvline(x=g1, color='b', label='axvline - full height', linestyle='dashed')
        ax.axvline(x=g2, color='b', label='axvline - full height', linestyle='dashed')
        ax.axvline(x=g3, color='b', label='axvline - full height', linestyle='dashed')
        ax.axvline(x=g4, color='b', label='axvline - full height', linestyle='dashed')
        ax.axvline(x=g5, color='b', label='axvline - full height', linestyle='dashed')
        ax.axvline(x=gGamma4, color='b', label='axvline - full height', linestyle='dashed')
        xlabpos = [gGamma1, g1, g2, g3, g4, g5, gGamma4]
        # labels = [r'$(0,0,0)$', r'$(1,0,0)$', r'$(2,0,0)$', r'$(2,1,0)$', r'$(2,2,0)$', r'$(1,1,0)$', r'$(0,0,0)$']
        labels = [r'$(0,0,0)$', r'$(1,1,0)$', r'$(2,2,0)$', r'$(2,2,1)$', r'$(2,2,2)$', r'$(1,1,1)$', r'$(0,0,0)$']
        ax.set_xticks(xlabpos, labels)
        ax.set_xlim([0, gGamma4])
        fig.colorbar(C)
        plt.savefig(filename+".pdf")
        plt.clf()
        plt.close()

    def DSSF_all_spin_components(DSSF, filename):
        com_string = np.array(["x", "y", "z"])
        for i in range(3):
            for j in range(3):
                DSSF_helper(DSSF[:,:,i,j], filename + "_"+ com_string[i]+com_string[j])

    dir_to_save = dir + "/results"
    if not os.path.isdir(dir_to_save):
        os.mkdir(dir_to_save)
    for i in range(4):
        for j in range (4):
            if not os.path.isdir(dir_to_save + "/"+str(i)+str(j)):
                os.mkdir(dir_to_save + "/"+str(i)+str(j))
            SSSF_helper(S_local[:,:,i,j], S_global[:,:,i,j], dir_to_save + "/"+str(i)+str(j))
            DSSF_all_spin_components(DSSF_local[:,:,i,j], dir_to_save + "/"+str(i)+str(j)+"/DSSF_local")
            DSSF_all_spin_components(DSSF_global[:,:,i,j], dir_to_save + "/"+str(i)+str(j)+"/DSSF_global")
    if not os.path.isdir(dir_to_save + "/SSSF"):
        os.mkdir(dir_to_save + "/SSSF")
    if not os.path.isdir(dir_to_save + "/DSSF_local"):
        os.mkdir(dir_to_save + "/DSSF_local")
    if not os.path.isdir(dir_to_save + "/DSSF_global"):
        os.mkdir(dir_to_save + "/DSSF_global")
    SSSF_helper(contract('ijxyab->ijab', S_local), contract('ijxyab->ijab', S_global), dir_to_save + "/SSSF")
    DSSF_all_spin_components(contract('ijxyab->ijab',DSSF_local), dir_to_save + "/DSSF_local/")
    DSSF_all_spin_components(contract('ijxyab->ijab',DSSF_global), dir_to_save + "/DSSF_global/")


def read_MD(dir, mag):
    directory = os.fsencode(dir)
    P = np.loadtxt(dir + "/pos.txt")
    T = np.loadtxt(dir + "/Time_steps.txt")

    S = np.loadtxt(dir + "/spin_t.txt").reshape((len(T), len(P), 3))
    S_local = np.zeros((50,50,4,4,3,3))
    S_global = np.zeros((50,50,4,4,3,3))
    if P.shape[1] == 2:
        SSSF2D(S[0],P, 100, dir, True)
    elif mag == "001":
        S_global = SSSFHK0(S[0], P, 50, dir, True)
        S_local = SSSFHK0(S[0], P, 50, dir, False)
    elif mag == "1-10":
        S_global = SSSFHHL(S[0], P, 50, dir, True)
        S_local = SSSFHHL(S[0], P, 50, dir, False)
    elif mag == "110":
        S_global = SSSFHnHL(S[0], P, 50, dir, True)
        S_local = SSSFHnHL(S[0], P, 50, dir, False)
    w0 = 0
    wmax = 10
    w = np.arange(w0, wmax, 1/600)
    DSSF_local = DSSF(w, DSSF_K, S, P, T, False)
    DSSF_global = DSSF(w, DSSF_K, S, P, T, True)
    # A = A / np.max(A)
    # np.savetxt(dir + "_DSSF.txt", A)
    # fig, ax = plt.subplots(figsize=(10,4))
    # C = ax.imshow(A, origin='lower', extent=[0, gGamma4, w0, wmax], aspect='auto', interpolation='lanczos', cmap='gnuplot2')
    # ax.axvline(x=gGamma1, color='b', label='axvline - full height', linestyle='dashed')
    # ax.axvline(x=g1, color='b', label='axvline - full height', linestyle='dashed')
    # ax.axvline(x=g2, color='b', label='axvline - full height', linestyle='dashed')
    # ax.axvline(x=g3, color='b', label='axvline - full height', linestyle='dashed')
    # ax.axvline(x=g4, color='b', label='axvline - full height', linestyle='dashed')
    # ax.axvline(x=g5, color='b', label='axvline - full height', linestyle='dashed')
    # ax.axvline(x=gGamma4, color='b', label='axvline - full height', linestyle='dashed')
    # xlabpos = [gGamma1, g1, g2, g3, g4, g5, gGamma4]
    # # labels = [r'$(0,0,0)$', r'$(1,0,0)$', r'$(2,0,0)$', r'$(2,1,0)$', r'$(2,2,0)$', r'$(1,1,0)$', r'$(0,0,0)$']
    # labels = [r'$(0,0,0)$', r'$(1,1,0)$', r'$(2,2,0)$', r'$(2,2,1)$', r'$(2,2,2)$', r'$(1,1,1)$', r'$(0,0,0)$']
    # ax.set_xticks(xlabpos, labels)
    # ax.set_xlim([0, gGamma4])
    # fig.colorbar(C)
    # plt.savefig(dir+"DSSF.pdf")
    # plt.clf()
    # C = ax.imshow(A, origin='lower', extent=[0, gGamma3, w0, wmax], aspect='auto', interpolation='lanczos', cmap='gnuplot2')
    # ax.axvline(x=gGamma1, color='b', label='axvline - full height', linestyle='dashed')
    # ax.axvline(x=gX, color='b', label='axvline - full height', linestyle='dashed')
    # ax.axvline(x=gW, color='b', label='axvline - full height', linestyle='dashed')
    # ax.axvline(x=gK, color='b', label='axvline - full height', linestyle='dashed')
    # ax.axvline(x=gGamma2, color='b', label='axvline - full height', linestyle='dashed')
    # ax.axvline(x=gL, color='b', label='axvline - full height', linestyle='dashed')
    # ax.axvline(x=gU, color='b', label='axvline - full height', linestyle='dashed')
    # ax.axvline(x=gW1, color='b', label='axvline - full height', linestyle='dashed')
    # ax.axvline(x=gX1, color='b', label='axvline - full height', linestyle='dashed')
    # ax.axvline(x=gGamma3, color='b', label='axvline - full height', linestyle='dashed')
    # xlabpos = [gGamma1, gX, gW, gK, gGamma2, gL, gU, gW1, gX1, gGamma3]
    # labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W^\prime$', r'$X^\prime$',
    #             r'$\Gamma$']
    # ax.set_xticks(xlabpos, labels)
    # ax.set_xlim([0, gGamma3])
    # fig.colorbar(C)
    # plt.savefig(dir+"DSSF.pdf") 
    # plt.clf()
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
    w = np.arange(-omega_range, omega_range, 1/abs(tau_end-tau_start))
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
            M_NL[int(info[2])] = M01[-domain:] - M0[-domain:] - M1[-domain:] 
    # gaussian_filter =  np.exp(-1e-6 * (contract('i,i,a->ia',T,T,np.ones(len(tau))) + contract('a,a,i->ia',tau,tau,np.ones(len(T)))))   
    ffactau = np.exp(-1j*contract('w,t->wt', w, tau))/len(tau)
    # M_NL_FF = contract('it, ti->it', M_NL, gaussian_filter)
    M_NL_FF = M_NL
    M_NL_FF = np.abs(contract('it, wi, ut->wu', M_NL_FF, ffactau, ffactt))
    M_NL_FF = np.log(M_NL_FF)
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


# read_2D_nonlinear("pyrochlore_h110=10_driven_001")

# obenton_to_xx_zz()
#
# dir = "CZO_h=4T"
dir = "CZO_MD_h110=8"
read_MD_tot(dir, "1-10", SSSFGraphHHL)
# parseDSSF(dir)
# fullread(dir, False, "111")
# fullread(dir, True, "111")
# parseSSSF(dir)
# parseDSSF(dir)
# read_0_field(70, "/scratch/y/ybkim/zhouzb79/MC_Phase_Diagram_0_field")
# A = np.loadtxt("test_Jpm=0.3/specific_heat.txt", unpack=True)
# plt.plot(A[0], A[1])
# plt.xscale('log')
# plt.savefig("test_Jpm=0.3/specific_heat.pdf")