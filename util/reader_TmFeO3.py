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


P1 = 2*np.pi * np.array([1, 0, 3])
P2 = 2*np.pi * np.array([3, 0, 3])
P3 = 2*np.pi * np.array([3, 0, 1])
P4 = 2*np.pi * np.array([3, 2, 1])
# P1 = 2*np.pi * np.array([-1, 1, 1])
# P2 = 2*np.pi * np.array([0, 1, 1])
# P3 = 2*np.pi * np.array([1, 1, 1])
# P4 = 2*np.pi * np.array([1, 1, 1])


# P1 = 2*np.pi * np.array([2, 0, 1])
# P2 = 2*np.pi * np.array([2, 1, 1])
# P3 = 2*np.pi * np.array([2, 2, 1])

# P1 = 2*np.pi * np.array([2, 1, 0])
# P2 = 2*np.pi * np.array([2, 1, 1])
# P3 = 2*np.pi * np.array([2, 1, 2])
# P4 = 2*np.pi * np.array([2, 1, 2])


# P1 = 2*np.pi * np.array([0, 1, -1])
# P2 = 2*np.pi * np.array([0, 1, 0])
# P3 = 2*np.pi * np.array([0, 1, 1])

# P1 = 2*np.pi * np.array([0, -1, 1])
# P2 = 2*np.pi * np.array([0, 0, 1])
# P3 = 2*np.pi * np.array([0, 1, 1])

graphres = 8


#Path to 1-10
P12 = np.linspace(P1, P2, calcNumSites(P1, P2, graphres))[1:-1]
P23 = np.linspace(P2, P3, calcNumSites(P2, P3, graphres))[1:-1]
P34 = np.linspace(P3, P4, calcNumSites(P3, P4, graphres))[1:-1]
g1 = 0
g2 = g1 + len(P12)
g3 = g2 + len(P23)
g4 = g3 + len(P34)

# DSSF_K = np.concatenate((GammaX, XW, WK, KGamma, GammaL, LU, UW1, W1X1, X1Gamma))

DSSF_K = np.concatenate((P12, P23, P34))
# DSSF_K = np.concatenate((P12, P23))



def Spin_global_pyrochlore(k,S,P):
    size = int(len(P)/4)
    tS = np.zeros((len(k),3), dtype=np.complex128)
    for i in range(4):
        ffact = np.exp(1j * contract('ik,jk->ij', k, P[i*size:(i+1)*size]))
        tS = tS + contract('js, ij, sp->ip', S[i*size:(i+1)*size], ffact, localframe[:,i,:])/np.sqrt(size)
    return tS

def Spin(k, S, P):
    ffact = np.exp(1j*contract('ik,jk->ij', k, P))
    N = len(S)
    return contract('js, ij->is', S, ffact)/np.sqrt(N)


def Spin_global_pyrochlore_t(k,S,P):
    size = int(len(P)/4)
    tS = np.zeros((len(S), 4, len(k),3), dtype=np.complex128)
    for i in range(4):
        ffact = np.exp(1j * contract('ik,jk->ij', k, P[i*size:(i+1)*size]))
        tS[:,i,:,:] = contract('tjs, ij, sp->tip', S[:,i*size:(i+1)*size,:], ffact, localframe[:,i,:])/np.sqrt(size)
    return tS

def Spin_t(k, S, P):
    ffact = np.exp(1j*contract('ik,jk->ij', k, P))
    N = len(S)
    results = np.zeros((len(S), len(k), S.shape[2]), dtype=np.complex128)
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
    ffactt = np.exp(1j*contract('w,t->wt', w, T))
    if gb:
        A = Spin_global_pyrochlore_t(k, S, P)
        Somega = contract('tnis, wt->wnis', A, ffactt)/np.sqrt(len(T))
        read = np.real(contract('wni, wmi, inm->wi', Somega[:,:,:,2], np.conj(Somega[:,:,:,2]), g(k)))
        return read
    else:
        A = Spin_t(k, S, P)
        Somega = contract('tis, wt->wis', A, ffactt)/np.sqrt(len(T))
        read = np.real(contract('wia, wib->wi', Somega, np.conj(Somega)))
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
    # SSSFGraphHK0(A, B, S[:,:,0,0], f1)
    # SSSFGraphHK0(A, B, S[:,:,1,1], f2)
    # SSSFGraphHK0(A, B, S[:,:,2,2], f3)
    # SSSFGraphHK0(A, B, S[:, :, 0, 1], f4)
    # SSSFGraphHK0(A, B, S[:, :, 0, 2], f5)
    # SSSFGraphHK0(A, B, S[:, :, 1, 2], f6)

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


def read_MD_tot(dir):
    print("Begin MD reading")
    directory = os.fsencode(dir)
    for file in sorted(os.listdir(directory)):
        print(file)
        filename = os.fsdecode(file)
        if os.path.isdir(dir + "/" + filename):
            w0 = 0
            wmax = 15
            t_evolved = 100
            SU2 = read_MD_SU2(dir + "/" + filename, w0, wmax, t_evolved)
            SU3 = read_MD_SU3(dir + "/" + filename, w0, wmax, t_evolved)
            A = SU2 + SU3
            np.savetxt(dir + "/DSSF.txt", A)
            fig, ax = plt.subplots(figsize=(10,4))
            C = ax.imshow(A, origin='lower', extent=[0, g4, w0, wmax], aspect='auto', interpolation='gaussian', cmap='gnuplot2')
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
            
def read_MD_SU2(dir, w0, wmax, t_evolved):
    directory = os.fsencode(dir)
    P = np.loadtxt(dir + "/pos_SU2.txt")
    T = np.loadtxt(dir + "/Time_steps.txt")
    S = np.loadtxt(dir + "/spin_t_SU2.txt")
    Slength = int(len(S)/len(P))
    S = S.reshape((Slength, len(P), 3))
    S = S[-len(T):]  # Ensure S has the same length as T

    w = np.arange(w0, wmax, 1/t_evolved)
    A = DSSF(w, DSSF_K, S, P, T, False)
    A = np.log(A)
    # A = A / np.max(A)
    np.savetxt(dir + "_DSSF_SU2.txt", A)
    fig, ax = plt.subplots(figsize=(10,4))
    C = ax.imshow(A, origin='lower', extent=[0, g4, w0, wmax], aspect='auto', interpolation='gaussian', cmap='gnuplot2')
    ax.axvline(x=g1, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=g2, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=g3, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=g4, color='b', label='axvline - full height', linestyle='dashed')
    xlabpos = [g1, g2, g3, g4]
    # labels = [r'$(0,0,1)$', r'$(0,1,1)$', r'$(0,2,1)$', r'$(0,3,1)$']
    labels = [r'$(0,0,0)$', r'$(0,0,1)$', r'$(0,1,1)$', r'$(1,1,1)$']
    # labels = [r'$(-1,1,1)$', r'$(0,1,1)$', r'$(1,1,1)$']
    # labels = [r'$(2,0,1)$', r'$(2,1,1)$', r'$(2,2,1)$']
    # labels = [r'$(2,1,0)$', r'$(2,1,1)$', r'$(2,1,2)$']
    # labels = [r'$(0,1,-1)$', r'$(0,1,0)$', r'$(0,1,1)$']
    # labels = [r'$(0,-1,1)$', r'$(0,0,1)$', r'$(0,1,1)$']
# 
    ax.set_xticks(xlabpos, labels)
    ax.set_xlim([0, g4])
    fig.colorbar(C)
    plt.savefig(dir+"DSSF_SU2.pdf")
    plt.clf()
    return A
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

def read_MD_SU3(dir, w0, wmax, t_evolved):
    directory = os.fsencode(dir)
    P = np.loadtxt(dir + "/pos_SU3.txt")
    T = np.loadtxt(dir + "/Time_steps.txt")
    S = np.loadtxt(dir + "/spin_t_SU3.txt")
    Slength = int(len(S)/len(P))
    S = S.reshape((Slength, len(P), 8))
    S = S[-len(T):]  # Ensure S has the same length as T

    w = np.arange(w0, wmax, 1/t_evolved)
    A = DSSF(w, DSSF_K, S, P, T, False)
    A = np.log(A)
    # A = A / np.max(A)
    np.savetxt(dir + "_DSSF_SU3.txt", A)
    fig, ax = plt.subplots(figsize=(10,4))
    C = ax.imshow(A, origin='lower', extent=[0, g4, w0, wmax], aspect='auto', interpolation='gaussian', cmap='gnuplot2')
    ax.axvline(x=g1, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=g2, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=g3, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=g4, color='b', label='axvline - full height', linestyle='dashed')
    xlabpos = [g1, g2, g3, g4]
    # labels = [r'$(0,0,1)$', r'$(0,1,1)$', r'$(0,2,1)$', r'$(0,3,1)$']
    # labels = [r'$(1,0,3)', r'$(3,0,3)$', r'$(3,0,1)$', r'$(3,2,1)$']
    labels = [r'$(0,0,0)$', r'$(0,0,1)$', r'$(0,1,1)$', r'$(1,1,1)$']
    # labels = [r'$(2,0,1)$', r'$(2,1,1)$', r'$(2,2,1)$']
    # labels = [r'$(2,1,0)$', r'$(2,1,1)$', r'$(2,1,2)$']
    # labels = [r'$(0,1,-1)$', r'$(0,1,0)$', r'$(0,1,1)$']
    # labels = [r'$(0,-1,1)$', r'$(0,0,1)$', r'$(0,1,1)$']
# 
    ax.set_xticks(xlabpos, labels)
    ax.set_xlim([0, g4])
    fig.colorbar(C)
    plt.savefig(dir+"DSSF_SU3.pdf")
    plt.clf()
    return A

def read_2D_nonlinear(dir):
    directory = os.fsencode(dir)
    tau_start, tau_end, tau_step, time_start, time_end, time_step = np.loadtxt(dir + "/param.txt")
    M0 = np.loadtxt(dir + "/M_time_0/M0/M_t.txt")[:,0]
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
            M1 = np.loadtxt(dir + "/" + filename + "/M1/M_t.txt")[:,0]
            M01 = np.loadtxt(dir + "/" + filename + "/M01/M_t.txt")[:,0]
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


def read_2D_nonlinear_adaptive_time_step(dir, readslice, fm):
    """
    Process 2D nonlinear spectroscopy data with adaptive time steps.
    
    Args:
        dir: Directory containing the spectroscopy data
    """
    directory = os.path.abspath(dir)  # Use absolute path for reliability
    
    if fm:
        readfile = "M_t_f.txt"
    else:
        readfile = "M_t.txt"
    
    # Load M0 data once
    m0_file = os.path.join(directory, "M_time_0.000000/M1/" + readfile)
    m0_time_file = os.path.join(directory, "M_time_0.000000/M1/Time_steps.txt")

    time_steps = np.min([len(np.loadtxt(m0_file)), len(np.loadtxt(m0_time_file))])

    try:
        M0 = np.loadtxt(m0_file)[-time_steps:,readslice]
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
    M0_w = np.dot(M0, M0_phase.T)
    
    # Initialize result array
    M_NL_FF = np.zeros((len(wp), len(w)), dtype=complex)
    
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
            M1 = np.loadtxt(os.path.join(base_path, "M1/" + readfile))[-time_steps:,readslice]
            M1_T = np.loadtxt(os.path.join(base_path, "M1/Time_steps.txt"))
            # Load M01 data
            M01 = np.loadtxt(os.path.join(base_path, "M01/" + readfile))[-time_steps:,readslice]
            M01_T = np.loadtxt(os.path.join(base_path, "M01/Time_steps.txt"))
            
            # Transform to frequency domain
            M1_phase = np.exp(1j * np.outer(wp, M1_T))
            M01_phase = np.exp(1j * np.outer(wp, M01_T))
            
            M1_w = np.dot(M1, M1_phase.T)
            M01_w = np.dot(M01, M01_phase.T)
            M_NL_here = M01_w - M0_w - M1_w
            
            # Apply phase factor
            ffactau = np.exp(-1j * w * current_tau) / len(tau)
            M_NL_FF += np.outer(M_NL_here, ffactau)
            
        except Exception as e:
            print(f"Error processing {subdir}: {e}")
            continue
    
    # Take absolute value for plotting
    M_NL_FF_abs = np.abs(M_NL_FF)

    # Suppress intensity near (0,0)
    # M_NL_FF_abs[len(wp)//2-2:len(wp)//2+2, 0:2] = 1e-15

    # Save raw data
    output_file = os.path.join(dir, "M_NL_FF.txt")
    np.savetxt(output_file, M_NL_FF_abs)
    
    # Create plots with shared setup
    plt.figure(figsize=(10, 8))
    extent = [0, omega_range, -omega_range, omega_range]
    
    # Linear scale plot
    plt.imshow(M_NL_FF_abs, origin='lower', extent=extent,
              aspect='auto', interpolation='lanczos', cmap='gnuplot2')
    plt.colorbar(label='Amplitude')
    plt.xlabel('Frequency (J1)')
    plt.ylabel('Frequency (J1)')
    plt.title('2D Nonlinear Spectrum')
    plt.savefig(f"{dir}/NLSPEC_{readslice}_{fm}.pdf", dpi=300, bbox_inches='tight')
    plt.clf()
    
    # Log scale plot
    plt.imshow(M_NL_FF_abs, origin='lower', extent=extent,
              aspect='auto', interpolation='lanczos', cmap='gnuplot2',
              norm=PowerNorm(gamma=0.5))
    plt.colorbar(label='Amplitude (sqrt scale)')
    plt.xlabel('Frequency (J1)')
    plt.ylabel('Frequency (J1)')
    plt.title('2D Nonlinear Spectrum')
    plt.savefig(f"{directory}/NLSPEC_{readslice}_{fm}_SU3_log.pdf", dpi=300, bbox_inches='tight')
    plt.clf()
    
    return M_NL_FF_abs



def read_2D_nonlinear_adaptive_time_step_SU3(dir, readslice, fm):
    """
    Process 2D nonlinear spectroscopy data with adaptive time steps.
    
    Args:
        dir: Directory containing the spectroscopy data
    """
    directory = os.path.abspath(dir)  # Use absolute path for reliability
    
    if fm:
        readfile = "M_t_f_SU3.txt"
    else:
        readfile = "M_t_SU3.txt"

    m0_file = os.path.join(directory, "M_time_0.000000/M1/" + readfile)
    m0_time_file = os.path.join(directory, "M_time_0.000000/M1/Time_steps.txt")
    
    time_steps = np.min([len(np.loadtxt(m0_file)), len(np.loadtxt(m0_time_file))])

    try:
        M0 = np.loadtxt(m0_file)[-time_steps:,readslice]
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
    M0_w = np.dot(M0, M0_phase.T)
    
    # Initialize result array
    M_NL_FF = np.zeros((len(wp), len(w)), dtype=complex)
    
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
            M1 = np.loadtxt(os.path.join(base_path, "M1/" + readfile))[-time_steps:,readslice]
            M1_T = np.loadtxt(os.path.join(base_path, "M1/Time_steps.txt"))
            # Load M01 data
            M01 = np.loadtxt(os.path.join(base_path, "M01/" + readfile))[-time_steps:,readslice]
            M01_T = np.loadtxt(os.path.join(base_path, "M01/Time_steps.txt"))
            
            # Transform to frequency domain
            M1_phase = np.exp(1j * np.outer(wp, M1_T))
            M01_phase = np.exp(1j * np.outer(wp, M01_T))
            
            M1_w = np.dot(M1, M1_phase.T)
            M01_w = np.dot(M01, M01_phase.T)
            M_NL_here = M01_w - M0_w - M1_w
            
            # Apply phase factor
            ffactau = np.exp(-1j * w * current_tau) / len(tau)
            M_NL_FF += np.outer(M_NL_here, ffactau)
            
        except Exception as e:
            print(f"Error processing {subdir}: {e}")
            continue
    
    # Take absolute value for plotting
    M_NL_FF_abs = np.abs(M_NL_FF)

    # M_NL_FF_abs[len(wp)//2-2:len(wp)//2+2, 0:2] = 1e-15

    # Save raw data
    output_file = os.path.join(directory, "M_NL_FF_SU3.txt")
    np.savetxt(output_file, M_NL_FF_abs)
    
    # Create plots with shared setup
    plt.figure(figsize=(10, 8))
    extent = [0, omega_range, -omega_range, omega_range]
    
    # Linear scale plot
    plt.imshow(M_NL_FF_abs, origin='lower', extent=extent,
              aspect='auto', interpolation='lanczos', cmap='gnuplot2')
    plt.colorbar(label='Amplitude')
    plt.xlabel('Frequency (J1)')
    plt.ylabel('Frequency (J1)')
    plt.title('2D Nonlinear Spectrum')
    plt.savefig(f"{directory}/NLSPEC_{readslice}_{fm}_SU3.pdf", dpi=300, bbox_inches='tight')
    plt.clf()
    
    # Log scale plot
    plt.imshow(M_NL_FF_abs, origin='lower', extent=extent,
              aspect='auto', interpolation='lanczos', cmap='gnuplot2',
              norm=PowerNorm(gamma=0.5))
    plt.colorbar(label='Amplitude (sqrt scale)')
    plt.xlabel('Frequency (J1)')
    plt.ylabel('Frequency (J1)')
    plt.title('2D Nonlinear Spectrum')
    plt.savefig(f"{directory}/NLSPEC_{readslice}_{fm}_SU3_log.pdf", dpi=300, bbox_inches='tight')
    plt.clf()
    
    return M_NL_FF_abs


def full_read_2DCS_TFO(dir):
    SU2x = read_2D_nonlinear_adaptive_time_step(dir, 0, True)
    SU2y = read_2D_nonlinear_adaptive_time_step(dir, 1, True)
    SU2z = read_2D_nonlinear_adaptive_time_step(dir, 2, True)
    SU32 = read_2D_nonlinear_adaptive_time_step_SU3(dir, 1, True)
    # SU35 = read_2D_nonlinear_adaptive_time_step_SU3(dir, 4, True)
    # SU37 = read_2D_nonlinear_adaptive_time_step_SU3(dir, 6, True)
    for i in range(8):
        if i == 1:
            continue
        read_2D_nonlinear_adaptive_time_step_SU3(dir, i, True)


    omega_range = 3

    xtotal = 5 * SU2x 
    ytotal = 5 * SU2y 
    ztotal = 5 * SU2z + 5.2 * SU32

    extent = [0, omega_range, -omega_range, omega_range]

    # Power-law normalized plot for xtotal
    plt.imshow(xtotal, origin='lower', extent=extent,
              aspect='auto', interpolation='lanczos', cmap='gnuplot2',
              norm=PowerNorm(gamma=0.5))
    plt.colorbar(label='Amplitude (sqrt scale)')
    plt.xlabel('Frequency (J1)')
    plt.ylabel('Frequency (J1)')
    plt.title('2D Nonlinear Spectrum (X Total)')
    plt.savefig(f"{dir}/NLSPEC_x_total_sqrt.pdf", dpi=300, bbox_inches='tight')
    plt.clf()

    # Power-law normalized plot for ytotal
    plt.imshow(ytotal, origin='lower', extent=extent,
              aspect='auto', interpolation='lanczos', cmap='gnuplot2',
              norm=PowerNorm(gamma=0.5))
    plt.colorbar(label='Amplitude (sqrt scale)')
    plt.xlabel('Frequency (J1)')
    plt.ylabel('Frequency (J1)')
    plt.title('2D Nonlinear Spectrum (Y Total)')
    plt.savefig(f"{dir}/NLSPEC_y_total_sqrt.pdf", dpi=300, bbox_inches='tight')
    plt.clf()

    # Power-law normalized plot for ztotal
    plt.imshow(ztotal, origin='lower', extent=extent,
              aspect='auto', interpolation='lanczos', cmap='gnuplot2',
              norm=PowerNorm(gamma=0.5))
    plt.colorbar(label='Amplitude (sqrt scale)')
    plt.xlabel('Frequency (J1)')
    plt.ylabel('Frequency (J1)')
    plt.title('2D Nonlinear Spectrum (Z Total)')
    plt.savefig(f"{dir}/NLSPEC_z_total_sqrt.pdf", dpi=300, bbox_inches='tight')
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
    full_read_2DCS_TFO(directory)
# read_MD_tot(dir)


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