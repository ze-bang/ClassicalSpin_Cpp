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


# P1 = 2*np.pi * np.array([1, 0, 3])
# P2 = 2*np.pi * np.array([3, 0, 3])
# P3 = 2*np.pi * np.array([3, 0, 1])
# P4 = 2*np.pi * np.array([3, 2, 1])
P1 = 2*np.pi * np.array([-1, 1, 1])
P2 = 2*np.pi * np.array([0, 1, 1])
P3 = 2*np.pi * np.array([1, 1, 1])
P4 = 2*np.pi * np.array([0, 3, 1])


# P1 = 2*np.pi * np.array([2, 0, 1])
# P2 = 2*np.pi * np.array([2, 1, 1])
# P3 = 2*np.pi * np.array([2, 2, 1])

P1 = 2*np.pi * np.array([2, 1, 0])
P2 = 2*np.pi * np.array([2, 1, 1])
P3 = 2*np.pi * np.array([2, 1, 2])

graphres = 8
stepN = np.linalg.norm(P2-P1)/graphres


#Path to 1-10
P12 = drawLine(P1, P2, stepN)[1:-1]
P23 = drawLine(P2, P3, stepN)[1:-1]
P34 = drawLine(P3, P4, stepN)[1:-1]



g1 = 0
g2 = g1 + magnitude_bi(P1, P2)
g3 = g2 + magnitude_bi(P2, P3)
g4 = g3 + magnitude_bi(P3, P4)


# DSSF_K = np.concatenate((GammaX, XW, WK, KGamma, GammaL, LU, UW1, W1X1, X1Gamma))

# DSSF_K = np.concatenate((P12, P23, P34))
DSSF_K = np.concatenate((P12, P23))



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
    return contract('tjs, ij->tis', S, ffact)/np.sqrt(N)

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
        read = np.real(contract('wia, wib->wiab', Somega, np.conj(Somega)))
        # read = contract('wiab, ab->wi', read, g(k))
        return read[:,:,0,0]

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
    directory = os.fsencode(dir)
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if os.path.isdir(dir + "/" + filename):
            read_MD(dir + "/" + filename)

def read_MD(dir):
    directory = os.fsencode(dir)
    P = np.loadtxt(dir + "/pos_SU2.txt")
    # T = np.loadtxt(dir + "/Time_steps.txt")
    S = np.loadtxt(dir + "/spin_t_SU2.txt")
    Slength = int(len(S)/len(P))
    S = S.reshape((Slength, len(P), 3))
    T = np.loadtxt("/scratch/y/ybkim/zhouzb79/MD_TmFeO3_xii=0/Time_steps.txt")[:len(S)]

    w0 = 0
    wmax = 15
    w = np.arange(w0, wmax, 1/10)[3:]
    A = DSSF(w, DSSF_K, S, P, T, False)
    A = np.log(A)
    A = A / np.max(A)
    np.savetxt(dir + "_DSSF.txt", A)
    fig, ax = plt.subplots(figsize=(10,4))
    C = ax.imshow(A, origin='lower', extent=[0, g3, w0, wmax], aspect='auto', interpolation='gaussian', cmap='gnuplot2')
    ax.axvline(x=g1, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=g2, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=g3, color='b', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=g4, color='b', label='axvline - full height', linestyle='dashed')
    xlabpos = [g1, g2, g3]
    # labels = [r'$(0,0,1)$', r'$(0,1,1)$', r'$(0,2,1)$', r'$(0,3,1)$']
    # labels = [r'$(-1,1,1)$', r'$(0,1,1)$', r'$(1,1,1)$']
    # labels = [r'$(2,0,1)$', r'$(2,1,1)$', r'$(2,2,1)$']
    labels = [r'$(2,1,0)$', r'$(2,1,1)$', r'$(2,1,2)$']

    ax.set_xticks(xlabpos, labels)
    ax.set_xlim([0, g3])
    fig.colorbar(C)
    plt.savefig(dir+"DSSF.pdf")
    plt.clf()
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


def read_2D_nonlinear_adaptive_time_step(dir):
    directory = os.fsencode(dir)
    tau_start, tau_end, tau_step, time_start, time_end, time_step = np.loadtxt(dir + "/param.txt")
    M0 = np.loadtxt(dir + "/M_time_0/M0/M_t.txt")[-20001:,1]
    M0_T = np.loadtxt(dir + "/M_time_0/M0/Time_steps.txt")
    M0_cutoff = np.where(M0_T >= 0)[0][0]


    omega_range = 5

    w = np.arange(-omega_range, omega_range, 0.2)
    M_NL_FF = np.zeros((len(w), len(w)))

    M0_w = contract('t, wt->w', M0[M0_cutoff:], np.exp(1j*contract('w, t->wt', w, M0_T[M0_cutoff:])))
    tau = np.linspace(tau_start, tau_end, int(tau_step))

    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if os.path.isdir(dir + "/" + filename):
            ph1, ph2, current_tau = filename.split("_")
            current_tau = float(current_tau)
            try:
                M1 = np.loadtxt(dir + "/" + filename + "/M1/M_t.txt")[-20001:,1]
                M1_T = np.loadtxt(dir + "/" + filename + "/M1/Time_steps.txt")
                M1_cutoff = np.where(M1_T >= 0)[0][0]

                M01 = np.loadtxt(dir + "/" + filename + "/M01/M_t.txt")[-20001:,1]
                M01_T = np.loadtxt(dir + "/" + filename + "/M01/Time_steps.txt")
                M01_cutoff = np.where(M01_T >= 0)[0][0]
                M1_w = contract('t, wt->w', M1[M1_cutoff:], np.exp(1j*contract('w, t->wt', w, M1_T[M1_cutoff:])))
                M01_w = contract('t, wt->w', M01[M01_cutoff:], np.exp(1j*contract('w, t->wt', w, M01_T[M01_cutoff:])))
                
                M_NL_here = M01_w - M0_w - M1_w
                ffactau = np.exp(-1j*w*current_tau)/len(tau)
                M_NL_FF = M_NL_FF + contract('w, t->wt', M_NL_here, ffactau)
            except:
                continue
    np.savetxt(dir + "/M_NL_FF.txt", M_NL_FF)
    plt.imshow(np.log(M_NL_FF), origin='lower', extent=[-omega_range, omega_range, -omega_range, omega_range], aspect='auto', interpolation='lanczos', cmap='gnuplot2', norm='linear')
    # plt.pcolormesh(w, w, np.log(M_NL_FF))
    plt.colorbar()
    plt.savefig(dir + "_NLSPEC.pdf")
    np.savetxt(dir + "_M_NL_FF.txt", M_NL_FF)
    plt.clf()
    plt.imshow(np.log(np.abs(M_NL_FF)), origin='lower', extent=[-omega_range, omega_range, -omega_range, omega_range], aspect='auto', interpolation='lanczos', cmap='gnuplot2', norm='linear')
    # plt.pcolormesh(w, w, np.log(M_NL_FF))
    plt.colorbar()
    plt.savefig(dir + "_NLSPEC_log.pdf")
    np.savetxt(dir + "_M_NL_FF_log.txt", M_NL_FF)
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
# dir = "TmFeO3_MD_Test"
# read_MD_tot(dir)
dir = "/scratch/y/ybkim/zhouzb79/MD_TmFeO3_xii=0"
read_MD_tot(dir)
# parseDSSF(dir)
# fullread(dir, False, "111")
# fullread(dir, True, "111")
# parseSSSF(dir)
# parseDSSF(dir)

# read_2D_nonlinear_adaptive_time_step("C://Users/raima/Downloads/TmFeO3_Fe_2DCS_Tzero_xii=0")
read_2D_nonlinear_adaptive_time_step("/scratch/y/ybkim/zhouzb79/TmFeO3_2DCS_Tzero_xii=0")

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

A = np.loadtxt("./TmFeO3_2DCS.txt", dtype=np.complex128)[25:-25, 25:-25]
plt.imshow(np.log(np.abs(A)), origin='lower', extent=[-5, 5, -5, 5], aspect='auto', interpolation='gaussian', cmap='gnuplot2', norm='linear')
plt.savefig("TmFeO3_2DCS.pdf")