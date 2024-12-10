import h5py
import numpy as np
from opt_einsum import contract
import matplotlib.pyplot as plt
import os
import matplotlib.tri as mtri 
from numba import njit
z = np.array([np.array([1,1,1])/np.sqrt(3), np.array([1,-1,-1])/np.sqrt(3), np.array([-1,1,-1])/np.sqrt(3), np.array([-1,-1,1])/np.sqrt(3)])
x = np.array([[-2,1,1],[-2,-1,-1],[2,1,-1], [2,-1,1]])/np.sqrt(6)
y = np.array([[0,-1,1],[0,1,-1],[0,-1,-1], [0,1,1]])/np.sqrt(2)
localframe = np.array([x,y,z])


def Spin_global_pyrochlore(k,S,P):
    tS = np.zeros((4,len(k),3), dtype=np.complex128)
    for i in range(4):
        ffact = np.exp(1j * contract('ik,jk->ij', k, P[i::4]))
        tS[i] = contract('js, ij->is', S[i::4], ffact)/np.sqrt(len(k)/4)
    return tS

def Spin(k, S, P):
    ffact = np.exp(1j*contract('ik,jk->ij', k, P))
    N = len(S)
    return contract('js, ij->is', S, ffact)/np.sqrt(N)

@njit
def g(q):
    M = np.zeros((len(q), 3, 3, 4,4))
    for k in range(len(q)):
        for x in range(3):
            for y in range(3):
                for i in range(4):
                    for j in range(4):
                        if not np.dot(q[k],q[k]) == 0:
                            M[k, x, y, i,j] = np.dot(localframe[x][i], localframe[y][j]) - np.dot(localframe[x][i],q[k]) * np.dot(localframe[y][j],q[k])/ np.dot(q[k],q[k])
                        else:
                            M[k, x, y, i,j] = 0
    return M


def SSSF_q(k, S, P, gb=False):
    if gb:
        A = Spin_global_pyrochlore(k, S, P)
        return np.real(contract('kia, lib, iabkl -> iab', A, np.conj(A), g(k)))
    else:
        A = Spin(k, S, P)
        return np.real(contract('ia, ib -> iab', A, np.conj(A)))

def hnhltoK(H, L, K=0):
    A = contract('ij,k->ijk',H, 2*np.array([np.pi,-np.pi,0])) \
        + contract('ij,k->ijk',L, 2*np.array([0,0,np.pi]))
    return A

def hhztoK(H, K):
    return contract('ij,k->ijk',H, 2*np.array([np.pi,0,0])) + contract('ij,k->ijk',K, 2*np.array([0,np.pi,0]))

def hnhkkn2ktoK(H, K, L=0):
    A = contract('ij,k->ijk',H, 2*np.array([np.pi,-np.pi,0])) \
        + contract('ij,k->ijk',K, 2*np.array([np.pi,np.pi,-2*np.pi]))
    return A

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

def SSSFGraphHH2K(A, B, d1, filename):
    plt.pcolormesh(A,B, d1)
    plt.colorbar()
    plt.ylabel(r'$(K,K,-2K)$')
    plt.xlabel(r'$(H,-H,0)$')
    plt.savefig(filename + ".pdf")
    plt.clf()

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
    SSSFGraphHK0(A, B, S[:,:,0,1], f4)
    SSSFGraphHK0(A, B, S[:,:,0,2], f5)
    SSSFGraphHK0(A, B, S[:,:,1,2], f6)


def SSSFHnHKKn2K(S, P, nK, filename, gb=False):
    H = np.linspace(-3, 3, nK)
    L = np.linspace(-3, 3, nK)
    A, B = np.meshgrid(H, L)
    K = hnhkkn2ktoK(A, B).reshape((nK*nK,3))
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
    SSSFGraphHH2K(A, B, S[:,:,0,0], f1)
    SSSFGraphHH2K(A, B, S[:,:,1,1], f2)
    SSSFGraphHH2K(A, B, S[:,:,2,2], f3)
    SSSFGraphHH2K(A, B, S[:,:,0,1], f4)
    SSSFGraphHH2K(A, B, S[:,:,0,2], f5)
    SSSFGraphHH2K(A, B, S[:,:,1,2], f6)

def SSSF_collect(S, P, nK, filename, field_dir, g=False):
    if not os.path.isdir(filename):
        os.mkdir(filename)

    if field_dir == "110":
        SSSFHnHL(S, P, nK, filename, g)
    elif field_dir == "001":
        SSSFHK0(S, P, nK, filename, g)
    else:
        SSSFHnHKKn2K(S, P, nK, filename, g)

    
def magnetization(S, n):
    A = np.zeros((4,3))
    # print(S.shape, len(S),size)
    for i in range (4):
        A[i] = contract('s, s->s',np.mean(S[i::4], axis=0), contract('xs,s->x',localframe[:,i,:],n))
    # mag = contract('ax, xas->s', A, localframe)/4
    return np.mean(A,axis=0)


def magnetization_local(S):
    return np.mean(S, axis=0)

def plot_lattice(P, S, filename):
    ax = plt.axes(projection='3d')
    ax.set_axis_off()
    ax.scatter(P[:,0],P[:,1],P[:,2], color='w', edgecolors='b', s=60,alpha=1)
    ax.quiver(P[:,0], P[:,1], P[:,2],S[:,0], S[:,1], S[:,2], color='red', length=0.3)
    plt.savefig(filename)
    plt.clf()

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
        ax.view_init(elev=20., azim=i)

    rot_animation = FuncAnimation(fig, animate, frames=np.arange(0, 362, 2), interval=100)
    rot_animation.save(filename+'.gif', dpi=80, writer='imagemagick')
    plt.clf()

def convert_index(value, start, end, num):
    return int((value-start)/(end-start)*(num))

def fullread(Jpm_start, Jpm_end, nJpm, H_start, H_end, nH, field_dir, dir):

    JPMS = np.linspace(Jpm_start, Jpm_end, nJpm)
    HS = np.linspace(H_start, H_end, nH)
    phase_diagram = np.zeros((nJpm,nH))
    entropy_diagram = np.zeros((nJpm, nH))
    if field_dir == "110":
        n = np.array([1,1,0])/np.sqrt(2)
    elif field_dir == "111":
        n = np.array([1,1,1])/np.sqrt(3)
    else:
        n = np.array([0,0,1])
    count = 0
    directory = os.fsencode(dir)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if os.path.isdir(dir + "/" + filename):
            print(dir + "/" + filename)
            info = filename.split("_")
            try:
                S = np.loadtxt(dir + "/" + filename + "/spin239.txt")
                P = np.loadtxt(dir + "/" + filename + "/pos.txt")
                mag = magnetization_local(S)
                phase_diagram[int(info[4]), int(info[5])] = np.linalg.norm(mag)
                heat_capacity = np.loadtxt(dir + "/" + filename + "/heat_capacity.txt", unpack=True)
                heat_capacity = np.flip(heat_capacity, axis=1)
                heat_capacity = heat_capacity[:,40:]
                heat_capacity[1] = heat_capacity[1] * 8.6173303e-2
                Cv_integrand = heat_capacity[1]/heat_capacity[0]
                S = np.zeros(len(heat_capacity[1])-1)
                for i in range(1,len(heat_capacity[1])):
                    S[i-1] = -np.trapz(Cv_integrand[i:], heat_capacity[0][i:]) + np.log(2)
                np.savetxt(dir + "/" + filename + "/entropy.txt", S)
                entropy_diagram[int(info[4]), int(info[5])] = S[0] if S[0] > 0 else 0
            except:
                phase_diagram[int(info[4]), int(info[5])] = np.nan
                entropy_diagram[int(info[4]), int(info[5])] = np.nan
            count = count + 1

    np.savetxt(dir+"_magnetization.txt", phase_diagram)
    np.savetxt(dir+"_entropy.txt", entropy_diagram)
    # plt.imshow(phase_diagram.T, origin="lower", aspect="auto", extent=[Jpm_start, Jpm_end, H_start, H_end])
    # plt.scatter(JPMS, HS, c=phase_diagram)
    plt.imshow(phase_diagram.T, extent=[Jpm_start, Jpm_end, H_start, H_end], origin='lower', aspect='auto')
    plt.colorbar()
    plt.savefig(dir+"_magnetization.pdf")
    plt.clf()
    plt.imshow(entropy_diagram.T, extent=[Jpm_start, Jpm_end, H_start, H_end], origin='lower', aspect='auto')
    plt.colorbar()
    plt.savefig(dir+"_entropy.pdf")
    plt.clf()


def read_MC(Jpm_start, Jpm_end, nJpm, H_start, H_end, nH, field_dir, dir, filename):

    JPMS = np.linspace(Jpm_start, Jpm_end, nJpm)
    HS = np.linspace(H_start, H_end, nH)
    phase_diagram = np.zeros((nJpm, nH))

    if field_dir == "110":
        n = np.array([1,1,0])/np.sqrt(2)
    elif field_dir == "111":
        n = np.array([1,1,1])/np.sqrt(3)
    else:
        n = np.array([0,0,1])

    S = np.loadtxt(dir + "/" + filename + "/spin.txt")
    P = np.loadtxt(dir + "/" + filename + "/pos.txt")
    S_global = np.zeros(S.shape)
    for i in range(4):
        S_global[i::4] = contract('js, sp->jp', S[i::4], localframe[:,i,:])

    mag = magnetization(S, n)
    print(mag)
    plot_lattice(P[0:4], S_global[0:4],  dir + filename + "_real_config.pdf")
    if not os.path.isdir( dir + filename + "/SSSF/"):
        os.mkdir( dir + filename + "/SSSF/")
    SSSF_collect(S, P, 50, dir + filename + "/SSSF/", field_dir, True)

#Jpm_0.285000_h_1.650000_index_195_55
#Jpm_0.054000_h_1.050000_index_118_35
#Jpm_-0.300000_h_1.320000_index_0_44
#Jpm_-0.300000_h_0.000000_index_0_0
# read_MC(-0.3, 0.3, 200, 0, 3.0, 100, "110", "/scratch/zhouzb79/MC_phase_diagram_CZO_110/", "Jpm_-0.300000_h_0.000000_index_0_0")
# fullread(-0.3, 0.3, 200, 0, 3.0, 100, "001", "/scratch/zhouzb79/MC_phase_diagram_CZO_001")
# fullread(-0.3, 0.3, 200, 0, 3.0, 100, "110", "/scratch/zhouzb79/MC_phase_diagram_CZO_110")
# fullread(-0.3, 0.3, 200, 0, 3.0, 100, "111", "/scratch/zhouzb79/MC_phase_diagram_CZO_111")
# fullread(-0.3, 0.3, 200, 0, 8.0, 100, "001", "/scratch/zhouzb79/MC_phase_diagram_CZO_001_XAIAO")
# fullread(-0.3, 0.3, 200, 0, 8.0 , 100, "110", "/scratch/zhouzb79/MC_phase_diagram_CZO_110_XAIAO")
# fullread(-0.3, 0.3, 200, 0, 8.0, 100, "111", "/scratch/zhouzb79/MC_phase_diagram_CZO_111_XAIAO")
fullread(-0.3, 0.3, 50, 0, 8.0, 20, "001", "/scratch/zhouzb79/MC_phase_diagram_XYZ_001_XAIAO")
fullread(-0.3, 0.3, 50, 0, 8.0 , 20, "110", "/scratch/zhouzb79/MC_phase_diagram_XYZ_110_XAIAO")
fullread(-0.3, 0.3, 50, 0, 8.0, 20, "111", "/scratch/zhouzb79/MC_phase_diagram_XYZ_111_XAIAO")
