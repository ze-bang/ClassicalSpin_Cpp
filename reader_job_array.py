import h5py
import numpy as np
from opt_einsum import contract
import matplotlib.pyplot as plt
import os
import matplotlib.tri as mtri 
from numba import njit
import matplotlib as mpl
plt.rcParams['text.usetex'] = True
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"


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


def magnetostriction(S, h, dir):
    tau = np.zeros((4,3,int(len(S)/4)))
    for i in range(4):
        for j in range(3):
            tau[i,j] = S[i::4][:,j]
    C_B = 1
    C_11 = 1
    C_22 = 0
    C_44 = 1
    g = np.array([4e-7, -8e-7, 12e-7, -2.6e-7, 0.27e-7, -0.8e-7, 0.5e-7, -0.7e-7, 0.43e-7, 0.51e-7])
    
    if dir == "111":
        L111_111 = h/(27*C_B)*((g[9]+2*g[8])*(3*tau[0][2]-tau[1][2]-tau[2][2]-tau[3][2]) + (g[3]+2*g[2])* (3*tau[0][0]-tau[1][0]-tau[2][0]-tau[3][0]))\
                +4/(27*C_44)*h*((8*np.sqrt(2)*g[0]-4*g[1])*(tau[1][0]+tau[2][0]+tau[3][0]) + (g[3]-g[2])* (9*tau[0][0]+tau[1][0]+tau[2][0]+tau[3][0])\
                +(8*np.sqrt(2)*g[6]-4*g[7])*(tau[1][2]+tau[2][2]+tau[3][2]) + (g[9]-g[8])* (9*tau[0][2]+tau[1][2]+tau[2][2]+tau[3][2]))
        
        L111_110 = h/(27*C_B)*((g[9]+2*g[8])*(3*tau[0][2]-tau[1][2]-tau[2][2]-tau[3][2]) + (g[3]+2*g[2])* (3*tau[0][0]-tau[1][0]-tau[2][0]-tau[3][0]))\
                + 1/(18*np.sqrt(3)*(C_11-C_22))*h*((np.sqrt(6)*g[0]+np.sqrt(3)*g[1])*(tau[1][0]+tau[2][0]-2*tau[3][0]) +(np.sqrt(6)*g[6]+np.sqrt(3)*g[7])*(tau[1][2]+tau[2][2]-2*tau[3][2]))\
                - 2/(9*C_44) * h * ((g[2]-g[3])*(3*tau[0][0]+tau[1][0]+tau[2][0]-tau[3][0]) + (g[1]-2*np.sqrt(2)*g[0])* (tau[1][0]+tau[2][0]+2*tau[3][0]) \
                                    +(g[8]-g[9])*(3*tau[0][2]+tau[1][2]+tau[2][2]-tau[3][2]) + (g[7]-2*np.sqrt(2)*g[6])* (tau[1][2]+tau[2][2]+2*tau[3][2])) \
                + ((np.sqrt(2)*g[4]-g[5])/(6*np.sqrt(3)*(C_11-C_22)) + 2*(2*np.sqrt(6)*g[4]+np.sqrt(3)*g[5])/(9*C_44))*h*(tau[1][1]-tau[2][1])
        
        L111_001 = h/(27*C_B)*((g[9]+2*g[8])*(3*tau[0][2]-tau[1][2]-tau[2][2]-tau[3][2]) + (g[3]+2*g[2])* (3*tau[0][0]-tau[1][0]-tau[2][0]-tau[3][0]))\
                - 1/(9*np.sqrt(3)*(C_11-C_22))*h*((np.sqrt(6)*g[0]+np.sqrt(3)*g[1])*(tau[1][0]+tau[2][0]-2*tau[3][0]) +(np.sqrt(6)*g[6]+np.sqrt(3)*g[7])*(tau[1][2]+tau[2][2]-2*tau[3][2])\
                + (3*np.sqrt(2)*g[4] - 3*g[5])*(tau[1][1]-tau[2][1]))
        
        A_1 = np.mean(3*tau[0][2]-tau[1][2]-tau[2][2]-tau[3][2])
        A_2 = np.mean(3*tau[0][0]-tau[1][0]-tau[2][0]-tau[3][0])
        A_3 = np.mean(tau[1][0]+tau[2][0]+tau[3][0])
        A_4 = np.mean(tau[1][2]+tau[2][2]+tau[3][2])
        A_5 = np.mean(tau[1][1]-tau[2][1])
        return np.array([np.mean(L111_111), np.mean(L111_110), np.mean(L111_001),A_1,A_2,A_3,A_4,A_5])
    elif dir == "110":
        L110_111 = np.sqrt(2)/(9*np.sqrt(3)*C_B)*h*((g[9]+2*g[8])*(tau[0][2]-tau[3][2]) + (g[3]+2*g[2])* (tau[0][0]-tau[3][0]))\
                + 2/(27*C_44)*h*(-2*np.sqrt(6)*(g[2]-g[3])*(3*tau[0][0]+tau[3][0]) + np.sqrt(3)*(4*g[0]-np.sqrt(2)*g[1])*(3*tau[1][0]+3*tau[2][0]+2*tau[3][0])\
                - 2*np.sqrt(6)*(g[8]-g[9])*(3*tau[0][2]+tau[3][2]) + np.sqrt(3)*(4*g[6]-np.sqrt(2)*g[7])*(3*tau[1][2]+3*tau[2][2]+2*tau[3][2])\
                + (12*g[4]+3*np.sqrt(2)*g[5])*(tau[1][1]-tau[2][1]))
        
        L110_110 = np.sqrt(2)/(9*np.sqrt(3)*C_B)*h*((g[9]+2*g[8])*(tau[0][2]-tau[3][2]) + (g[3]+2*g[2])* (tau[0][0]-tau[3][0]))\
                + 1/(12*np.sqrt(3)*(C_11-C_22))*h*((2*g[0]+np.sqrt(2)*g[1])*(tau[0][0]-tau[3][0]) +(2*g[6]+np.sqrt(2)*g[7])*(tau[0][2]-tau[3][2])\
                                                + (2*np.sqrt(3)*g[4]-np.sqrt(6)*g[5])*(tau[1][1]-tau[2][1]))\
                + 1/(9*C_44) * h * (np.sqrt(3)*(-4*g[0]+np.sqrt(2)*g[1]-2*np.sqrt(2)*g[2]+2*np.sqrt(2)*g[3])*(tau[0][0]-tau[3][0])\
                                    +np.sqrt(3)*(-4*g[6]+np.sqrt(2)*g[7]-2*np.sqrt(2)*g[8]+2*np.sqrt(2)*g[9])*(tau[0][2]-tau[3][2])\
                                    +(12*g[4]+3*np.sqrt(2)*g[5])*(tau[1][1]-tau[2][1]))
        
        L110_001 = np.sqrt(2)/(9*np.sqrt(3)*C_B)*h*((g[9]+2*g[8])*(tau[0][2]-tau[3][2]) + (g[3]+2*g[2])* (tau[0][0]-tau[3][0]))\
                + 1/(6*np.sqrt(3))*h*((-2*g[0]-np.sqrt(2)*g[1])*(tau[0][0]-tau[3][0]) +(-2*g[6]-np.sqrt(2)*g[7])*(tau[0][2]-tau[3][2])\
                + (-2*np.sqrt(3)*g[4]+np.sqrt(6)*g[5])*(tau[1][1]-tau[2][1]))
        A_1 = np.mean(tau[0][2]-tau[3][2])
        A_2 = np.mean(tau[0][0]-tau[3][0])
        A_3 = np.mean(3*tau[0][0]+tau[3][0])
        A_4 = np.mean(3*tau[0][2]+tau[3][2])
        A_5 = np.mean(tau[1][1]-tau[2][1])
        return np.array([np.mean(L110_111), np.mean(L110_110), np.mean(L110_001),A_1,A_2,A_3,A_4,A_5])
    else:
        L001_111 = 1/(9*np.sqrt(3)*C_B)*h*(2*g[2]+g[3])*(tau[0][0]-tau[1][0]-tau[2][0]+tau[3][0]) + (2*g[8]+g[9])*(tau[0][2]-tau[1][2]-tau[2][2]+tau[3][2])\
                - 4/(27*C_44)*h*((np.sqrt(3)*g[2]-np.sqrt(3)*g[3])*(3*tau[0][0]+tau[1][0]+tau[2][0]-tau[3][0])-(2*np.sqrt(6)*g[0]-np.sqrt(3)*g[1])*(tau[1][0]+tau[2][0]+2*tau[3][0])\
                +(np.sqrt(3)*g[8]-np.sqrt(3)*g[9])*(3*tau[0][2]+tau[1][2]+tau[2][2]- tau[3][2])-(2*np.sqrt(6)*g[6]-np.sqrt(3)*g[7])*(tau[1][2]+tau[2][2]+2*tau[3][2])\
                +(6*np.sqrt(2)*g[4]+3*g[5])*(tau[1][1]-tau[2][1]))
        
        L001_110 = 1/(9*np.sqrt(3)*C_B)*h*(2*g[2]+g[3])*(tau[0][0]-tau[1][0]-tau[2][0]+tau[3][0]) + (2*g[8]+g[9])*(tau[0][2]-tau[1][2]-tau[2][2]+tau[3][2])\
                + 1/(6*np.sqrt(3)*(C_11-C_22))*h*((-np.sqrt(2)*g[0]-g[1])*(tau[0][0]-tau[1][0]-tau[2][0]+tau[3][0])\
                                            +(-np.sqrt(2)*g[6]-g[7])*(tau[0][2]-tau[1][2]-tau[2][2]+tau[3][2]))\
                - 2/(3*np.sqrt(3)*C_44)*h*((-2*np.sqrt(2)*g[0]+g[1]+g[2]-g[3])*(tau[0][0]+tau[1][0]+tau[2][0]+tau[3][0])\
                                        +(-2*np.sqrt(2)*g[6]+g[7]+g[8]-g[9])*(tau[0][2]+tau[1][2]+tau[2][2]+tau[3][2]))
        
        L001_001 = 1/(9*np.sqrt(3)*C_B)*h*(2*g[2]+g[3])*(tau[0][0]-tau[1][0]-tau[2][0]+tau[3][0]) + (2*g[8]+g[9])*(tau[0][2]-tau[1][2]-tau[2][2]+tau[3][2])\
                + 1/(3*np.sqrt(3)*(C_11-C_22)) *h *((np.sqrt(2)*g[0]+g[1])*(tau[0][0]-tau[1][0]-tau[2][0]+tau[3][0]) + (np.sqrt(2)*g[6]+g[7])*(tau[0][2]-tau[1][2]-tau[2][2]+tau[3][2]))
        A_1 = np.mean(tau[0][0]-tau[1][0]-tau[2][0]+tau[3][0])
        A_2 = np.mean(tau[0][2]-tau[1][2]-tau[2][2]+tau[3][2])
        A_3 = np.mean(tau[1][0]+tau[2][0]+2*tau[3][0])
        A_4 = np.mean(tau[1][2]+tau[2][2]+2*tau[3][2])
        A_5 = np.mean(tau[1][1]-tau[2][1])
        return np.array([np.mean(L001_111), np.mean(L001_110), np.mean(L001_001),A_1,A_2,A_3,A_4,A_5])

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

def fullread(Jpm_start, Jpm_end, nJpm, H_start, H_end, nH, field_dir, dir, xorz):

    JPMS = np.linspace(Jpm_start, Jpm_end, nJpm)
    HS = np.linspace(H_start, H_end, nH)
    Energies = np.zeros((nJpm, nH))
    phase_diagram = np.zeros((nJpm,nH))
    entropy_diagram = np.zeros((nJpm, nH))
    mag_diagram = np.zeros((nJpm, nH))
    magnetostrictions = np.zeros((nJpm, nH, 8))

    magnetostriction_string = np.array(["111", "110", "001"])

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
            S = np.loadtxt(dir + "/" + filename + "/spin.txt")
            # E = np.loadtxt(dir + "/" + filename + "/energy199.txt")
            # P = np.loadtxt(dir + "/" + filename + "/pos.txt")
            # try:
            mag = magnetization_local(S)
            phase_diagram[int(info[5]), int(info[6])] = np.abs(mag[xorz])

            magnetostrictions[int(info[5]), int(info[6])] = magnetostriction(S, HS[int(info[6])], field_dir)

                # num_lines = sum(1 for _ in open(dir + "/" + filename + "/heat_capacity.txt"))
                # if num_lines == 600:
                #     # heat_capacity = np.loadtxt(dir + "/" + filename + "/heat_capacity.txt", unpack=True,skiprows=200,max_rows=200)
                #     heat_capacity = np.loadtxt(dir + "/" + filename + "/heat_capacity.txt", unpack=True,skiprows=400,max_rows=200)
                # elif num_lines == 400:
                #     heat_capacity = np.loadtxt(dir + "/" + filename + "/heat_capacity.txt", unpack=True,skiprows=200,max_rows=200)
                # else:
                #     heat_capacity = np.loadtxt(dir + "/" + filename + "/heat_capacity.txt", unpack=True,max_rows=200)                   
                #         # heat_capacity_2 = np.loadtxt(dir + "/" + filename + "/heat_capacity.txt", unpack=True,max_rows=200)
                # # print(num_lines)
                # # heat_capacity[1] = (heat_capacity_2[1]+heat_capacity[1])/2
                # # print(heat_capacity.shape)
                # heat_capacity = np.flip(heat_capacity, axis=1)
                # heat_capacity = heat_capacity[:,10:]
                # heat_capacity[1] = heat_capacity[1] * 8.6173303e-2
                # Cv_integrand = heat_capacity[1]/heat_capacity[0]
                # entropy = np.zeros(len(heat_capacity[1])-1)
                # for i in range(1,len(heat_capacity[1])):
                #     entropy[i-1] = -np.trapz(Cv_integrand[i:], heat_capacity[0][i:]) + np.log(2)
                # np.savetxt(dir + "/" + filename + "/entropy.txt", entropy)
                # entropy_diagram[int(info[4]), int(info[5])] = entropy[0] 
                # mag_diagram[int(info[4]), int(info[5])] = np.abs(magnetization(S, n)[2])
                # Energies[int(info[4]), int(info[5])] = np.mean(E)
            # except:
            #     phase_diagram[int(info[5]), int(info[6])] = np.nan
            #     # entropy_diagram[int(info[4]), int(info[5])] = np.nan
            #     # mag_diagram[int(info[4]), int(info[5])] = np.nan
            #     # Energies[int(info[4]), int(info[5])] = np.nan
            #     magnetostrictions[int(info[5]), int(info[6])] = np.nan
            count = count + 1

    np.savetxt(dir+"_magnetization.txt", phase_diagram)
    np.savetxt(dir+"_entropy.txt", entropy_diagram)
    np.savetxt(dir+"_global_mag.txt", mag_diagram)
    np.savetxt(dir+"_energy.txt", Energies)
    # plt.imshow(phase_diagram.T, origin="lower", aspect="auto", extent=[Jpm_start, Jpm_end, H_start, H_end])
    # plt.scatter(JPMS, HS, c=phase_diagram)
    if not nJpm == 1 and not nH == 1:

        plt.imshow(phase_diagram.T, extent=[Jpm_start, Jpm_end, H_start, H_end], origin='lower', aspect='auto')
        plt.colorbar()
        plt.savefig(dir+"_magnetization.pdf")
        plt.clf()
        # plt.imshow(entropy_diagram.T, extent=[Jpm_start, Jpm_end, H_start, H_end], origin='lower', aspect='auto')
        # plt.colorbar()
        # plt.savefig(dir+"_entropy.pdf")
        # plt.clf()
        # plt.imshow(mag_diagram.T, extent=[Jpm_start, Jpm_end, H_start, H_end], origin='lower', aspect='auto')
        # plt.colorbar()
        # plt.savefig(dir+"_global_mag.pdf")
        # plt.clf()
        # plt.imshow(Energies.T, extent=[Jpm_start, Jpm_end, H_start, H_end], origin='lower', aspect='auto')
        # plt.colorbar()
        # plt.savefig(dir+"_energy.pdf")
        # plt.clf()
        for i in range (3):
            np.savetxt(dir+"_magnetostriction_"+magnetostriction_string[i]+".txt", magnetostrictions[:,:,i])
            plt.imshow(magnetostrictions[:,:,i].T, extent=[Jpm_start, Jpm_end, H_start, H_end], origin='lower', aspect='auto')
            plt.colorbar()
            plt.savefig(dir+"_magnetostriction_"+magnetostriction_string[i]+".pdf")
            plt.clf()
        for i in range (5):
            np.savetxt(dir+"_magnetostriction_"+str(3+i)+".txt", magnetostrictions[:,:,3+i])
            plt.imshow(magnetostrictions[:,:,3+i].T, extent=[Jpm_start, Jpm_end, H_start, H_end], origin='lower', aspect='auto')
            plt.colorbar()
            plt.savefig(dir+"_magnetostriction_"+str(3+i)+".pdf")
            plt.clf()
            
    elif nJpm == 1:
        phase_diagram = phase_diagram.flatten()
        # entropy_diagram = entropy_diagram.flatten()
        # mag_diagram = mag_diagram.flatten()
        # Energies = Energies.flatten()
        magnetostrictions = magnetostrictions.reshape((nH,3))
        plt.plot(HS, phase_diagram)
        plt.savefig(dir+"_magnetization.pdf")
        plt.clf()
        # plt.plot(HS, entropy_diagram)
        # plt.savefig(dir+"_entropy.pdf")
        # plt.clf()
        # plt.plot(HS, mag_diagram)
        # plt.savefig(dir+"_global_mag.pdf")
        # plt.clf()
        # plt.plot(HS, Energies)
        # plt.savefig(dir+"_energy.pdf")
        # plt.clf()
        for i in range (3):
            np.savetxt(dir+"_magnetostriction_"+magnetostriction_string[i]+".txt", magnetostrictions[:,i])
        plt.scatter(HS, magnetostrictions[:,0])
        plt.scatter(HS, magnetostrictions[:,1])
        plt.scatter(HS, magnetostrictions[:,2])
        plt.legend(["111", "110", "001"])
        plt.savefig(dir+"_magnetostriction.pdf")
        plt.clf()

def lineread(H_start, H_end, nH, field_dir, dir, xorz,ax,imp=False):
    # fig, ax = plt.subplots(constrained_layout="True")
    HS = np.linspace(H_start, H_end, nH)
    Energies = np.zeros(nH)
    phase_diagram = np.zeros(nH)
    entropy_diagram = np.zeros(nH)
    mag_diagram = np.zeros(nH)
    magnetostrictions = np.zeros((nH, 8))

    magnetostriction_string = np.array(["111", "110", "001"])

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
            S = np.loadtxt(dir + "/" + filename + "/spin.txt")
            # E = np.loadtxt(dir + "/" + filename + "/energy199.txt")
            # P = np.loadtxt(dir + "/" + filename + "/pos.txt")
            mag = magnetization_local(S)
            phase_diagram[int(info[3])] = np.abs(mag[xorz])

            magnetostrictions[int(info[3])] = magnetostriction(S, HS[int(info[3])], field_dir)
            count = count + 1
            # except:
            #     phase_diagram[int(info[3])] = np.nan
            #     magnetostrictions[int(info[3])] = np.nan
            #     count = count + 1

    np.savetxt(dir+"/magnetization.txt", phase_diagram)
    np.savetxt(dir+"/entropy.txt", entropy_diagram)
    np.savetxt(dir+"/global_mag.txt", mag_diagram)
    np.savetxt(dir+"/energy.txt", Energies)
    
    phase_diagram = phase_diagram.flatten()
    entropy_diagram = entropy_diagram.flatten()
    mag_diagram = mag_diagram.flatten()
    Energies = Energies.flatten()
    magnetostrictions = magnetostrictions.reshape((nH,8))
    # ax.plot(HS, phase_diagram)
    # ax.savefig(dir+"/magnetization.pdf")
    # ax.clf()
    # for i in range (3):
    #     np.savetxt(dir+"/magnetostriction_"+magnetostriction_string[i]+".txt", magnetostrictions[:,i])
    # for i in range (5):
    #     np.savetxt(dir+"/magnetostriction_"+str(3+i)+".txt", magnetostrictions[:,3+i])
    #     ax.scatter(HS, magnetostrictions[:,3+i])
    #     ax.colorbar()
    #     ax.savefig(dir+"/magnetostriction_"+str(3+i)+".pdf")
    #     ax.clf()
    if imp:
        ax.scatter(HS, magnetostrictions[:,3])
    else:
        ax.scatter(HS, magnetostrictions[:,0])
        ax.scatter(HS, magnetostrictions[:,1])
        ax.scatter(HS, magnetostrictions[:,2])
        ax.legend([r"$L^{(" + field_dir +")}_{[111]}$", r"$L^{(" + field_dir +")}_{[110]}$", r"$L^{(" + field_dir +")}_{[001]}$"], fontsize="16")
    # plt.set_ylabel(r"$\Delta L/L$")
    # plt.set_xlabel(r"$h/J_{yy}$")
    # plt.savefig(dir+"/magnetostriction.pdf")
    # plt.clf()


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
# fullread(-0.3, 0.3, 50, 0, 8.0, 20, "001", "/scratch/y/ybkim/zhouzb79/MC_phase_diagram_XYZ_001_XAIAO", 0)
# fullread(-0.3, 0.3, 50, 0, 15.0 , 20, "110", "/scratch/y/ybkim/zhouzb79/MC_phase_diagram_XYZ_110_XAIAO_High_field", 0)
# fullread(-0.3, 0.3, 50, 0, 8.0, 20, "111", "/scratch/y/ybkim/zhouzb79/MC_phase_diagram_XYZ_111_XAIAO", 0)
# fullread(-0.3, 0.3, 50, 0, 8.0, 20, "001", "/scratch/y/ybkim/zhouzb79/MC_phase_diagram_XYZ_001_ZAIAO", 2)
# fullread(-0.3, 0.3, 50, 0, 8.0 , 20, "110", "/scratch/y/ybkim/zhouzb79/MC_phase_diagram_XYZ_110_ZAIAO", 2)
# fullread(-0.3, 0.3, 50, 0, 8.0, 20, "111", "/scratch/y/ybkim/zhouzb79/MC_phase_diagram_XYZ_111_ZAIAO", 2)
# fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 8), sharey=True, sharex=True, constrained_layout=True)
# ax[0,0].set_title("CSO Octupolar")
# ax[0,1].set_title("CSO Dipolar")
# ax[1,0].set_title("CHO Octupolar")
# ax[1,1].set_title("CHO Dipolar")

directory = "/Users/zhengbangzhou/Library/CloudStorage/OneDrive-UniversityofToronto/PhD Stuff/Projects/PSG_Pyrochlore/XYZ_project/magnetostriction"


def graph_magnetostriction(filename):
    mpl.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(ncols=3, nrows=2,constrained_layout=True, figsize=(16,8))
    ax[0,0].text(.01, .99, r"$(\mathrm{a})$", ha='left', va='top', transform=ax[0,0].transAxes,
                zorder=10,color='black')
    ax[0,1].text(.01, .99, r"$(\mathrm{b})$", ha='left', va='top', transform=ax[0,1].transAxes,
                zorder=10,color='black')
    ax[0,2].text(.01, .99, r"$(\mathrm{c})$", ha='left', va='top', transform=ax[0,2].transAxes,
                zorder=10,color='black')
    ax[1,0].text(.01, .99, r"$(\mathrm{d})$", ha='left', va='top', transform=ax[1,0].transAxes,
                zorder=10,color='black')
    ax[1,1].text(.01, .99, r"$(\mathrm{e})$", ha='left', va='top', transform=ax[1,1].transAxes,
                zorder=10,color='black')
    ax[1,2].text(.01, .99, r"$(\mathrm{f})$", ha='left', va='top', transform=ax[1,2].transAxes,
                zorder=10,color='black')
    
    lineread(0, 8, 20, "111", filename+"_octupolar_111", 0, ax[0,0])
    ax[0,0].set_title(r"$B\parallel (111)$")
    ax[0,0].set_ylabel(r"$\Delta L/L$")
    lineread(0, 8, 20, "110", filename+"_octupolar_110", 0, ax[0,1])
    ax[0,1].set_title(r"$B\parallel (110)$")
    lineread(0, 8, 20, "001", filename+"_octupolar_001", 0, ax[0,2])
    ax[0,2].set_title(r"$B\parallel (001)$")
    lineread(0, 8, 20, "111", filename+"_dipolar_111", 0, ax[1,0])
    ax[1,0].set_ylabel(r"$\Delta L/L$")
    lineread(0, 8, 20, "110", filename+"_dipolar_110", 0, ax[1,1])
    lineread(0, 8, 20, "001", filename+"_dipolar_001", 0, ax[1,2])
    ax[1,0].set_xlabel(r"$h/J_{yy}$")
    ax[1,1].set_xlabel(r"$h/J_{yy}$")
    ax[1,2].set_xlabel(r"$h/J_{yy}$")
    plt.savefig(filename+"_magnetostriction.pdf")
# graph_magnetostriction(directory+"/CZO")
# graph_magnetostriction(directory+"/CHO")
graph_magnetostriction(directory+"/CSO")

mpl.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8, 8), constrained_layout=True)
ax[0,0].text(.01, .99, r"$(\mathrm{a})$", ha='left', va='top', transform=ax[0,0].transAxes,
            zorder=10,color='black')
ax[0,1].text(.01, .99, r"$(\mathrm{b})$", ha='left', va='top', transform=ax[0,1].transAxes,
            zorder=10,color='black')
ax[1,0].text(.1, .99, r"$(\mathrm{c})$", ha='left', va='top', transform=ax[1,0].transAxes,
            zorder=10,color='black')
ax[1,1].text(.01, .99, r"$(\mathrm{d})$", ha='left', va='top', transform=ax[1,1].transAxes,
            zorder=10,color='black')
# graph_magnetostriction(directory+"/CZO")
ax[0,0].set_ylabel(r"$\Delta L/L$")
# ax[0].set_xlabel(r"$h/J_{yy}$")
lineread(0, 8, 20, "001", directory+"/CHO_octupolar_001", 0, ax[0,0])
# lineread(0, 8, 20, "111", directory+"/CHO_dipolar_111", 0)
# lineread(0, 8, 20, "110", directory+"/CHO_dipolar_110", 0)
lineread(0, 8, 20, "001", directory+"/CHO_dipolar_001", 0, ax[0,1])
# ax[1].set_ylabel(r"$\Delta L/L$")
# ax[1].set_xlabel(r"$h/J_{yy}$")
lineread(0, 8, 20, "001", directory+"/CHO_octupolar_001", 0, ax[1,0],True)
ax[1,0].set_ylabel(r"$S_0^x-S_1^x-S_2^x+S_3^x$")
ax[1,0].set_xlabel(r"$h/J_{yy}$")
# lineread(0, 8, 20, "111", directory+"/CHO_dipolar_111", 0)
# lineread(0, 8, 20, "110", directory+"/CHO_dipolar_110", 0)
lineread(0, 8, 20, "001", directory+"/CHO_dipolar_001", 0, ax[1,1],True)
ax[1,1].set_xlabel(r"$h/J_{yy}$")

plt.savefig("magnetostriction.pdf")

# ax[0,0].text(0.1, 0.95, "(a)", transform=ax[0,0].transAxes, fontsize=12)
# ax[0,1].text(0.1, 0.95, "(b)", transform=ax[0,1].transAxes, fontsize=12)
# ax[1,0].text(0.1, 0.95, "(c)", transform=ax[1,0].transAxes, fontsize=12)
# ax[1,1].text(0.1, 0.95, "(d)", transform=ax[1,1].transAxes, fontsize=12)
# lineread(0, 8, 20, "111", "CSO_octupolar_111", 0)
# lineread(0, 8, 20, "110", "CSO_octupolar_110", 0)
# lineread(0, 8, 20, "001", "CSO_octupolar_001", 0)
# lineread(0, 8, 20, "111", "CSO_dipolar_111", 0)
# lineread(0, 8, 20, "110", "CSO_dipolar_110", 0)
# lineread(0, 8, 20, "001", "CSO_dipolar_001", 0)
# lineread(0, 8, 20, "111", "CHO_octupolar_111", 0)
# lineread(0, 8, 20, "110", "CHO_octupolar_110_test", 0)
# lineread(0, 8, 20, "110", "CHO_octupolar_110", 0)
# lineread(0, 8, 20, "001", "CHO_octupolar_001", 0)
# lineread(0, 8, 20, "111", "CHO_dipolar_111", 0)
# lineread(0, 8, 20, "110", "CHO_dipolar_110", 0)
# lineread(0, 8, 20, "001", "CHO_dipolar_001", 0)
# plt.savefig("magnetostriction.pdf")
# 0.1375 0.1375 1 0.2375