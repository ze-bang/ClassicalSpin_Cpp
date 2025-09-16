import h5py
import numpy as np
from opt_einsum import contract
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, ensures 3D projection is registered
import os
import sys
from math import gcd
from functools import reduce
# plt.rcParams['text.usetex'] = True
import re
from scipy.optimize import minimize
from scipy.spatial import Delaunay, cKDTree
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, generate_binary_structure, label as ndi_label
from scipy.stats import zscore
def honeycomb_reciprocal_basis():
    """
    Calculate reciprocal lattice vectors for a honeycomb lattice.
    
    Real space basis vectors:
    a1 = (1, 0, 0)
    a2 = (1/2, sqrt(3)/2, 0)
    
    Returns:
        numpy.ndarray: Reciprocal lattice vectors b1 and b2
    """
    # a1 = np.array([0, 1, 0])
    # a2 = np.array([np.sqrt(3)/2, 1/2, 0])
    # a3 = np.array([0, 0, 1])  # Third basis vector (perpendicular to plane)
    
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

kitaevLocal = np.array([[1/np.sqrt(6),1/np.sqrt(6),-2/np.sqrt(6)],[-1/np.sqrt(2),1/np.sqrt(2),0],[1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)]])

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
    # Plot the first Brillouin zone (a hexagon)
    # The vertices of the hexagon in units of 2*pi/a
    # We assume a=1, so the units are 2*pi
    # The user's plot seems to be in units of (kx/2pi, ky/2pi)
    # So we need to divide the standard BZ coordinates by 2*pi
    
    # Get the reciprocal lattice vectors from the honeycomb_reciprocal_basis function
    reciprocal_basis = honeycomb_reciprocal_basis()
    b1 = reciprocal_basis[0] # First reciprocal lattice vector
    b2 = reciprocal_basis[1]  # Second reciprocal lattice vector

    # Create the hexagonal Brillouin zone using the reciprocal lattice vectors
    # Vertices of the hexagon in reciprocal space
    bz_vertices = np.array([
        b1 * (-1/3) + b2 * (1/3),      # K point
        b1 * (1/3) + b2 * (2/3),     # K' point
        b1 * (2/3) + b2 * (1/3),    # -K point
        b1 * (1/3) + b2 * (-1/3),    # -K point
        b1 * (-1/3) + b2 * (-2/3),     # -K' point
        b1 * (-2/3) + b2 * (-1/3),    # -K point
        b1 * (-1/3) + b2 * (1/3),      # K point
    ])

    # High-symmetry points
    gamma_point = np.array([0, 0, 0])
    k_point = b1 * (-1/3) + b2 * (1/3)
    m_point = b1 * 0 + b2 * (1/2)

    # Extract only x and y components for 2D plotting
    bz_vertices_plot = bz_vertices[:, :2]
    gamma_plot = gamma_point[:2]
    k_plot = k_point[:2]
    m_plot = m_point[:2]

    plt.plot(bz_vertices_plot[:, 0], bz_vertices_plot[:, 1], 'w--', lw=1.5)

    # Plot symmetry points and labels
    plt.scatter([gamma_plot[0], k_plot[0], m_plot[0]], [gamma_plot[1], k_plot[1], m_plot[1]], c='white', s=50, zorder=5)
    plt.text(gamma_plot[0] + 0.02, gamma_plot[1] + 0.02, r'$\Gamma$', color='white', fontsize=14)
    plt.text(k_plot[0] + 0.02, k_plot[1] + 0.02, 'K', color='white', fontsize=14)
    plt.text(m_plot[0] + 0.02, m_plot[1] + 0.02, 'M', color='white', fontsize=14)

    plt.pcolormesh(A, B, d1)
    plt.colorbar()
    plt.ylabel(r'$K_y$')
    plt.xlabel(r'$K_x$')
    plt.savefig(filename + ".pdf")
    plt.clf()

def SSSFGraph2D_flat(A, B, d1, filename):

    # Create the hexagonal Brillouin zone using the reciprocal lattice vectors
    # Vertices of the hexagon in reciprocal space
    bz_vertices = np.array([
        [-1/3, 1/3],      # K point
        [1/3, 2/3],     # K' point
        [2/3, 1/3],    # -K point
        [1/3, -1/3],    # -K point
        [-1/3, -2/3],     # -K' point
        [-2/3, -1/3],    # -K point
        [-1/3, 1/3]       # Close the hexagon
    ])

    # High-symmetry points
    gamma_point = np.array([0, 0, 0])
    k_point = [-1/3, 1/3]
    m_point = [0, 1/2]

    # Extract only x and y components for 2D plotting
    bz_vertices_plot = bz_vertices
    gamma_plot = gamma_point
    k_plot = k_point
    m_plot = m_point

    plt.plot(bz_vertices_plot[:, 0], bz_vertices_plot[:, 1], 'w--', lw=1.5)

    # Plot symmetry points and labels
    plt.scatter([gamma_plot[0], k_plot[0], m_plot[0]], [gamma_plot[1], k_plot[1], m_plot[1]], c='white', s=50, zorder=5)
    plt.text(gamma_plot[0] + 0.02, gamma_plot[1] + 0.02, r'$\Gamma$', color='white', fontsize=14)
    plt.text(k_plot[0] + 0.02, k_plot[1] + 0.02, 'K', color='white', fontsize=14)
    plt.text(m_plot[0] + 0.02, m_plot[1] + 0.02, 'M', color='white', fontsize=14)

    plt.pcolormesh(A, B, d1)
    plt.colorbar()
    plt.ylabel(r'$h$')
    plt.xlabel(r'$k$')
    plt.savefig(filename + ".pdf")
    plt.clf()

def hnhltoK(H, L, K=0):
    A = contract('ij,k->ijk',H, 2*np.array([np.pi,-np.pi,0])) \
        + contract('ij,k->ijk',L, 2*np.array([0,0,np.pi]))
    return A

def hhztoK(H, K):
    return contract('ij,k->ijk',H, np.array([1,0,0])) + contract('ij,k->ijk',K, np.array([0,1,0]))

def hk2d(H,K):
    return contract('ij,k->ijk',H, 2*np.array([np.pi,0, 0])) + contract('ij,k->ijk',K, 2*np.array([0,np.pi, 0]))

def hhknk(H,K):
    A = contract('ij,k->ijk',H,  np.array([1,1, 0])) + contract('ij,k->ijk',K, np.array([1,-1, 0]))
    return contract('ijk,ka->ija', A, KBasis)

def hk0(H,K):
    A = contract('ij,k->ijk',H,  np.array([1,0, 0])) + contract('ij,k->ijk',K, np.array([0,1, 0]))
    return contract('ijk,ka->ija', A, KBasis)

def hhknk_2D(H,K):
    return contract('ij,k->ijk', H, np.array([1,0])) + contract('ij,k->ijk',K, np.array([0,1]))

def hk0_rlu(H, K):
    A = contract('ij,k->ijk',H,  np.array([1,0, 0])) + contract('ij,k->ijk',K, np.array([0,1, 0]))
    return A

def SSSF2D(S, P, nK, dir, gb=False):
    H = np.linspace(-2*np.pi, 2*np.pi, nK)
    L = np.linspace(-2*np.pi, 2*np.pi, nK)
    A, B = np.meshgrid(H, L)
    K = hhztoK(A, B).reshape((nK*nK,3))
    SSSF = SSSF_q(K, S, P, gb)
    SSSF = SSSF.reshape((nK, nK, 3, 3))
    total_SSSF = contract('ijab->ij', SSSF)
    
    SSSFGraph2D(A, B, total_SSSF, dir+"/SSSF_tot")

    def save_top_points(SSSF_target, SSSF, dir, file_prefix):
        # Save the top N points (total SSSF)
        N_TOP = 1  # Number of top points to save
        
        # Get the top N indices
        flat_indices = np.argpartition(SSSF_target.flatten(), -N_TOP)[-N_TOP:]
        flat_indices = flat_indices[np.argsort(-SSSF_target.flatten()[flat_indices])]  # Sort by intensity
        
        # Convert flat indices to 2D indices
        top_indices_2d = np.unravel_index(flat_indices, SSSF_target.shape)
        
        # Prepare data for all top N points
        component_labels = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
        header = "Kx Ky Kz Intensity " + " ".join([f"SSSF_{lbl}" for lbl in component_labels])
        
        top_points_data = []
        top_points_rlu_data = []
        
        for idx, (i, j) in enumerate(zip(top_indices_2d[0], top_indices_2d[1])):
            point = K[i * SSSF_target.shape[1] + j]
            intensity = contract('ijab->ij', SSSF)[i, j]
            components = SSSF[i, j, :, :].flatten()
            
            # Save in Cartesian coordinates (divided by 2pi)
            row = np.concatenate([point, [intensity], components])
            top_points_data.append(row)
            
            # Convert to RLU
            point_rlu = np.linalg.solve(KBasis.T, point)
            row_rlu = np.concatenate([point_rlu, [intensity], components])
            top_points_rlu_data.append(row_rlu)
            
            if idx == 0:  # First point is the maximum
                print(f"Maximum total SSSF = {intensity:.6f} at H={point[0]:.4f}, K={point[1]:.4f}, L={point[2]:.4f}")
                print(f"Maximum total SSSF in RLU = {intensity:.6f} at h={point_rlu[0]:.4f}, k={point_rlu[1]:.4f}, l={point_rlu[2]:.4f}")
        
        # Save all top N points
        np.savetxt(dir+f"/SSSF_{file_prefix}_top{N_TOP}_points.txt", 
                np.array(top_points_data) / (2*np.pi), 
                header=header)
        
        np.savetxt(dir+f"/SSSF_{file_prefix}_top{N_TOP}_points_rlu.txt", 
                np.array(top_points_rlu_data), 
                header="h k l Intensity " + " ".join([f"SSSF_{lbl}" for lbl in component_labels]))
    
    save_top_points(total_SSSF, SSSF, dir, "total")

    # Find and plot the maximum point for each component (xx, yy, zz)
    for i, component in enumerate(['x', 'y', 'z']):
        for j, comp in enumerate(['x', 'y', 'z']):
            save_top_points(SSSF[:,:,i,j], SSSF, dir, f"{component}{comp}")
            SSSFGraph2D(A, B, SSSF[:,:,i,j], dir+f"/SSSF_{component}{comp}")
    

    # For yy component
    yy_component = SSSF[:,:,1,1].flatten()
    yy_top_50_indices = np.argsort(yy_component)[-50:][::-1]  # Top 50 indices in descending order
    yy_top_50_points = K[yy_top_50_indices] 
    yy_top_50_intensities = yy_component[yy_top_50_indices]
    xx_intensities_at_yy_top = SSSF.reshape(nK*nK, 3, 3)[yy_top_50_indices, 0, 0]
    
    # Save top 50 yy points with corresponding xx intensities
    yy_analysis_data = np.column_stack([yy_top_50_points/ (2*np.pi), yy_top_50_intensities, xx_intensities_at_yy_top])
    np.savetxt(dir+"/SSSF_yy_top50_analysis.txt", 
              yy_analysis_data, 
              header="H K L SSSF_yy SSSF_xx_at_yy_point")
    
    # For xx component
    xx_component = SSSF[:,:,0,0].flatten()
    xx_top_50_indices = np.argsort(xx_component)[-50:][::-1]  # Top 50 indices in descending order
    xx_top_50_points = K[xx_top_50_indices]
    xx_top_50_intensities = xx_component[xx_top_50_indices]
    yy_intensities_at_xx_top = SSSF.reshape(nK*nK, 3, 3)[xx_top_50_indices, 1, 1]
    
    # Save top 50 xx points with corresponding yy intensities
    xx_analysis_data = np.column_stack([xx_top_50_points/ (2*np.pi), xx_top_50_intensities, yy_intensities_at_xx_top])
    np.savetxt(dir+"/SSSF_xx_top50_analysis.txt", 
              xx_analysis_data, 
              header="H K L SSSF_xx SSSF_yy_at_xx_point")
    
    # Create subplot figure showing top points on both xx and yy plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top-left: SSSF_yy with top yy points labeled
    im1 = axes[0,0].pcolormesh(A, B, SSSF[:,:,1,1], shading='auto')
    axes[0,0].scatter(yy_top_50_points[:5,0], yy_top_50_points[:5,1], 
                     c='red', s=100, marker='x', linewidths=3)
    for i in range(5):
        axes[0,0].annotate(f'{i+1}', (yy_top_50_points[i,0], yy_top_50_points[i,1]), 
                          xytext=(5, 5), textcoords='offset points', 
                          fontsize=12, color='red', weight='bold')
        
    axes[0,0].set_title('SSSF_yy with Top 5 YY Points')
    axes[0,0].set_xlabel('H')
    axes[0,0].set_ylabel('K')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Top-right: SSSF_xx with top yy points labeled
    im2 = axes[0,1].pcolormesh(A, B, SSSF[:,:,0,0], shading='auto')
    axes[0,1].scatter(yy_top_50_points[:5,0], yy_top_50_points[:5,1], 
                     c='red', s=100, marker='x', linewidths=3)
    for i in range(5):
        axes[0,1].annotate(f'{i+1}', (yy_top_50_points[i,0], yy_top_50_points[i,1]), 
                          xytext=(5, 5), textcoords='offset points', 
                          fontsize=12, color='red', weight='bold')
    axes[0,1].set_title('SSSF_xx with Top 5 YY Points')
    axes[0,1].set_xlabel('H')
    axes[0,1].set_ylabel('K')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Bottom-left: SSSF_yy with top xx points labeled
    im3 = axes[1,0].pcolormesh(A, B, SSSF[:,:,1,1], shading='auto')
    axes[1,0].scatter(xx_top_50_points[:5,0], xx_top_50_points[:5,1], 
                     c='blue', s=100, marker='o', linewidths=2, facecolors='none')
    for i in range(5):
        axes[1,0].annotate(f'{i+1}', (xx_top_50_points[i,0], xx_top_50_points[i,1]), 
                          xytext=(5, 5), textcoords='offset points', 
                          fontsize=12, color='blue', weight='bold')
    axes[1,0].set_title('SSSF_yy with Top 5 XX Points')
    axes[1,0].set_xlabel('H')
    axes[1,0].set_ylabel('K')
    plt.colorbar(im3, ax=axes[1,0])
    
    # Bottom-right: SSSF_xx with top xx points labeled
    im4 = axes[1,1].pcolormesh(A, B, SSSF[:,:,0,0], shading='auto')
    axes[1,1].scatter(xx_top_50_points[:5,0], xx_top_50_points[:5,1], 
                     c='blue', s=100, marker='o', linewidths=2, facecolors='none')
    for i in range(5):
        axes[1,1].annotate(f'{i+1}', (xx_top_50_points[i,0], xx_top_50_points[i,1]), 
                          xytext=(5, 5), textcoords='offset points', 
                          fontsize=12, color='blue', weight='bold')
    axes[1,1].set_title('SSSF_xx with Top 5 XX Points')
    axes[1,1].set_xlabel('H')
    axes[1,1].set_ylabel('K')
    plt.colorbar(im4, ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig(dir+"/SSSF_top_points_analysis.pdf")
    plt.clf()
    plt.close()
    
    print(f"Top 5 yy component points:")
    for i in range(5):
        h, k, l = yy_top_50_points[i]
        print(f"  {i+1}. H={h:.4f}, K={k:.4f}, L={l:.4f}: SSSF_yy={yy_top_50_intensities[i]:.6f}, SSSF_xx={xx_intensities_at_yy_top[i]:.6f}")
    
    print(f"Top 5 xx component points:")
    for i in range(5):
        h, k, l = xx_top_50_points[i]
        print(f"  {i+1}. H={h:.4f}, K={k:.4f}, L={l:.4f}: SSSF_xx={xx_top_50_intensities[i]:.6f}, SSSF_yy={yy_intensities_at_xx_top[i]:.6f}")

    return SSSF

def ordering_q_SSSF2D(SSSF, K):
    maxindx = np.argmax(SSSF)

    return K[maxindx]

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

def ordering_q_SSSF(SSSF):
    S = np.abs(SSSF)
    Szz = S[:,0,0]
    max = np.max(Szz)
    if max < 1e-13:
        qzz = np.array([np.NaN, np.NaN, np.NaN])
    else:
        indzz = np.array([])
        tempindzz = np.where(np.abs(Szz-max)<1e-13)[0]
        indzz = np.concatenate((indzz, tempindzz))
        indzz = np.array(indzz.flatten(),dtype=int)
        qzz = SSSF[indzz]
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

def magnetic_moment_along(P, S, dir):
    directions = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [1/2, np.sqrt(3)/2, 0],
        [-1/2, np.sqrt(3)/2, 0],
        [1/2, -np.sqrt(3)/2, 0],
        [-1/2, -np.sqrt(3)/2, 0]
    ])

    z_axis = np.array([0, 0, 1])

    # Normalize chain directions
    u_dirs = directions

    # In-plane perpendicular directions to each chain (moment directions)
    moment_dir = np.cross(u_dirs, z_axis)

    # For each chain direction, group sites into parallel chains and average spin·moment_dir
    for i in range(len(u_dirs)):
        u = u_dirs[i]                   # chain direction (unit)
        v_perp = moment_dir[i]          # perpendicular (unit), also moment_dir
        # Coordinates along and across the chain
        s_A = P[::2] @ u
        s_B = (P[1::2] - np.array([0, 1/np.sqrt(3), 0])) @ u

        proj_A = S[::2] @ v_perp
        proj_B = S[1::2] @ v_perp

        # Group identical s_A (with tolerance) and average proj_A over those groups
        keys = np.round(s_A, 4)
        uniq_keys, inv = np.unique(keys, return_inverse=True)
        counts = np.bincount(inv)
        sums = np.bincount(inv, weights=proj_A)
        proj_A_avg = sums / counts

        # Use grouped/averaged A-values for plotting
        s_A = uniq_keys
        proj_A = proj_A_avg


        # Group identical s_B (with tolerance) and average proj_B over those groups
        keys = np.round(s_B, 4)
        uniq_keys, inv = np.unique(keys, return_inverse=True)
        counts = np.bincount(inv)
        sums = np.bincount(inv, weights=proj_B)
        proj_B_avg = sums / counts

        # Use grouped/averaged A-values for plotting
        s_B = uniq_keys
        proj_B = proj_B_avg


        # Spin projection onto the moment direction for this family
        fig, ax = plt.subplots(figsize=(20, 4))

        ax.plot(s_A*10, proj_A, 'o')
        ax.plot(s_B*10, proj_B, 'o')
        ax.set_xlabel("Position along chain")
        ax.set_ylabel("Spin projection")
        plt.savefig(f"{dir}/chain_{i+1}.pdf")
        plt.close()



def regnault_magnetic_moment_reconstruction(P, SSSF, Q):
    # Read the SSSF_max_point.txt file and extract Kx, Ky, Kz, and SSSF components
    file_path = os.path.join(SSSF, f"SSSF_{Q}_top1_points_rlu.txt")

    values = np.loadtxt(file_path, skiprows=1)
    # Extract Kx, Ky, Kz and SSSF components
    Kx, Ky, Kz = values[0:3]
    print(f"Using K-point: ({Kx:.4f}, {Ky:.4f}, {Kz:.4f})")
    Intensity = values[3]
    SSSF_xx, SSSF_xy, SSSF_xz, SSSF_yx, SSSF_yy, SSSF_yz, SSSF_zx, SSSF_zy, SSSF_zz = values[4:13]

    # Single-Q ansatz fitting for honeycomb lattice
    print(f"\nReconstructing magnetic moment using single-Q ansatz at Q=({Kx:.4f}, {Ky:.4f}, {Kz:.4f})")
    
    # Q in reciprocal space (assuming Kx, Ky are in reduced units)
    Q_vec = contract('i, ik->k', np.array([Kx, Ky, Kz]), honeycomb_reciprocal_basis())[0:2]  # Use only x,y components for 2D lattice
    # Create SSSF tensor from the components
    SSSF_target = np.array([
        [SSSF_xx, SSSF_xy, SSSF_xz],
        [SSSF_yx, SSSF_yy, SSSF_yz],
        [SSSF_zx, SSSF_zy, SSSF_zz]
    ])
    
    # Normalize SSSF_target for better convergence
    SSSF_norm = np.linalg.norm(SSSF_target)
    if SSSF_norm > 1e-10:
        SSSF_target_normalized = SSSF_target / SSSF_norm
    else:
        print("Warning: SSSF tensor has very small norm, using unnormalized version")
        SSSF_target_normalized = SSSF_target
        SSSF_norm = 1.0
    
    def generate_spin_configuration(params, positions, Q_vec):
        """
        Generate spin configuration using constrained ansatz:
        S_x(R_i) = m_b * sin(gamma) * cos(Q·R + phi_i^a)
        S_y(R_i) = m_b * cos(gamma) * cos(Q·R + phi_i^b)
        S_z(R_i) = m_c * cos(Q·R + phi_i^c)
        
        With constraints:
        - m_b, m_c same for both sublattices
        - gamma same for both sublattices
        - phi_1^c - phi_1^b = -(phi_2^c - phi_2^b)
        - phi_1^b - phi_1^a = -(phi_2^b - phi_2^a)
        
        params: [m_b, m_c, gamma, phi_1^a, delta_ba, delta_cb, phi_2^a]
        where:
        - delta_ba = phi_1^b - phi_1^a = -(phi_2^b - phi_2^a)
        - delta_cb = phi_1^c - phi_1^b = -(phi_2^c - phi_2^b)
        - phi_2^a is free
        """
        m_b, m_c, gamma, phi_1_a, delta_ba, delta_cb, phi_2_a = params
        
        # Sublattice A (1) phases
        phi_1_b = phi_1_a + delta_ba
        phi_1_c = phi_1_b + delta_cb
        
        # Sublattice B (2) phases with constraints
        # delta_ba = -(phi_2^b - phi_2^a) => phi_2^b = phi_2^a - delta_ba
        # delta_cb = -(phi_2^c - phi_2^b) => phi_2^c = phi_2^b - delta_cb
        phi_2_b = phi_2_a - delta_ba
        phi_2_c = phi_2_b - delta_cb
        
        N = len(positions)
        spins = np.zeros((N, 3))
        
        for i in range(N):
            pos = positions[i, :2]  # Use only x,y components for 2D lattice
            Q_dot_R = np.dot(Q_vec, pos)
            
            if i % 2 == 0:  # A sublattice (1)
                spins[i, 0] = m_b * np.sin(gamma) * np.cos(Q_dot_R + phi_1_a)
                spins[i, 1] = m_b * np.cos(gamma) * np.cos(Q_dot_R + phi_1_b)
                spins[i, 2] = m_c * np.cos(Q_dot_R + phi_1_c)
            else:  # B sublattice (2)
                spins[i, 0] = m_b * np.sin(gamma) * np.cos(Q_dot_R + phi_2_a)
                spins[i, 1] = m_b * np.cos(gamma) * np.cos(Q_dot_R + phi_2_b)
                spins[i, 2] = m_c * np.cos(Q_dot_R + phi_2_c)
            
            # Normalize the spin
            norm = np.linalg.norm(spins[i])
            if norm > 1e-10:
                spins[i] = spins[i] / norm
        
        return spins
    
    def compute_sssf_at_q(spins, positions, Q_eval):
        """Compute SSSF at specific Q"""
        # Use only x,y components of positions for 2D
        pos_2d = positions[:, :2]
        ffact = np.exp(1j * np.dot(pos_2d, Q_eval))
        N = len(spins)
        S_q = np.dot(spins.T, ffact) / np.sqrt(N)
        
        # Compute SSSF tensor
        SSSF = np.real(np.outer(S_q, np.conj(S_q)))
        return SSSF
    
    def objective_function(params, positions, Q_vec, SSSF_target_normalized):
        """Objective function to minimize using normalized SSSF"""
        # Generate spin configuration
        spins = generate_spin_configuration(params, positions, Q_vec)
        
        # Compute SSSF at the ordering wavevector
        SSSF_calc = compute_sssf_at_q(spins, positions, Q_vec)
        
        # Normalize calculated SSSF
        SSSF_calc_norm = np.linalg.norm(SSSF_calc)
        if SSSF_calc_norm > 1e-10:
            SSSF_calc_normalized = SSSF_calc / SSSF_calc_norm
        else:
            SSSF_calc_normalized = SSSF_calc
        
        # Calculate mean squared error between normalized tensors
        mse = np.sum((SSSF_calc_normalized - SSSF_target_normalized)**2)
        
        return mse
    
    # Parameter bounds
    bounds = [
        (0, 1.0),  # m_b
        (0, 1.0),  # m_c
        (0, 2*np.pi),  # gamma
        (0, 2*np.pi),  # phi_1^a
        (-2*np.pi, 2*np.pi),  # delta_ba
        (-2*np.pi, 2*np.pi),  # delta_cb
        (0, 2*np.pi),  # phi_2^a (free parameter)
    ]
    
    # Multiple optimization attempts with random initial guesses
    best_result = None
    best_cost = np.inf
    
    n_attempts = 30  # Increased attempts for constrained problem
    for attempt in range(n_attempts):
        # Random initial guess
        initial_params = np.array([
            np.random.uniform(0, 1),  # m_b
            np.random.uniform(0, 1),  # m_c
            np.random.uniform(0, 2*np.pi),  # gamma
            np.random.uniform(0, 2*np.pi),  # phi_1^a
            np.random.uniform(-np.pi, np.pi),  # delta_ba
            np.random.uniform(-np.pi, np.pi),  # delta_cb
            np.random.uniform(0, 2*np.pi),  # phi_2^a
        ])
        
        # Optimize
        result = minimize(
            objective_function,
            initial_params,
            args=(P, Q_vec, SSSF_target_normalized),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 3000, 'ftol': 1e-12}
        )
        
        if result.fun < best_cost:
            best_result = result
            best_cost = result.fun
            
        # Early termination if good enough solution found
        if best_cost < 1e-8:
            print(f"Good solution found after {attempt+1} attempts")
            break
    
    if best_result is not None and best_cost < 0.1:  # Relaxed convergence criterion for normalized fitting
        print("Optimization successful!")
        print(f"Final cost (normalized): {best_cost:.6e}")
        
        # Extract optimized parameters
        opt_params = best_result.x
        m_b, m_c, gamma, phi_1_a, delta_ba, delta_cb, phi_2_a = opt_params
        
        # Compute all phases
        phi_1_b = phi_1_a + delta_ba
        phi_1_c = phi_1_b + delta_cb
        phi_2_b = phi_2_a - delta_ba
        phi_2_c = phi_2_b - delta_cb
        
        # Verify constraints
        constraint1 = (phi_1_c - phi_1_b) + (phi_2_c - phi_2_b)
        constraint2 = (phi_1_b - phi_1_a) + (phi_2_b - phi_2_a)
        
        print("\nOptimized parameters (with constraints):")
        print(f"Common: m_b={m_b:.4f}, m_c={m_c:.4f}, gamma={gamma:.4f}")
        print(f"Phase differences: delta_ba={delta_ba:.4f}, delta_cb={delta_cb:.4f}")
        print(f"Sublattice A (1): phi_a={phi_1_a:.4f}, phi_b={phi_1_b:.4f}, phi_c={phi_1_c:.4f}")
        print(f"Sublattice B (2): phi_a={phi_2_a:.4f}, phi_b={phi_2_b:.4f}, phi_c={phi_2_c:.4f}")
        print(f"\nConstraint verification:")
        print(f"phi_1^c - phi_1^b = {phi_1_c - phi_1_b:.4f}")
        print(f"phi_2^c - phi_2^b = {phi_2_c - phi_2_b:.4f}")
        print(f"Sum (should be ~0): {constraint1:.6f}")
        print(f"phi_1^b - phi_1^a = {phi_1_b - phi_1_a:.4f}")
        print(f"phi_2^b - phi_2^a = {phi_2_b - phi_2_a:.4f}")
        print(f"Sum (should be ~0): {constraint2:.6f}")
        
        # Generate final spin configuration
        final_spins = generate_spin_configuration(opt_params, P, Q_vec)
        
        # Verify by computing SSSF
        SSSF_fitted = compute_sssf_at_q(final_spins, P, Q_vec)
        
        # Scale fitted SSSF to match original norm
        SSSF_fitted_scaled = SSSF_fitted * (SSSF_norm / np.linalg.norm(SSSF_fitted))
        
        print("\nTarget SSSF tensor (original scale):")
        print(SSSF_target)
        print("\nFitted SSSF tensor (scaled to match):")
        print(SSSF_fitted_scaled)
        print("\nRelative error:")
        print(np.abs(SSSF_fitted_scaled - SSSF_target) / (np.abs(SSSF_target) + 1e-10))
        
        # Save results
        save_path = os.path.dirname(file_path) if isinstance(file_path, str) else "."
        np.savetxt(os.path.join(save_path, f"fitted_spins_constrained_{Q}.txt"), final_spins)
        np.savetxt(os.path.join(save_path, f"fitted_spins_constrained_{Q}.txt"), P)
        
        # Save parameters
        full_params = np.array([m_b, m_c, gamma, phi_1_a, phi_1_b, phi_1_c, phi_2_a, phi_2_b, phi_2_c, delta_ba, delta_cb])
        param_names = "m_b m_c gamma phi_1_a phi_1_b phi_1_c phi_2_a phi_2_b phi_2_c delta_ba delta_cb"
        np.savetxt(os.path.join(save_path, f"fitted_parameters_constrained_{Q}.txt"), 
                   full_params.reshape(1, -1),
                   header=param_names)
        
        # Save fitting quality metrics
        fitting_metrics = np.array([
            SSSF_norm,  # Original SSSF norm
            np.linalg.norm(SSSF_fitted),  # Fitted SSSF norm
            best_cost,  # Normalized cost
            np.linalg.norm(SSSF_fitted_scaled - SSSF_target)  # Absolute error
        ])
        np.savetxt(os.path.join(save_path, "fitting_metrics_constrained.txt"),
                   fitting_metrics,
                   header="original_norm fitted_norm normalized_cost absolute_error")
        
        # Plot the fitted spin configuration
        plot_spin_config_2d(P, final_spins, os.path.join(save_path, f"fitted_spin_config_constrained_{Q}.pdf"))

        magnetic_moment_along(P, final_spins, save_path)
        print("\nConstrained spin configuration plots saved.")
        
        return {
            "params": opt_params,
            "spins": final_spins,
            "positions": P,
            "SSSF_fitted": SSSF_fitted_scaled,
            "cost": best_cost
        }
    else:
        print(f"Optimization did not converge sufficiently after {n_attempts} attempts!")
        print(f"Best cost achieved: {best_cost:.6e}")
        return None



def compute_skyrmion_chirality(P, S):
    """Compute per-triangle skyrmion (scalar chirality) density using Berg–Lüscher.
    Returns triangulation, densities per triangle, total Q, and triangle centroids.
    """
    # Normalize spins
    S_norm = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    pts2d = P[:, :2]
    tri = Delaunay(pts2d)
    tris = tri.simplices
    q = np.zeros(len(tris))
    centroids = np.zeros((len(tris), 2))
    for idx, (i, j, k) in enumerate(tris):
        r0, r1, r2 = pts2d[i], pts2d[j], pts2d[k]
        # Oriented area sign to keep a consistent mapping orientation
        area2 = (r1[0]-r0[0])*(r2[1]-r0[1]) - (r1[1]-r0[1])*(r2[0]-r0[0])
        area_sign = 1.0 if area2 == 0 else np.sign(area2)
        s0, s1, s2 = S_norm[i], S_norm[j], S_norm[k]
        num = np.dot(s0, np.cross(s1, s2))
        den = 1.0 + np.dot(s0, s1) + np.dot(s1, s2) + np.dot(s2, s0)
        omega = 2.0 * np.arctan2(num, den)
        q[idx] = area_sign * omega / (4.0 * np.pi)
        centroids[idx] = (r0 + r1 + r2) / 3.0
    Q_total = float(np.sum(q))
    return tri, q, Q_total, centroids


def plot_chirality_real_space(P, S, out_base):
    """Plot skyrmion chirality density in real space and save artifacts.
    Saves: out_base_density.pdf, out_base_density.txt, out_base_Q.txt
    """
    tri, q, Q_total, centroids = compute_skyrmion_chirality(P, S)
    import matplotlib.tri as mtri
    triang = mtri.Triangulation(P[:, 0], P[:, 1], tri.simplices)

    # Ensure output directory exists
    out_dir = os.path.dirname(out_base)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 8))
    tpc = plt.tripcolor(triang, facecolors=q, shading='flat', cmap='RdBu_r')
    plt.colorbar(tpc, label='Skyrmion density q_t')
    # plt.quiver(P[:, 0], P[:, 1], S[:, 0], S[:, 1], S[:, 2], cmap='viridis',
    #            scale_units='xy', angles='xy', scale=1, width=0.002, headwidth=3, headlength=4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Chirality density, Q≈{Q_total:.3f}')
    plt.tight_layout()
    plt.savefig(out_base + '_density.pdf')
    plt.clf()
    plt.close()

    # Save per-triangle data and total Q
    np.savetxt(out_base + '_density.txt', np.column_stack([centroids, q]), header='x y q_t')
    with open(out_base + '_Q.txt', 'w') as f:
        f.write(f'{Q_total}\n')


def energy_density_by_section(filename, P, energy_landscape, num_sec_x, num_sec_y):
    """Compute energy density in specified sections of the lattice."""
    x_edges = np.linspace(np.min(P[:, 0]), np.max(P[:, 0]), num_sec_x + 1)
    y_edges = np.linspace(np.min(P[:, 1]), np.max(P[:, 1]), num_sec_y + 1)
    energy_density = np.zeros((num_sec_x, num_sec_y))

    for i in range(num_sec_x):
        for j in range(num_sec_y):
            # Define the section boundaries
            x_mask = (P[:, 0] >= x_edges[i]) & (P[:, 0] < x_edges[i + 1])
            y_mask = (P[:, 1] >= y_edges[j]) & (P[:, 1] < y_edges[j + 1])
            mask = x_mask & y_mask

            if np.any(mask):
                energy_density[i, j] = np.mean(energy_landscape[mask])
    energy_density[energy_density == 0] = np.nan  # Mark empty sections as NaN
    # Save the energy density data
    np.savetxt(filename.replace('.pdf', '.txt'), energy_density)

    # plotting
    plt.figure(figsize=(8, 6))
    plt.imshow(energy_density.T, origin='lower', aspect='auto',
               extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
               cmap='viridis')
    plt.colorbar(label='Energy Density')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Energy Density by Section')
    plt.savefig(filename)
    plt.close()

    energy_mean = np.mean(energy_landscape)
    energy_lower = np.where(energy_density < energy_mean, energy_density, np.nan)

    plt.figure(figsize=(8, 6))
    plt.imshow(energy_lower.T, origin='lower', aspect='auto',
               extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
               cmap='viridis')
    plt.colorbar(label='Energy Density')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Energy Density by Section')
    plt.savefig(filename.replace('.pdf', '_lower.pdf'))
    plt.close()


    return energy_density


def _segment_defect_free_region(P, S, grid_res=256, sigma=1.0, q_rel_thresh=0.15, grad_rel_thresh=0.35):
    """
    Segment 'defect-free' region via low |q| and low |∇S| on the continuum grid.
    Returns: dict with grid mask, site mask, and diagnostic data.
    """
    cont = compute_continuum_chirality(P, S, grid_res=grid_res, sigma=sigma)  # uses existing compute
    X, Y, Sxg, Syg, Szg, q, mask = cont['X'], cont['Y'], cont['Sx'], cont['Sy'], cont['Sz'], cont['q'], cont['mask']
    dy, dx = cont['dy'], cont['dx']

    # Smoothness measure: ||∇S||^2 = sum_c (|∂x S_c|^2 + |∂y S_c|^2)
    dSx_dy, dSx_dx = np.gradient(Sxg, dy, dx)
    dSy_dy, dSy_dx = np.gradient(Syg, dy, dx)
    dSz_dy, dSz_dx = np.gradient(Szg, dy, dx)
    grad2 = dSx_dx**2 + dSx_dy**2 + dSy_dx**2 + dSy_dy**2 + dSz_dx**2 + dSz_dy**2

    # Relative thresholds w.r.t. valid region
    absq = np.abs(q)
    q_max = np.nanmax(absq[mask]) if np.any(mask) else np.nanmax(absq)
    g_med = np.nanmedian(grad2[mask])
    g_max = np.nanmax(grad2[mask])
    g_thr = g_med + grad_rel_thresh * (g_max - g_med)

    free_grid_mask = (mask &
                      (absq <= q_rel_thresh * (q_max + 1e-12)) &
                      (grad2 <= g_thr))

    # Keep the largest connected component for robustness
    labeled, ncomp = ndi_label(free_grid_mask.astype(int))
    if ncomp > 0:
        # Find largest
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        keep = np.argmax(sizes)
        free_grid_mask = (labeled == keep)

    # Map grid mask back to sites using nearest grid cell
    x0, y0 = X[0, 0], Y[0, 0]
    nx, ny = X.shape[1], X.shape[0]
    ix = np.clip(np.floor((P[:, 0] - x0) / dx + 0.5).astype(int), 0, nx - 1)
    iy = np.clip(np.floor((P[:, 1] - y0) / dy + 0.5).astype(int), 0, ny - 1)
    site_free_mask = free_grid_mask[iy, ix]

    return {
        'cont': cont,
        'free_grid_mask': free_grid_mask,
        'site_free_mask': site_free_mask,
        'grad2': grad2,
    }


def energetics_argument(P, S, energy_landscape, out_dir, J=1.0, grid_res=256, sigma=1.0):
    """
    Cleave into defect-free vs. rest and compare energies.
    Assumes energy_landscape is per-site and averages it by region.

    Saves:
      out_dir/energetics_summary.txt
      out_dir/region_masks_scatter.pdf
      out_dir/region_q_overlay.pdf
    """
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    seg = _segment_defect_free_region(P, S, grid_res=grid_res, sigma=sigma)
    site_free = seg['site_free_mask']
    cont = seg['cont']
    X, Y, q = cont['X'], cont['Y'], cont['q']
    free_grid_mask = seg['free_grid_mask']

    # Region sizes
    n_free = int(site_free.sum())
    n_def = int((~site_free).sum())

    # Energy analysis using provided energy landscape
    if len(energy_landscape) == len(P):
        e_free = float(np.nanmean(energy_landscape[site_free])) if n_free else np.nan
        e_def = float(np.nanmean(energy_landscape[~site_free])) if n_def else np.nan
        e_boundary = np.nan  # not defined for per-site energy
        summary_mode = "site_energy"
    else:
        # Fallback if energy landscape doesn't match sites
        e_free = e_def = e_boundary = np.nan
        summary_mode = "no_matching_energy"

    # Save summary
    with open(os.path.join(out_dir, "energetics_summary.txt"), "w") as f:
        f.write(f"mode: {summary_mode}\n")
        f.write(f"N_sites: {len(P)}\n")
        f.write(f"N_free: {n_free}\n")
        f.write(f"N_defective: {n_def}\n")
        f.write(f"E_free_mean: {e_free}\n")
        f.write(f"E_def_mean: {e_def}\n")
        f.write(f"E_boundary_mean: {e_boundary}\n")
        if summary_mode == "site_energy":
            f.write(f"E_diff: {e_def - e_free}\n")
            f.write(f"E_free_std: {float(np.nanstd(energy_landscape[site_free])) if n_free else np.nan}\n")
            f.write(f"E_def_std: {float(np.nanstd(energy_landscape[~site_free])) if n_def else np.nan}\n")

    # Plot site mask scatter
    plt.figure(figsize=(7, 6))
    plt.scatter(P[~site_free, 0], P[~site_free, 1], s=6, c='crimson', label='defective')
    plt.scatter(P[site_free, 0], P[site_free, 1], s=6, c='royalblue', label='defect-free', alpha=0.7)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.xlabel('x'); plt.ylabel('y'); plt.title('Region segmentation (sites)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "region_masks_scatter.pdf"))
    plt.clf(); plt.close()

    # Plot q overlay with free-region contour
    plt.figure(figsize=(7, 6))
    im = plt.pcolormesh(X, Y, q, shading='auto', cmap='RdBu_r')
    plt.colorbar(im, label='q(x,y)')
    # contour of free region
    try:
        plt.contour(X, Y, free_grid_mask.astype(float), levels=[0.5], colors='k', linewidths=1.2)
    except Exception:
        pass
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x'); plt.ylabel('y'); plt.title('Continuum chirality and free-region')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "region_q_overlay.pdf"))
    plt.clf(); plt.close()

    # Plot energy distribution comparison
    if summary_mode == "site_energy":
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(energy_landscape[site_free], bins=30, alpha=0.7, label='defect-free', color='royalblue')
        plt.hist(energy_landscape[~site_free], bins=30, alpha=0.7, label='defective', color='crimson')
        plt.xlabel('Energy')
        plt.ylabel('Count')
        plt.legend()
        plt.title('Energy distributions')
        
        plt.subplot(1, 2, 2)
        plt.scatter(P[:, 0], P[:, 1], c=energy_landscape, s=4, cmap='viridis')
        plt.colorbar(label='Energy')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x'); plt.ylabel('y'); plt.title('Energy landscape')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "energy_analysis.pdf"))
        plt.clf(); plt.close()

    return {
        'N_free': n_free,
        'N_def': n_def,
        'E_free_mean': e_free,
        'E_def_mean': e_def,
        'E_boundary_mean': e_boundary,
        'site_free_mask': site_free,
    }


def _edges_from_triangulation(points):
    """Build unique undirected edges from Delaunay triangulation of points (N,2)."""
    tri = Delaunay(points)
    simplices = tri.simplices
    edges = set()
    for a, b, c in simplices:
        e = [(a, b), (b, c), (c, a)]
        for i, j in e:
            if i > j:
                i, j = j, i
            edges.add((i, j))
    edges = np.array(sorted(list(edges)), dtype=int)
    return edges


def _psi6_and_spacing(points, edges):
    """
    For each point, compute:
      - degree (neighbor count from edges)
      - psi6 complex order parameter using neighbor angles
      - median neighbor spacing
    Returns degree (N,), psi6 (N,), spacing (N,)
    """
    N = points.shape[0]
    nbrs = [[] for _ in range(N)]
    for i, j in edges:
        nbrs[i].append(j)
        nbrs[j].append(i)
    degree = np.array([len(n) for n in nbrs], dtype=int)
    psi6 = np.zeros(N, dtype=np.complex128)
    spacing = np.zeros(N, dtype=float)
    for i in range(N):
        ns = nbrs[i]
        if len(ns) == 0:
            psi6[i] = 0.0
            spacing[i] = np.nan
            continue
        vecs = points[np.array(ns)] - points[i]
        thetas = np.arctan2(vecs[:, 1], vecs[:, 0])
        psi6[i] = np.mean(np.exp(1j * 6.0 * thetas))
        dists = np.linalg.norm(vecs, axis=1)
        spacing[i] = np.median(dists)
    return degree, psi6, spacing


def _largest_coherent_component(edges, good_mask, psi6, ori_thr_deg=15.0):
    """
    Among nodes with good_mask True, build components requiring neighboring psi6 phase
    difference < ori_thr_deg. Return indices of the largest component.
    """
    N = good_mask.size
    # Build adjacency list filtered by good_mask and orientation similarity
    adj = [[] for _ in range(N)]
    thr = np.deg2rad(ori_thr_deg)
    phases = np.angle(psi6)
    for i, j in edges:
        if not (good_mask[i] and good_mask[j]):
            continue
        dphi = np.angle(np.exp(1j * (phases[i] - phases[j])))  # wrap to [-pi,pi]
        if abs(dphi) <= thr:
            adj[i].append(j)
            adj[j].append(i)
    # BFS/DFS to get components
    visited = np.zeros(N, dtype=bool)
    best_comp = []
    for s in range(N):
        if visited[s] or not (good_mask[s]):
            continue
        comp = []
        stack = [s]
        visited[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        if len(comp) > len(best_comp):
            best_comp = comp
    return np.array(best_comp, dtype=int)


def segment_skyrmion_lattice(P, S, grid_res=256, sigma=1.0, psi6_thr=0.75, sp_rel_thr=0.20, ori_thr_deg=15.0):
    """
    Skyrmion-lattice-aware segmentation.
    Steps:
      1) Detect skyrmion cores from continuum chirality |q| peaks.
      2) Build Delaunay graph over core positions and compute psi6, spacing.
      3) Mark cores as 'good' if degree==6, |psi6|>=psi6_thr, spacing close to global median (±sp_rel_thr).
      4) Keep the largest orientation-coherent connected component as ordered domain.
      5) Assign lattice sites to nearest core within 0.6*a_skx and inherit ordered/defective label.
    Returns dict with masks and diagnostics.
    """
    cont = compute_continuum_chirality(P, S, grid_res=grid_res, sigma=sigma)
    X, Y, q, Sz, mask = cont['X'], cont['Y'], cont['q'], cont['Sz'], cont['mask']
    cores = detect_skyrmion_cores_from_grid(X, Y, Sz, q, mask, q_rel_thresh=0.2, sz_prominence=0.2, neighborhood=9)
    core_xy = np.array([(x, y) for (x, y, qv, szv, kind) in cores['q_peaks']], dtype=float)
    if core_xy.shape[0] < 3:
        # Fallback: no reliable core detection, return everything as ordered
        site_ordered_mask = np.ones(P.shape[0], dtype=bool)
        return {
            'cont': cont,
            'cores': cores,
            'core_xy': core_xy,
            'site_ordered_mask': site_ordered_mask,
            'core_good_mask': np.ones(core_xy.shape[0], dtype=bool),
            'psi6': np.array([], dtype=np.complex128),
            'degree': np.array([], dtype=int),
            'spacing': np.array([], dtype=float),
            'a_skx': np.nan,
            'main_component': np.array([], dtype=int),
        }

    # Build lattice graph and metrics
    edges = _edges_from_triangulation(core_xy)
    degree, psi6, spacing = _psi6_and_spacing(core_xy, edges)
    a_skx = np.nanmedian(spacing)
    spacing_ok = np.isfinite(spacing) & (np.abs(spacing - a_skx) <= sp_rel_thr * (a_skx + 1e-12))
    psi6_ok = (np.abs(psi6) >= psi6_thr)
    degree_ok = (degree == 6)
    core_good_mask = spacing_ok & psi6_ok & degree_ok

    # Largest coherent component among good cores
    main_comp = _largest_coherent_component(edges, core_good_mask, psi6, ori_thr_deg=ori_thr_deg)
    core_in_main = np.zeros(core_xy.shape[0], dtype=bool)
    core_in_main[main_comp] = True

    # Assign sites to nearest core and label ordered if nearest core in main component
    tree = cKDTree(core_xy)
    dists, idxs = tree.query(P[:, :2], k=1)
    r_assign = 0.6 * (a_skx if np.isfinite(a_skx) else np.nanmedian(spacing))
    near_any = dists <= (r_assign if np.isfinite(r_assign) else np.inf)
    site_ordered_mask = near_any & core_in_main[idxs]

    return {
        'cont': cont,
        'cores': cores,
        'core_xy': core_xy,
        'edges': edges,
        'psi6': psi6,
        'degree': degree,
        'spacing': spacing,
        'a_skx': a_skx,
        'core_good_mask': core_good_mask,
        'main_component': main_comp,
        'site_ordered_mask': site_ordered_mask,
    }


def energetics_argument_skyrmion(P, S, energy_landscape, out_dir, grid_res=256, sigma=1.0,
                                 psi6_thr=0.75, sp_rel_thr=0.20, ori_thr_deg=15.0):
    """
    Energetics argument using skyrmion-lattice-aware segmentation.
    Treat the well-ordered skyrmion lattice domain as 'ordered', and deviations
    (dislocations, grain boundaries, vacancies/interstitials) as 'defective'.
    """
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    seg = segment_skyrmion_lattice(P, S, grid_res=grid_res, sigma=sigma,
                                   psi6_thr=psi6_thr, sp_rel_thr=sp_rel_thr, ori_thr_deg=ori_thr_deg)
    site_ordered = seg['site_ordered_mask']
    cont = seg['cont']
    X, Y, q = cont['X'], cont['Y'], cont['q']

    n_ord = int(site_ordered.sum())
    n_def = int((~site_ordered).sum())

    if len(energy_landscape) == len(P):
        e_ord = float(np.nanmean(energy_landscape[site_ordered])) if n_ord else np.nan
        e_def = float(np.nanmean(energy_landscape[~site_ordered])) if n_def else np.nan
        mode = 'site_energy'
    else:
        e_ord = e_def = np.nan
        mode = 'no_matching_energy'

    with open(os.path.join(out_dir, 'energetics_summary.txt'), 'w') as f:
        f.write(f"mode: {mode}\n")
        f.write(f"N_sites: {len(P)}\n")
        f.write(f"N_ordered: {n_ord}\n")
        f.write(f"N_defective: {n_def}\n")
        f.write(f"E_ordered_mean: {e_ord}\n")
        f.write(f"E_def_mean: {e_def}\n")
        if mode == 'site_energy':
            f.write(f"E_diff: {e_def - e_ord}\n")

    # Plot: site classification
    plt.figure(figsize=(7, 6))
    plt.scatter(P[~site_ordered, 0], P[~site_ordered, 1], s=6, c='crimson', label='defective')
    plt.scatter(P[site_ordered, 0], P[site_ordered, 1], s=6, c='royalblue', label='ordered', alpha=0.7)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.xlabel('x'); plt.ylabel('y'); plt.title('SkX-aware segmentation (sites)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'skx_region_masks_scatter.pdf'))
    plt.clf(); plt.close()

    # Plot: q overlay
    plt.figure(figsize=(7, 6))
    im = plt.pcolormesh(X, Y, q, shading='auto', cmap='RdBu_r')
    plt.colorbar(im, label='q(x,y)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x'); plt.ylabel('y'); plt.title('q(x,y) for SkX segmentation')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'skx_q_overlay.pdf'))
    plt.clf(); plt.close()

    # Plot: cores and psi6
    core_xy = seg['core_xy']
    if core_xy.shape[0] >= 3:
        psi6 = seg['psi6']
        degree = seg['degree']
        core_good = seg['core_good_mask']
        edges = seg.get('edges')
        plt.figure(figsize=(7, 6))
        if edges is not None and edges.size:
            for i, j in edges:
                xi, yi = core_xy[i]
                xj, yj = core_xy[j]
                plt.plot([xi, xj], [yi, yj], color='lightgray', lw=0.5, zorder=1)
        sc = plt.scatter(core_xy[:, 0], core_xy[:, 1], c=np.abs(psi6), s=25, cmap='viridis', zorder=2)
        plt.colorbar(sc, label='|psi6|')
        bad = ~(core_good)
        if np.any(bad):
            plt.scatter(core_xy[bad, 0], core_xy[bad, 1], facecolors='none', edgecolors='r', s=60, label='irregular cores', zorder=3)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x'); plt.ylabel('y'); plt.title('Skyrmion cores and |psi6|')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'skx_cores_psi6.pdf'))
        plt.clf(); plt.close()

    # Energy plots
    if mode == 'site_energy':
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(energy_landscape[site_ordered], bins=30, alpha=0.7, label='ordered', color='royalblue')
        plt.hist(energy_landscape[~site_ordered], bins=30, alpha=0.7, label='defective', color='crimson')
        plt.xlabel('Energy'); plt.ylabel('Count'); plt.legend(); plt.title('Energy distributions')
        plt.subplot(1, 2, 2)
        plt.scatter(P[:, 0], P[:, 1], c=energy_landscape, s=4, cmap='inferno')
        plt.colorbar(label='Energy')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x'); plt.ylabel('y'); plt.title('Energy landscape')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'skx_energy_analysis.pdf'))
        plt.clf(); plt.close()

    return {
        'N_ordered': n_ord,
        'N_def': n_def,
        'E_ordered_mean': e_ord,
        'E_def_mean': e_def,
        'site_ordered_mask': site_ordered,
        'a_skx': seg['a_skx'],
    }

def _interpolate_spins_to_grid(P, S, grid_res=256, method='linear'):
    """Interpolate spins onto a regular XY grid.
    Returns X, Y, Sxg, Syg, Szg, mask, and grid spacings dx, dy.
    """
    pts = P[:, :2]
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    # Pad a bit to reduce edge effects
    pad_x = 0.02 * (x_max - x_min + 1e-12)
    pad_y = 0.02 * (y_max - y_min + 1e-12)
    x = np.linspace(x_min - pad_x, x_max + pad_x, grid_res)
    y = np.linspace(y_min - pad_y, y_max + pad_y, grid_res)
    X, Y = np.meshgrid(x, y)
    # Primary interp
    Sx_lin = griddata(pts, S[:, 0], (X, Y), method=method)
    Sy_lin = griddata(pts, S[:, 1], (X, Y), method=method)
    Sz_lin = griddata(pts, S[:, 2], (X, Y), method=method)
    # Fallback fill with nearest for NaNs
    Sx_near = griddata(pts, S[:, 0], (X, Y), method='nearest')
    Sy_near = griddata(pts, S[:, 1], (X, Y), method='nearest')
    Sz_near = griddata(pts, S[:, 2], (X, Y), method='nearest')
    Sxg = np.where(np.isnan(Sx_lin), Sx_near, Sx_lin)
    Syg = np.where(np.isnan(Sy_lin), Sy_near, Sy_lin)
    Szg = np.where(np.isnan(Sz_lin), Sz_near, Sz_lin)
    # Normalize spins per-gridpoint to unit length to mitigate interpolation artifacts
    norm = np.sqrt(Sxg**2 + Syg**2 + Szg**2) + 1e-12
    Sxg, Syg, Szg = Sxg / norm, Syg / norm, Szg / norm
    mask = ~np.isnan(Sx_lin) & ~np.isnan(Sy_lin) & ~np.isnan(Sz_lin)
    dx = (x_max - x_min + 2 * pad_x) / (grid_res - 1)
    dy = (y_max - y_min + 2 * pad_y) / (grid_res - 1)
    return X, Y, Sxg, Syg, Szg, mask, dx, dy


def compute_continuum_chirality(P, S, grid_res=256, sigma=1.0, method='linear'):
    """Continuum chirality q(x,y) = (1/4pi) S · (∂x S × ∂y S) on a regular grid.
    Returns dict with X,Y,Sx,Sy,Sz,q,mask,dx,dy,Q_total.
    """
    # Normalize input spins
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    X, Y, Sxg, Syg, Szg, mask, dx, dy = _interpolate_spins_to_grid(P, S, grid_res, method)
    if sigma and sigma > 0:
        # Smooth each component a bit to stabilize derivatives
        Sxg = gaussian_filter(Sxg, sigma=sigma)
        Syg = gaussian_filter(Syg, sigma=sigma)
        Szg = gaussian_filter(Szg, sigma=sigma)
        # Renormalize after smoothing
        norm = np.sqrt(Sxg**2 + Syg**2 + Szg**2) + 1e-12
        Sxg, Syg, Szg = Sxg / norm, Syg / norm, Szg / norm

    # Finite differences (np.gradient handles uneven but we pass dx,dy scalars)
    dSx_dy, dSx_dx = np.gradient(Sxg, dy, dx)
    dSy_dy, dSy_dx = np.gradient(Syg, dy, dx)
    dSz_dy, dSz_dx = np.gradient(Szg, dy, dx)
    # Cross product ∂x S × ∂y S
    cx = dSy_dx * dSz_dy - dSz_dx * dSy_dy
    cy = dSz_dx * dSx_dy - dSx_dx * dSz_dy
    cz = dSx_dx * dSy_dy - dSy_dx * dSx_dy
    q = (Sxg * cx + Syg * cy + Szg * cz) / (4.0 * np.pi)

    # Only integrate within valid mask region
    q_masked = np.where(mask, q, 0.0)
    Q_total = float(np.sum(q_masked) * dx * dy)
    return {
        'X': X,
        'Y': Y,
        'Sx': Sxg,
        'Sy': Syg,
        'Sz': Szg,
        'q': q,
        'mask': mask,
        'dx': dx,
        'dy': dy,
        'Q_total': Q_total,
    }


def _find_local_extrema(arr, mode='max', size=5, mask=None, threshold=None):
    """Find local maxima or minima indices in 2D array using morphological filters.
    - mode: 'max' or 'min'
    - size: neighborhood size (odd integer)
    - mask: optional boolean mask to limit valid region
    - threshold: optional absolute threshold for arr values (for 'max') or -arr for 'min'
    Returns list of (i,j) indices.
    """
    if mask is None:
        mask = np.ones_like(arr, dtype=bool)
    footprint = generate_binary_structure(2, 2)
    footprint = np.pad(footprint, ((0, 0), (0, 0)), mode='constant', constant_values=0)
    if mode == 'max':
        filt = maximum_filter(arr, size=size)
        candidates = (arr == filt) & mask
        if threshold is not None:
            candidates &= (arr >= threshold)
    else:
        inv = -arr
        filt = maximum_filter(inv, size=size)
        candidates = (inv == filt) & mask
        if threshold is not None:
            candidates &= (arr <= threshold)
    ii, jj = np.where(candidates)
    return list(zip(ii, jj))


def detect_skyrmion_cores_from_grid(X, Y, Sz, q, mask, q_rel_thresh=0.2, sz_prominence=None, neighborhood=7):
    """Detect skyrmion/antiskyrmion cores as local maxima of |q| and extrema of Sz.
    Returns dict with core lists and a combined table.
    """
    # |q| peaks
    absq = np.abs(q)
    q_max = np.nanmax(absq[mask]) if np.any(mask) else np.nanmax(absq)
    q_thr = q_rel_thresh * (q_max + 1e-12)
    peaks = _find_local_extrema(absq, mode='max', size=neighborhood, mask=mask, threshold=q_thr)
    q_cores = []
    for i, j in peaks:
        x, y = X[i, j], Y[i, j]
        q_val = q[i, j]
        sz_val = Sz[i, j]
        kind = 'skyrmion' if q_val > 0 else 'antiskyrmion'
        q_cores.append((x, y, q_val, sz_val, kind))

    # Sz minima and maxima (optional context)
    sz_min_thr = None
    sz_max_thr = None
    if sz_prominence is not None:
        smin = np.nanmin(Sz[mask])
        smax = np.nanmax(Sz[mask])
        rng = smax - smin + 1e-12
        sz_min_thr = smin + sz_prominence * rng
        sz_max_thr = smax - sz_prominence * rng
    sz_mins = _find_local_extrema(Sz, mode='min', size=neighborhood, mask=mask, threshold=sz_min_thr)
    sz_maxs = _find_local_extrema(Sz, mode='max', size=neighborhood, mask=mask, threshold=sz_max_thr)
    sz_min_list = [(X[i, j], Y[i, j], q[i, j], Sz[i, j], 'min_Sz') for i, j in sz_mins]
    sz_max_list = [(X[i, j], Y[i, j], q[i, j], Sz[i, j], 'max_Sz') for i, j in sz_maxs]

    # Merge for convenience
    combined = q_cores + sz_min_list + sz_max_list
    return {
        'q_peaks': q_cores,
        'sz_mins': sz_min_list,
        'sz_maxs': sz_max_list,
        'combined': combined,
    }


def plot_continuum_chirality_and_cores(res, cores, out_base):
    """Plot q(x,y) and Sz with detected cores and save data tables."""
    X, Y, q, Sz, mask, Q_total = res['X'], res['Y'], res['q'], res['Sz'], res['mask'], res['Q_total']
    out_dir = os.path.dirname(out_base)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # q heatmap
    plt.figure(figsize=(7, 6))
    im = plt.pcolormesh(X, Y, q, shading='auto', cmap='RdBu_r')
    plt.colorbar(im, label='Continuum chirality q(x,y)')
    # for x, y, qv, szv, kind in cores['q_peaks']:
    #     plt.plot(x, y, 'ko' if kind == 'skyrmion' else 'ks', markersize=6, fillstyle='none')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'Continuum chirality, Q≈{Q_total:.3f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(out_base + '_q.pdf')
    plt.clf()
    plt.close()

    # Sz heatmap
    plt.figure(figsize=(7, 6))
    im2 = plt.pcolormesh(X, Y, Sz, shading='auto', cmap='viridis')
    plt.colorbar(im2, label='S_z(x,y)')
    # Overlay Sz minima/maxima
    for x, y, qv, szv, _ in cores['sz_mins']:
        plt.plot(x, y, 'rv', markersize=6)
    for x, y, qv, szv, _ in cores['sz_maxs']:
        plt.plot(x, y, 'r^', markersize=6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('S_z field with local extrema')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(out_base + '_Sz.pdf')
    plt.clf()
    plt.close()

    # Save grids and core tables
    np.savetxt(out_base + '_q_grid.txt', np.column_stack([X.ravel(), Y.ravel(), q.ravel()]), header='x y q')
    np.savetxt(out_base + '_Sz_grid.txt', np.column_stack([X.ravel(), Y.ravel(), Sz.ravel()]), header='x y Sz')
    with open(out_base + '_Q_total.txt', 'w') as f:
        f.write(f'{Q_total}\n')
    if cores['combined']:
        core_arr = np.array(cores['combined'], dtype=object)
        # Save as CSV-like text
        with open(out_base + '_cores.txt', 'w') as f:
            f.write('x y q sz kind\n')
            for x, y, qv, szv, kind in cores['combined']:
                f.write(f'{x} {y} {qv} {szv} {kind}\n')



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
    nK = 100
    A = np.zeros((8, nK, nK))
    w = np.array([0.1, 1, 2, 3, 4, 5, 6, 7])
    w0 = 0
    wmax = 5
    w_line = np.arange(w0, wmax, 1/100)[1:]
    B = np.zeros((len(w_line), len(DSSF_K)))
    SSSF = np.zeros((nK, nK, 3, 3))

    H = np.linspace(0, 1, nK)
    L = np.linspace(0, 1, nK)
    C, D = np.meshgrid(H, L)

    for file in sorted(os.listdir(directory)):  
        filename = os.fsdecode(file)
        if os.path.isdir(dir + "/" + filename):
            try:
                B += read_MD(dir + "/" + filename, w_line)
                S = np.loadtxt(dir + "/" + filename + "/spin.txt")
                P = np.loadtxt(dir + "/" + filename + "/pos.txt")
                SSSF += SSSF2D(S, P, nK, dir + "/" + filename)
            except:
                print("Error reading file: " + filename)
                continue
    # plot the SSSF
    SSSFGraph2D(C, D, contract('ijab->ij', SSSF), dir + "/SSSF_tot")


    A = np.transpose(A, (0, 2, 1))
    
    
    fig, ax = plt.subplots(figsize=(10,4))

    C = ax.imshow(B, origin='lower', extent=[0, gK1, 0, 15], aspect='auto', interpolation='lanczos', cmap='gnuplot2')
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
    plt.savefig(dir+"/DSSF_line.pdf")
    plt.clf()

    for file in sorted(os.listdir(directory)):  
        filename = os.fsdecode(file)
        if os.path.isdir(dir + "/" + filename):
            try:
                A += read_MD_slice(dir + "/" + filename, nK, w)
            except:
                print("Error reading file: " + filename)
                continue


    for i in range(2):
        fig11 = plt.figure(figsize=(8, 8), constrained_layout=False)
        grid = fig11.add_gridspec(2, 2, wspace=0, hspace=0)
        ax = [fig11.add_subplot(grid[0, 0]), fig11.add_subplot(grid[0, 1]), fig11.add_subplot(grid[1, 1]), fig11.add_subplot(grid[1, 0])]
        ax[0].set_title(str(w[4*i]) +'meV')
        ax[1].set_title(str(w[4*i+1]) +'meV')
        ax[2].set_title(str(w[4*i+2]) +'meV', y=-0.2)
        ax[3].set_title(str(w[4*i+3]) +'meV', y=-0.2)

        ax[0].set_xticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[2].set_yticks([])


        # Top-left quadrant
        C = ax[0].imshow(A[4*i], origin='lower', extent=[-1.0, 1.0, -1.5, 1.5], aspect='auto', cmap='gnuplot2')
        ax[0].set_xlim([-1.0, 0])
        ax[0].set_ylim([0, 1.5])
        
        # Top-right quadrant
        C = ax[1].imshow(A[4*i+1], origin='lower', extent=[-1.0, 1.0, -1.5, 1.5], aspect='auto', cmap='gnuplot2')
        ax[1].set_xlim([0, 1.0])
        ax[1].set_ylim([0, 1.5])
        
        # Bottom-right quadrant
        C = ax[2].imshow(A[4*i+2], origin='lower', extent=[-1.0, 1.0, -1.5, 1.5], aspect='auto', cmap='gnuplot2')
        ax[2].set_xlim([0, 1.0])
        ax[2].set_ylim([-1.5, 0])
        
        # Bottom-left quadrant
        C = ax[3].imshow(A[4*i+3], origin='lower', extent=[-1.0, 1.0, -1.5, 1.5], aspect='auto', cmap='gnuplot2')
        ax[3].set_xlim([-1.0, 0])
        ax[3].set_ylim([-1.5, 0])


        plt.savefig(dir + "/DSSF_" + str(i) + ".pdf")
        plt.clf()
        plt.close()

    fig11 = plt.figure(figsize=(8, 8), constrained_layout=False)
    grid = fig11.add_gridspec(2, 2, wspace=0, hspace=0)
    ax = [fig11.add_subplot(grid[0, 0]), fig11.add_subplot(grid[0, 1]), fig11.add_subplot(grid[1, 1]), fig11.add_subplot(grid[1, 0])]
    ax[0].set_title('0.1meV')
    ax[1].set_title('2meV')
    ax[2].set_title('4meV', y=-0.2)
    ax[3].set_title('7meV', y=-0.2)

    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].set_yticks([])

    C = ax[0].imshow(A[0][0:int(nK/2), int(nK/2):nK], origin='lower', extent=[-1.0, 0, 0, 1.5], aspect='auto', cmap='gnuplot2')
    C = ax[1].imshow(A[2][int(nK/2):nK, int(nK/2):nK], origin='lower', extent=[0, 1.0, 0, 1.5], aspect='auto', cmap='gnuplot2')
    C = ax[2].imshow(A[4][int(nK/2):nK, 0:int(nK/2)], origin='lower', extent=[0, 1.0, -1.5, 0], aspect='auto', cmap='gnuplot2')
    C = ax[3].imshow(A[7][0:int(nK/2), 0:int(nK/2)], origin='lower', extent=[-1.0, 0, -1.5, 0], aspect='auto', cmap='gnuplot2')

    plt.savefig(dir + "/DSSF_exp.pdf")
    plt.clf()
    plt.close()


    for i in range(len(w)):
        fig, ax = plt.subplots(figsize=(5,5))
        C = ax.imshow(A[i], origin='lower', extent=[-1.0, 1.0, -1.5, 1.5], aspect='auto', cmap='gnuplot2')

        plt.savefig(dir + "/DSSF_" + str(w[i]) + ".pdf")
        plt.clf()
        plt.close()

        
def read_MD(dir, w):
    directory = os.fsencode(dir)
    P = np.loadtxt(dir + "/pos.txt")
    T = np.loadtxt(dir + "/Time_steps.txt")

    S = np.loadtxt(dir + "/spin_t.txt").reshape((len(T), len(P), 3))

    A = DSSF(w, DSSF_K, S, P, T, True)
    np.savetxt(dir + "_DSSF.txt", A)
    return A

def read_MD_slice(dir, nK, w):
    directory = os.fsencode(dir)
    P = np.loadtxt(dir + "/pos.txt")
    T = np.loadtxt(dir + "/Time_steps.txt")

    S = np.loadtxt(dir + "/spin_t.txt").reshape((len(T), len(P), 3))

    nK = 100
    H = np.linspace(-1.5, 1.5, nK)
    L = np.linspace(-1.0, 1.0, nK)
    A, B = np.meshgrid(H, L)
    K = hhknk(A, B).reshape((nK*nK,3))

    A = DSSF(w, K, S, P, T, True)
    np.savetxt(dir + "_DSSF_sliced.txt", A)
    A = A.reshape((len(w), nK, nK))
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
# dir = "BCAO_zero_field_1.7K"
# dir = "Kitaev_BCAO"
# read_MD_tot(dir)
# read_MD_tot("BCAO_zero_field_5K_sasha")
# read_MD_tot("BCAO_zero_field_15K")

def plot_spin_config_2d(P, S, filename, zoom_frac=None):
    """
    Graphs the spin configuration projected on the 2D xy, xz, and yz planes.

    Args:
        P (numpy.ndarray): Array of site positions (N, 3).
        S (numpy.ndarray): Array of spin vectors (N, 3).
        filename (str): Base path to save the output plots. Projections will be appended.
    """
    base_filename, ext = os.path.splitext(filename)

    # Helper to compute axis limits based on zoom fraction
    def _limits_from_zoom(x, y, frac):
        if frac is None or frac >= 1.0:
            return (np.min(x), np.max(x)), (np.min(y), np.max(y))
        # Centered crop
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        rx = 0.5 * (xmax - xmin) * frac
        ry = 0.5 * (ymax - ymin) * frac
        rx = max(rx, 1e-6)
        ry = max(ry, 1e-6)
        return (cx - rx, cx + rx), (cy - ry, cy + ry)

    # --- XY Projection ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(P[:, 0], P[:, 1], c='lightblue', edgecolors='k', s=2, zorder=1)
    colors_xy = S[:, 2]
    q_xy = ax.quiver(P[:, 0], P[:, 1], S[:, 0], S[:, 1], colors_xy,
                     cmap='viridis', scale_units='xy', angles='xy', scale=1,
                     width=0.002, headwidth=3, headlength=4, zorder=2)
    cbar_xy = fig.colorbar(q_xy, ax=ax, shrink=0.8)
    cbar_xy.set_label('Spin z-component')
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')
    ax.set_title('Spin Configuration (2D XY Projection)')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)
    # Apply zoom if requested
    xlim, ylim = _limits_from_zoom(P[:, 0], P[:, 1], zoom_frac)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.savefig(f"{base_filename}_xy{ext}")
    plt.clf()
    plt.close(fig)

    # --- XZ Projection ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(P[:, 0], P[:, 1], c='lightblue', edgecolors='k', s=2, zorder=1)
    colors_xz = S[:, 1]
    q_xz = ax.quiver(P[:, 0], P[:, 1], S[:, 0], S[:, 2], colors_xz,
                     cmap='viridis', scale_units='xy', angles='xy', scale=1,
                     width=0.002, headwidth=3, headlength=4, zorder=2)
    cbar_xz = fig.colorbar(q_xz, ax=ax, shrink=0.8)
    cbar_xz.set_label('Spin y-component')
    ax.set_xlabel('x position')
    ax.set_ylabel('z position')
    ax.set_title('Spin Configuration (2D XZ Projection)')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)
    xlim, ylim = _limits_from_zoom(P[:, 0], P[:, 1], zoom_frac)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.savefig(f"{base_filename}_xz{ext}")
    plt.clf()
    plt.close(fig)

    # --- YZ Projection ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(P[:, 0], P[:, 1], c='lightblue', edgecolors='k', s=2, zorder=1)
    colors_yz = S[:, 0]
    q_yz = ax.quiver(P[:, 0], P[:, 1], S[:, 1], S[:, 2], colors_yz,
                     cmap='viridis', scale_units='xy', angles='xy', scale=1,
                     width=0.002, headwidth=3, headlength=4, zorder=2)
    cbar_yz = fig.colorbar(q_yz, ax=ax, shrink=0.8)
    cbar_yz.set_label('Spin x-component')
    ax.set_xlabel('y position')
    ax.set_ylabel('z position')
    ax.set_title('Spin Configuration (2D YZ Projection)')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)
    xlim, ylim = _limits_from_zoom(P[:, 0], P[:, 1], zoom_frac)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.savefig(f"{base_filename}_yz{ext}")
    plt.clf()
    plt.close(fig)

def plot_spin_config_3d(P, S, filename, color_by='z', subsample=None, zoom_frac=None):
    """
    Plot 3D spin orientations at their real-space positions.

    Args:
        P (ndarray): Positions, shape (N, 2 or 3). If 2D, z is set to 0.
        S (ndarray): Spins, shape (N, 3).
        filename (str): Output file path (e.g., path/to/spin_config_3d.pdf/png).
        color_by (str): One of {'z','magnitude','x','y'}. Color arrows by this component/magnitude.
        subsample (int|None): If set and N > subsample, uniformly subsample to this many arrows.
    """
    if P.shape[1] == 2:
        P3 = np.column_stack([P[:, 0], P[:, 1], np.zeros(len(P))])
    else:
        P3 = P.copy()

    N = len(P3)
    if subsample is not None and N > subsample:
        idx = np.linspace(0, N - 1, subsample, dtype=int)
        P3 = P3[idx]
        S = S[idx]

    # Choose colors
    if color_by == 'magnitude':
        colors = np.linalg.norm(S, axis=1)
        cbar_label = '|S|'
    elif color_by in ('x', 'X'):
        colors = S[:, 0]
        cbar_label = 'Sx'
    elif color_by in ('y', 'Y'):
        colors = S[:, 1]
        cbar_label = 'Sy'
    else:  # 'z' default
        colors = S[:, 2]
        cbar_label = 'Sz'

    # Heuristic arrow length based on typical neighbor spacing
    # Use median nearest-neighbor distance
    try:
        from sklearn.neighbors import NearestNeighbors  # optional, skip if not available
        nn = NearestNeighbors(n_neighbors=min(2, len(P3)))
        nn.fit(P3)
        dists, _ = nn.kneighbors(P3)
        # dists[:,0] is 0 (self); use dists[:,1] if available
        if dists.shape[1] > 1:
            med_nn = float(np.median(dists[:, 1]))
        else:
            med_nn = float(np.median(dists))
    except Exception:
        # Fallback: range-based scale
        span = np.ptp(P3, axis=0)
        med_nn = float(np.mean(span) / max(10, np.cbrt(len(P3))))
    arrow_len = 0.4 * med_nn if med_nn > 0 else 0.1

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter sites
    ax.scatter(P3[:, 0], P3[:, 1], P3[:, 2], c='lightgray', s=10, depthshade=True, linewidths=0)
    # Quiver spins
    q = ax.quiver(
        P3[:, 0], P3[:, 1], P3[:, 2],
        S[:, 0], S[:, 1], S[:, 2],
        length=arrow_len, normalize=True, cmap='viridis', lw=0.6
    )
    # Matplotlib's 3D quiver doesn't directly support array colors, so set via set_array
    try:
        q.set_array(colors)
        cbar = fig.colorbar(q, ax=ax, shrink=0.8)
        cbar.set_label(cbar_label)
    except Exception:
        pass

    # Set equal aspect ratio
    def _set_axes_equal(ax_3d):
        """Set equal aspect for 3D axes"""
        x_limits = ax_3d.get_xlim3d()
        y_limits = ax_3d.get_ylim3d()
        z_limits = ax_3d.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])
        ax_3d.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax_3d.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax_3d.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    _set_axes_equal(ax)
    # Apply zoom by shrinking displayed ranges around center
    if zoom_frac is not None and zoom_frac < 1.0:
        xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
        cx = 0.5 * (xlim[0] + xlim[1]); cy = 0.5 * (ylim[0] + ylim[1]); cz = 0.5 * (zlim[0] + zlim[1])
        rx = 0.5 * (xlim[1] - xlim[0]) * zoom_frac
        ry = 0.5 * (ylim[1] - ylim[0]) * zoom_frac
        rz = 0.5 * (zlim[1] - zlim[0]) * zoom_frac
        rx = max(rx, 1e-6); ry = max(ry, 1e-6); rz = max(rz, 1e-6)
        ax.set_xlim3d(cx - rx, cx + rx)
        ax.set_ylim3d(cy - ry, cy + ry)
        ax.set_zlim3d(cz - rz, cz + rz)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Spin configuration (3D)')
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
    plt.close(fig)

def compute_regional_SSSF(P, S, nK, output_dir, n_regions_x=3, n_regions_y=3, overlap=0.1, gb=False):
    """
    Split the spin configuration into regions and compute full analysis for each region.
    
    Args:
        P: Position array (N, 2 or 3)
        S: Spin array (N, 3)
        nK: Number of k-points for SSSF grid
        output_dir: Directory to save results
        n_regions_x: Number of regions in x direction
        n_regions_y: Number of regions in y direction
        overlap: Fraction of overlap between regions (0 to 1)
        gb: Whether to use global basis for SSSF
    
    Returns:
        Dictionary containing regional SSSF results
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get spatial extent
    x_min, x_max = np.min(P[:, 0]), np.max(P[:, 0])
    y_min, y_max = np.min(P[:, 1]), np.max(P[:, 1])
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Calculate region sizes with overlap
    region_width = x_range / n_regions_x * (1 + overlap)
    region_height = y_range / n_regions_y * (1 + overlap)
    
    # Step size between region centers
    x_step = x_range / n_regions_x
    y_step = y_range / n_regions_y
    
    # Store results
    regional_results = {}
    all_SSSF = []
    region_info = []
    all_Q_total = []
    all_energies = []
    
    # Create figure for regional SSSF overview
    fig_sssf, axes_sssf = plt.subplots(n_regions_y, n_regions_x, 
                                figsize=(4*n_regions_x, 4*n_regions_y))
    if n_regions_x == 1 and n_regions_y == 1:
        axes_sssf = np.array([[axes_sssf]])
    elif n_regions_x == 1:
        axes_sssf = axes_sssf.reshape(-1, 1)
    elif n_regions_y == 1:
        axes_sssf = axes_sssf.reshape(1, -1)
    
    # Create figure for regional chirality overview
    fig_chir, axes_chir = plt.subplots(n_regions_y, n_regions_x, 
                                figsize=(4*n_regions_x, 4*n_regions_y))
    if n_regions_x == 1 and n_regions_y == 1:
        axes_chir = np.array([[axes_chir]])
    elif n_regions_x == 1:
        axes_chir = axes_chir.reshape(-1, 1)
    elif n_regions_y == 1:
        axes_chir = axes_chir.reshape(1, -1)
    
    # Process each region
    for i in range(n_regions_y):
        for j in range(n_regions_x):
            # Define region boundaries
            x_center = x_min + (j + 0.5) * x_step
            y_center = y_min + (i + 0.5) * y_step
            
            x_region_min = x_center - region_width / 2
            x_region_max = x_center + region_width / 2
            y_region_min = y_center - region_height / 2
            y_region_max = y_center + region_height / 2
            
            # Select spins in this region
            mask = ((P[:, 0] >= x_region_min) & (P[:, 0] <= x_region_max) &
                    (P[:, 1] >= y_region_min) & (P[:, 1] <= y_region_max))
            
            n_spins = np.sum(mask)
            
            if n_spins < 10:  # Skip regions with too few spins
                print(f"Region ({i},{j}): Too few spins ({n_spins}), skipping")
                # Clear the subplot
                axes_sssf[i, j].text(0.5, 0.5, f'Region ({i},{j})\nInsufficient data\nN={n_spins}', 
                                        ha='center', va='center', transform=axes_sssf[i, j].transAxes)
                axes_sssf[i, j].set_xticks([])
                axes_sssf[i, j].set_yticks([])
                
                axes_chir[i, j].text(0.5, 0.5, f'Region ({i},{j})\nInsufficient data\nN={n_spins}', 
                                        ha='center', va='center', transform=axes_chir[i, j].transAxes)
                axes_chir[i, j].set_xticks([])
                axes_chir[i, j].set_yticks([])
                continue
            
            P_region = P[mask]
            S_region = S[mask]
            
            # Create region directory
            region_key = f"region_{i}_{j}"
            region_dir = os.path.join(output_dir, region_key)
            if not os.path.exists(region_dir):
                os.makedirs(region_dir)
            
            print(f"\nAnalyzing region ({i},{j}): {n_spins} spins")
            
            # 1. Compute SSSF for this region
            H = np.linspace(-2*np.pi, 2*np.pi, nK)
            L = np.linspace(-2*np.pi, 2*np.pi, nK)
            A, B = np.meshgrid(H, L)
            K = hhztoK(A, B).reshape((nK*nK, 3))
            
            SSSF_region = SSSF_q(K, S_region, P_region, gb)
            SSSF_region = SSSF_region.reshape((nK, nK, 3, 3))
            
            # Plot total SSSF for this region
            total_SSSF = contract('ijab->ij', SSSF_region)
            im = axes_sssf[i, j].pcolormesh(A, B, total_SSSF, cmap='viridis')
            axes_sssf[i, j].set_title(f'Region ({i},{j})\nN={n_spins}', fontsize=10)
            axes_sssf[i, j].set_xlabel('$k_x$')
            axes_sssf[i, j].set_ylabel('$k_y$')
            
            # Save SSSF components
            for ii, component_i in enumerate(['x', 'y', 'z']):
                for jj, component_j in enumerate(['x', 'y', 'z']):
                    np.savetxt(os.path.join(region_dir, f'SSSF_{component_i}{component_j}.txt'), 
                                SSSF_region[:,:,ii,jj])
            
            # 2. Plot 2D spin configuration
            base2d = os.path.join(region_dir, "spin_config_2d.pdf")
            plot_spin_config_2d(P_region, S_region, base2d)
            
            # 3. Plot 3D spin configuration
            base3d = os.path.join(region_dir, "spin_config_3d.pdf")
            plot_spin_config_3d(P_region, S_region, base3d, color_by='z', subsample=None)
            
            # 4. Compute and plot chirality
            plot_chirality_real_space(P_region, S_region, os.path.join(region_dir, "chirality"))
            
            # 5. Compute continuum chirality and detect cores
            cont = compute_continuum_chirality(P_region, S_region, grid_res=128, sigma=1.0)
            cores = detect_skyrmion_cores_from_grid(cont['X'], cont['Y'], cont['Sz'], cont['q'], 
                                                    cont['mask'], q_rel_thresh=0.2, 
                                                    sz_prominence=0.2, neighborhood=7)
            plot_continuum_chirality_and_cores(cont, cores, 
                                                os.path.join(region_dir, "continuum_chirality"))
            
            # Plot continuum chirality in overview figure
            X, Y, q = cont['X'], cont['Y'], cont['q']
            im_chir = axes_chir[i, j].pcolormesh(X, Y, q, shading='auto', cmap='RdBu_r')
            axes_chir[i, j].set_title(f'Region ({i},{j})\nQ={cont["Q_total"]:.3f}', fontsize=10)
            axes_chir[i, j].set_xlabel('x')
            axes_chir[i, j].set_ylabel('y')
            axes_chir[i, j].set_aspect('equal')
            
            # 6. Compute magnetization
            M = magnetization(S_region)
            np.savetxt(os.path.join(region_dir, "magnetization.txt"), M)
            
            # 7. Compute ordering wavevector
            ordering_q_result = ordering_q(S_region, P_region)
            np.savetxt(os.path.join(region_dir, "ordering_wave.txt"), ordering_q_result)
            
            # Store results
            regional_results[region_key] = {
                'SSSF': SSSF_region,
                'n_spins': n_spins,
                'bounds': (x_region_min, x_region_max, y_region_min, y_region_max),
                'center': (x_center, y_center),
                'positions': P_region,
                'spins': S_region,
                'magnetization': M,
                'Q_total': cont['Q_total'],
                'n_cores': len(cores['q_peaks']),
                'ordering_q': ordering_q_result
            }
            
            all_SSSF.append(SSSF_region)
            all_Q_total.append(cont['Q_total'])
            region_info.append({
                'i': i, 'j': j,
                'n_spins': n_spins,
                'center': (x_center, y_center),
                'Q_total': cont['Q_total'],
                'n_cores': len(cores['q_peaks'])
            })
            
            # Save region summary
            with open(os.path.join(region_dir, 'region_summary.txt'), 'w') as f:
                f.write(f"Region ({i},{j}) Summary\n")
                f.write(f"========================\n")
                f.write(f"Center: ({x_center:.4f}, {y_center:.4f})\n")
                f.write(f"Bounds: x=[{x_region_min:.4f}, {x_region_max:.4f}], ")
                f.write(f"y=[{y_region_min:.4f}, {y_region_max:.4f}]\n")
                f.write(f"Number of spins: {n_spins}\n")
                f.write(f"Magnetization: [{M[0]:.6f}, {M[1]:.6f}, {M[2]:.6f}]\n")
                f.write(f"|M|: {np.linalg.norm(M):.6f}\n")
                f.write(f"Total topological charge Q: {cont['Q_total']:.6f}\n")
                f.write(f"Number of skyrmion cores: {len(cores['q_peaks'])}\n")
    
    # Save overview figures
    fig_sssf.tight_layout()
    fig_sssf.savefig(os.path.join(output_dir, 'regional_SSSF_overview.pdf'))
    plt.close(fig_sssf)
    
    fig_chir.tight_layout()
    fig_chir.savefig(os.path.join(output_dir, 'regional_chirality_overview.pdf'))
    plt.close(fig_chir)
    
    # Compute statistics across regions
    if all_SSSF:
        all_SSSF = np.array(all_SSSF)
        mean_SSSF = np.mean(all_SSSF, axis=0)
        std_SSSF = np.std(all_SSSF, axis=0)
        
        # Plot mean and standard deviation
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Mean SSSF
        total_mean = contract('ijab->ij', mean_SSSF)
        im1 = axes[0, 0].pcolormesh(A, B, total_mean, cmap='viridis')
        axes[0, 0].set_title('Mean SSSF across regions')
        axes[0, 0].set_xlabel('$k_x$')
        axes[0, 0].set_ylabel('$k_y$')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Std SSSF
        total_std = contract('ijab->ij', std_SSSF)
        im2 = axes[0, 1].pcolormesh(A, B, total_std, cmap='hot')
        axes[0, 1].set_title('Std Dev of SSSF across regions')
        axes[0, 1].set_xlabel('$k_x$')
        axes[0, 1].set_ylabel('$k_y$')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Q distribution
        axes[1, 0].hist(all_Q_total, bins=min(10, len(all_Q_total)), edgecolor='black')
        axes[1, 0].set_xlabel('Topological charge Q')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title(f'Q distribution (mean={np.mean(all_Q_total):.3f})')
        
        # Magnetization magnitudes
        mag_magnitudes = [np.linalg.norm(regional_results[f"region_{info['i']}_{info['j']}"]['magnetization']) 
                            for info in region_info if f"region_{info['i']}_{info['j']}" in regional_results]
        axes[1, 1].hist(mag_magnitudes, bins=min(10, len(mag_magnitudes)), edgecolor='black')
        axes[1, 1].set_xlabel('|M|')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title(f'Magnetization distribution (mean={np.mean(mag_magnitudes):.3f})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'regional_statistics.pdf'))
        plt.close()
        
        # Save statistics
        np.savetxt(os.path.join(output_dir, 'SSSF_mean_total.txt'), total_mean)
        np.savetxt(os.path.join(output_dir, 'SSSF_std_total.txt'), total_std)
        
        regional_results['statistics'] = {
            'mean_SSSF': mean_SSSF,
            'std_SSSF': std_SSSF,
            'mean_Q': np.mean(all_Q_total),
            'std_Q': np.std(all_Q_total),
            'n_regions': len(all_SSSF)
        }
    
    # Create spatial map showing regions with properties
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Subplot 1: Region boundaries
    axes[0, 0].scatter(P[:, 0], P[:, 1], c='lightgray', s=1, alpha=0.5)
    for info in region_info:
        i, j = info['i'], info['j']
        key = f"region_{i}_{j}"
        if key in regional_results:
            bounds = regional_results[key]['bounds']
            rect = plt.Rectangle((bounds[0], bounds[2]), 
                                bounds[1] - bounds[0],
                                bounds[3] - bounds[2],
                                linewidth=2, edgecolor='red', 
                                facecolor='none', alpha=0.7)
            axes[0, 0].add_patch(rect)
    axes[0, 0].set_xlabel('x position')
    axes[0, 0].set_ylabel('y position')
    axes[0, 0].set_title(f'Regional division: {n_regions_x}×{n_regions_y} regions')
    axes[0, 0].set_aspect('equal')
    
    # Subplot 2: Q values by region
    axes[0, 1].scatter(P[:, 0], P[:, 1], c='lightgray', s=1, alpha=0.5)
    for info in region_info:
        i, j = info['i'], info['j']
        key = f"region_{i}_{j}"
        if key in regional_results:
            center = regional_results[key]['center']
            Q = regional_results[key]['Q_total']
            color = 'blue' if Q > 0 else 'red' if Q < 0 else 'gray'
            size = min(abs(Q) * 1000, 200) + 10
            axes[0, 1].scatter(center[0], center[1], c=color, s=size, alpha=0.7)
            axes[0, 1].text(center[0], center[1], f'{Q:.2f}', 
                            ha='center', va='center', fontsize=8)
    axes[0, 1].set_xlabel('x position')
    axes[0, 1].set_ylabel('y position')
    axes[0, 1].set_title('Topological charge Q by region')
    axes[0, 1].set_aspect('equal')
    
    # Subplot 3: Magnetization by region
    axes[1, 0].scatter(P[:, 0], P[:, 1], c='lightgray', s=1, alpha=0.5)
    for info in region_info:
        i, j = info['i'], info['j']
        key = f"region_{i}_{j}"
        if key in regional_results:
            center = regional_results[key]['center']
            M = regional_results[key]['magnetization']
            # Plot magnetization as arrow
            axes[1, 0].quiver(center[0], center[1], M[0], M[1], 
                            angles='xy', scale_units='xy', scale=10,
                            width=0.003, headwidth=3, headlength=4,
                            color='blue', alpha=0.7)
            axes[1, 0].text(center[0], center[1], f'|M|={np.linalg.norm(M):.2f}', 
                            ha='center', va='bottom', fontsize=7)
    axes[1, 0].set_xlabel('x position')
    axes[1, 0].set_ylabel('y position')
    axes[1, 0].set_title('Magnetization by region')
    axes[1, 0].set_aspect('equal')
    
    # Subplot 4: Number of cores by region
    axes[1, 1].scatter(P[:, 0], P[:, 1], c='lightgray', s=1, alpha=0.5)
    for info in region_info:
        i, j = info['i'], info['j']
        key = f"region_{i}_{j}"
        if key in regional_results:
            center = regional_results[key]['center']
            n_cores = regional_results[key]['n_cores']
            axes[1, 1].scatter(center[0], center[1], c='purple', s=50 + n_cores*20, alpha=0.7)
            axes[1, 1].text(center[0], center[1], str(n_cores), 
                            ha='center', va='center', fontsize=10, color='white')
    axes[1, 1].set_xlabel('x position')
    axes[1, 1].set_ylabel('y position')
    axes[1, 1].set_title('Number of skyrmion cores by region')
    axes[1, 1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regional_properties_map.pdf'))
    plt.close()
    
    # Save comprehensive summary
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
        f.write(f"Regional Analysis Summary\n")
        f.write(f"=========================\n")
        f.write(f"Total regions analyzed: {len(region_info)}\n")
        f.write(f"Grid: {n_regions_x}×{n_regions_y} with {overlap:.1%} overlap\n\n")
        
        if regional_results.get('statistics'):
            f.write(f"Statistical Summary:\n")
            f.write(f"Mean Q: {regional_results['statistics']['mean_Q']:.6f}\n")
            f.write(f"Std Q: {regional_results['statistics']['std_Q']:.6f}\n\n")
        
        f.write(f"Per-Region Details:\n")
        f.write(f"{'Region':<12} {'N_spins':<10} {'Q':<10} {'|M|':<10} {'N_cores':<10}\n")
        f.write("-" * 52 + "\n")
        for info in region_info:
            i, j = info['i'], info['j']
            key = f"region_{i}_{j}"
            if key in regional_results:
                data = regional_results[key]
                f.write(f"{key:<12} {data['n_spins']:<10} ")
                f.write(f"{data['Q_total']:<10.4f} ")
                f.write(f"{np.linalg.norm(data['magnetization']):<10.4f} ")
                f.write(f"{data['n_cores']:<10}\n")
    
    print(f"\nRegional analysis complete. Results saved to {output_dir}")
    print(f"Analyzed {len(region_info)} regions with {n_regions_x}×{n_regions_y} division")
    
    return regional_results


def parse_spin_config(directory):
    nK = 101
    SSSF = np.zeros((nK, nK, 3, 3))

    H = np.linspace(-1, 1, nK)
    L = np.linspace(-1, 1, nK)
    C, D = np.meshgrid(H, L)
    for file in sorted(os.listdir(directory)):  
        filename = os.fsdecode(file)
        # print(filename)
        if os.path.isdir(directory + "/" + filename):
            S = np.loadtxt(directory + "/" + filename + "/spin.txt")
            P = np.loadtxt(directory + "/" + filename + "/pos.txt")
            SSSF += SSSF2D(S, P, nK, directory + "/" + filename )
            base2d = directory + "/" + filename + "/spin_config_2d.pdf"
            print("Computing 2D spin configuration plot")
            plot_spin_config_2d(P, S, base2d)
            # Zoomed 2D projections (50% window around center)
            plot_spin_config_2d(P, S, base2d.replace('.pdf', '_zoom.pdf'), zoom_frac=0.5)
            # 3D orientation plot
            print("Computing 3D configuration")
            # base3d = directory + "/" + filename + "/spin_config_3d.pdf"
            # plot_spin_config_3d(P, S, base3d, color_by='z', subsample=None)
            # Zoomed 3D view (50% window around center)
            # plot_spin_config_3d(P, S, base3d.replace('.pdf', '_zoom.pdf'), color_by='z', subsample=None, zoom_frac=0.5)
            # Load energy landscape data (assumes two columns: index and energy)
            print("Computing chirality plot")
            # plot_chirality_real_space(P, S, directory + "/" + filename + "/chirality")
            # Continuum chirality on grid + core detection
            print("Computing coarse grained chirality")
            # cont = compute_continuum_chirality(P, S, grid_res=256, sigma=1.0)
            # cores = detect_skyrmion_cores_from_grid(cont['X'], cont['Y'], cont['Sz'], cont['q'], cont['mask'], q_rel_thresh=0.2, sz_prominence=0.2, neighborhood=9)
            # plot_continuum_chirality_and_cores(cont, cores, directory + "/" + filename + "/continuum_chirality")
            # regnault_magnetic_moment_reconstruction(P, directory + "/" + filename, 'xx')
            # regnault_magnetic_moment_reconstruction(P, directory + "/" + filename, 'yy')
            # compute_regional_SSSF(P, S, 101, directory + "/" + filename + "/region", 5, 5)
            energy_landscape_path = os.path.join(directory, filename, "energy_landscape.txt")
            if os.path.exists(energy_landscape_path):
                energy_data = np.loadtxt(energy_landscape_path, comments=['E', '/'])  # skip header lines
                # If the file has two columns, use the second as energy
                if energy_data.ndim == 2 and energy_data.shape[1] >= 2:
                    energy_landscape = energy_data[:, 1]
                else:
                    energy_landscape = energy_data
                # Plot energy landscape as a heatmap using the 0 and 1 components of P as coordinates
                plt.figure(figsize=(8, 6))
                sc = plt.scatter(P[:, 0], P[:, 1], c=energy_landscape, cmap='inferno', s=20)
                plt.colorbar(sc, label='Energy')
                plt.xlabel('x position')
                plt.ylabel('y position')
                plt.title('Energy Landscape')
                plt.tight_layout()
                plt.savefig(directory + "/" + filename + "/energy_landscape_heatmap.pdf")
                plt.clf()
                plt.close()

                np.savetxt(directory + "/" + filename + "/energy_density.txt", np.array([np.mean(energy_landscape)]))

                energy_density_by_section(directory + "/" + filename + "/energy_coarse_grained.pdf", P, energy_landscape, 5, 5)
                
                # Perform energetics argument analysis
                print("Computing energetics argument (defect-free vs defective regions)")
                energetics_result = energetics_argument(P, S, energy_landscape, directory + "/" + filename + "/energetics_analysis", 
                                                      J=1.0, grid_res=256, sigma=1.0)
                print(f"Found {energetics_result['N_free']} defect-free sites and {energetics_result['N_def']} defective sites")
                print(f"Average energy: defect-free = {energetics_result['E_free_mean']:.6f}, defective = {energetics_result['E_def_mean']:.6f}")

                # Skyrmion-lattice-aware energetics that does not classify cores as defects
                print("Computing skyrmion-lattice-aware energetics (ordered vs defective)")
                skx_res = energetics_argument_skyrmion(P, S, energy_landscape, directory + "/" + filename + "/energetics_skx",
                                                       grid_res=256, sigma=1.0, psi6_thr=0.75, sp_rel_thr=0.20, ori_thr_deg=15.0)
                print(f"SkX ordered: {skx_res['N_ordered']} sites; defective: {skx_res['N_def']} sites")
                print(f"Energy means: ordered = {skx_res['E_ordered_mean']:.6f}, defective = {skx_res['E_def_mean']:.6f}")
            else:
                print(f"Warning: {energy_landscape_path} not found, skipping energy landscape plot.")
                energy_landscape = np.zeros(P.shape[0])



    SSSF = SSSF / len(os.listdir(directory))
    SSSFGraph2D(C, D, contract('ijab->ij', SSSF), directory + "/SSSF_tot")

    # Plot each component of SSSF
    for i, component_i in enumerate(['x', 'y', 'z']):
        for j, component_j in enumerate(['x', 'y', 'z']):
            SSSFGraph2D(C, D, SSSF[:,:,i,j], directory + f"/SSSF_{component_i}{component_j}")


def parse_spin_config_file(directory):
    nK = 101
    SSSF = np.zeros((nK, nK, 3, 3))

    H = np.linspace(-1, 1, nK)
    L = np.linspace(-1, 1, nK)
    C, D = np.meshgrid(H, L)

    S = np.loadtxt(directory + "/spin.txt")
    P = np.loadtxt(directory + "/pos.txt")
    SSSF += SSSF2D(S, P, nK, directory )
    base2d = directory + "/spin_config_2d.pdf"
    print("Computing 2D spin configuration plot")
    plot_spin_config_2d(P, S, base2d)
    # Zoomed 2D projections (50% window around center)
    plot_spin_config_2d(P, S, base2d.replace('.pdf', '_zoom.pdf'), zoom_frac=0.5)
    # 3D orientation plot
    print("Computing 3D configuration")
    # base3d = directory + "/spin_config_3d.pdf"
    # plot_spin_config_3d(P, S, base3d, color_by='z', subsample=None)
    # Zoomed 3D view (50% window around center)
    # plot_spin_config_3d(P, S, base3d.replace('.pdf', '_zoom.pdf'), color_by='z', subsample=None, zoom_frac=0.5)
    # Load energy landscape data (assumes two columns: index and energy)
    print("Computing chirality plot")
    plot_chirality_real_space(P, S, directory + "/chirality")
    # Continuum chirality on grid + core detection
    print("Computing coarse grained chirality")
    cont = compute_continuum_chirality(P, S, grid_res=256, sigma=1.0)
    cores = detect_skyrmion_cores_from_grid(cont['X'], cont['Y'], cont['Sz'], cont['q'], cont['mask'], q_rel_thresh=0.2, sz_prominence=0.2, neighborhood=9)
    plot_continuum_chirality_and_cores(cont, cores, directory + "/continuum_chirality")
    # regnault_magnetic_moment_reconstruction(P, directory, 'xx')
    # regnault_magnetic_moment_reconstruction(P, directory, 'yy')
    # compute_regional_SSSF(P, S, 101, directory + "/region", 5, 5)
    energy_landscape_path = os.path.join(directory, "energy_landscape.txt")
    if os.path.exists(energy_landscape_path):
        energy_data = np.loadtxt(energy_landscape_path, comments=['E', '/'])  # skip header lines
        # If the file has two columns, use the second as energy
        if energy_data.ndim == 2 and energy_data.shape[1] >= 2:
            energy_landscape = energy_data[:, 1]
        else:
            energy_landscape = energy_data
        # Plot energy landscape as a heatmap using the 0 and 1 components of P as coordinates
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(P[:, 0], P[:, 1], c=energy_landscape, cmap='inferno', s=20)
        plt.colorbar(sc, label='Energy')
        plt.xlabel('x position')
        plt.ylabel('y position')
        plt.title('Energy Landscape')
        plt.tight_layout()
        plt.savefig(directory + "/energy_landscape_heatmap.pdf")
        plt.clf()
        plt.close()

        np.savetxt(directory + "/energy_density.txt", np.array([np.mean(energy_landscape)]))

        energy_density_by_section(directory + "/energy_coarse_grained.pdf", P, energy_landscape, 5, 5)
        
        # Perform energetics argument analysis
        print("Computing energetics argument (defect-free vs defective regions)")
        energetics_result = energetics_argument(P, S, energy_landscape, directory + "/energetics_analysis", 
                                                J=1.0, grid_res=256, sigma=1.0)
        print(f"Found {energetics_result['N_free']} defect-free sites and {energetics_result['N_def']} defective sites")
        print(f"Average energy: defect-free = {energetics_result['E_free_mean']:.6f}, defective = {energetics_result['E_def_mean']:.6f}")

        # Skyrmion-lattice-aware energetics that does not classify cores as defects
        print("Computing skyrmion-lattice-aware energetics (ordered vs defective)")
        skx_res = energetics_argument_skyrmion(P, S, energy_landscape, directory + "/energetics_skx",
                                                grid_res=256, sigma=1.0, psi6_thr=0.75, sp_rel_thr=0.20, ori_thr_deg=15.0)
        print(f"SkX ordered: {skx_res['N_ordered']} sites; defective: {skx_res['N_def']} sites")
        print(f"Energy means: ordered = {skx_res['E_ordered_mean']:.6f}, defective = {skx_res['E_def_mean']:.6f}")
    else:
        print(f"Warning: {energy_landscape_path} not found, skipping energy landscape plot.")
        energy_landscape = np.zeros(P.shape[0])


def read_field_scan(directory):
    h_values = []
    m_values = []
    m_stds = []
    for subdir in sorted(os.listdir(directory)):
        full_path = os.path.join(directory, subdir)
        if os.path.isdir(full_path) and subdir.startswith("h_"):
            try:
                h_str = subdir.split('_')[1]
                h = float(h_str)
                spin_file = os.path.join(full_path, "0/spin.txt")
                print(spin_file)
                if os.path.exists(spin_file):
                    M = np.loadtxt(spin_file)
                    M_mean = np.mean(M, axis=0)
                    # M_stdev = np.std(M, axis=0)
                    h_values.append(h)
                    m_values.append(M_mean)
                    # m_stds.append(M_stdev)
                else:
                    print(f"Magnetization file not found in {full_path}")

            except (IndexError, ValueError) as e:
                print(f"Could not parse field value from directory {subdir}: {e}")
            except Exception as e:
                print(f"An error occurred while processing {full_path}: {e}")

    if not h_values:
        print(f"No magnetization data found in {directory}")
        return

    # Sort values by field strength for a clean plot
    sorted_indices = np.argsort(h_values)
    h_values_sorted = np.array(h_values)[sorted_indices]
    m_values_sorted = np.array(m_values)[sorted_indices]
    # m_stds_sorted = np.array(m_stds)[sorted_indices]

    # Save magnetization as a function of h
    # output_data = np.c_[h_values_sorted, m_values_sorted, m_stds_sorted]
    output_data = np.c_[h_values_sorted, m_values_sorted]
    output_filename = os.path.join(directory, "magnetization_vs_field.txt")
    np.savetxt(output_filename, output_data, header="h Mx My Mz", fmt='%f %f %f %f')

    plt.figure(figsize=(10, 6))
    # plt.errorbar(h_values_sorted, m_values_sorted[:,0], yerr=m_stds_sorted[:,0], fmt='-o', label='Mx')
    # plt.errorbar(h_values_sorted, m_values_sorted[:,1], yerr=m_stds_sorted[:,1], fmt='-o', label='My')
    # plt.errorbar(h_values_sorted, m_values_sorted[:,2], yerr=m_stds_sorted[:,2], fmt='-o', label='Mz')
    plt.plot(h_values_sorted, m_values_sorted[:,0], '-o', label='Mx')
    plt.plot(h_values_sorted, m_values_sorted[:,1], '-o', label='My')
    plt.plot(h_values_sorted, m_values_sorted[:,2], '-o', label='Mz')
    plt.xlabel("Field Strength (h)")
    plt.ylabel("Magnetization Magnitude |M|")
    plt.legend(["Mx", "My", "Mz"], loc='upper right')
    plt.title(f"Magnetization vs. Field Strength in {os.path.basename(directory)}")
    plt.grid(True)
    plt.savefig(os.path.join(directory, "magnetization_vs_field.pdf"))
    plt.clf()
    plt.close()
    
if __name__ == "__main__":

    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    field_scan = sys.argv[2] if len(sys.argv) > 2 else "field_scan"
    if field_scan.lower() == "true":
        read_field_scan(base_dir)
    else:
        if os.path.isdir(base_dir):
            for subdir in sorted(os.listdir(base_dir)):
                full_path = os.path.join(base_dir, subdir)
                if os.path.isdir(full_path):
                    print(f"Processing directory: {full_path}")
                    try:
                        print("Computing spin configuration information...")
                        parse_spin_config(full_path)
                        # read_MD_tot(full_path)
                    except Exception as e:
                        print(f"Could not process {full_path}: {e}")



# dir = "BCAO_sasha_phase/J3_1.308000_Jzp_0.000000"
# S = np.loadtxt(dir + "/spins.txt")
# P = np.loadtxt(dir + "/pos.txt")
# SSSF2D(S, P, 100, dir)

# read_MD_tot("BCAO_J1J3")
# parseDSSF(dir)

# A = np.loadtxt("BCAO_Sasha_Sweep_ab.txt")
# plt.plot(A[:,0], A[:,1]/0.086)
# plt.xlabel(r"$\theta$ (from a to b)")
# plt.ylabel("B/T")
# plt.savefig("BCAO_Sasha_Sweep_ab.pdf")
# plt.clf()
# A = np.loadtxt("BCAO_Sasha_Sweep_ac.txt")
# plt.plot(A[:,0], A[:,1]/0.086)
# plt.xlabel(r"$\theta$ (from a to c)")
# plt.ylabel("B/T")
# plt.savefig("BCAO_Sasha_Sweep_ac.pdf")

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