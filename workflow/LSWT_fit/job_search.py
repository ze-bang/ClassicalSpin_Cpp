import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import hamiltonian_init_new as ham
from joblib import Parallel, delayed
import sys

E_array=np.linspace(0,2,50)
Nid=int(sys.argv[1])



Np3=50
Np =50

Np2 = 100

S=0.5

a_g=-3.067
b_g=-7.423


B = 3


b_direction='b'

df1 = pd.read_csv("plot-data big1.csv")
df2 = pd.read_csv("plot-data big2.csv")
df3 = pd.read_csv("plot-data big3.csv")
df4 = pd.read_csv("plot-data big3 up.csv")


column_keys = df1.columns
print(column_keys)

column_keys = df2.columns
print(column_keys)

# Access a specific column
column_data_1 = df1[["x", " y"]].values.tolist()
column_data_1=np.array(column_data_1)

# Access a specific column
column_data_2 = df2[["x", " y"]].values.tolist()

column_data_2=np.array(column_data_2)

# Access a specific column
column_data_3 = df3[["x", " y"]].values.tolist()

column_data_3=np.array(column_data_3)

# Access a specific column
column_data_4 = df4[["x", " y"]].values.tolist()

column_data_4=np.array(column_data_4)

data_all_gamma=np.vstack((column_data_1,column_data_2,column_data_3,column_data_4))

data_all_gamma[48,0]-=0.01
data_all_gamma[58,0]-=0.01


pi=np.pi
sq=np.sqrt(3)

L=4*pi/3+8*pi/(3*sq)+2*pi/3+2*pi/(3*sq)

r1=(2*pi/3)/L
r2=(4*pi/3)/L
r3=(4*pi/3+4*pi/(3*sq))/L
r4=(4*pi/3+8*pi/(3*sq))/L
r5=(6*pi/3+8*pi/(3*sq))/L
r6=(6*pi/3+8*pi/(3*sq)+2*pi/(3*sq))/L


ratios=[r1,r2,r3,r4,r5,r6]


L1_x=[]
L1_y=[]
L2_x=[]
L2_y=[]
L3_x=[]
L3_y=[]
L4_x=[]
L4_y=[]
L5_x=[]
L5_y=[]
L6_x=[]
L6_y=[]
L7_x=[]
L7_y=[]
L8_x=[]
L8_y=[]
for i in range(len(data_all_gamma[:,0])):
    if(data_all_gamma[i,0]<r1):
        L1_x.append(2*pi/3*data_all_gamma[i,0]/r1)
        L1_y.append(data_all_gamma[i,1])
    elif(data_all_gamma[i,0]<r2):
        L2_x.append(2*pi/3*(data_all_gamma[i,0]-r1)/(r2-r1)-2*pi/3)
        L2_y.append(data_all_gamma[i,1])
    elif(data_all_gamma[i,0]<r3):
        L3_x.append(4*pi/(3*sq)*(data_all_gamma[i,0]-r2)/(r3-r2))
        L3_y.append(data_all_gamma[i,1])
    elif(data_all_gamma[i,0]<r4):
        L4_x.append(-(4*pi/(3*sq)*(data_all_gamma[i,0]-r3)/(r4-r3)-4*pi/(3*sq)))
        L4_y.append(data_all_gamma[i,1])
    elif(data_all_gamma[i,0]<r5 and data_all_gamma[i,1]<8):
        L5_x.append(2*pi/3*(data_all_gamma[i,0]-r4)/(r5-r4))
        L5_y.append(data_all_gamma[i,1])
    elif(data_all_gamma[i,0]<r5 and data_all_gamma[i,1]>8):
        L6_x.append(2*pi/3*(data_all_gamma[i,0]-r4)/(r5-r4))
        L6_y.append(data_all_gamma[i,1])
    elif(data_all_gamma[i,0]<r6 and data_all_gamma[i,1]<9):
        L7_x.append(2*pi/(3*sq)*(data_all_gamma[i,0]-r5)/(r6-r5))
        L7_y.append(data_all_gamma[i,1])
    elif(data_all_gamma[i,0]<r6 and data_all_gamma[i,1]>9):
        L8_x.append(2*pi/(3*sq)*(data_all_gamma[i,0]-r5)/(r6-r5))
        L8_y.append(data_all_gamma[i,1])

data1=np.zeros((3,len(L1_x)))
data2=np.zeros((3,len(L2_x)))
data3=np.zeros((3,len(L3_x)))
data4=np.zeros((3,len(L4_x)))
data5=np.zeros((3,len(L5_x)))
data6=np.zeros((3,len(L6_x)))
data7=np.zeros((3,len(L7_x)))
data8=np.zeros((3,len(L8_x)))

data1[2]=L1_y
data1[1]=L1_x
#plt.scatter(data1[0],data1[1],c=data1[2],vmin=0,vmax=8)
data2[2]=L2_y
data2[0]=np.array(L2_x)*0.5*sq
data2[1]=-np.array(L2_x)*0.5
#plt.scatter(data2[0],data2[1],c=data2[2],vmin=0,vmax=8)
#print(data2[2])

data3[2]=L3_y
data3[0]=-np.array(L3_x)*0.5
data3[1]=np.array(L3_x)*0.5*sq
#plt.scatter(data3[0],data3[1],c=data3[2],vmin=0,vmax=8)
data4[2]=np.array(L4_y)
data4[0]=np.array(L4_x)
#plt.scatter(data4[0],data4[1],c=data4[2],vmin=0,vmax=8)
data5[2]=np.array(L5_y)
data5[1]=np.array(L5_x)[::-1]+2*pi/3
#plt.scatter(data5[0],data5[1],c=data5[2],vmin=0,vmax=8)

data7[2]=np.array(L7_y)
data7[0]=-np.array(L7_x)
data7[1]=np.ones(len(L7_x))*2*pi/3
#plt.scatter(data7[0],data7[1],c=data7[2],vmin=0,vmax=8)
#plt.gca().set_aspect('equal')

#plt.show()


data6[2]=np.array(L6_y)
data6[1]=np.array(L6_x)[::-1]+2*pi/3
#plt.scatter(data6[0],data6[1],c=data6[2],vmin=8,vmax=14)
data8[2]=np.array(L8_y)
data8[0]=-np.array(L8_x)
data8[1]=np.ones(len(L8_x))*2*pi/3
#plt.scatter(data8[0],data8[1],c=data8[2],vmin=8,vmax=14)
#plt.gca().set_aspect('equal')
#plt.show()

data_all_lower=np.hstack((data1,data2,data3,data4,data5,data7))
data_all_upper=np.hstack((data6,data8))
#print(np.shape(data_all_lower))
#print(np.shape(data_all_upper))


def J3(J1,S,b_g):
    return 1/(3*S)*(b_g-3*J1*S)

def J1z(J1,S,a_g,b_g,c_M1_b,c_M1_a):
    return 1/(8*S)*(c_M1_b-c_M1_a+2*a_g+2*b_g-8*J1*S)

def J3z(J1,S,a_g,b_g,c_M1_b,c_M1_a):
    return 1/(8*S)*(c_M1_a-c_M1_b+2/3*a_g-2*b_g+8*J1*S)

def D(S,c_M1_a,c_M1_b):
    return 1/(4*S)*(c_M1_b+c_M1_a)

def F2(J1,S,b_g,c_M1_a,c_M1_b,b_M1_a):
    return np.max([(1/(16*S**2)*(c_M1_a**2-4*b_M1_a-(c_M1_b-8*J1*S)*(4*b_g+c_M1_b-8*J1*S))),0])

def G2(J1,S,b_g,c_M1_a,c_M1_b,b_M1_b):
    return np.max([(1/(16*S**2)*(c_M1_b**2-4*b_M1_b+(c_M1_a+8*J1*S)*(4*b_g-c_M1_a-8*J1*S))),0])


def F2_old(J1,S,b_g,c_M1_a,c_M1_b,b_M1_a):
    return (1/(16*S**2)*(c_M1_a**2-4*b_M1_a-(c_M1_b-8*J1*S)*(4*b_g+c_M1_b-8*J1*S)))

def G2_old(J1,S,b_g,c_M1_a,c_M1_b,b_M1_b):
    return (1/(16*S**2)*(c_M1_b**2-4*b_M1_b+(c_M1_a+8*J1*S)*(4*b_g-c_M1_a-8*J1*S)))


def f_a(x):
    return 0.12817723*x -0.04713861

def f_b(x):
    return -(0.12153832*x -1.09268781)

b_M1_a=np.linspace(30,60,200)
c_M1_a=f_a(b_M1_a)
b_M1_b=np.linspace(50,80,200)
c_M1_b=f_b(b_M1_b)

#print(f_a(42.098),f_b(62.617))


J1_x = np.linspace(-9, -2, Np2)

b_M1_a = np.linspace(30, 60, Np)
c_M1_a = f_a(b_M1_a)
b_M1_b = np.linspace(50, 80, Np)
c_M1_b = f_b(b_M1_b)

deltaF = np.zeros((Np, Np))
deltaG = np.zeros((Np, Np))
F_se = np.zeros((2, Np, Np))
G_se = np.zeros((2, Np, Np))

for i1 in range(Np):
    print(i1)
    ca_i = c_M1_a[i1]
    ba_i = b_M1_a[i1]
    for j1 in range(Np):
        cb_j = c_M1_b[j1]
        bb_j = b_M1_b[j1]

        # Precompute for current i1,j1 pair
        J3_x = J3(J1_x, S, b_g)
        J1z_x = J1z(J1_x, S, a_g, b_g, cb_j, ca_i)
        J3z_x = J3z(J1_x, S, a_g, b_g, cb_j, ca_i)
        D_x = D(S, ca_i, cb_j)

        # Vectorized F_x and G_x
        F_x = np.sqrt([F2_old(j1x, S, b_g, ca_i, cb_j, ba_i) for j1x in J1_x])
        G_x = np.sqrt([G2_old(j1x, S, b_g, ca_i, cb_j, bb_j) for j1x in J1_x])

        # Find valid ranges (avoid nans)
        valid_F = ~np.isnan(F_x)
        valid_G = ~np.isnan(G_x)

        if np.any(valid_F):
            Fs = J1_x[np.argmax(valid_F)]
            Fe = J1_x[len(valid_F) - 1 - np.argmax(valid_F[::-1])]
        else:
            Fs = Fe = J1_x[0]

        if np.any(valid_G):
            Gs = J1_x[np.argmax(valid_G)]
            Ge = J1_x[len(valid_G) - 1 - np.argmax(valid_G[::-1])]
        else:
            Gs = Ge = J1_x[0]

        # Store results
        deltaF[i1, j1] = Fe - Fs
        deltaG[i1, j1] = Ge - Gs
        F_se[:, i1, j1] = [Fs, Fe]
        G_se[:, i1, j1] = [Gs, Ge]

def ends(G_se,F_se,Np):
    J1_ends=np.zeros((Np,Np,2))
    for i1 in range(Np):
        for j1 in range(Np):
            J1_ends[i1,j1,0]=np.max([G_se[0,i1,j1],F_se[0,i1,j1]])
            J1_ends[i1,j1,1]=np.min([G_se[1,i1,j1],F_se[1,i1,j1]])
    return J1_ends

J1_ends=ends(G_se,F_se,Np)








#####################
print('all_+ve')

err_all = np.zeros((Np3, Np, Np, 2))
kitaev_all = np.zeros((Np3, Np, Np, 2))


# Create reusable base for parameter dict
base_params = {
    'g_b': 5.0,
    'g_a': 5.0,
    'g_z': 2.7,
    'mu_B': 0.05788381,
    'S': 0.5,
    'J2': 0.0,
    'J2z': 0.0,
    'E': E_array[Nid],
    'delta_1': np.array([[0, 1], [-np.sqrt(3)/2, -1/2], [np.sqrt(3)/2, -1/2]]),
    'delta_2': np.array([[np.sqrt(3), 0], [np.sqrt(3)/2, 1.5], [-np.sqrt(3)/2, 1.5],
                         [-np.sqrt(3), 0], [-np.sqrt(3)/2, -1.5], [np.sqrt(3)/2, -1.5]]),
    'delta_3': np.array([[0, -2], [np.sqrt(3), 1], [-np.sqrt(3), 1]]),
    'phi': np.array([0, 2*np.pi/3, -2*np.pi/3])
}

# Preallocate reusable arrays
data_t_lower = np.zeros((3, 82))
data_t_upper = np.zeros((3, 17))


def compute_kitaev_and_error(i1, j1, k1, J1_val, c_a, c_b, b_a, b_b):
    J3_x = J3(J1_val, S, b_g)
    J1z_x = J1z(J1_val, S, a_g, b_g, c_b, c_a)
    J3z_x = J3z(J1_val, S, a_g, b_g, c_b, c_a)
    D_x = D(S, c_a, c_b)
    F_x = np.sqrt(F2(J1_val, S, b_g, c_a, c_b, b_a))
    G_x = np.sqrt(G2(J1_val, S, b_g, c_a, c_b, b_b))

    kitaev_vals = [D_x - np.sqrt(2) * F_x, D_x + np.sqrt(2) * F_x]

    parameters = base_params.copy()
    parameters.update({
        'J1': J1_val,
        'J3': J3_x,
        'J1z': J1z_x,
        'J3z': J3z_x,
        'D': D_x,
        'F': F_x,
        'G': G_x
    })

    data_t_lower = np.zeros(82)
    data_t_upper = np.zeros(17)

    for ii in range(82):
        kvec = data_all_lower[:2, ii]
        data_t_lower[ii] = np.real(ham.hamiltonian_diagonalization(kvec, 'b', parameters, B)[1])

    for ii in range(17):
        kvec = data_all_upper[:2, ii]
        data_t_upper[ii] = np.real(ham.hamiltonian_diagonalization(kvec, 'b', parameters, B)[0])

    # R²-like scores
    diff_lower = data_t_lower - data_all_lower[2]
    diff_upper = data_t_upper - data_all_upper[2]
    var_lower = data_all_lower[2] - np.mean(data_all_lower[2])
    var_upper = data_all_upper[2] - np.mean(data_all_upper[2])

    ss_res_lower = np.sum(diff_lower ** 2)
    ss_res_upper = np.sum(diff_upper ** 2)
    ss_tot_lower = np.sum(var_lower ** 2)
    ss_tot_upper = np.sum(var_upper ** 2)

    err0 = 1 - (ss_res_lower + ss_res_upper) / (ss_tot_lower + ss_tot_upper)
    err1 = 1 - ss_res_lower / ss_tot_lower

    return (k1, i1, j1, kitaev_vals, err0, err1)

results = Parallel(n_jobs=-1, verbose=5)(
    delayed(compute_kitaev_and_error)(
        i1, j1, k1, J1_val, 
        c_M1_a[i1], c_M1_b[j1],
        b_M1_a[i1], b_M1_b[j1]
    )
    for i1 in range(Np)
    for j1 in range(Np)
    for k1, J1_val in enumerate(np.linspace(J1_ends[i1, j1, 0], J1_ends[i1, j1, 1], Np3))
)


for k1, i1, j1, kitaev_vals, err0, err1 in results:
    kitaev_all[k1, i1, j1, 0] = kitaev_vals[0]
    kitaev_all[k1, i1, j1, 1] = kitaev_vals[1]
    err_all[k1, i1, j1, 0] = err0
    err_all[k1, i1, j1, 1] = err1


data_all_p_p={}
data_all_p_p['kitaev']=kitaev_all
data_all_p_p['err_all']=err_all

np.save('data_all_p_p_E_'+str(Nid)+'.npy', data_all_p_p)


#####################



#####################
print('+ve_-ve')

err_all = np.zeros((Np3, Np, Np, 2))
kitaev_all = np.zeros((Np3, Np, Np, 2))


# Create reusable base for parameter dict
base_params = {
    'g_b': 5.0,
    'g_a': 5.0,
    'g_z': 2.7,
    'mu_B': 0.05788381,
    'S': 0.5,
    'J2': 0.0,
    'J2z': 0.0,
    'E': E_array[Nid],
    'delta_1': np.array([[0, 1], [-np.sqrt(3)/2, -1/2], [np.sqrt(3)/2, -1/2]]),
    'delta_2': np.array([[np.sqrt(3), 0], [np.sqrt(3)/2, 1.5], [-np.sqrt(3)/2, 1.5],
                         [-np.sqrt(3), 0], [-np.sqrt(3)/2, -1.5], [np.sqrt(3)/2, -1.5]]),
    'delta_3': np.array([[0, -2], [np.sqrt(3), 1], [-np.sqrt(3), 1]]),
    'phi': np.array([0, 2*np.pi/3, -2*np.pi/3])
}

# Preallocate reusable arrays
data_t_lower = np.zeros((3, 82))
data_t_upper = np.zeros((3, 17))


def compute_kitaev_and_error(i1, j1, k1, J1_val, c_a, c_b, b_a, b_b):
    J3_x = J3(J1_val, S, b_g)
    J1z_x = J1z(J1_val, S, a_g, b_g, c_b, c_a)
    J3z_x = J3z(J1_val, S, a_g, b_g, c_b, c_a)
    D_x = D(S, c_a, c_b)
    F_x = np.sqrt(F2(J1_val, S, b_g, c_a, c_b, b_a))
    G_x =-np.sqrt(G2(J1_val, S, b_g, c_a, c_b, b_b))

    kitaev_vals = [D_x - np.sqrt(2) * F_x, D_x + np.sqrt(2) * F_x]

    parameters = base_params.copy()
    parameters.update({
        'J1': J1_val,
        'J3': J3_x,
        'J1z': J1z_x,
        'J3z': J3z_x,
        'D': D_x,
        'F': F_x,
        'G': G_x
    })

    data_t_lower = np.zeros(82)
    data_t_upper = np.zeros(17)

    for ii in range(82):
        kvec = data_all_lower[:2, ii]
        data_t_lower[ii] = np.real(ham.hamiltonian_diagonalization(kvec, 'b', parameters, B)[1])

    for ii in range(17):
        kvec = data_all_upper[:2, ii]
        data_t_upper[ii] = np.real(ham.hamiltonian_diagonalization(kvec, 'b', parameters, B)[0])

    # R²-like scores
    diff_lower = data_t_lower - data_all_lower[2]
    diff_upper = data_t_upper - data_all_upper[2]
    var_lower = data_all_lower[2] - np.mean(data_all_lower[2])
    var_upper = data_all_upper[2] - np.mean(data_all_upper[2])

    ss_res_lower = np.sum(diff_lower ** 2)
    ss_res_upper = np.sum(diff_upper ** 2)
    ss_tot_lower = np.sum(var_lower ** 2)
    ss_tot_upper = np.sum(var_upper ** 2)

    err0 = 1 - (ss_res_lower + ss_res_upper) / (ss_tot_lower + ss_tot_upper)
    err1 = 1 - ss_res_lower / ss_tot_lower

    return (k1, i1, j1, kitaev_vals, err0, err1)

results = Parallel(n_jobs=-1, verbose=5)(
    delayed(compute_kitaev_and_error)(
        i1, j1, k1, J1_val, 
        c_M1_a[i1], c_M1_b[j1],
        b_M1_a[i1], b_M1_b[j1]
    )
    for i1 in range(Np)
    for j1 in range(Np)
    for k1, J1_val in enumerate(np.linspace(J1_ends[i1, j1, 0], J1_ends[i1, j1, 1], Np3))
)


for k1, i1, j1, kitaev_vals, err0, err1 in results:
    kitaev_all[k1, i1, j1, 0] = kitaev_vals[0]
    kitaev_all[k1, i1, j1, 1] = kitaev_vals[1]
    err_all[k1, i1, j1, 0] = err0
    err_all[k1, i1, j1, 1] = err1


data_all_p_p={}
data_all_p_p['kitaev']=kitaev_all
data_all_p_p['err_all']=err_all

np.save('data_all_p_n_E_'+str(Nid)+'.npy', data_all_p_p)


#####################



#####################
print('-ve_+ve')

err_all = np.zeros((Np3, Np, Np, 2))
kitaev_all = np.zeros((Np3, Np, Np, 2))


# Create reusable base for parameter dict
base_params = {
    'g_b': 5.0,
    'g_a': 5.0,
    'g_z': 2.7,
    'mu_B': 0.05788381,
    'S': 0.5,
    'J2': 0.0,
    'J2z': 0.0,
    'E': E_array[Nid],
    'delta_1': np.array([[0, 1], [-np.sqrt(3)/2, -1/2], [np.sqrt(3)/2, -1/2]]),
    'delta_2': np.array([[np.sqrt(3), 0], [np.sqrt(3)/2, 1.5], [-np.sqrt(3)/2, 1.5],
                         [-np.sqrt(3), 0], [-np.sqrt(3)/2, -1.5], [np.sqrt(3)/2, -1.5]]),
    'delta_3': np.array([[0, -2], [np.sqrt(3), 1], [-np.sqrt(3), 1]]),
    'phi': np.array([0, 2*np.pi/3, -2*np.pi/3])
}

# Preallocate reusable arrays
data_t_lower = np.zeros((3, 82))
data_t_upper = np.zeros((3, 17))


def compute_kitaev_and_error(i1, j1, k1, J1_val, c_a, c_b, b_a, b_b):
    J3_x = J3(J1_val, S, b_g)
    J1z_x = J1z(J1_val, S, a_g, b_g, c_b, c_a)
    J3z_x = J3z(J1_val, S, a_g, b_g, c_b, c_a)
    D_x = D(S, c_a, c_b)
    F_x =-np.sqrt(F2(J1_val, S, b_g, c_a, c_b, b_a))
    G_x = np.sqrt(G2(J1_val, S, b_g, c_a, c_b, b_b))

    kitaev_vals = [D_x - np.sqrt(2) * F_x, D_x + np.sqrt(2) * F_x]

    parameters = base_params.copy()
    parameters.update({
        'J1': J1_val,
        'J3': J3_x,
        'J1z': J1z_x,
        'J3z': J3z_x,
        'D': D_x,
        'F': F_x,
        'G': G_x
    })

    data_t_lower = np.zeros(82)
    data_t_upper = np.zeros(17)

    for ii in range(82):
        kvec = data_all_lower[:2, ii]
        data_t_lower[ii] = np.real(ham.hamiltonian_diagonalization(kvec, 'b', parameters, B)[1])

    for ii in range(17):
        kvec = data_all_upper[:2, ii]
        data_t_upper[ii] = np.real(ham.hamiltonian_diagonalization(kvec, 'b', parameters, B)[0])

    # R²-like scores
    diff_lower = data_t_lower - data_all_lower[2]
    diff_upper = data_t_upper - data_all_upper[2]
    var_lower = data_all_lower[2] - np.mean(data_all_lower[2])
    var_upper = data_all_upper[2] - np.mean(data_all_upper[2])

    ss_res_lower = np.sum(diff_lower ** 2)
    ss_res_upper = np.sum(diff_upper ** 2)
    ss_tot_lower = np.sum(var_lower ** 2)
    ss_tot_upper = np.sum(var_upper ** 2)

    err0 = 1 - (ss_res_lower + ss_res_upper) / (ss_tot_lower + ss_tot_upper)
    err1 = 1 - ss_res_lower / ss_tot_lower

    return (k1, i1, j1, kitaev_vals, err0, err1)

results = Parallel(n_jobs=-1, verbose=5)(
    delayed(compute_kitaev_and_error)(
        i1, j1, k1, J1_val, 
        c_M1_a[i1], c_M1_b[j1],
        b_M1_a[i1], b_M1_b[j1]
    )
    for i1 in range(Np)
    for j1 in range(Np)
    for k1, J1_val in enumerate(np.linspace(J1_ends[i1, j1, 0], J1_ends[i1, j1, 1], Np3))
)


for k1, i1, j1, kitaev_vals, err0, err1 in results:
    kitaev_all[k1, i1, j1, 0] = kitaev_vals[0]
    kitaev_all[k1, i1, j1, 1] = kitaev_vals[1]
    err_all[k1, i1, j1, 0] = err0
    err_all[k1, i1, j1, 1] = err1


data_all_p_p={}
data_all_p_p['kitaev']=kitaev_all
data_all_p_p['err_all']=err_all

np.save('data_all_n_p_E_'+str(Nid)+'.npy', data_all_p_p)


#####################


#####################
print('-ve_-ve')

err_all = np.zeros((Np3, Np, Np, 2))
kitaev_all = np.zeros((Np3, Np, Np, 2))


# Create reusable base for parameter dict
base_params = {
    'g_b': 5.0,
    'g_a': 5.0,
    'g_z': 2.7,
    'mu_B': 0.05788381,
    'S': 0.5,
    'J2': 0.0,
    'J2z': 0.0,
    'E': E_array[Nid],
    'delta_1': np.array([[0, 1], [-np.sqrt(3)/2, -1/2], [np.sqrt(3)/2, -1/2]]),
    'delta_2': np.array([[np.sqrt(3), 0], [np.sqrt(3)/2, 1.5], [-np.sqrt(3)/2, 1.5],
                         [-np.sqrt(3), 0], [-np.sqrt(3)/2, -1.5], [np.sqrt(3)/2, -1.5]]),
    'delta_3': np.array([[0, -2], [np.sqrt(3), 1], [-np.sqrt(3), 1]]),
    'phi': np.array([0, 2*np.pi/3, -2*np.pi/3])
}

# Preallocate reusable arrays
data_t_lower = np.zeros((3, 82))
data_t_upper = np.zeros((3, 17))


def compute_kitaev_and_error(i1, j1, k1, J1_val, c_a, c_b, b_a, b_b):
    J3_x = J3(J1_val, S, b_g)
    J1z_x = J1z(J1_val, S, a_g, b_g, c_b, c_a)
    J3z_x = J3z(J1_val, S, a_g, b_g, c_b, c_a)
    D_x = D(S, c_a, c_b)
    F_x =-np.sqrt(F2(J1_val, S, b_g, c_a, c_b, b_a))
    G_x =-np.sqrt(G2(J1_val, S, b_g, c_a, c_b, b_b))

    kitaev_vals = [D_x - np.sqrt(2) * F_x, D_x + np.sqrt(2) * F_x]

    parameters = base_params.copy()
    parameters.update({
        'J1': J1_val,
        'J3': J3_x,
        'J1z': J1z_x,
        'J3z': J3z_x,
        'D': D_x,
        'F': F_x,
        'G': G_x
    })

    data_t_lower = np.zeros(82)
    data_t_upper = np.zeros(17)

    for ii in range(82):
        kvec = data_all_lower[:2, ii]
        data_t_lower[ii] = np.real(ham.hamiltonian_diagonalization(kvec, 'b', parameters, B)[1])

    for ii in range(17):
        kvec = data_all_upper[:2, ii]
        data_t_upper[ii] = np.real(ham.hamiltonian_diagonalization(kvec, 'b', parameters, B)[0])

    # R²-like scores
    diff_lower = data_t_lower - data_all_lower[2]
    diff_upper = data_t_upper - data_all_upper[2]
    var_lower = data_all_lower[2] - np.mean(data_all_lower[2])
    var_upper = data_all_upper[2] - np.mean(data_all_upper[2])

    ss_res_lower = np.sum(diff_lower ** 2)
    ss_res_upper = np.sum(diff_upper ** 2)
    ss_tot_lower = np.sum(var_lower ** 2)
    ss_tot_upper = np.sum(var_upper ** 2)

    err0 = 1 - (ss_res_lower + ss_res_upper) / (ss_tot_lower + ss_tot_upper)
    err1 = 1 - ss_res_lower / ss_tot_lower

    return (k1, i1, j1, kitaev_vals, err0, err1)

results = Parallel(n_jobs=-1, verbose=5)(
    delayed(compute_kitaev_and_error)(
        i1, j1, k1, J1_val, 
        c_M1_a[i1], c_M1_b[j1],
        b_M1_a[i1], b_M1_b[j1]
    )
    for i1 in range(Np)
    for j1 in range(Np)
    for k1, J1_val in enumerate(np.linspace(J1_ends[i1, j1, 0], J1_ends[i1, j1, 1], Np3))
)


for k1, i1, j1, kitaev_vals, err0, err1 in results:
    kitaev_all[k1, i1, j1, 0] = kitaev_vals[0]
    kitaev_all[k1, i1, j1, 1] = kitaev_vals[1]
    err_all[k1, i1, j1, 0] = err0
    err_all[k1, i1, j1, 1] = err1


data_all_p_p={}
data_all_p_p['kitaev']=kitaev_all
data_all_p_p['err_all']=err_all

np.save('data_all_n_n_E_'+str(Nid)+'.npy', data_all_p_p)

#####################
