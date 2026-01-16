import numpy as np
from numpy import linalg as LA

def hamiltonian_general(q,which_direction,parameters,B):
    A_q=Aq(q,which_direction,parameters,B)
    A_neg_q=Aq(-q,which_direction,parameters,B)
    B_q=Bq(q,which_direction,parameters,B)
    B_neg_q=Bq(-q,which_direction,parameters,B)
    C_q=Cq(q,which_direction,parameters,B)
    C_neg_q=Cq(-q,which_direction,parameters,B)
    F_q=Fq(q,which_direction,parameters,B)

    #print(Aq,Bq,Cq,C_neg_q,F_q)
    hamiltonian=np.zeros((4,4),dtype='complex')
    hamiltonian[0,0]=A_q
    hamiltonian[1,1]=A_q
    hamiltonian[2,2]=A_neg_q
    hamiltonian[3,3]=A_neg_q
    hamiltonian[0,1]=B_q
    hamiltonian[2,3]=np.conjugate(B_neg_q)
    hamiltonian[1,0]=np.conjugate(B_q)
    hamiltonian[3,2]=B_neg_q
    hamiltonian[0,2]=F_q
    hamiltonian[1,3]=F_q
    hamiltonian[2,0]=F_q
    hamiltonian[3,1]=F_q
    hamiltonian[0,3]=C_q
    hamiltonian[3,0]=np.conjugate(C_q)
    hamiltonian[1,2]=C_neg_q
    hamiltonian[2,1]=np.conjugate(C_neg_q)
    return hamiltonian

def hamiltonian_diagonalization(q,which_direction,parameters,B):
    hamiltonian=2*hamiltonian_general(q,which_direction,parameters,B)
    metric=np.zeros((4,4))
    metric[0,0]=1
    metric[1,1]=1
    metric[2,2]=-1
    metric[3,3]=-1
    g_h=np.matmul(metric,hamiltonian)
    eigenvalues, _=LA.eig(g_h)
    eigenvalues=np.sort(eigenvalues)[::-1]
    return eigenvalues


def hamiltonian_cholesky(q,which_direction,parameters,B):
    A=2*hamiltonian_general(q,which_direction,parameters,B)
    B=LA.cholesky(A, upper=True)
    It=np.zeros((4,4),dtype=complex)
    for i in range(2):
        It[i][i]=1.0
    for i in range(2,4):
        It[i][i]=-1.0
    Bt=np.transpose(np.conjugate(B))
    Temp=np.matmul(B,It)
    C=np.matmul(Temp,Bt)
    w, v = LA.eig(C)
    ws = np.sort(w)
    #print(np.round(ws,3),"**")
    vs = v[:, w.argsort()]
    #print(np.round(vs,3),"***")
    #print (ws.real)
    U=np.zeros((4,4),dtype=complex)
    for i in range(int(4)):
        for j in range(int(2)):
            U[i][j]=vs[i][4-1-j]
    for i in range(int(4)):
        for j in range(int(2),int(4)):
            U[i][j]=vs[i][j-2]
    
    #print(U,"****")
    En=np.zeros((4,4),dtype=complex)
    K3=np.zeros((4,4),dtype=complex)
    K3=LA.inv(B)  
    #print(ws)
    for i in range(0,int(2)):
        t=ws[4-1-i].real
        En[i][i]=np.sqrt(t)
        En[i+2][i+2]=np.sqrt(t)
    Temp2=np.matmul(K3,U)
    InvT=np.matmul(Temp2,En)
    #print(En,'*')

    return InvT

def gen_BZ(G,N):  #Generate BZ, okay
    N1=N[0]
    N2=N[1]
    #if(N1!=N2):
    #    print("Error")
    G1=G[0]
    G2=G[1]
    K=np.zeros((2,N1,N2))
    #for i,j in it.product(range(N1),range(N2)):
    for i in range(N1):
        for j in range(N2):
            K[0,i,j]=G1[0]/N1*i+G2[0]/N2*j#+G1[0]/(2*N1)+G2[0]/(2*N2) #centered grid
            K[1,i,j]=G1[1]/N1*i+G2[1]/N2*j#+G1[1]/(2*N1)+G2[1]/(2*N2) 
    return K

def reduced_moment(which_direction,parameters,B,G,grid_N):
    S=parameters['S']
    moment=S
    Nx,Ny=grid_N[0],grid_N[1]
    K_grid=gen_BZ(G,grid_N)
    for i1 in range(Nx):
        for j1 in range(Ny):
            U=LA.inv(hamiltonian_cholesky(K_grid[:,i1,j1],which_direction,parameters,B))
            U_dag=np.conjugate(np.transpose(U))
            for m in range(2,4):
                moment-=1/(2*Nx*Ny)*(U_dag[m,0]*U[0,m]+U_dag[m,1]*U[1,m])
    return moment

def Aq(q,which_direction,parameters,B):
    g_b=parameters['g_b']
    g_a=parameters['g_a']
    mu_B=parameters['mu_B']
    S=parameters['S']
    J1=parameters['J1']
    J2=parameters['J2']
    J3=parameters['J3']
    J2z=parameters['J2z']

    g2=0.5*(gamma2(q,parameters)+np.conjugate(gamma2(q,parameters)))

    if(which_direction=='b'):
        return g_b*mu_B*B/2-3*S/2*(J1+J3)+3*S/2*((J2+J2z)*g2-2*J2)
    elif(which_direction=='a'):
        return g_a*mu_B*B/2-3*S/2*(J1+J3)+3*S/2*((J2+J2z)*g2-2*J2)
    

def Bq(q,which_direction,parameters,B):
    S=parameters['S']
    J1=parameters['J1']
    J3=parameters['J3']
    J1z=parameters['J1z']
    J3z=parameters['J3z']
    D=parameters['D']
    E=parameters['E']
    g1=gamma(q,parameters)
    g3=gamma3(q,parameters)
    gp2=gamma_prime2(q,parameters)
    gpp2=gamma_prime_prime2(q,parameters)

    if(which_direction=='b'):
        return 3*S/4*((J1+J1z)*g1+(J3+J3z)*g3-D*gp2+E*gpp2)
    elif(which_direction=='a'):
        return 3*S/4*((J1+J1z)*g1+(J3+J3z)*g3+D*gp2-E*gpp2)
    

def Cq(q,which_direction,parameters,B):
    S=parameters['S']
    J1=parameters['J1']
    J3=parameters['J3']
    J1z=parameters['J1z']
    J3z=parameters['J3z']
    D=parameters['D']
    F=parameters['F']
    E=parameters['E']
    G=parameters['G']
    g1=gamma(q,parameters)
    g3=gamma3(q,parameters)
    gp=gamma_prime(q,parameters)
    gp2=gamma_prime2(q,parameters)
    gpp=gamma_prime_prime(q,parameters)
    gpp2=gamma_prime_prime2(q,parameters)
    if(which_direction=='b'):
        return 3*S/4*((J1z-J1)*g1+(J3z-J3)*g3+D*gp2-2.0j*F*gpp-E*gpp2-2.0j*G*gp)
    elif(which_direction=='a'):
        return 3*S/4*((J1-J1z)*g1+(J3-J3z)*g3+D*gp2-2.0j*F*gp-E*gpp2+2.0j*G*gpp)
    

def Fq(q,which_direction,parameters,B):
    S=parameters['S']
    J2=parameters['J2']
    J2z=parameters['J2z']
    g2=gamma2(q,parameters)
    if(which_direction=='b'):
        return 3*S/2*(J2z-J2)*g2
    elif(which_direction=='a'):
        return 3*S/2*(J2-J2z)*g2
    

def gamma(q,parameters):
    delta_1=parameters['delta_1']
    g=0.0+0.0j
    for i in range(3):
        g+=(1/3)*np.exp(1.0j*np.dot(q,delta_1[i]))
    return g


def gamma3(q,parameters):
    delta_3=parameters['delta_3']
    g=0.0+0.0j
    for i in range(3):
        g+=(1/3)*np.exp(1.0j*np.dot(q,delta_3[i]))
    return g


def gamma2(q,parameters):
    delta_2=parameters['delta_2']
    g=0.0+0.0j
    for i in range(6):
        g+=(1/6)*np.exp(1.0j*np.dot(q,delta_2[i]))
    return g

def gamma_prime(q,parameters):
    delta_1=parameters['delta_1']
    phi=parameters['phi']
    g=0.0+0.0j
    for i in range(3):
        g+=(1/3)*np.exp(1.0j*np.dot(q,delta_1[i]))*np.cos(phi[i]) #Not changed yet
    return g

def gamma_prime2(q,parameters):
    delta_1=parameters['delta_1']
    phi=parameters['phi']
    g=0.0+0.0j
    for i in range(3):
        g+=(1/3)*np.exp(1.0j*np.dot(q,delta_1[i]))*np.cos(2*phi[i]) #Not changed yet
    return g

def gamma_prime_prime(q,parameters):
    delta_1=parameters['delta_1']
    phi=parameters['phi']
    g=0.0+0.0j
    for i in range(3):
        g+=(1/3)*np.exp(1.0j*np.dot(q,delta_1[i]))*np.sin(phi[i])
    return g

def gamma_prime_prime2(q,parameters):
    delta_1=parameters['delta_1']
    phi=parameters['phi']
    g=0.0+0.0j
    for i in range(3):
        g+=(1/3)*np.exp(1.0j*np.dot(q,delta_1[i]))*np.sin(2*phi[i])
    return g




