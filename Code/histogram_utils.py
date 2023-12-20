# histogram_utils.py

import numpy as np

def Perimetre_Hermine(X,t):
    [M,N] = np.shape(X)
    L_1 = np.sum((X[0:M-1,0:N]>t)*(X[1:M,0:N]<=t))
    L_2 = np.sum((X[0:M,0:N-1]>t)*(X[0:M,1:N]<=t))
    L_3 = np.sum((X[0:M-1,0:N]<=t)*(X[1:M,0:N]>t))
    L_4 = np.sum((X[0:M,0:N-1]<=t)*(X[0:M,1:N]>t))
    L = L_1 + L_2 + L_3 + L_4
    return L

def Area(X, t):
    return np.sum(X>=t)

def Euler(image_binary):
    M,N = np.shape(image_binary)
    u = np.zeros([M+2,M+2])
    u[1:M+1,1:M+1] = image_binary
    M,N = np.shape(u)
    u_c = np.append(u[1:M,1:M],np.zeros((M-1,1)), axis = 1)
    u_cc = np.vstack([u_c,np.zeros((1,np.shape(u_c)[1]))])
    # Matrice S d'Ebner :
    mtri = u + np.vstack([u[1:M,0:N],np.zeros((1,N))]) + np.append(u[0:M,1:N],np.zeros((M,1)), axis = 1)  + u_cc
    # Configuration en crois :
    ind = np.where(mtri == 2)
    # Cas ou S(i,j) = 0
    list_l = ind[0]
    list_c = ind[1]
    c_z = list_c[np.where(u[ind]==0)[0]]
    l_z = list_l[np.where(u[ind]==0)[0]]
    # pour le compter il faut que S(i+1,j+1) soit aussi égale à 0
    S_0 = sum(u[l_z + 1, c_z + 1]==0)
    # Cas ou S(i,j) = 1
    l_o = list_l[np.where(u[ind]==1)[0]]
    c_o = list_c[np.where(u[ind]==1)[0]]
    # pour le compter il faut que S(i+1,j+1) = 1 soit aussi égale à 1
    S_1 = sum(u[l_o+1,c_o + 1] == 1)
    # Calcul de la caractéristique d'Euler
    E = (1/4)*np.sum(mtri == 1) - (1/4)*np.sum(mtri == 3) - (0.5*S_0 + 0.5*S_1)
    return E


def calculate_feature(i, U_rgb, U, XX, n):
    UU = U_rgb[i]
    U1 = UU[:, :, 0]
    U2 = UU[:, :, 1]
    U3 = UU[:, :, 2]
    U_gray = U[i]
    L1 = np.linspace(np.min(U1), np.max(U1), n)
    L2 = np.linspace(np.min(U2), np.max(U2), n)
    L3 = np.linspace(np.min(U3), np.max(U3), n)
    L_g = np.linspace(np.min(U_gray), np.max(U_gray), n)
    
    temp_array = np.zeros((12 * n,))
    for k in range(n):
        temp_array[k] = np.sum(U1 >= L1[k])
        temp_array[k + n] = np.sum(U2 >= L2[k])
        temp_array[k + 2 * n] = np.sum(U3 >= L3[k])
        temp_array[k + 3 * n] = np.sum(U_gray >= L_g[k])
        temp_array[k + 4 * n] = Perimetre_Hermine(U1, L1[k])
        temp_array[k + 5 * n] = Perimetre_Hermine(U2, L2[k])
        temp_array[k + 6 * n] = Perimetre_Hermine(U3, L3[k])
        temp_array[k + 7 * n] = Perimetre_Hermine(U_gray, L_g[k])
        temp_array[k + 8 * n] = Euler(U1 >= L1[k])
        temp_array[k + 9 * n] = Euler(U2 >= L2[k])
        temp_array[k + 10 * n] = Euler(U3 >= L3[k])
        temp_array[k + 11 * n] = Euler(U_gray >= L_g[k])
    XX[i] = temp_array
