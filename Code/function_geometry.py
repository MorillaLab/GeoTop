# +
import numpy as np 
import matplotlib.pyplot as plt 
import os 
from scipy.stats import norm
import scipy.stats as sps
import cv2
from scipy.ndimage import convolve
import skimage
from scipy import ndimage
from scipy.spatial.distance import cdist
from nibabel.testing import data_path
import nibabel as nib
from sklearn.mixture import GaussianMixture
import os
import pandas as pd
import skimage.measure
import seaborn as sns
import scipy as sp
import cv2
import gudhi as gd
import random

from scipy import ndimage


# -

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

def minmax(U, U_mask, labels_infile):
    minn = []
    maxx = []
    for i in np.arange(len(U)):
        UU = U[i]
        real_idx = np.where(labels_infile == i)[0][0]
        UU_mask = U_mask[real_idx]
        M,NN,l = np.shape(UU)
        proportion = np.zeros(l)
        for q  in np.arange(l):
            proportion[q] = np.sum(UU_mask[:,:,q] > 0)/M*NN
        j = np.argmax(proportion)
        UU_mask[:,:,j][UU_mask[:,:,j] > 0] = 1 
        Z =  UU_mask[:,:,j]*UU[:,:,j]
        area = np.sum(UU_mask[:,:,j])
        zeros_cord = np.where(UU_mask[:,:,j] == 0)
        min_n = np.min(Z)
        Z = Z - np.min(Z)
        Z[zeros_cord] = 0
        minn.append(min(Z[Z > 0]))
        maxx.append(max(Z[Z > 0]))
    return max(minn), min(maxx)

def function_EAP(UU_mask, UU): #On doit centrer l'image 
    N = 100
    M, NN,l = np.shape(UU)
    proportion = np.zeros(l)
    for q  in range(l):
        proportion[q] = np.sum(UU_mask[:,:,q] > 0)/M*NN
    j = np.argmax(proportion)
    CG = np.zeros((N, 3))
    UU_mask[:,:,j][UU_mask[:,:,j] > 0] = 1 
    Z =  UU_mask[:,:,j]*UU[:,:,j]
    Z = (Z-np.mean(Z))/np.std(Z)
    area = np.sum(UU_mask[:,:,j])
    zeros_cord = np.where(UU_mask[:,:,j] == 0)
    min_n = np.min(Z)
    Z = Z - np.min(Z)
    Z[zeros_cord] = 0
    quantile = len(zeros_cord[0])/(512*512) #proportion de 0 dans l'image
    T = []
    Q = [k for k in np.linspace(0,1,N)]
    T = np.array([np.quantile(Z[Z > 0], k) for k in Q])
    T= np.linspace(min(T), max(T), N)
    for i in range(N):
#         CG[i, 0] = Area(Z, T[i])/(area + 1*(area ==0))  
#         CG[i, 1] = Perimetre_Hermine(Z, T[i])/(area + 1*(area ==0)) + CG[i, 0]*(4/np.sqrt((area + 1*(area ==0))))
#         CG[i, 2] = Euler(Z>=T[i])/(area + 1*(area ==0)) + (4/((area + 1*(area ==0))*np.pi))*CG[i, 1] + CG[i, 0]*(1/np.sqrt((area + 1*(area ==0))))
#         Thresh[i] = T[i]
        CG[i, 0] = Area(Z, T[i])
        CG[i, 1] = Perimetre_Hermine(Z, T[i])
        CG[i, 2] = Euler(Z>=T[i])
    return CG, T

def function_EAP1(UU_mask, UU, minn, maxx):
    N = 100
    M, NN,l = np.shape(UU)
    proportion = np.zeros(l)
    for q  in range(l):
        proportion[q] = np.sum(UU_mask[:,:,q] > 0)/M*NN
    j = np.argmax(proportion)
    CG = np.zeros((N, 3))
    UU_mask[:,:,j][UU_mask[:,:,j] > 0] = 1 
    Z =  UU_mask[:,:,j]*UU[:,:,j]
#     Z = (Z-np.mean(Z))/np.std(Z)
    area = np.sum(UU_mask[:,:,j])
    zeros_cord = np.where(UU_mask[:,:,j] == 0)
    min_n = np.min(Z)
    Z = Z - np.min(Z)
    Z[zeros_cord] = 0
    T= np.linspace(minn, maxx, N)
    Thresh = np.zeros(N)
#     for i in range(N):
#         CG[i, 0] = Area(Z, T[i])
#         CG[i, 1] = Perimetre_Hermine(Z, T[i])
#         CG[i, 2] = Euler(Z>=T[i])
#         Thresh[i] = T[i]
    return CG, Thresh, T

def prep(U_mask, U, labels_infile, k):
    minn, maxx = minmax(U, U_mask, labels_infile)
    real_idx = np.where(labels_infile == k)[0][0]
    UU_mask = U_mask[real_idx]
    UU = U[k]
    N = 100
    M, NN,l = np.shape(UU)
    proportion = np.zeros(l)
    for q  in range(l):
        proportion[q] = np.sum(UU_mask[:,:,q] > 0)/M*NN
    j = np.argmax(proportion)
    CG = np.zeros((N, 3))
    UU_mask[:,:,j][UU_mask[:,:,j] > 0] = 1 
    Z =  UU_mask[:,:,j]*UU[:,:,j]
    # Z = (Z-np.mean(Z))/np.std(Z)
    area = np.sum(UU_mask[:,:,j])
    zeros_cord = np.where(UU_mask[:,:,j] == 0)
    min_n = np.min(Z)
    Z = Z - np.min(Z)
    Z[zeros_cord] = 0
    L = np.linspace(minn, maxx, N)[::-1]
    plt.imshow(Z)
    plt.show()
    return Z, L, j

def function_persistance(field, L, plot):
    """field (numpy array) : image of size (mxm) 
       L  (numpy array)  : thresholds from maximum to minimum
       plot (Boolean)  : if it is equal to True: will plot the level sets """
    per = {}
    area = {}
    euler = {}
    connected_comp = {}
    gem1 = {}
    connected_comp = np.zeros(len(L))
    persistence = {}
    Per_total = np.zeros(len(L))
    Area_total = np.zeros(len(L))
    euler_total = np.zeros(len(L))
    # L[0] = maximum of field. First time seeing connected components.
    U0 = np.uint8(field >=L[0])
    Per_total[0] = Perimetre_Hermine(U0, 0.5)
    Area_total[0] = Area(U0, 0.5)
    euler_total[0] = Euler(U0)
    numb_connect, labeled_comp = cv2.connectedComponents(U0, connectivity=8) #labeling the connected components 
    list_of_labels = np.delete(np.unique(labeled_comp), np.where(np.unique(labeled_comp) ==0))
    connected_comp[0] = len(list_of_labels)

    #For the first excursion set we see fot t = maximum of field.
    # for each label of each connected component we initialize an entry in gem1, per, area and the euler characteristic. 
    for i in list_of_labels:
        gem1[i] = (np.where(labeled_comp == i)[0][0], np.where(labeled_comp == i)[1][0])
        matrix_label = np.zeros(labeled_comp.shape)
        matrix_label [labeled_comp == i] = 1
        per[gem1[i]] = [Perimetre_Hermine(matrix_label , 0.5)]
        area[gem1[i]] = [Area(matrix_label , 0.5)]
        euler[gem1[i]] = [Euler(matrix_label)]
    wp= 1
    death = []
    death_idx = []
    keys_gem1 = list(gem1.keys())
    key_dead = {}
    link = {}
    #then we 
    for t in L[1:]:
        U1 = np.uint8(field >= t)
        Per_total[wp] = Perimetre_Hermine(U1, 0.5)
        Area_total[wp] = Area(U1, 0.5)
        euler_total[wp] = Euler(U1)
        numb_connect, labeled_comp = cv2.connectedComponents(U1, connectivity=8)
        list_of_labels = np.delete(np.unique(labeled_comp), np.where(np.unique(labeled_comp) ==0))
        connected_comp[wp] = len(list_of_labels)
        wp = wp+1
        gem2 = {}
        # either its a new component or components merged
        # all new components in this new excursion set are stored in gem2
        for i in list_of_labels:
            gem2[i] = (np.where(labeled_comp == i)[0][0], np.where(labeled_comp == i)[1][0])
    
        # and now we compare with gem1 to see which one has merged 
        
        #check merge 
        for i, idx1 in zip(gem1.keys(), gem1.values()): 
            for f, idx2 in zip(gem1.keys(), gem1.values()):
                if labeled_comp[idx1] == labeled_comp[idx2] and not (idx1==idx2) and i < f:
                    key_idx = [idx1,idx2]
                    key_dic = [i,f]
                    list_zeros = [per[idx1].count(0), per[idx2].count(0)]
                    death.append(key_dic[np.argmax(list_zeros)]) #deaths old label
                    death_idx.append(key_idx[np.argmax(list_zeros)]) #deaths old index 
                    key_dead[key_dic[np.argmax(list_zeros)]] = key_idx[np.argmax(list_zeros)]
                    if min(list_zeros) < max(list_zeros):
                        if not key_idx[np.argmin(list_zeros)] in link.keys(): 
                            link[key_idx[np.argmin(list_zeros)]] = [key_idx[np.argmax(list_zeros)]]
                        else:
                            link[key_idx[np.argmin(list_zeros)]] = link[key_idx[np.argmin(list_zeros)]] + [key_idx[np.argmax(list_zeros)]]
                    else:  
                        list_idxx = key_idx.copy()
                        list_idxx.remove(key_idx[np.argmin(list_zeros)])
                        if key_idx[np.argmin(list_zeros)] in link.keys(): 
                            link[key_idx[np.argmin(list_zeros)]] = link[key_idx[np.argmin(list_zeros)]] + [list_idxx[0]]
                        else: # Faut checker que pas de doublons entre pere, fils grand pere.
                            link[key_idx[np.argmin(list_zeros)]] = [list_idxx[0]]

         
        # kill the merged componont it in per and area, cheking what happened to the old ones in the new matrix                                                          
        list_death = np.unique(death)
        for i in list_death:
            per[key_dead[i]] = per[key_dead[i]] + [0]
            area[key_dead[i]] = area[key_dead[i]] + [0]
            euler[key_dead[i]] = euler[key_dead[i]] + [0]
            if i in gem1.keys(): del gem1[i]   
                
        #compare gem2 and gem1 to look for new components.
        w = 0
        for j in gem2.values():
            for k in gem1.values():
                if labeled_comp[k] == labeled_comp[j]:
                    w = w+1
            if w == 0: #if new initialise it in gem1
                #it's a new element and should create a whole new thing for it.
                gem1[max(keys_gem1) +1] = j
                keys_gem1.append(max(keys_gem1) +1)
                kk = list(gem1.keys())[0]
                per[j] = [0]*len(per[gem1[kk]]) 
                area[j] = [0]*len(per[gem1[kk]]) 
                euler[j] = [0]*len(per[gem1[kk]])
            w = 0
        if plot:
            print("wp", wp, "level", t) 
            print("connected_components", len(list_of_labels))
            plt.imshow(labeled_comp/max(list_of_labels), cmap = "gray")
            plt.show()
            
        #now add the new value in the perimeter 
        for k in gem1.values():
            matrix_label = np.zeros(labeled_comp.shape)
            matrix_label[labeled_comp == labeled_comp[k]] = 1
            per[k] = per[k] + [Perimetre_Hermine(matrix_label , 0.5)]
            area[k] = area[k] + [Area(matrix_label , 0.5)]
            euler[k] = euler[k] + [Euler(matrix_label)]

    #At the end of the procedure getting the barecode and persistence
    life = {}
    barcode = {}
    for idx in per.keys(): #premier non zeros = date de naissance, taille de L - premiers non zeros 
        if len(L) - next((i for i, x in enumerate(per[idx][::-1]) if x), None) < len(L):
            life[idx] = (L[next((i for i, x in enumerate(per[idx]) if x), None)], L[len(L) - next((i for i, x in enumerate(per[idx][::-1]) if x), None)])
            barcode[idx] = (next((i for i, x in enumerate(per[idx]) if x), None)+1, len(L) - next((i for i, x in enumerate(per[idx][::-1]) if x), None)+1)
            persistence[idx] = np.abs(next((i for i, x in enumerate(per[idx]) if x), None) - (len(L) - next((i for i, x in enumerate(per[idx][::-1]) if x), None)))
        else:
            life[idx] = (L[next((i for i, x in enumerate(per[idx]) if x), None)], L[-1])
            barcode[idx] = (next((i for i, x in enumerate(per[idx]) if x), None)+1, len(L) - next((i for i, x in enumerate(per[idx][::-1]) if x), None)+1)
            persistence[idx] = len(L)
 
    return life, barcode, persistence, connected_comp, Per_total, Area_total, euler_total, per, area, euler 


# Quantile with the not unique method 
def function_persistance_quantile(field, T):
    per = {}
    area = {}
    connected_comp = {}
    gem1 = {}
    quantile = np.linspace(0,1, T)
    I = []
    for q in quantile:
        I.append(np.quantile(field.flatten(), q))
    L = I[::-1]
    connected_comp = np.zeros(len(L))
    persistence = {}
    Per_total = np.zeros(len(L))
    Area_total = np.zeros(len(L))
    euler_total = np.zeros(len(L))
    U0 = np.uint8(field >=L[0])
    Per_total[0] = Perimetre_Hermine(U0, 0.5)
    Area_total[0] = Area(U0, 0.5)
    euler_total[0] = Euler(U0)
    numb_connect, labeled_comp = cv2.connectedComponents(U0, connectivity=8)
    list_of_labels = np.delete(np.unique(labeled_comp), np.where(np.unique(labeled_comp) ==0))
    connected_comp[0] = len(list_of_labels)

    # First time creating new components

    for i in list_of_labels:
        gem1[i] = (np.where(labeled_comp == i)[0][0], np.where(labeled_comp == i)[1][0])
        matrix_label = np.zeros(labeled_comp.shape)
        matrix_label [labeled_comp == i] = 1
        per[gem1[i]] = [Perimetre_Hermine(matrix_label , 0.5)]
        area[gem1[i]] = [Area(matrix_label , 0.5)]

    wp= 1
    death = []
    death_idx = []
    link = {}
    keys_gem1 = list(gem1.keys())
    key_dead = {}
    for t in L[1:]:
        U1 = np.uint8(field >= t)
        Per_total[wp] = Perimetre_Hermine(U1, 0.5)
        Area_total[wp] = Area(U1, 0.5)
        euler_total[wp] = Euler(U1)
        numb_connect, labeled_comp = cv2.connectedComponents(U1, connectivity=8)
        list_of_labels = np.delete(np.unique(labeled_comp), np.where(np.unique(labeled_comp) ==0))
        connected_comp[wp] = len(list_of_labels)
        wp = wp+1
        gem2 = {}
        # either its a new component or components merged
        for i in list_of_labels:
            gem2[i] = (np.where(labeled_comp == i)[0][0], np.where(labeled_comp == i)[1][0])

        #check merge 
        for i, idx1 in zip(gem1.keys(), gem1.values()): 
            for f, idx2 in zip(gem1.keys(), gem1.values()):
                if labeled_comp[idx1] == labeled_comp[idx2] and not (idx1==idx2) and i < f:
                    key_idx = [idx1,idx2]
                    key_dic = [i,f]
                    list_zeros = [per[idx1].count(0), per[idx2].count(0)]
                    death.append(key_dic[np.argmax(list_zeros)]) #deaths old label
                    death_idx.append(key_idx[np.argmax(list_zeros)]) #deaths old index 
                    key_dead[key_dic[np.argmax(list_zeros)]] = key_idx[np.argmax(list_zeros)]
                    if min(list_zeros) < max(list_zeros):
                        if not key_idx[np.argmin(list_zeros)] in link.keys(): 
                            link[key_idx[np.argmin(list_zeros)]] = [key_idx[np.argmax(list_zeros)]]
                        else:
                            link[key_idx[np.argmin(list_zeros)]] = link[key_idx[np.argmin(list_zeros)]] + [key_idx[np.argmax(list_zeros)]]
                    else:  
                        list_idxx = key_idx.copy()
                        list_idxx.remove(key_idx[np.argmin(list_zeros)])
                        if key_idx[np.argmin(list_zeros)] in link.keys(): 
                            link[key_idx[np.argmin(list_zeros)]] = link[key_idx[np.argmin(list_zeros)]] + [list_idxx[0]]
                        else: # Faut checker que pas de doublons entre pere, fils grand pere.
                            link[key_idx[np.argmin(list_zeros)]] = [list_idxx[0]]

         # kill it in per and area, cheking what happened to the old ones in the new matrix                                                          
        list_death = np.unique(death)
        for i in list_death:
            per[key_dead[i]] = per[key_dead[i]] + [0]
            area[key_dead[i]] = area[key_dead[i]] + [0]
            if i in gem1.keys(): del gem1[i]       
        #new components.
        w = 0
        for j in gem2.values():
            for k in gem1.values():
                if labeled_comp[k] == labeled_comp[j]:
                    w = w+1
            if w == 0:
                #it's a new element and should create a whole new thing for it.
                gem1[max(keys_gem1) +1] = j
                keys_gem1.append(max(keys_gem1) +1)
                kk = list(gem1.keys())[0]
                per[j] = [0]*len(per[gem1[kk]]) 
                area[j] = [0]*len(per[gem1[kk]]) 
            w = 0
        print("wp", wp, "level", t) 
        print("connected_components", len(list_of_labels))
        plt.imshow(labeled_comp/max(list_of_labels), cmap = "gray")
        plt.show()
        #get the perimeter of the old ones for which no merge has happened and new

        for k in gem1.values():
            matrix_label = np.zeros(labeled_comp.shape)
            matrix_label[labeled_comp == labeled_comp[k]] = 1
            per[k] = per[k] + [Perimetre_Hermine(matrix_label , 0.5)]
            area[k] = area[k] + [Area(matrix_label , 0.5)]

    life = {}
    barcode = {}
    for idx in per.keys(): #premier non zeros = date de naissance, taille de L - premiers non zeros 
        if len(L) - next((i for i, x in enumerate(per[idx][::-1]) if x), None) < len(L):
            life[idx] = (L[next((i for i, x in enumerate(per[idx]) if x), None)], L[len(L) - next((i for i, x in enumerate(per[idx][::-1]) if x), None)])
            barcode[idx] = (next((i for i, x in enumerate(per[idx]) if x), None)+1, len(L) - next((i for i, x in enumerate(per[idx][::-1]) if x), None)+1)
            persistence[idx] = np.abs(next((i for i, x in enumerate(per[idx]) if x), None) - (len(L) - next((i for i, x in enumerate(per[idx][::-1]) if x), None)))
        else:
            life[idx] = (L[next((i for i, x in enumerate(per[idx]) if x), None)], L[-1])
            barcode[idx] = (next((i for i, x in enumerate(per[idx]) if x), None)+1, len(L) - next((i for i, x in enumerate(per[idx][::-1]) if x), None)+1)
            persistence[idx] = len(L)

    return life, barcode, persistence, connected_comp, Per_total, Area_total, euler_total, per, area


def resume_geometrique(field, L):
    per = {}
    area = {}
    connected_comp = {}
    gem1 = {}
    U0 = np.uint8(field >=L[0])
    numb_connect, labeled_comp = cv2.connectedComponents(U0, connectivity=8)
    list_of_labels = np.delete(np.unique(labeled_comp), np.where(np.unique(labeled_comp) ==0))

    # First time creating new components

    for i in list_of_labels:
        gem1[i] = (np.where(labeled_comp == i)[0][0], np.where(labeled_comp == i)[1][0])
        matrix_label = np.zeros(labeled_comp.shape)
        matrix_label [labeled_comp == i] = 1
        per[gem1[i]] = [Perimetre_Hermine(matrix_label , 0.5)]
        area[gem1[i]] = [Area(matrix_label , 0.5)]

    wp= 1
    death = []
    death_idx = []
    link = {}
    keys_gem1 = list(gem1.keys())
    key_dead = {}
    for t in L[1:]:
        U1 = np.uint8(field >= t)
        numb_connect, labeled_comp = cv2.connectedComponents(U1, connectivity=8)
        list_of_labels = np.delete(np.unique(labeled_comp), np.where(np.unique(labeled_comp) ==0))
        gem2 = {}
        # either its a new component or components merged
        for i in list_of_labels:
            gem2[i] = (np.where(labeled_comp == i)[0][0], np.where(labeled_comp == i)[1][0])

        #check merge 
        for i, idx1 in zip(gem1.keys(), gem1.values()): ## ennumarate ? i < f ??
            for f, idx2 in zip(gem1.keys(), gem1.values()): 
                if labeled_comp[idx1] == labeled_comp[idx2] and not (idx1==idx2) and i < f:
                    key_idx = [idx1,idx2]
                    key_dic = [i,f]
                    list_zeros = [per[idx1].count(0), per[idx2].count(0)]
                    if not (per[idx1].count(0) == per[idx2].count(0)):
                        death.append(key_dic[np.argmax(list_zeros)]) #deaths old label
                        death_idx.append(key_idx[np.argmax(list_zeros)]) #deaths old index 
                        key_dead[key_dic[np.argmax(list_zeros)]] = key_idx[np.argmax(list_zeros)]
                    else: 
                        list_per = [sum(per[idx1]), sum(per[idx2])]
                        death.append(key_dic[np.argmin(list_per)]) #deaths old label
                        death_idx.append(key_idx[np.argmin(list_per)]) #deaths old index 
                        key_dead[key_dic[np.argmin(list_per)]] = key_idx[np.argmin(list_per)]

         # kill it in per and area, cheking what happened to the old ones in the new matrix                                                          
        list_death = np.unique(death)
        for i in list_death:
            per[key_dead[i]] = per[key_dead[i]] + [0]
            area[key_dead[i]] = area[key_dead[i]] + [0]
            if i in gem1.keys(): del gem1[i]       
        #new components.
        w = 0
        for j in gem2.values():
            for k in gem1.values():
                if labeled_comp[k] == labeled_comp[j]:
                    w = w+1
            if w == 0:
                #it's a new element and should create a whole new thing for it.
                gem1[max(keys_gem1) +1] = j
                keys_gem1.append(max(keys_gem1) +1)
                kk = list(gem1.keys())[0]
                per[j] = [0]*len(per[gem1[kk]]) 
                area[j] = [0]*len(per[gem1[kk]]) 
            w = 0
        #get the perimeter of the old ones for which no merge has happened and new

        for k in gem1.values():
            matrix_label = np.zeros(labeled_comp.shape)
            matrix_label[labeled_comp == labeled_comp[k]] = 1
            per[k] = per[k] + [Perimetre_Hermine(matrix_label , 0.5)]
            area[k] = area[k] + [Area(matrix_label , 0.5)]

    persistence_diagram = {}
    barcode = {}
    resume_geometrique = []
    for idx in per.keys(): #premier non zeros = date de naissance, taille de L - premiers non zeros 
        if len(L) - next((i for i, x in enumerate(per[idx][::-1]) if x), None) < len(L):
            persistence_diagram[idx] = (L[next((i for i, x in enumerate(per[idx]) if x), None)], L[len(L) - next((i for i, x in enumerate(per[idx][::-1]) if x), None)])
            barcode[idx] = (next((i for i, x in enumerate(per[idx]) if x), None)+1, len(L) - next((i for i, x in enumerate(per[idx][::-1]) if x), None)+1)
#             persistence[idx] = np.abs(next((i for i, x in enumerate(per[idx]) if x), None) - (len(L) - next((i for i, x in enumerate(per[idx][::-1]) if x), None)))
        else:
            persistence_diagram[idx] = (L[next((i for i, x in enumerate(per[idx]) if x), None)], L[-1])
            barcode[idx] = (next((i for i, x in enumerate(per[idx]) if x), None)+1, len(L) - next((i for i, x in enumerate(per[idx][::-1]) if x), None)+1)
#             persistence[idx] = len(L)
            
        resume_geometrique.append((sum(area[idx]), sum(per[idx])))

    return resume_geometrique, persistence_diagram
