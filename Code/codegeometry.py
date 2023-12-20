#Script by Mariem Abaach
#Universités Paris Cité et Côte D'Azur
##

import numpy as np
import cv2


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

    persistence_diagram = []
    barcode = {}
    resume_geometrique = []
    for idx in per.keys(): #premier non zeros = date de naissance, taille de L - premiers non zeros 
        if len(L) - next((i for i, x in enumerate(per[idx][::-1]) if x), None) < len(L):
            #persistence_diagram[idx] = (L[next((i for i, x in enumerate(per[idx]) if x), None)], L[len(L) - next((i for i, x in enumerate(per[idx][::-1]) if x), None)])
            persistence_diagram.append((0, (L[next((i for i, x in enumerate(per[idx]) if x), None)], L[len(L) - next((i for i, x in enumerate(per[idx][::-1]) if x), None)])))
            barcode[idx] = (next((i for i, x in enumerate(per[idx]) if x), None)+1, len(L) - next((i for i, x in enumerate(per[idx][::-1]) if x), None)+1)
#             persistence[idx] = np.abs(next((i for i, x in enumerate(per[idx]) if x), None) - (len(L) - next((i for i, x in enumerate(per[idx][::-1]) if x), None)))
        else:
            persistence_diagram.append((0, (L[next((i for i, x in enumerate(per[idx]) if x), None)], np.inf)))
            barcode[idx] = (next((i for i, x in enumerate(per[idx]) if x), None)+1, len(L) - next((i for i, x in enumerate(per[idx][::-1]) if x), None)+1)
#             persistence[idx] = len(L)
            
        resume_geometrique.append((sum(area[idx]), sum(per[idx])))

    return resume_geometrique, persistence_diagram
