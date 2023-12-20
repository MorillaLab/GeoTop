# feature_utils.py

# Script by Mariem Abaach et Ian Morilla
# Universités Paris Cité, Côte D'Azur et Sorbonne Paris Nord
##

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocessing_data(path, T):
    U = []
    w = 0
    for i in os.listdir(path)[:T]:
        img = os.path.join(path, i)
        M = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        U.append((-1) * M - np.min((-1) * M))
        w += 1
    return U

def preprocessing_data_rgb_na(path, T):
    U = []
    w = 0
    for i in os.listdir(path)[:T]:
        img = os.path.join(path, i)
        M = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        Z = np.zeros(M.shape)
        Z[:, :, 0] = (-1) * M[:, :, 0] - np.min((-1) * M[:, :, 0])
        Z[:, :, 1] = (-1) * M[:, :, 1] - np.min((-1) * M[:, :, 1])
        Z[:, :, 2] = (-1) * M[:, :, 2] - np.min((-1) * M[:, :, 2])
        U.append(Z)
        w += 1
    return U
 
def preprocessing_data_rgb(path, T, augmentation=True):
    U = []
    w = 0
    
    # Define data augmentation parameters
    if augmentation:
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=11.45916,  # Random rotation
            width_shift_range=0.2,  # Random horizontal shift
            height_shift_range=0.2,  # Random vertical shift
            shear_range=0.2,  # Random shear
            zoom_range=0.2,  # Random zoom
            horizontal_flip=True,  # Random horizontal flip
            vertical_flip=True  # Random vertical flip
        )
    
    for i in os.listdir(path)[:T]:
        img = os.path.join(path, i)
        M = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        
        # Apply data augmentation
        if augmentation:
            M = datagen.random_transform(M)
        
        Z = np.zeros(M.shape)
        Z[:, :, 0] = (-1) * M[:, :, 0] - np.min((-1) * M[:, :, 0])
        Z[:, :, 1] = (-1) * M[:, :, 1] - np.min((-1) * M[:, :, 1])
        Z[:, :, 2] = (-1) * M[:, :, 2] - np.min((-1) * M[:, :, 2])
        U.append(Z)
        w += 1
    return U

#source: https://datacarpentry.org/image-processing/aio/index.html
# Only keep high-intensity pixels in an image.
def preprocess_image(img_path, target_path):
    threshold = 128
    image = load_img(img_path)
    r = np.copy(image[:, :, 0])
    g = np.copy(image[:, :, 1])
    b = np.copy(image[:, :, 2])
    r[r < threshold] = 0
    g[g < threshold] = 0
    b[b < threshold] = 0
    grid = np.array([r, g, b])
    grid = np.moveaxis(grid, 0, 2)
    image = np.where((grid == [0, 0, 0]).all(axis=-1)[..., None], [0, 0, 0], image)
    cv2.imwrite(target_path, image)
    

def Perimetre_Hermine(X,t):
    [M,N] = np.shape(X)
    L_1 = np.sum((X[0:M-1,0:N]>t)*(X[1:M,0:N]<=t))
    L_2 = np.sum((X[0:M,0:N-1]>t)*(X[0:M,1:N]<=t))
    L_3 = np.sum((X[0:M-1,0:N]<=t)*(X[1:M,0:N]>t))
    L_4 = np.sum((X[0:M,0:N-1]<=t)*(X[0:M,1:N]>t))
    L = L_1 + L_2 + L_3 + L_4
    return L

def Area(X, t):
    return np.sum(X >= t)

def Euler_2(image_binary):
    M, N = image_binary.shape
    u = np.zeros((M + 2, N + 2))
    u[1:M + 1, 1:N + 1] = image_binary
    M, N = u.shape
    u_c = np.append(u[1:M, 1:N], np.zeros((M - 1, 1)), axis=1)
    u_cc = np.vstack([u_c, np.zeros((1, u_c.shape[1]))])
    # Matrice S d'Ebner:
    mtri = u + np.vstack([u[1:M, 0:N], np.zeros((1, N))]) + np.append(u[0:M, 1:N], np.zeros((M, 1)), axis=1) + u_cc
    # Configuration en crois:
    ind = np.where(mtri == 2)
    # Cas ou S(i,j) = 0
    list_l = ind[0]
    list_c = ind[1]
    c_z = list_c[np.where(u[ind] == 0)[0]]
    l_z = list_l[np.where(u[ind] == 0)[0]]
    # pour le compter il faut que S(i+1,j+1) soit aussi égale à 0
    S_0 = np.sum(u[l_z + 1, c_z + 1] == 0)
    # Cas ou S(i,j) = 1
    l_o = list_l[np.where(u[ind] == 1)[0]]
    c_o = list_c[np.where(u[ind] == 1)[0]]
    # pour le compter il faut que S(i+1,j+1) = 1 soit aussi égale à 1
    S_1 = np.sum(u[l_o + 1, c_o + 1] == 1)
    # Calcul de la caractéristique d'Euler
    E = (1 / 4) * np.sum(mtri == 1) - (1 / 4) * np.sum(mtri == 3) - (0.5 * S_0 + 0.5 * S_1)
    return E

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


def calculate_feature(args):
    i, U_rgb, U, XX, n = args
    UU = U_rgb[i]
    U1 = UU[:, :, 0]
    U2 = UU[:, :, 1]
    U3 = UU[:, :, 2]
    U_gray = U[i]
    L1 = np.linspace(np.min(U1), np.max(U1), n)
    L2 = np.linspace(np.min(U2), np.max(U2), n)
    L3 = np.linspace(np.min(U3), np.max(U3), n)
    L_g = np.linspace(np.min(U_gray), np.max(U_gray), n)
    for k in range(n):
        XX[i, k] = np.sum(U1 >= L1[k])
        XX[i, k + n] = np.sum(U2 >= L2[k])
        XX[i, k + 2 * n] = np.sum(U3 >= L3[k])
        XX[i, k + 3 * n] = np.sum(U_gray >= L_g[k])
        XX[i, k + 4 * n] = Perimetre_Hermine(U1, L1[k])
        XX[i, k + 5 * n] = Perimetre_Hermine(U2, L2[k])
        XX[i, k + 6 * n] = Perimetre_Hermine(U3, L3[k])
        XX[i, k + 7 * n] = Perimetre_Hermine(U_gray, L_g[k])
        XX[i, k + 8 * n] = Euler(U1 >= L1[k])
        XX[i, k + 9 * n] = Euler(U2 >= L2[k])
        XX[i, k + 10 * n] = Euler(U3 >= L3[k])
        XX[i, k + 11 * n] = Euler(U_gray >= L_g[k])
