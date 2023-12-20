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
from function_geometry import * 
from PIL import Image


def preprocessing_data(path):
    U = []
    for i in os.listdir(path): 
        img = path+i
        M = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        U.append((-1)*M - np.min((-1)*M)) #pour avoir les bonnes valeurs en haut 
    return U