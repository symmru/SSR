import numpy as np 
from PIL import Image
import pickle
from scipy.interpolate import RegularGridInterpolator
from scipy import sparse
import os
import argparse
# import torch

def xyz2latlong(vertices):
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    long = np.arctan2(y, x)
    xy2 = x**2 + y**2
    lat = np.arctan2(z, np.sqrt(xy2))
    return lat, long

def interp_r2tos2(sig_r2, V, method="linear", dtype=np.float32):
    """
    sig_r2: rectangular shape of (lat, long, n_channels)
    V: array of spherical coordinates of shape (n_vertex, 3)
    method: interpolation method. "linear" or "nearest"
    """
    ele, azi = xyz2latlong(V)
    nlat, nlong = sig_r2.shape[0], sig_r2.shape[1]
    dlat, dlong = np.pi/(nlat-1), 2*np.pi/nlong
    lat = np.linspace(-np.pi/2, np.pi/2, nlat)
    long = np.linspace(-np.pi, np.pi, nlong+1)
    sig_r2 = np.concatenate((sig_r2, sig_r2[:, 0:1]), axis=1)
    intp = RegularGridInterpolator((lat, long), sig_r2, method=method)
    s2 = np.array([ele, azi]).T
    sig_s2 = intp(s2).astype(dtype)
    return sig_s2


def change(VE, i):
    M = RM[i]
    M = np.asarray(M)
    M = M.reshape(3,-1)
    new_V = M.dot(VE.T)
    V = new_V.T
    return V

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str)
args = parser.parse_args()
video = args.video
global RM
RM = np.load('RM.npy')
src_path = video+'_img'
dest_path = video+'_data/'
p = pickle.load(open('./mesh_files/partial_icosphere_9.pkl', "rb"))
V = p['V']
j = 0

print('#########To Numpy File#############')
files = sorted(os.listdir(src_path))
for imgname in files:
    if imgname=='.DS_Store':
        continue
    imgfile = src_path+'/'+imgname
    # print(imgfile)
    img = Image.open(imgfile)
    img.load()
    data = np.asarray(img, dtype='int32')
    data = data/255.0
    for i in range(80):
        new_V = change(V, i)
        res = interp_r2tos2(data, new_V, method = 'linear', dtype = np.float32)
        np.savez(dest_path+str(j)+'.npz',label = res, DI = imgname, PI = i)
        j += 1

