#import necessary libraries

import numpy as np
import cv2 as cv
from scipy.interpolate import LinearNDInterpolator
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import sys

#this script is used to render the images after rough elastic alignment

mesh_dir = '' #dir where mesh saved
load_dir = '' #dir where images are
savedir = '' #dir where you want to save images
savedir_ds = '' #dir where you want to save downsampled copies
start = 0 #start index of images

mesh = np.load(mesh_dir + 'relaxed_mesh_0_200.npy', allow_pickle=True)[()] #load the mesh

def matches_to_flow(nodes, offsets):
    '''uses bilinear interpolation to convert the mesh displacements into a flow map on the whole image'''
    
    x = nodes[:,0]
    y = nodes[:,1]
    
    x_off = offsets[:,0]
    y_off = offsets[:,1]

    #finding x flow field

    X = np.linspace(0, 20000, 20000)
    Y = np.linspace(0, 20000, 20000)

    X, Y = np.meshgrid(X, Y)

    interp1 = LinearNDInterpolator(list(zip(x, y)), x_off)
    X_OFF = interp1(X, Y)
    print('test1')
    
    #finding y flow field

    interp2 = LinearNDInterpolator(list(zip(x, y)), y_off)
    Y_OFF = interp2(X, Y)
    print('test2')

    return X, Y, X_OFF, Y_OFF

def load_img(i):
    '''loads the image, i is the image index'''

    index = str(i).zfill(4)
    
    img = Image.open(load_dir + 'global_affine_crop_' + index + '.tif').convert('L')
    
    return img

for index in [int(sys.argv[1])]:

    i = index - start
    
    mesh_nodes = mesh[i][0]
    tri_points = mesh[i][1]
    
    offsets = mesh_nodes - tri_points
    X, Y, X_OFF, Y_OFF = matches_to_flow(tri_points * 4, offsets * 4) #scale displacements by 4 to account for image downsampling during alignment

    map_x = np.add(X_OFF.T, X)
    print('moge')
    map_y = np.add(Y_OFF.T, Y)
    print('my beloved')
            
    MAP_X = map_x.astype('float32')
    MAP_Y = map_y.astype('float32')

    img = np.array(load_img(index))
    img_ds = np.array(load_img(index).resize((1000,1000)))
    cv.imwrite(savedir_ds + f'pre_ds_{i}.tif', img_ds)
    cv.imwrite(savedir_ds + f'pre_zoom_{i}.tif', img[4000:5000, 4000:5000])
    
    img = cv.remap(img, MAP_X, MAP_Y, cv.INTER_LINEAR)
    img_ds = cv.resize(img, (1000,1000))
    
    cv.imwrite(savedir + f'post_{i}.tif', img)
    cv.imwrite(savedir_ds + f'post_ds_{i}.tif', img_ds)
    cv.imwrite(savedir_ds + f'post_zoom_{i}.tif', img[4000:5000, 4000:5000])