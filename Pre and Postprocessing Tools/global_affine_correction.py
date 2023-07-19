import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import img_as_ubyte
import os

filedir = 'D:/twk40/'
savedir = 'D:/affine_twk40/'

image_list = sorted(os.listdir(filedir))
print(image_list[205])

def affine_alignment(img1, img2, mask1, mask2, ds_factor):

    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)


    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img1_gray = clahe.apply(img1_gray)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img2_gray = clahe.apply(img2_gray)

    empty_array_1 = np.zeros(img1_gray.shape[:2], dtype=np.uint8)
    empty_array_2 = np.zeros(img2_gray.shape[:2], dtype=np.uint8)

    row1, col1 = img1_gray.shape
    row2, col2 = img2_gray.shape

    sift = cv.SIFT_create()

    kp1, d1 = sift.detectAndCompute(img1_gray, mask = mask1)
    kp2, d2 = sift.detectAndCompute(img2_gray, mask = mask2)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(d1, d2, k = 2)

    ratio = 0.8

    good_matches = []
    for i, j in matches:
        if i.distance < j.distance * ratio:
            good_matches.append([i])

    source = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    destination = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    source = source
    destination = destination

    M, mask = cv.estimateAffine2D(source, destination)

    M[0,2] *= ds_factor
    M[1,2] *= ds_factor

    return M

def load_img(index):

    img = cv.imread(filedir + image_list[index])

    return img

def load_mask(index):

    index = str(index).zfill(4)
    mask = cv.imread(filedir + 'mask_' + index + '.tif')
    pad = 1356
    mask = cv.copyMakeBorder(mask, pad, pad, pad, pad, cv.BORDER_CONSTANT)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

    return mask

start = 0
end = len(image_list) - 1
ds_factor = 8
affine_motions = []

'''
for i in range(start, end):

    print('finding affine motion', i)

    if i == start:

        img1_full = load_img(i)
        img2_full = load_img(i + 1)
        #mask1 = load_mask(i)
        #mask2 = load_mask(i + 1)
        mask1 = None
        mask2 = None

        width1 = int(img1_full.shape[1] / ds_factor)
        height1 = int(img1_full.shape[0] / ds_factor)

        width2 = int(img2_full.shape[1] / ds_factor)
        height2 = int(img2_full.shape[0] / ds_factor)

        img1 = cv.resize(img1_full, (width1, height1))
        img2 = cv.resize(img2_full, (width2, height2))
        #mask1 = cv.resize(mask1, (width, height))
        #mask2 = cv.resize(mask2, (width, height))

        M = affine_alignment(img2, img1, mask2, mask1, ds_factor)

        affine_motions.append(M)

    else:

        img1_full = img2_full
        img2_full = load_img(i + 1)
        #mask1 = load_mask(i)
        #mask2 = load_mask(i + 1)
        mask1 = None
        mask2 = None

        width1 = int(img1_full.shape[1] / ds_factor)
        height1 = int(img1_full.shape[0] / ds_factor)

        width2 = int(img2_full.shape[1] / ds_factor)
        height2 = int(img2_full.shape[0] / ds_factor)

        img1 = cv.resize(img1_full, (width1, height1))
        img2 = cv.resize(img2_full, (width2, height2))
        #mask1 = cv.resize(mask1, (width, height))
        #mask2 = cv.resize(mask2, (width, height))

        M = affine_alignment(img2, img1, mask2, mask1, ds_factor)

        affine_motions.append(M)
'''
       

#something that deforms images

affine_motions = np.load('D:/' + 'affine_motions.npy', allow_pickle=True)

#np.save('D:/' + 'affine_motions.npy', affine_motions)
print('motions saved, begin rendering')

#save affine motions

def compose_affine_maps(M1, M2):
    #composition of M2 circ M1

    M1_new = np.zeros((3,3))
    M2_new = np.zeros((3,3))

    M1_new[0:2, :] = M1
    M1_new[2,2] = 1
    M2_new[0:2, :] = M2
    M2_new[2,2] = 1

    return np.matmul(M1_new, M2_new)[0:2, :]

#apply affine motions

prev_map = None

for i in range(start, end):
    
    print('applying affine motion', i)

    if prev_map is None:
        
        img_full = load_img(i+1)
        M = affine_motions[i - start]
        prev_map = M
        dst = cv.warpAffine(img_full, M, (28672, 28672))
        dst = cv.resize(dst, (1024, 1024))
        index = str(i+1).zfill(4)
        cv.imwrite(savedir + f'global_affine_' + index + '.tif', dst)

    else:

        img_full = load_img(i+1)
        #img_full = cv.resize(img_full, (500, 500))
        M = compose_affine_maps(prev_map, affine_motions[i - start])
        prev_map = M
        dst = cv.warpAffine(img_full, M, (28672, 28672))
        dst = cv.resize(dst, (1024, 1024))
        index = str(i+1).zfill(4)
        cv.imwrite(savedir + f'global_affine_' + index + '.tif', dst)

#am = np.load(amdir + 'affine_motions.npy', allow_pickle=True)

#am[732] = I

#np.save(amdir + 'affine_motions.npy', am)