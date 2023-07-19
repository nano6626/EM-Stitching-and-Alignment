import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import img_as_ubyte
import os

filedir = 'D:/twk40/' #directory where images are
savedir = 'D:/affine_twk40/' #where you want to save them

image_list = sorted(os.listdir(filedir))

def affine_alignment(img1, img2, mask1, mask2, ds_factor):
    '''
    a script to perform affine alignment of two images

    Parameters
    ----------

    img1, img2 : opencv images
                 the images you want to align

    mask1, mask2 : np arrays of size img1, img2 respectively, dtype is uint8
                   the masks of the region of interest (usually the mask of the worm)

    ds_factor : int
                how much img1 and img2 are downsampled

    Returns
    -------

    M : np array
        a matrix storing the affine motion (2 * 3 in size)
    '''

    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

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
    '''load the image'''

    img = cv.imread(filedir + image_list[index])

    return img

def load_mask(index):
    '''load the mask'''

    index = str(index).zfill(4)
    mask = cv.imread(filedir + 'mask_' + index + '.tif')
    pad = 1356
    mask = cv.copyMakeBorder(mask, pad, pad, pad, pad, cv.BORDER_CONSTANT)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

    return mask


start = 0 #index where you start
end = 540 #index where you end alignment
ds_factor = 12 #downsmapling factor
affine_motions = [] #stores affine motions

#in this case, the affine motions were already calculated, so we load them instead
#otherwise, you should use commented code below to generate affine motions
affine_motions = np.load('D:/' + 'affine_motions.npy', allow_pickle=True)

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

        width = int(img1_full.shape[1] / ds_factor)
        height = int(img1_full.shape[0] / ds_factor)

        img1 = cv.resize(img1_full, (width, height))
        img2 = cv.resize(img2_full, (width, height))
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

        width = int(img1_full.shape[1] / ds_factor)
        height = int(img1_full.shape[0] / ds_factor)

        img1 = cv.resize(img1_full, (width, height))
        img2 = cv.resize(img2_full, (width, height))
        #mask1 = cv.resize(mask1, (width, height))
        #mask2 = cv.resize(mask2, (width, height))

        M = affine_alignment(img2, img1, mask2, mask1, ds_factor)

        affine_motions.append(M)
'''

#np.save('D:/' + 'affine_motions.npy', affine_motions)
print('motions saved, begin rendering')

#save affine motions

def compose_affine_maps(M1, M2):
    '''composition of M2 circ M1'''

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
        #prev_map = M
        dst = cv.warpAffine(img_full, M, (img_full.shape[1], img_full.shape[0]))
        dst = cv.resize(dst, (1024, 1024))
        index = str(i+1).zfill(4)
        cv.imwrite(savedir + f'global_affine_' + index + '.tif', dst)

    else:

        img_full = load_img(i+1)
        #img_full = cv.resize(img_full, (500, 500))
        M = compose_affine_maps(prev_map, affine_motions[i - start])
        #prev_map = M
        dst = cv.warpAffine(img_full, M, (img_full.shape[1], img_full.shape[0]))
        dst = cv.resize(dst, (1024, 1024))
        index = str(i+1).zfill(4)
        cv.imwrite(savedir + f'global_affine_' + index + '.tif', dst)