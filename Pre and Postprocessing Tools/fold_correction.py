import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import img_as_ubyte

filedir = 'C:\\Education\\University of Toronto\\Year 4\\Zhen Lab\\Z Alignment\\Dauer2 Test Images\\fold_correction\\'
savedir = filedir


def affine_alignment(img1, img2, ds_factor, mask):

    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    width = int(img1_gray.shape[1] / ds_factor)
    height = int(img1_gray.shape[0] / ds_factor)

    img1_gray = cv.resize(img1_gray, (width, height))
    img2_gray = cv.resize(img2_gray, (width, height))
    mask = cv.resize(mask, (width, height))

    empty_array_1 = np.zeros(img1_gray.shape[:2], dtype=np.uint8)
    empty_array_2 = np.zeros(img2_gray.shape[:2], dtype=np.uint8)

    row1, col1 = img1_gray.shape
    row2, col2 = img2_gray.shape

    sift = cv.SIFT_create()

    kp1, d1 = sift.detectAndCompute(img1_gray, None)
    kp2, d2 = sift.detectAndCompute(img2_gray, mask)

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

'''
pad = 1356

img_733 = cv.imread(filedir + 'global_affine_0733.tif')
img_736_1 = cv.imread(filedir + 'Region_0736_1.tif')
img_736_1 = cv.copyMakeBorder(img_736_1, pad, pad, pad, pad, cv.BORDER_CONSTANT)
img_736_2 = cv.imread(filedir + 'Region_0736_2.tif')
img_736_2 = cv.copyMakeBorder(img_736_2, pad, pad, pad, pad, cv.BORDER_CONSTANT)

mask1 = cv.imread(filedir + 'mask_0733.tif')
mask1 = cv.cvtColor(mask1, cv.COLOR_BGR2GRAY)
mask2 = cv.imread(filedir + 'mask_0733_2.tif')
mask2 = cv.cvtColor(mask2, cv.COLOR_BGR2GRAY)


M_1 = affine_alignment(img_736_1, img_733, 5, mask2)
print('1')
M_2 = affine_alignment(img_736_2, img_733, 5, mask1)
print('2')

dst1 = cv.warpAffine(img_736_1, M_1, (15000, 15000))
dst2 = cv.warpAffine(img_736_2, M_2, (15000, 15000))
'''

dst1 = cv.imread(filedir + '0736_1.tif')
dst2 = cv.imread(filedir + '0736_2.tif')

dst = dst1 + dst2

cv.imwrite(savedir + 'global_affine_0736.tif', dst)

