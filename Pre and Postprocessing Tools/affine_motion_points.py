import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import img_as_ubyte
import os

dir = 'C:/Education/University of Toronto/Year 4/Zhen Lab/Z Alignment/twk40 Alignment/'

#img1_full = cv.imread(dir + 'global_affine_0069.tif')
#img2_full = cv.imread(dir + 'ribbon_01_section_070_1.tif')
#img3_full = cv.imread(dir + 'ribbon_01_section_070_2.tif')

img1_full = cv.imread(dir + 'global_affine_0399.tif')
image_list = sorted(os.listdir('D:/twk40/'))
img2_name = 'D:/twk40/' + image_list[400]
img2_full = cv.imread(img2_name)

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

ds_factor = 10

width1 = int(img1_full.shape[1] / ds_factor)
height1 = int(img1_full.shape[0] / ds_factor)

width2 = int(img2_full.shape[1] / ds_factor)
height2 = int(img2_full.shape[0] / ds_factor)

#width3 = int(img3_full.shape[1] / ds_factor)
#height3 = int(img3_full.shape[0] / ds_factor)

img1 = cv.resize(img1_full, (width1, height1))
img2 = cv.resize(img2_full, (width2, height2))
#img3 = cv.resize(img3_full, (width3, height3))

M1 = affine_alignment(img2, img1, None, None, ds_factor)
#M2 = affine_alignment(img3, img1, None, None, ds_factor)

dst = cv.warpAffine(img2_full, M1, (20000, 20000))
#dst2 = cv.warpAffine(img3_full, M2, (28672, 28672))
#dst = dst1 + dst2
dst_ds = cv.resize(dst, (1000, 1000))
img1_ds = cv.resize(img1, (1000, 1000))

cv.imwrite(dir + 'global_affine_0400.tif', dst)
cv.imwrite(dir + 'global_affine_0400_ds.tif', dst_ds)
#cv.imwrite(dir + '0069_ds.tif', img1_ds)
