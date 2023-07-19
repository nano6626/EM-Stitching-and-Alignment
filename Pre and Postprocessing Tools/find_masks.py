import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import img_as_ubyte

filedir = 'C:\\Education\\University of Toronto\\Year 4\\Zhen Lab\\Z Alignment\\Dauer2 Test Images\\raw_images\\'
savedir = 'C:\\Education\\University of Toronto\\Year 4\\Zhen Lab\\Z Alignment\\Dauer2 Test Images\\global_affine\\'

def load_img(index):

    index = str(index).zfill(4)
    img = cv.imread(filedir + 'Region_' + index + '_r1-c1.tif')
    pad = 1356
    img = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_CONSTANT)

    return img

def find_mask(edges, p2):

    if p2 == 0:
        print('Cannot find mask')
        return False

    try:

        circles = cv.HoughCircles(image=edges, method=cv.HOUGH_GRADIENT, dp=1,  
                    minDist=10, param1=200, param2=p2, minRadius=80, 
                    maxRadius=190)

        if len(circles[0, :]) < 5:

            print(len(circles[0, :]), 'Too few circles')
            return find_mask(edges, p2 - 1)

        elif len(circles[0, :]) > 9:

            print(len(circles[0, :]), 'Too many circles')
            return find_mask(edges, p2 + 1)

        else:

            print(len(circles[0, :]), 'circles detected')

            circle_array = np.zeros((15000, 15000), dtype = np.uint8)
            for circle in circles[0, :]:
                a, b = int(circle[0]), int(circle[1])
                a *= 15
                b *= 15
                radius = int(circle[2])
                radius *= 15
                cv.circle(img=circle_array, center=(a, b), radius=radius, color=255, 
                        thickness=-1)
        
            kernel = np.ones((5, 5), np.uint8)

            return cv.dilate(circle_array, kernel, iterations = 11)

    except:

        print('no circles detected, reducing p2')
        return find_mask(edges, p2 - 1)

'''
for i in [0]:

    img = load_img(100)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(img_gray)

    blur = cv.GaussianBlur(cl,(5,5),0)
    th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

    edges = cv.Canny(th,230,255)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    erode = cv.dilate(edges, kernel, iterations=2)

    edges_ds = cv.resize(erode, (1000, 1000)) #ds factor of 15
'''

img = load_img(100)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = clahe.apply(img_gray)
blur = cv.GaussianBlur(cl,(5,5),0)
th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
edges = cv.Canny(th,240,255)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
dilate = cv.dilate(edges, kernel, iterations=31)

img_ds = cv.resize(dilate, (1000, 1000))

plt.imshow(dilate)
plt.show()
