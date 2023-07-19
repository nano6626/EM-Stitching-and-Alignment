#importing libraries

import functools as ft
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import PIL
import cv2 as cv
from jax.lib import xla_bridge
PIL.Image.MAX_IMAGE_PIXELS = 400000000
print(xla_bridge.get_backend().platform)
count = cv.cuda.getCudaEnabledDeviceCount()
print(count)

from sofima.flow_field import JAXMaskedXCorrWithStatsCalculator
from sofima.flow_utils import clean_flow
from sofima.warp import ndimage_warp
from sofima import mesh
from itertools import combinations
from math import comb
import os

'''
This script was designed for the fine elastic alignment of the liver EM dataset. It works in an iterative
manner. We start by finding flow vector fields between images we want to align, but use a large mesh size. 
We use these flows to further refine the flow into a smaller mesh size, until it's sufficiently small
for optimization. This iterative approach seemed to work significantly better than what SOFIMA had done. I 
had also written this script before SOFIMA added decent/formal z-alignment functionality. I had a lot of help from 
https://github.com/google-research/sofima/issues/3 in developing this. 
'''

def correlation(img1, img2, patch_size, step, batch_size):
    '''find a flow between two images using cross correlation'''
    maskcorr = JAXMaskedXCorrWithStatsCalculator()
    flow = maskcorr.flow_field(img1, img2, patch_size, step, batch_size = batch_size)
    return flow
    
def clean_displacements(flow, min_peak_ratio, min_peak_sharpness, k0, stride):
    '''
    filter displacements according to various parameters, relax flow field using spring mesh system

    Parameters
    ----------
    
    min_peak_ratio : float
                     after cross correlation, we calculate the first and second highest peaks. Their ratio
                     is the peak ratio. The min_peak_ratio is the smallest value we allow this to be
                     until we consider the match a false match. 

    min_peak_sharpness : float
                         obtained by calculating curvature around peak (hessian matrix typically).
    
    k0 : float
         spring constant of mesh

    stride : float
             in this case, corresponds to how far nodes in the mesh are from each other
    '''

    clean_output = clean_flow(flow, min_peak_ratio=min_peak_ratio, min_peak_sharpness=min_peak_sharpness, max_magnitude=0, max_deviation=0)
    #plt.matshow(clean_output[0])
    clean_output_compatible = []
    for i in range(2):
        clean_output_compatible.append(np.array([clean_output[i,:]]))
    clean_output_compatible = np.array(clean_output_compatible)

    config = mesh.IntegrationConfig(dt=0.001, gamma=0., k0=k0, k=0.1, stride=stride,
                                num_iters=1000, max_iters=500000, stop_v_max=0.000001,
                                dt_max=100, prefer_orig_order=True,
                                start_cap=0.1, final_cap=10)
    x = np.zeros_like(clean_output_compatible)
    x,ekin,t = np.array(mesh.relax_mesh(x, clean_output_compatible, config))
    #plt.matshow(x[0,0])

    clean_output_temp = []
    for i in range(2):
        clean_output_temp.append(x[i,0])
    clean_output_temp = np.array(clean_output_temp)

    return clean_output_temp
    

def find_nbhd(target_img, x, y, epsilon):
    '''find the neighbourhood of a point in an image. in hindsight, I could have just padded the image :)'''

    col, row = target_img.shape

    #there are 9 cases LOL
    #case 1, x - epsilon and y - epsilon is less than 0 ###
    #case 2, x - epsilon is less than 0 (only) ###
    #case 3, x - epsilon is less than 0 and y + epsilon > row ###
    #case 4, y + epsilon > row only ###
    #case 5, y + epsilon > row and x + epsilon > col ###
    #case 6, x + epsilon > col only ###
    #case 7, x + epsilon > col and y - epsilon less than 0 ###
    #case 8, y - epsilon less than 0 only
    #case 9, everything else

    if x < epsilon and y < epsilon:
        return target_img[0:x+epsilon, 0:y+epsilon], [0,0]
    
    elif x < epsilon and y + epsilon > row:
        return target_img[0:x+epsilon, y-epsilon:row], [0, y-epsilon]
    
    elif x < epsilon:
        return target_img[0:x+epsilon, y-epsilon:y+epsilon], [0, y-epsilon]

    elif x + epsilon > col and y < epsilon:
        return target_img[x-epsilon:col, 0:y+epsilon], [x-epsilon, 0]
    
    elif x + epsilon > col and y + epsilon > row:
        return target_img[x-epsilon:col, y-epsilon:row], [x-epsilon, y-epsilon]

    elif x + epsilon > col:
        return target_img[x-epsilon:col, y-epsilon:y+epsilon], [x-epsilon, y-epsilon]

    elif y + epsilon > row:
        return target_img[x-epsilon:x+epsilon, y-epsilon:row], [x-epsilon, y-epsilon]

    elif y < epsilon:
        return target_img[x-epsilon:x+epsilon, 0:y+epsilon], [x-epsilon, 0]

    else:
        return target_img[x-epsilon:x+epsilon, y-epsilon:y+epsilon], [x-epsilon, y-epsilon]
        
def find_disp_vfield_iterative(target_img, source_img, flow, epsilon, threshold, stride):
  '''refine the displacement vector field (flow field) by inputting one flow field, and refining
  it into half its original resolution'''

  x_res, y_res = flow[:2,...][0].shape

  #find top left corners in the source image

  refined_flow = np.zeros((4, x_res * 2, y_res * 2))

  tlc = np.zeros_like(flow[:2,...])
  for i in range(x_res):
    for j in range(y_res):
      tlc_x = stride * i
      tlc_y = stride * j
      centre_x = tlc_x + stride//2
      centre_y = tlc_y + stride//2
      radius = max(centre_x, centre_y) + epsilon

      x_disp = flow[0][j,i]
      y_disp = flow[1][j,i]
      
      try:
        x = int(x_disp + centre_x)
        y = int(y_disp + centre_y)
      except ValueError:
        x = int(centre_x)
        y = int(centre_y)

      row, col = target_img.shape

      if x > row:
        x = row
      
      if y > col:
        y = col

      if x < 0:
        x = 0
      
      if y < 0:
        y = 0

      
      epsilon = stride // 2

      nbhd, raymoo = find_nbhd(target_img, y, x, epsilon)
      template = source_img[tlc_y:tlc_y + stride, tlc_x:tlc_x + stride]

      #split template into 4 images

      #template1 = template[0:stride//2, 0:stride//2]
      #template2 = template[stride//2:stride + 1, 0:stride//2]
      #template3 = template[0:stride//2, stride//2:stride + 1]
      #template4 = template[stride//2:stride + 1, stride//2:stride + 1]

      #fine_1 = correlation(nbhd, template1, 100, 100, 1)
      #fine_2 = correlation(nbhd, template2, stride//2, stride//2, 1)
      #fine_3 = correlation(nbhd, template3, stride//2, stride//2, 1)
      #fine_4 = correlation(nbhd, template4, stride//2, stride//2, 1)

      fine_flow = correlation(nbhd, template, stride // 2, stride // 2, 1)
      #fine_flow = correlation(template, nbhd, stride // 2, stride // 2, 1)
      fine_flow[0] = fine_flow[0] + flow[0][j,i]
      fine_flow[1] = fine_flow[1] + flow[1][j,i]

      #obtain a fine flow from template to nbhd
      #if the original flow is x_res, y_res resolution
      #the new flow will be 2 * x_res, 2 * y_res in resolution
      #create empty array with that size, update it each time
      #to update, multiply i,j by 2
      #update displacements i*2,j*2 to i*2 + 1, j*2 + 1

      refined_flow[:, j * 2: j * 2 + 2, i * 2: i * 2 + 2] = fine_flow

      #forgor to add the extra flow skull emoji

  return refined_flow

def find_disp_vfield_iterative_N(target_img, source_img, flow, epsilon, threshold, stride, N):
  '''refine the disp vfield (flow field) into an N-th of its original resolution'''

  x_res, y_res = flow[:2,...][0].shape

  #find top left corners in the source image

  refined_flow = np.zeros((4, x_res * N, y_res * N))

  tlc = np.zeros_like(flow[:2,...])
  for i in range(x_res):
    for j in range(y_res):
      tlc_x = stride * i
      tlc_y = stride * j
      centre_x = tlc_x + stride//2
      centre_y = tlc_y + stride//2
      radius = max(centre_x, centre_y) + epsilon

      x_disp = flow[0][j,i]
      y_disp = flow[1][j,i]
      
      try:
        x = int(x_disp + centre_x)
        y = int(y_disp + centre_y)
      except ValueError:
        x = int(centre_x)
        y = int(centre_y)

      row, col = target_img.shape

      if x > row:
        x = row
      
      if y > col:
        y = col

      if x < 0:
        x = 0
      
      if y < 0:
        y = 0

      
      epsilon = stride // 2

      nbhd, raymoo = find_nbhd(target_img, y, x, epsilon)
      template = source_img[tlc_y:tlc_y + stride, tlc_x:tlc_x + stride]

      #split template into 4 images

      #template1 = template[0:stride//2, 0:stride//2]
      #template2 = template[stride//2:stride + 1, 0:stride//2]
      #template3 = template[0:stride//2, stride//2:stride + 1]
      #template4 = template[stride//2:stride + 1, stride//2:stride + 1]

      #fine_1 = correlation(nbhd, template1, 100, 100, 1)
      #fine_2 = correlation(nbhd, template2, stride//2, stride//2, 1)
      #fine_3 = correlation(nbhd, template3, stride//2, stride//2, 1)
      #fine_4 = correlation(nbhd, template4, stride//2, stride//2, 1)

      fine_flow = correlation(nbhd, template, stride // N, stride // N, 1)
      #fine_flow = correlation(template, nbhd, stride // 2, stride // 2, 1)
      fine_flow[0] = fine_flow[0] + flow[0][j,i]
      fine_flow[1] = fine_flow[1] + flow[1][j,i]

      #obtain a fine flow from template to nbhd
      #if the original flow is x_res, y_res resolution
      #the new flow will be 2 * x_res, 2 * y_res in resolution
      #create empty array with that size, update it each time
      #to update, multiply i,j by 2
      #update displacements i*2,j*2 to i*2 + 1, j*2 + 1

      refined_flow[:, j * N: j * N + N, i * N: i * N + N] = fine_flow

      #forgor to add the extra flow skull emoji

  return refined_flow

def euclidean_norm(v1, v2):
    '''euclidean distance of vectors'''
    v1 = v1.flatten()
    v2 = v2.flatten()

    return np.sqrt(np.sum((v1 - v2)**2))

def vector_voting(vector_fields, T = 100):
  '''implementation of vector voting, see 
  Petascale pipeline for precise alignment of images from serial section electron microscopy
  by Sergiy Popovych et al. 
  https://www.biorxiv.org/content/10.1101/2022.03.25.485816v1
  '''

  #number of vector_fields

  n = len(vector_fields)

  #organize n input vector fields into simple majority subsets
  #the size of subsets we need to choose is given by m

  m = int(np.floor((n/2) + 1))

  #we now choose all size-m subsets out of the flow array

  subset_indices = list(combinations(np.arange(n), m))

  #we now loop through subset_indices

  softmin_distances = []
  for i in subset_indices:

    #choose all pairs of indices in this

    pairs = combinations(i, 2)

    norms = []
    for j in pairs:

      norms.append(euclidean_norm(vector_fields[j[0]], vector_fields[j[1]]))
      #print(euclidean_norm(vector_fields[j[0]], vector_fields[j[1]]))

    norms = np.array(norms)
    distance = np.sum(norms) * (1 / comb(m, 2))
    softmin_distance = np.exp(distance / T)
    softmin_distances.append(softmin_distance)

  softmin_distances = np.array(softmin_distances)
  weights = softmin_distances / (np.sum(softmin_distances))

  #finally, compute vector field according to these weights

  summands = []

  for k in range(len(weights)):

    #print(np.sum(np.array([vector_fields[l] for l in subset_indices[k]]), axis = 0))

    summand = (weights[k] / m) * (np.sum(np.array([vector_fields[l] for l in subset_indices[k]]), axis = 0))
    summands.append(summand)

  summands = np.array(summands)
  
  return np.sum(summands, axis = 0)

#implementing filters

def dot_prod_sq(v1, v2):
  '''dot product of vectors, normalized'''

  v1 = v1 / np.sqrt(v1[0]**2 + v1[1]**2)
  v2 = v2 / np.sqrt(v2[0]**2 + v2[1]**2)

  return (v1[0] * v2[0] + v1[1] * v2[1]) #dot product with magnitudes normalized

def dot_prod_filter(flow):
  '''filter according to dot products'''

  filtered_flow = flow

  cirno_funke, x_res, y_res = flow.shape

  #H = np.zeros((x_res, y_res))

  for i in range(1, x_res-1):
    for j in range(1, y_res-1):
      #computing dot products squared
      v1 = dot_prod_sq(flow[:2,i-1,j-1], flow[:2,i,j])
      v2 = dot_prod_sq(flow[:2,i-1,j], flow[:2,i,j])
      v3 = dot_prod_sq(flow[:2,i-1,j+1], flow[:2,i,j])
      v4 = dot_prod_sq(flow[:2,i,j-1], flow[:2,i,j])
      v5 = dot_prod_sq(flow[:2,i,j+1], flow[:2,i,j])
      v6 = dot_prod_sq(flow[:2,i+1,j-1], flow[:2,i,j])
      v7 = dot_prod_sq(flow[:2,i+1,j], flow[:2,i,j])
      v8 = dot_prod_sq(flow[:2,i+1,j+1], flow[:2,i,j])

      hamiltonian = v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8
      if hamiltonian < 0:
        filtered_flow[0,i,j] = np.nan
        filtered_flow[1,i,j] = np.nan

  return filtered_flow

def mean_filter(flow, N):
  '''an averaging filter'''
  cirno_funke, x_res, y_res = flow.shape
  x_ind = np.array_split(np.array(range(x_res)), N)
  y_ind = np.array_split(np.array(range(y_res)), N)

  for i in range(len(x_ind)):
    for j in range(len(y_ind)):
      flow_res = flow[:2, x_ind[i][0]:x_ind[i][-1], y_ind[j][0]:y_ind[j][-1]]

      #flow_res contains NxN vectors, take each vector, find the mean of the magnitudes
      #once magnitide mean is found, find stdev of magnitudes
      #remove dudes who are 3*stdev + mean bigger than mean

      magnitudes = np.sqrt(flow_res[0]**2 + flow_res[1]**2)
      mean = np.mean(magnitudes)
      stdev = np.std(magnitudes)

      for k in range(magnitudes.shape[0]):
        for l in range(magnitudes.shape[1]):
          if magnitudes[k,l] > mean + 3*stdev:
            flow_res[0,k,l] = np.nan
            flow_res[1,k,l] = np.nan
      
      flow[:2, x_ind[i][0]:x_ind[i][-1], y_ind[j][0]:y_ind[j][-1]] = flow_res

  return flow

def find_vfield(img1, img2):
    '''the script that finds the flow vector fields between two images'''
    
    img1 = np.array(img1)
    img2 = np.array(img2)
    
    initial_flow = correlation(img1, img2, 400, 400, 1)
    
    #initial_flow = dot_prod_filter(initial_flow)
    #initial_flow = mean_filter(initial_flow, 4)
    #plt.matshow(initial_flow[0])
    
    refined_flow = find_disp_vfield_iterative(img1, img2, initial_flow, 10, 10, 400)
    #refined_flow = dot_prod_filter(refined_flow)
    #refined_flow = mean_filter(refined_flow, 4)
    #plt.matshow(refined_flow[0])
    
    refined_flow2 = find_disp_vfield_iterative(img1, img2, refined_flow, 10, 10, 200)
    #refined_flow2 = dot_prod_filter(refined_flow2)
    #refined_flow2 = mean_filter(refined_flow2, 4)
    #plt.matshow(refined_flow2[0])
    
    refined_flow3 = find_disp_vfield_iterative(img1, img2, refined_flow2, 10, 10, 100)
    #refined_flow3 = dot_prod_filter(refined_flow3)
    #refined_flow3 = mean_filter(refined_flow3, 4)
    #plt.matshow(refined_flow3[0])
    
    clean_flows = clean_displacements(refined_flow3, 1.7, 1.7, 0.01, 50)
    
    return clean_flows #from img1 to img2
  
#this is the code for no vector voting

start = 105 #start index of  img
end = 212 #final index 
save_dir = ''
load_dir = ''
        

for i in range(start, end + 1):
    print(i)
    
    img1 = PIL.Image.open(save_dir + f'Aligned_Layer{i-1}.tif')
    img2 = PIL.Image.open(load_dir + f'Aligned_Layer{i}.tif')
    
    flow = find_vfield(img1, img2)
    
    warped_img2 = ndimage_warp(np.array(img2), -flow, stride=(50, 50), work_size=(250, 250), overlap=(0,0))
    cv.imwrite(save_dir + f'Aligned_Layer{i}.tif', warped_img2)

#if you want to use more vector voting, here's an implementation for n = 3

'''
for i in range(start, end + 1):
    
    print(i)
    
    if i - start == 1:
        
        img1 = PIL.Image.open(save_dir + f'Aligned_Layer{start}.tif')
        img2 = PIL.Image.open(load_dir + f'Aligned_Layer{start + 1}.tif')
            
        flow = find_vfield(img1, img2)
        
        warped_img2 = ndimage_warp(np.array(img2), -flow, stride=(50, 50), work_size=(250, 250), overlap=(0,0))
        cv.imwrite(save_dir + f'Aligned_Layer{i}.tif', warped_img2)
        #plt.figure(figsize=(15,10))
        #plt.imshow(warped_img2, cmap = plt.cm.Greys_r)
        
    elif i - start == 2:
        
        img1 = img1
        img2 = PIL.Image.fromarray(warped_img2)
            
        img3 = PIL.Image.open(load_dir + f'Aligned_Layer{i}.tif')
            
        flow13 = find_vfield(img1, img3)
        flow23 = find_vfield(img2, img3)
        
        vector_fields_x = np.array([flow13[0], flow23[0]])
        vector_fields_y = np.array([flow13[1], flow23[1]])
        
        flow = np.zeros_like(flow23)
        flow[0] = vector_voting(vector_fields_x)
        flow[1] = vector_voting(vector_fields_y)
        
        #flow = np.array([vector_voting(vector_fields_x), vector_voting(vector_fields_y)])
        #plt.matshow(flow[0])
        #plt.matshow(flow[1])
        #print('hi')
        
        warped_img3 = ndimage_warp(np.array(img3), -flow, stride=(50, 50), work_size=(250, 250), overlap=(0,0))
        cv.imwrite(save_dir + f'Aligned_Layer{i}.tif', warped_img3)
        #plt.figure(figsize=(15,10))
        #plt.imshow(warped_img3, cmap = plt.cm.Greys_r)
        
    elif i - start == 3:
        
        img1 = img1
        img2 = img2
        img3 = PIL.Image.fromarray(warped_img3)
        
        img4 = PIL.Image.open(load_dir + f'Aligned_Layer{i}.tif')
        
        flow14 = find_vfield(img1, img4)
        flow24 = find_vfield(img2, img4)
        flow34 = find_vfield(img3, img4)
        
        vector_fields_x = np.array([flow14[0], flow24[0], flow34[0]])
        vector_fields_y = np.array([flow14[1], flow24[1], flow34[1]])
        
        flow = np.zeros_like(flow24)
        flow[0] = vector_voting(vector_fields_x)
        flow[1] = vector_voting(vector_fields_y)
        
        #plt.matshow(flow[0])
        #plt.matshow(flow[1])
        #print('hi')
        
        warped_img4 = ndimage_warp(np.array(img4), -flow, stride=(50, 50), work_size=(250, 250), overlap=(0,0))
        cv.imwrite(save_dir + f'Aligned_Layer{i}.tif', warped_img4)
        #plt.figure(figsize=(15,10))
        #plt.imshow(warped_img4, cmap = plt.cm.Greys_r)
        
    elif i - start > 3:
        
        img1 = img2
        img2 = img3
        img3 = PIL.Image.fromarray(warped_img4)
        
        img4 = PIL.Image.open(load_dir + f'Aligned_Layer{i}.tif')
        
        flow14 = find_vfield(img1, img4)
        flow24 = find_vfield(img2, img4)
        flow34 = find_vfield(img3, img4)
        
        vector_fields_x = np.array([flow14[0], flow24[0], flow34[0]])
        vector_fields_y = np.array([flow14[1], flow24[1], flow34[1]])
        
        flow = np.zeros_like(flow24)
        flow[0] = vector_voting(vector_fields_x)
        flow[1] = vector_voting(vector_fields_y)
        
        #plt.matshow(flow[0])
        #plt.matshow(flow[1])
        #print('hi')
        
        warped_img4 = ndimage_warp(np.array(img4), -flow, stride=(50, 50), work_size=(250, 250), overlap=(0,0))
        cv.imwrite(save_dir + f'Aligned_Layer{i}.tif', warped_img4)
        #plt.figure(figsize=(15,10))
        #plt.imshow(warped_img4, cmap = plt.cm.Greys_r)

'''