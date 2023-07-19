#importing
#this script is very similar to that made by SOFIMA, can be found at https://github.com/google-research/sofima/blob/main/notebooks/em_alignment.ipynb
#this script is good at fine elastic alignment
#however, by design the script optimizes images one at a time and is quite poor at dealing at images that differ by significant warping

import tensorstore as ts
import numpy as np
from PIL import Image
from concurrent import futures
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import cv2 as cv

from connectomics.common import bounding_box
from sofima import flow_field
from sofima import flow_utils
from sofima import map_utils
from sofima import mesh
from sofima import warp
from tqdm.notebook import tqdm
Image.MAX_IMAGE_PIXELS = 400000000

assert jax.devices()[0].platform == 'gpu'

#loading images, load_img_ds also applies downsampling

load_dir = ''
savedir = ''
savedir_ds = ''

def load_img(z):
    '''load images'''

    index = str(z).zfill(4)
    
    img = np.array(Image.open(load_dir + f'post_{z}' + '.tif'))
    
    return img

def load_img_ds(z):
    '''load downsampled copies'''

    index = str(z).zfill(4)
    
    img = np.array(Image.open(load_dir + f'post_{z}' + '.tif').resize((10000, 10000)))
    
    return img

patch_size = 160 #size (in pixels) of the patches that are compared to each other via cross correlation
stride = 40      #distance between patches (can think of this as mesh size)
start = 0        #start index
end = 250        #end index

time1 = time.time()

def _compute_flow(downsampled):
    '''finds the flow maps between an image z+1 and image z, for all z'''
    
    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
    flows = []
    
    if downsampled: load = load_img_ds
    else: load = load_img
        
    prev = load(start)
    
    fs = []
    with futures.ThreadPoolExecutor() as tpe:
        for z in range(start + 1, end + 1):
            fs.append(tpe.submit(lambda z = z: load(z)))
            
        fs = fs[::-1]
        
        for z in tqdm(range(start + 1, end + 1)):
            curr = fs.pop().result()
            
            flows.append(mfc.flow_field(prev, curr, (patch_size, patch_size),
                                  (stride, stride), batch_size=16))
            
            prev = curr
            
    return flows

#flows found for downsampled and full-res images

flows1x = np.array(_compute_flow(downsampled = False))
flows2x = np.array(_compute_flow(downsampled = True))

# Convert to [channels, z, y, x].
flows2x = np.transpose(flows2x, [1, 0, 2, 3])
flows1x = np.transpose(flows1x, [1, 0, 2, 3])

# Pad to account for the edges of the images where there is insufficient context to estimate flow.
pad = patch_size // 2 // stride
flows1x = np.pad(flows1x, [[0, 0], [0, 0], [pad, pad], [pad, pad]], constant_values=np.nan)
flows2x = np.pad(flows2x, [[0, 0], [0, 0], [pad, pad], [pad, pad]], constant_values=np.nan)

#filtering out false matches
f1 = flow_utils.clean_flow(flows1x, min_peak_ratio=1.9, min_peak_sharpness=1.8, max_magnitude=50, max_deviation=30)
f2 = flow_utils.clean_flow(flows2x, min_peak_ratio=1.9, min_peak_sharpness=1.8, max_magnitude=50, max_deviation=30)

f2_hires = np.zeros_like(f1)

scale = 0.5
oy, ox = np.ogrid[:f2.shape[-2], :f2.shape[-1]]
oy = oy.ravel() / scale
ox = ox.ravel() / scale

box1x = bounding_box.BoundingBox(start=(0, 0, 0), size=(f1.shape[-1], f1.shape[-2], 1))
box2x = bounding_box.BoundingBox(start=(0, 0, 0), size=(f2.shape[-1], f2.shape[-2], 1))

for z in tqdm(range(f2.shape[1])):
  # Upsample and scale spatial components.
  resampled = map_utils.resample_map(
      f2[:, z:z + 1, ...],  #
      box2x, box1x, 1 / scale, 1)
  f2_hires[:, z:z + 1, ...] = resampled / scale

final_flow = flow_utils.reconcile_flows((f1, f2_hires), max_gradient=0, max_deviation=50, min_patch_size=400)

config = mesh.IntegrationConfig(dt=0.001, gamma=0.0, k0=0.001, k=0.1, stride=stride, num_iters=1000,
                                max_iters=100000, stop_v_max=0.005, dt_max=1000, start_cap=0.01,
                                final_cap=10, prefer_orig_order=True)

solved = [np.zeros_like(final_flow[:, 0:1, ...])]
origin = jnp.array([0., 0.])

for z in tqdm(range(0, final_flow.shape[1])):
    prev = map_utils.compose_maps_fast(final_flow[:, z:z+1, ...], origin, stride,
                                     solved[-1], origin, stride)
    x = np.zeros_like(solved[0])
    x, e_kin, num_steps = mesh.relax_mesh(x, prev, config)
    x = np.array(x)
    solved.append(x)

time2 = time.time()
print(time2 - time1, 'vfields are found')
np.save('solved_300_400.npy', np.array(solved))

#typically, it's good to save the flow maps, visualize them and warp afterwards
#in this script, we render the final images immediately 
from sofima.warp import ndimage_warp

solved = np.concatenate(solved, axis=1)
inv_map = map_utils.invert_map(solved, box1x, box1x, stride)

for i in range(inv_map.shape[1]):

    print(i)

    if i == 0:

        index = str(i).zfill(4)
    
        warped_img = load_img(i + start)
        warped_img_ds = cv.resize(warped_img, (1000,1000))
        cv.imwrite(savedir + 'post_' + index + '.tif', warped_img)
        cv.imwrite(savedir_ds + 'post_' + index + '.tif', warped_img_ds)
        cv.imwrite(savedir_ds + 'post_zoom_' + index + '.tif', warped_img[3000:5000, 3000:5000])
    
    else:

        index = str(i).zfill(4)
    
        flow = inv_map[:,i,...]
    
        warped_img = ndimage_warp(load_img(i + start), flow, stride=(40,40), work_size=(250,250), overlap=(0,0))
        warped_img_ds = cv.resize(warped_img, (1000,1000))
        cv.imwrite(savedir + 'post_' + index + '.tif', warped_img)
        cv.imwrite(savedir_ds + 'post_' + index + '.tif', warped_img_ds)
        cv.imwrite(savedir + 'post_zoom_' + index + '.tif', warped_img[3000:5000, 3000:5000])

time3 = time.time()
print(time3 - time2, 'images rendered')
