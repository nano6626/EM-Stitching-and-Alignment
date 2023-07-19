import numpy as np

load_dir = 'C:/Education/University of Toronto/Year 4/Zhen Lab/Z Alignment/twk40 Alignment/'
mpts = np.load(load_dir + 'mpts_0_540.npy', allow_pickle=True)[()]

for pair in list(mpts.keys()):

    matching_points = mpts[pair].copy()

    chopping_block = []

    for i in range(len(matching_points)):

        mpt = matching_points[i]

        x1 = mpt[0][0]
        x2 = mpt[0][1]
        x3 = mpt[1][0]
        x4 = mpt[1][1]

        if x1 < 0 or x2 < 0 or x3 < 0 or x4 < 0:

            print(i, 'delete', '0')

            chopping_block.append(i)

        if x1 > 5000 or x2 > 5000 or x3 > 5000 or x4 > 5000:

            print(i, 'delete', '1')

            chopping_block.append(i)

    if len(chopping_block) == 0:
        
        continue

    else:

        matching_points = np.delete(matching_points, chopping_block, axis = 0)
        mpts[pair] = matching_points.copy()

np.save(load_dir + 'mpts_0_540_filt.npy', mpts)