#import the necessary libraries

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
import itertools
import multiprocess as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from PIL import Image, ImageOps
import itertools
import time
import os

'''
This script is to be used for rough elastic alignment. The script takes in a volume of EM images and
their corresponding matching points. Each image is modeled as a spring mesh and forces are applied to 
the nodes of the mesh according to the matching points. Via gradient descent with momentum, the mesh
is relaxed to its minimum energy state. The minumum energy state of the mesh corresponds to the 
alignment of the images. 
'''

class Mesh():

    def __init__(self, img_size, index, mesh_size, k_internal, k_external, momentum):

        '''
        stores information about each image's mesh

        Parameters
        ----------

        img_size : list of length 2
                   the image size

        index : int
                the index of hte image

        mesh_size : int (can be float for subpixel precision)
                    how long is a mesh element in pixels

        k_internal : float
                     the spring constant of springs within the image's mesh

        k_external : float
                     the spring constant of springs between image meshes

        momentum : float
                   the momentum term in gradient descent with momentum (how much previous
                   steps affect the new step)

        Returns
        -------

        None
        
        '''

        self.k_internal = k_internal
        self.k_external = k_external
        self.momentum = momentum
        self.mesh_size = mesh_size
        self.index = index

        #generating the mesh properties, including its nodes, simplicies and neighbour list
        self.mesh_nodes, self.mesh_simplices, self.neighbours, self.tri = self.generate_mesh(img_size)
        self.resting_spring_lengths = self.find_resting_spring_lengths()
        self.previous_step = np.zeros((len(self.mesh_nodes), 2))
        self.prev_mesh_nodes = None

    def generate_mesh(self, img_size):
        '''
        inputs an image, returns the properties of the mesh

        Parameters
        ----------

        img_size : list of length 2
                   size of image

        Returns
        -------

        nodes : numpy array
                contains the indices of the pairs of nodes that make up the mesh

        simplicies : numpy array
                     stores information about which nodes form simplicies (triangles) in the mesh

        neighbours : numpy array
                     stores information about every node's neighbours

        tri : scipy Delaunay object
              combines everything above into one object
        '''

        row, col = img_size

        step_col = col // self.mesh_size
        step_row = row // self.mesh_size

        col_ind = np.linspace(0, col, step_col)
        row_ind = np.linspace(0, row, step_row)

        nodes = []
        for i in col_ind:
            for j in row_ind:
                nodes.append([i,j])

        nodes = np.array(nodes)

        #create a Delaunay object (use Delaunay triangulation to create mesh)
        tri = Delaunay(nodes)
        simplices = tri.simplices

        #obtain neighbour information 
        indptr, indices = tri.vertex_neighbor_vertices
        neighbours = [indices[indptr[k]:indptr[k+1]] for k in range(len(nodes))]
        neighbours = np.array(neighbours)

        return nodes, simplices, neighbours, tri

    def point_to_barycentric(self, points):
        '''when given a point, finds the triangle which it belongs to, as well as barycentric weights, taken from adi peleg's pipeline'''

        p = points.copy()
        p[p < 0] = 0.01
        simplex_indices = self.tri.find_simplex(p)
        assert not np.any(simplex_indices == -1)

        #http://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points
        X = self.tri.transform[simplex_indices, :2]
        Y = points - self.tri.transform[simplex_indices, 2]
        b = np.einsum('ijk,ik->ij', X, Y)
        pt_indices = self.tri.simplices[simplex_indices].astype(np.uint32)
        barys = np.c_[b, 1 - b.sum(axis=1)]

        return self.tri.simplices[simplex_indices].astype(np.uint32), barys

    def barycentric_to_point(self, simplex, weights):
        '''given a simplex index and barycentric weights, returns the coordinate'''

        coords = self.mesh_nodes[simplex]

        return np.sum(np.array([coords[i] * weights[i] for i in range(3)]), axis = 0)

    def find_distances(self, node_index):
        '''
        
        Calculates a signed distance between a particular node and its neighbours. The
        precise steps are:

        1) Look at node and its neighbours
        2) Find coordinate of node of interest and its neighbours
        3) Subtract to find distances (neighbour - node is convention)
        4) Take sum of distances, we use this to find forces
        '''

        node = self.mesh_nodes[node_index]
        neighbours = self.mesh_nodes[self.neighbours[node_index]]
        
        distances = np.array([neighbours[i] - node for i in range(len(neighbours))])

        return distances

    def find_resting_spring_lengths(self):
        '''store resting lengths as a list of length values corresponding to neighbours at node'''

        resting_lengths = [self.find_distances(i) for i in range(len(self.mesh_nodes))]

        return np.array(resting_lengths)
    
    def find_internal_node_force_and_energy(self, node_index):
        '''calculates net force and energy on a single node'''
        
        distances = self.find_distances(node_index)
        resting_lengths = self.resting_spring_lengths[node_index]

        forces = []
        energies = []
        for i in range(len(distances)):
            magnitude = np.sqrt(distances[i][0]**2 + distances[i][1]**2)
            resting_magnitude = np.sqrt(resting_lengths[i][0]**2 + resting_lengths[i][1]**2)
            force_x = distances[i][0] * (1 - ((resting_magnitude) / (magnitude))) * self.k_internal
            force_y = distances[i][1] * (1 - ((resting_magnitude) / (magnitude))) * self.k_internal
            forces.append([force_x, force_y])

            energy = (force_x**2 + force_y**2) / (2*self.k_internal)
            energies.append(energy)
        
        return np.sum(np.array(forces), axis = 0), np.sum(np.array(energies), axis = 0)

    def energy_minimization(self, external_force, step_size):
        '''performs the energy minimization process via gradient descent with momentum'''

        def find_force_and_energy():
            '''calculates force and energy of the whole mesh'''

            internal_force = []
            internal_energy = []

            for i in range(len(self.mesh_nodes)):

                internal = self.find_internal_node_force_and_energy(i)
                internal_force.append(internal[0])
                internal_energy.append(internal[1])

            internal_force = np.array(internal_force)
            internal_energy = np.array(internal_energy)
 
            external_energy = np.array([(i[0]**2 + i[1]**2) / (2 * self.k_external) for i in external_force])

            nodal_force = internal_force + external_force
            nodal_energy = internal_energy + external_energy
            total_energy = np.sum(nodal_energy)
            max_energy = np.max(nodal_energy)

            return nodal_force, total_energy, max_energy

        #find force and energy of the mesh (max energy is found for debugging purposes)
        nodal_force, total_energy, max_energy = find_force_and_energy()
        #normalize the nodal force, ensures we don't step too far during mesh update
        normalized_nodal_force = nodal_force / np.max(np.abs(nodal_force))
        #copy the current mesh nodes, mesh_config is for debugging purposes
        mesh_config = self.mesh_nodes.copy()
        self.prev_mesh_nodes = self.mesh_nodes.copy()

        #loop through mesh nodes and update to their new position (update previous step too)
        for i in range(len(self.mesh_nodes)):

            mesh_config[i] += normalized_nodal_force[i] * step_size + self.momentum * self.previous_step[i]
            self.previous_step[i] = normalized_nodal_force[i] * step_size + self.momentum * self.previous_step[i]

        self.mesh_nodes = mesh_config

        return total_energy, self.mesh_nodes
    
    def undo_prev_step(self):
        '''this allows us to undo the step if something goes wrong'''
        
        self.mesh_nodes = self.prev_mesh_nodes.copy()

class Optimizer():

    def __init__(self, indices, mpts, start_ind):

        '''
        an object that handles the mesh optimization process

        Parameters
        ----------

        indices : list
                  list of indices of images

        mpts : dictionary
               dictionary where the key is (i, j) and the value corresponding to the key is a list of matching
               coordinates in image i and image j
        
        start_ind : int
                    the index where we start creating the mesh
        '''

        self.img_size = [5000, 5000] #the xy dimensions of the image
        self.mesh_size = 100 #the pixel size of the mesh 
        self.k_internal = 0.1 #spring constant of internal mesh nodes (i.e. mesh within image)
        self.k_external = 0.05 #spring constant of image-to-image connections
        self.momentum = 0.5 #the momentum for gradient descent with momentum
        self.indices = indices 
        self.mpts = mpts
        self.prev_energy = None #the previous total energy (we use this to check if energy is decreasing)
        self.start_ind = start_ind
        
        #find forces on mesh, as well as the matching points in barycentric coords
        self.global_forces, self.barycentric_representation, self.meshes = self.generate_mesh()

    def generate_mesh(self):
        '''create mesh for each image in the stack'''
        
        global_forces = {}
        meshes = []
        for index in self.indices:
            mesh_index = Mesh(self.img_size, index, self.mesh_size, self.k_internal, self.k_external, self.momentum)
            meshes.append(mesh_index)
            num_of_nodes = len(mesh_index.mesh_nodes)
            global_forces[index] = np.full((num_of_nodes, 2), np.array([0,0], dtype = np.float32))

        #calculate forces and energies

        barycentric_representation = dict.fromkeys(self.mpts.keys(),[])

        for pair in list(self.mpts.keys()):
            
            bary_coords = []

            Mesh0 = meshes[pair[0]-self.start_ind]
            Mesh1 = meshes[pair[1]-self.start_ind]

            matches = self.mpts[pair]

            for match in matches:

                force = self.k_external * (match[0] - match[1])

                simplex0, weights0 = Mesh0.point_to_barycentric(np.array([match[0]]))
                simplex1, weights1 = Mesh1.point_to_barycentric(np.array([match[1]]))

                global_forces[pair[0]][simplex0[0]] += [w * force*-1 for w in weights0[0]]
                global_forces[pair[1]][simplex1[0]] += [w * force for w in weights1[0]]

                bary_coords.append([simplex0, weights0, simplex1, weights1])
                
            barycentric_representation[pair] = bary_coords

        return global_forces, barycentric_representation, meshes

    def recalculate_global_forces(self):
        '''
        
        update the forces on the mesh
        
        Notes
        -----

        In our problem, every image is meshed. The issue here is that every image is given the freedom to 
        move during mesh relaxation. Therefore, the forces between images are not constant. With this in
        mind, we must recalculate the nodal forces everytime the mesh is updated. We take advantage of the
        fact that in barycentric coordinates, the matching points are invariant to mesh updates. Therefore,
        by representing mpts in barycentric coords, we can easily recalculate the nodal forces every iteration.
        
        '''

        global_forces = {}

        for i in self.global_forces.keys():
            global_forces[i] = np.zeros_like(self.global_forces[i])

        for pair in list(self.mpts.keys()):

            Mesh0 = self.meshes[pair[0]-self.start_ind]
            Mesh1 = self.meshes[pair[1]-self.start_ind]

            bary_matches = self.barycentric_representation[pair]

            for bary_match in bary_matches:

                simplex0 = bary_match[0][0]
                weights0 = bary_match[1][0]
                simplex1 = bary_match[2][0]
                weights1 = bary_match[3][0]

                match0 = Mesh0.barycentric_to_point(simplex0, weights0)
                match1 = Mesh1.barycentric_to_point(simplex1, weights1)

                force = (match0 - match1) * self.k_external

                global_forces[pair[0]][simplex0] += [force*weights0[i]*-1 for i in range(3)]
                global_forces[pair[1]][simplex1] += [force*weights1[i] for i in range(3)]

        return global_forces

    def optimize_mesh(self, step_size):
        '''
        
        performs the mesh relaxation procedure

        Parameters
        ----------
        
        step_size : float
                    a scaling factor to control how quickly optimization occurs

        Notes
        -----
        
        When the object Optimizer is initiated, the object creates a mesh on each image. It also calculates
        forces on this mesh with the matching points. At the same time, it converts these matching points
        into a barycentric coordinate representation. During optimize_mesh, which we use to update the mesh,
        we loop through the mesh and update them according to their external force. Everytime we update a 
        single mesh in the stack, we recalculate global forces. This procedure can probably be made faster by
        only updating the forces in the meshes that were affected by the update. 
        '''

        energies = []
        mesh_debug = []

        for mesh in self.meshes:

            ind = mesh.index
            energy, new_mesh = mesh.energy_minimization(self.global_forces[ind], step_size)
            energies.append(energy)
            mesh_debug.append(new_mesh)
            self.global_forces = self.recalculate_global_forces()

        return energies, mesh_debug
    
    def undo_step(self):
        '''undoes previous step'''
        
        for mesh in self.meshes:
            
            mesh.undo_prev_step()
            
    def find_bary(self, dense_mpts):
        '''
        Finds barycentric representation of matching points. This can be used when you first want to
        optimize the mesh with a sparse distribution of matching points, and subsequently optimize with a 
        dense distribution of matching points. The way the code is currently structured just converts the
        mpts to barycentric coords. If you want to optimize with two different mpts, the steps to do this 
        are as follows:
        1) Create a set of sparse and dense matching pts using the find_matching_points script
        2) Create an optimizer object and use the object to find the barycentric coordinates of the dense_mpts
        3) Perform mesh relaxation with the sparse matching points
        4) Input the dense matching points into the object and convert to xy coordinates, resume optimization
        with reoptimize
        '''
        
        barycentric_representation = dict.fromkeys(self.mpts.keys(),[])

        for pair in list(self.mpts.keys()):
            
            bary_coords = []

            Mesh0 = self.meshes[pair[0]-self.start_ind]
            Mesh1 = self.meshes[pair[1]-self.start_ind]

            matches = self.mpts[pair]

            for match in matches:

                force = self.k_external * (match[0] - match[1])

                simplex0, weights0 = Mesh0.point_to_barycentric(np.array([match[0]]))
                simplex1, weights1 = Mesh1.point_to_barycentric(np.array([match[1]]))

                bary_coords.append([simplex0, weights0, simplex1, weights1])
                
            barycentric_representation[pair] = bary_coords

        return barycentric_representation
    
    def bary_to_coords(self, barycentric_rep):
        '''converts barycentric coordinate representation of mpts to xy coordinates'''
        
        coordinate_representation = {}
        
        for pair in list(self.mpts.keys()):
                
            source_target_pairs = []

            Mesh0 = self.meshes[pair[0]-self.start_ind]
            Mesh1 = self.meshes[pair[1]-self.start_ind]

            bary_matches = barycentric_rep[pair]

            for bary_match in bary_matches:

                simplex0 = bary_match[0][0]
                weights0 = bary_match[1][0]
                simplex1 = bary_match[2][0]
                weights1 = bary_match[3][0]

                match0 = Mesh0.barycentric_to_point(simplex0, weights0)
                match1 = Mesh1.barycentric_to_point(simplex1, weights1)
                
                source_target_pair = np.array([[match0[0], match0[1]],[match1[0], match1[1]]])
                source_target_pairs.append(source_target_pair)
                
            source_target_pairs = np.array(source_target_pairs)
                
            coordinate_representation[pair[0], pair[1]] = source_target_pairs
        
        return coordinate_representation

    def optimize(self):
        '''the function that does the optimizing part'''

        num_of_iters = 250 #max number of iterations
        step_size = 0.01 #step size

        #we use an adaptive step size that starts small and then grows to a max value of 1
        for i in range(num_of_iters):
            
            step_size *= 1.1
            if step_size > 1:
                step_size = 1
                
            if self.prev_energy == None or i < 50:

                energies, mesh_debug = self.optimize_mesh(step_size)
                print(i, np.sum(energies))
                self.prev_energy = np.sum(energies)
                plt.scatter(i, np.sum(energies))
                
            else:
                
                energies, mesh_debug = self.optimize_mesh(step_size)
                
                while np.sum(energies) > self.prev_energy:
                    
                    #when the new energy is greater than prev energy, reduce step size and try again
                    step_size *= 0.1
                    energies, mesh_debug = self.optimize_mesh(step_size)
                    if np.sum(energies) > self.prev_energy:
                        self.undo_step()
                    print(i, np.sum(energies), 'too high')
                    
                print(i, np.sum(energies))
                self.prev_energy = np.sum(energies)
                plt.scatter(i, np.sum(energies))

        plt.show()
        
    def reoptimize(self, mpts_new):
        '''
        This is the function that allows you to redo optimization with a new set of mpts. You can also
        restart optimization with your older mpts. Sometimes it helps to perform optimization multiple times. 
        '''

        self.barycentric_representation = mpts_new
        self.global_forces = self.recalculate_global_forces()
        self.momentum = np.zeros_like(self.momentum)

        num_of_iters = 250
        step_size = 0.01

        for i in range(num_of_iters):

            step_size *= 1.1
            if step_size > 1:
                step_size = 1

            if self.prev_energy == None or i < 50:

                energies, mesh_debug = self.optimize_mesh(step_size)
                print(i, np.sum(energies))
                self.prev_energy = np.sum(energies)
                plt.scatter(i, np.sum(energies))

            else:
                
                energies, mesh_debug = self.optimize_mesh(step_size)
                
                while np.sum(energies) > self.prev_energy:
                    
                    step_size *= 0.1
                    energies, mesh_debug = self.optimize_mesh(step_size)
                    if np.sum(energies) > self.prev_energy:
                        self.undo_step()
                    print(i, np.sum(energies), 'too high')
                    #print(self.prev_energy)
                    
                print(i, np.sum(energies))
                self.prev_energy = np.sum(energies)
                plt.scatter(i, np.sum(energies))

        plt.show()

#for the sake of parallelism, the following script can be used
def opti_mesh(x):
    '''a function that updates mesh'''
    
    mesh, global_forces, step_size = x

    ind = mesh.index
    energy, new_mesh = mesh.energy_minimization(global_forces[ind], step_size)

    return mesh, energy

if __name__ == '__main__':

    print(mp.cpu_count(), 'number of cpu cores available')
    #ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1)) #for use in clusters
    #print(ncpus)

    start = 0 #start index of images
    end = 200 #end index of images
    depth = 1 #the radius of image-to-image connections (depth = 2 means image i is connected to i-2, i-1, i+1 and i+2)
    num_of_nodes = 100 #how many nodes we split image into
    nbhd_size = 100 #the size of neighbourhoods during matching point calculation
    mesh_thresh = 100 #the peak displacement during cross correlation filtering
    pr = 1.7 #the peak ratio during cross correlation filtering
    
    #typically, you'd find the matching points here
    #however, since they were already calculated, we load them
    mpts_dir = '' #directory where you save mpts
    mpts = np.load('mpts.npy', allow_pickle = True)[()]
    

    #this is typically done when your start index doesn't match the start index of your matching points
    pairs = []
    i = start
    while i < end:
    
        if end - i < depth:
            depth -= 1
    
        for j in range(1, depth + 1):
            pairs.append((i, i + j))
    
        i += 1
      
    new_mpts = {}
    
    for pair in pairs:
        
        new_mpts[pair] = mpts[pair]
        
    mpts = new_mpts
    
    arr = [i for i in range(start, end + 1)]
    #create optimizer object, calculate mpts in barycentric coords
    opt = Optimizer(arr, mpts, start)
    bary1 = opt.find_bary(mpts)
    
    num_of_iters = 125
    step_size = 0.01
    processes = int(mp.cpu_count())
    prev_output = None
    min_step = 0.01

    begin = time.time()

    #this is just the optimize mesh script written out in a way that can be done in parallel
    
    for i in range(num_of_iters):
        
        if step_size < min_step:
            
            break
        
        step_size *= 1.1
        if step_size > 1:
            step_size = 1
            
        if opt.prev_energy == None or i < 50:
            
            p = mp.Pool(processes)
            #perform mesh optimization of each mesh, calculate forces after each mesh has finished moving
            output = p.map(opti_mesh, [(mesh, opt.global_forces, step_size) for mesh in opt.meshes])
            output = np.array(output)
            opt.meshes = output[:,0]
            energies = output[:,1]
            opt.global_forces = opt.recalculate_global_forces()
            print(i, np.sum(energies))
            opt.prev_energy = np.sum(energies)
            prev_output = output.copy()
            plt.scatter(i, np.sum(energies))
             
        else:
            
            p = mp.Pool(processes)
            #perform mesh optimization of each mesh, calculate forces after each mesh has finished moving
            output = p.map(opti_mesh, [(mesh, opt.global_forces, step_size) for mesh in opt.meshes])
            output = np.array(output)
            
            if np.sum(output[:,1]) < opt.prev_energy:
                
                opt.meshes = output[:,0]
                energies = output[:,1]
                opt.global_forces = opt.recalculate_global_forces()
                print(i, np.sum(energies))
                opt.prev_energy = np.sum(energies)
                prev_output = output.copy()
                plt.scatter(i, np.sum(energies))
                
            else:
                
                while np.sum(output[:,1]) >= opt.prev_energy:
                    
                    if step_size < min_step:
                        
                        output = prev_output.copy()
                        
                        break
                    
                    step_size *= 0.1
                    p = mp.Pool(processes)
                    output = p.map(opti_mesh, [(mesh, opt.global_forces, step_size) for mesh in opt.meshes])
                    output = np.array(output)
                    print(i, np.sum(output[:,1]), 'too high')
                        
                opt.meshes = output[:,0]
                energies = output[:,1]
                opt.global_forces = opt.recalculate_global_forces()
                print(i, np.sum(energies))
                opt.prev_energy = np.sum(energies)
                prev_output = output.copy()
                plt.scatter(i, np.sum(energies))
      
    plt.show()
    
    ending = time.time()
    print(ending-begin, 'runtime in s')
    
    post_mpts = opt.bary_to_coords(bary1) #returns matching points post relaxation
    
    relaxed_mesh = {} #saving mesh information into dictionary
    
    for i in range(len(opt.meshes)):
    
        nodes = opt.meshes[i].mesh_nodes
        relaxed_mesh[i] = np.array([opt.meshes[i].mesh_nodes, opt.meshes[i].tri.points])
        
    #saving into npy files
    
    np.save('post_mpts_0_200.npy', post_mpts)
    np.save('relaxed_mesh_0_200.npy', relaxed_mesh)