#!/usr/bin/env python3
# Implement Sampler class based on the DeepSDF article description

import os
import trimesh
import math
import numpy as np
from trimesh.sample import sample_surface_sphere

class MeshSampler:
    ''' 
    Class that performs sampling from the given mesh.

    Performs sampling 3 times.
    Two bigger sets are formed after sampling near the mesh surface performing 
    Gaussian noise on the points from the mesh faces. These sets make up 95% 
    points.
    Third set contains points distributed uniformly in the bounding cube.
    '''

    def __init__(self, mesh, n_points=10000, sigma1=0.005**0.5, sigma2=0.005**0.5/10, bounding_cube_dim=2):
        self.mesh = mesh
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sampled_points = None
        self.bounding_cube_dim = bounding_cube_dim
        self.n_points = n_points
    
    def sample_points(self):
        ''' Get points near the object.
        '''
        n1 = n2 = int(self.n_points / 2.1)
        n3 = self.n_points - n1 - n2
        all_points = (
            self.sample_near_surface(n1, self.sigma1),
            self.sample_near_surface(n2, self.sigma2),
            self.sample_uniform(n3)
        )
        self.sampled_points = np.concatenate(all_points, axis=0)

    def sample_near_surface(self, n_points, sigma):
        points, faces = trimesh.sample.sample_surface(self.mesh, count=n_points)
        noisy_points = self._add_noise(points, sigma)
        return noisy_points

    def sample_uniform(self, n_points):
        low = - self.bounding_cube_dim / 2
        high = -low
        uniform_points = np.random.uniform(low, high, (n_points, 3))
        return uniform_points
    
    def _add_noise(self, points, sigma):
        noise = np.random.normal(0, sigma, points.shape)
        noisy_points = points + noise
        return noisy_points

    def compute_sdf(self):
        ''' Return SDF values for sampled points.
        '''
        sdf = self.mesh.nearest.signed_distance(self.sampled_points)
        sdf = sdf.reshape(-1, 1)
        sdf = -sdf  # trimesh inverts SDF

        # Filter our nan values
        mask = np.any(np.isnan(sdf), axis=1)
        points = self.sampled_points[~mask]
        sdf = sdf[~mask]

        return points, sdf
