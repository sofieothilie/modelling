# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:36:51 2024

@author: kasperah
"""

import numpy as np
import scipy

def createSleipnerModel(nz, waterColumn, vel_reservoir, basementDepth, vel_basement):
    """
    Creates a velocity model for the Sleipner model

    Parameters:
    - waterColumn (int):  the depth until the model is reached (assuming its submerged in water).
    - nz (int): Nr of depth increments.
    - reservoir_mat: indicates the material inside the reservoir.
    
    Returns:
    - vel_matrix: a 3D matrix specifying the layer velocities.
    - vel_matrix_2D: a 2D matrix where the z-dimension is summed.
    """
    # sys.path.insert(0, 'C:/Users/Kaspe/OneDrive - NTNU/Documents/Post-doc, NTNU/CO2 Lab/Models/Sleipner/')# Desktop
    # sys.path.insert(0, 'C:/Users/kasperah/OneDrive - NTNU/Documents/Post-doc, NTNU/CO2 Lab/Models/Sleipner/') # Laptop
    import getpass
    username = getpass.getuser()
    mat_file = f'C:/Users/{username}/OneDrive - NTNU/Documents/Post-doc, NTNU/CO2 Lab/Models/Sleipner/zzz.mat' # Desktop
    data = scipy.io.loadmat(mat_file)
    modelThickness = 0.2
    caprockDepth = data['zzz'] + modelThickness
    caprockDepth = np.flip(caprockDepth)
    [nx,ny] = np.shape(caprockDepth)
    # extent_x = 3 # 3 meters in x-direction
    # extent_y = 1 # 1 meter in y-direction
    # dx = extent_x/nx
    # dy = extent_y/ny
    dz = np.max(caprockDepth)/nz # Step in z-direction
    vel_matrix_model = np.zeros([nx,ny,nz])
    vel_plastic = 2607
    vel_water = 1500
    
    
    # Add the water column above the model
    nz_w = int(waterColumn/dz)
    vel_matrix_waterColumn = np.ones([nx,ny,nz_w])*vel_water
    
    # Create the plastic model matrix
    for xi in range(nx):
        for yi in range(ny):
            for zi in range(nz):
                depth_i = zi*dz
                if depth_i < caprockDepth[xi,yi]:
                    vel_matrix_model[xi,yi,zi] = vel_plastic
                else:
                    vel_matrix_model[xi,yi,zi] = vel_reservoir
    
    # Add the basement
    nz_b = nz_w = int(basementDepth/dz) # Add a basement below the model
    vel_matrix_basement = np.ones([nx,ny,nz_b])*vel_basement
    
    # Add the basement
    
    
    # Stack them all together
    vel_matrix = np.concatenate((vel_matrix_waterColumn,vel_matrix_model,vel_matrix_basement),axis=2)
    
    # Create a 2D birds view representation
    vel_matrix_2D = np.sum(vel_matrix,axis=2)
    
    depth = np.shape(vel_matrix)[2]*dz
    return vel_matrix, vel_matrix_2D, depth