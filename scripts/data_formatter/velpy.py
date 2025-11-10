# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:53:24 2024

@author: kasperah
"""
import scipy
import math
import sys
import numpy as np
from scipy.stats import iqr
import segyio
from shutil import copyfile
import matplotlib
# matplotlib.use('QtAgg')
from matplotlib import pyplot as plt
import time
import multiprocessing as mp
from collections import deque
import pickle
from scipy.interpolate import griddata
from scipy.ndimage import generic_filter, gaussian_filter, sobel



# def horizontal_migration(injection_sim, injection_point, vel_reservoir, vel_CO2, vel_caprock):
    
class Velocity_model:
    def __init__(self):
        self.vel_matrix = None # 3D matrix with velocity in each voxel
        self.extent = None # The spatial extent of the model in 3D
        self.spatial_arrs = None # 3D matrix with spatial position, corresponding to vel_matrix
        self.vel_matrix_rms = None # RMS values of vel_matrix
        self.vel_caprock = 2607 # Speed of sound in the caprock (plastic)
        self.vel_reservoir = 1500 # Speed of sound in the reservoir (1500 is default for water)
        self.vel_CO2 = 300 # Speed of sound in air
        self.nvx_reservoir = None # Number of voxels within the reservoir
        self.vx_vol = None # Volume of each voxel (for volume calculations)

   
    def retrieveSleipnerModel(self, nz, waterColumn, vel_reservoir, basementDepth, vel_basement):
        """
        The function retreives the caprock-reservoir interface, described by the matlab file 'zzz.mat', and creates a 3D matrix based on it. In addition, it adds a layer on top of the mode (water column), and a layer below (basement). 
        The values inside the matrix represent the speed of sound in the various media, which isa suitable description of media for acoustical purposes.

        Parameters
        ----------
        nz : TYPE
            Number of pixels in the depth direction.
        waterColumn : TYPE
            Thickness (in m) of water above the caprock layer.
        vel_reservoir : TYPE
            Velocity of sound in the reservoir (set to slightly below water velocity).
        basementDepth : TYPE
            Depth of the model below the reservoir.
        vel_basement : TYPE
            Speed of sound in the basement - cannot be identical to the reservoir (overlap in algorithms).

        Returns
        -------
        None.

        """
        sys.path.insert(0,'C:/Users/kasperah/OneDrive - NTNU/Documents/Post-doc, NTNU/CO2 Lab/Models/Sleipner')
        sys.path.insert(0,'C:/Users/Verasonics/OneDrive - NTNU/Documents/Post-doc, NTNU/CO2 Lab/Models/Sleipner')
        from createSleipnerModel import createSleipnerModel     
        print('Retrieving model...')
        self.vel_reservoir = vel_reservoir
        self.vel_matrix,_,depth = createSleipnerModel(nz,waterColumn,vel_reservoir, basementDepth, vel_basement)
        self.extent = (3, 1, depth)
        
        # Calculate spatial matrix
        x_arr = np.linspace(0,self.extent[0], len(self.vel_matrix[:,0,0]))
        y_arr = np.linspace(0,self.extent[1], len(self.vel_matrix[0,:,0]))
        z_arr = np.linspace(0,self.extent[2], len(self.vel_matrix[0,0,:]))
        self.spatial_arrs = [x_arr,y_arr,z_arr]
        
        # Calculate rms velocities in the time domain (in spatial dimensions)
        print('Calculating RMS values...')
        self.vel_matrix_rms = np.zeros_like(self.vel_matrix)
        z = self.spatial_arrs[2]
        dz = z[1] - z[0]
        # Calculate the rms velocities
        for xi in range(np.shape(self.vel_matrix)[0]):
            for yi in range(np.shape(self.vel_matrix)[1]):
                for zi in range(1,np.shape(self.vel_matrix)[2]):
                    self.vel_matrix_rms[xi,yi,zi] = z[zi]**(-1)*(dz*self.vel_matrix[xi,yi,zi] + z[zi-1]*self.vel_matrix_rms[xi,yi,zi-1])
        self.vel_matrix_rms[:,:,0] = self.vel_matrix_rms[:,:,1] # Set first element, which is zero, to the second element
        print('Done')
    
    def retrieveRMSVelAtPoint(self,x,y,z):
        # Retrieves the velocity information at point (x,y,z)
        index_x = np.argmin(np.abs(self.spatial_arrs[0] - x))
        index_y = np.argmin(np.abs(self.spatial_arrs[1] - y))
        index_z = np.argmin(np.abs(self.spatial_arrs[2] - z))
        return self.vel_matrix_rms[index_x,index_y,index_z]
    
    def retrieveRMSVelAtTime(self, x, y):
        """
        Compute RMS velocity for a continuous velocity-depth model.
    
        Parameters
        ----------
        depth : array-like
            Depth samples (meters)
        velocity : array-like
            Velocities (m/s) at each depth sample
    
        Returns
        -------
        t : ndarray
            Two-way travel time (s) at each depth
        V_rms : ndarray
            RMS velocity (m/s) at each depth
        """
        # depth = np.array(depth)
        xi = abs(self.spatial_arrs[0] - x).argmin()
        yi = abs(self.spatial_arrs[1] - y).argmin()
        velocity = self.vel_matrix[xi, yi,:]
        
        # Compute incremental dz
        dz = self.spatial_arrs[2][1] - self.spatial_arrs[2][0]
        
        # Compute one-way travel time increments
        dt_one_way = dz / velocity[:-1]
        
        # Two-way time cumulative
        t = 2 * np.cumsum(dt_one_way)
        
        # Compute cumulative RMS velocity
        v_squared = velocity[:-1]**2
        cumulative_integral = np.cumsum(v_squared * (2 * dt_one_way))  # two-way time increment
        V_rms = np.sqrt(cumulative_integral / t)
        
        return t, V_rms
    
    def _travel_time_equation(self, Z, x, y, t0):
        return t0 - (2 * Z) / self.retrieveRMSVelAtPoint(x, y, Z)

    def _find_depth_for_travel_time(self, x0, y0, t0, initial_guess):
        from scipy.optimize import root_scalar
        bracket=[0, 10000]
        a, b = bracket  # Initial bracket
        fa = self._travel_time_equation(a, x0, y0, t0)
        fb = self._travel_time_equation(b, x0, y0, t0)
    
        if fa * fb > 0:
            raise ValueError(f"f(a) and f(b) are: {fa} and {fb}. Adjust the bracket.")
            
        solution = root_scalar(self._travel_time_equation, args=(x0, y0, t0), x0=initial_guess, bracket=bracket)
        return solution.root

    
    def createConstantVelModel(self, nx, ny, nz, extent, c):
        print('Creating model...')
        self.vel_reservoir = c
        self.vel_matrix = np.ones([nx,ny,nz])*c
        self.extent = extent
        
        # Calculate spatial matrix
        x_arr = np.linspace(0,self.extent[0], len(self.vel_matrix[:,0,0]))
        y_arr = np.linspace(0,self.extent[1], len(self.vel_matrix[0,:,0]))
        z_arr = np.linspace(0,self.extent[2], len(self.vel_matrix[0,0,:]))
        self.spatial_arrs = [x_arr,y_arr,z_arr]
        
        # Calculate rms velocities in the time domain (in spatial dimensions)
        print('Calculating RMS values...')
        self.vel_matrix_rms = np.zeros_like(self.vel_matrix)
        z = self.spatial_arrs[2]
        dz = z[1] - z[0]
        # Calculate the rms velocities
        for xi in range(np.shape(self.vel_matrix)[0]):
            for yi in range(np.shape(self.vel_matrix)[1]):
                for zi in range(1,np.shape(self.vel_matrix)[2]):
                    self.vel_matrix_rms[xi,yi,zi] = z[zi]**(-1)*(dz*self.vel_matrix[xi,yi,zi] + z[zi-1]*self.vel_matrix_rms[xi,yi,zi-1])
        self.vel_matrix_rms[:,:,0] = self.vel_matrix_rms[:,:,1] # Set first element, which is zero, to the second element
        print('Done')
        
    
    def plotVelocityModel_2D(self,x,y, CMAP = 'RdBu_r', rms=False, VMIN = 1400, VMAX = 3000, DPI = 300):
        if rms is True:
            data = self.vel_matrix_rms
        elif rms is False:
            data = self.vel_matrix
        fig_extent=(0,self.extent[1],self.extent[2],0)
        
        index_x = np.argmin(np.abs(self.spatial_arrs[0] - x))
        index_y = np.argmin(np.abs(self.spatial_arrs[1] - y))
        
        plt.figure(dpi = DPI)
        plt.imshow(data[index_x,:,:].T,aspect='auto', extent=fig_extent, cmap=CMAP, vmin = VMIN, vmax = VMAX)
        plt.title('X position (m): ' + str(x))
        plt.xlabel('Y distance (m)')
        plt.ylabel('Depth (m)')
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.001)

        fig_extent=(0,self.extent[0],self.extent[2],0)
        plt.figure(dpi = DPI)
        plt.imshow(data[:,index_y,:].T,aspect='auto',  extent=fig_extent, cmap=CMAP, vmin = VMIN, vmax = VMAX)
        plt.title('Y position (m): ' + str(y))
        plt.xlabel('X distance (m)')
        plt.ylabel('Depth (m)')
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.001)

        d = np.linspace(0,self.extent[2],len(data[0,0,:]))
        plt.figure(dpi = DPI)
        plt.plot(d,self.vel_matrix_rms[index_x,index_y,:])
        plt.title('RMS velocity')
        plt.xlabel('Depth (m)')
        plt.show(block=False)
        plt.pause(0.001)

        plt.figure(dpi = DPI)
        plt.plot(d,self.vel_matrix[index_x,index_y,:])
        plt.title('Layer velocity')
        plt.xlabel('Depth (m)')
        plt.show(block=False)
        plt.pause(0.001)
    
    def plotVelocityModel_3D(self, rms=False):
        if rms is False:
            data = self.vel_matrix
        elif rms is True:
            data = self.vel_matrix_rms
        # Create a 2D birds view representation
        vel_matrix_2D = np.sum(data,axis=2)
        fig_extent=(0,self.extent[1],self.extent[0],0)
        plt.figure()
        plt.imshow(vel_matrix_2D,aspect=1, extent=fig_extent)
        plt.ylabel('Distance (m)')
        plt.xlabel('Distance (m)')
        plt.show()
    
    def plotVelocityModel_3D_by_layer(self):
        data = self.vel_matrix
        # Loop over z-axis (axis=2)
        nz = data.shape[2]  # number of layers
        for k in range(nz):
            vel_matrix_2D = data[:, :, k]  # take one layer
            
            fig_extent = (0, self.extent[1], self.extent[0], 0)
            plt.figure()
            plt.imshow(vel_matrix_2D, aspect=1, extent=fig_extent)
            plt.title(f"Layer {k}")
            plt.ylabel("Distance (m)")
            plt.xlabel("Distance (m)")
            plt.colorbar(label="Velocity")  # optional
            plt.show()
    
    def plot_velocity_model_surface_3D(self, view_axis, interpolationF = 1):
        from seispy import plot_surface
        # Difference along Z axis
        matrix = self.vel_matrix.astype(np.float32)
        xlim = np.max(self.spatial_arrs[0])
        ylim = np.max(self.spatial_arrs[1])
        zlim = np.max(self.spatial_arrs[2])
        dx = np.max(self.spatial_arrs[0])/np.shape(matrix)[0]  # depth step size
        dy = np.max(self.spatial_arrs[1])/np.shape(matrix)[1]  # depth step size
        dz = np.max(self.spatial_arrs[2])/np.shape(matrix)[2]  # depth step size
        dm = np.diff(matrix, axis=2)  # shape: (Z-1, Y, X)
        
        threshold = 0
        mask = np.abs(dm) > threshold

        
        # Parameters
        max_interfaces = 2  # how many interfaces to extract
        X, Y, Z1 = mask.shape
        
        # Initialize array to store interface depths
        interfaces = np.full((X, Y, max_interfaces), np.nan)
        
        # Loop through each (x, y) and collect interface z-indices
        for i in range(X):
            for j in range(Y):
                z_hits = np.where(mask[i, j, :])[0]
                if z_hits.size > 0:
                    count = min(z_hits.size, max_interfaces)
                    interfaces[i, j, :count] = z_hits[:count]
        interfaces_m = interfaces * dz  # convert to meters
        
        x_vals = np.arange(X)*dx
        y_vals = np.arange(Y)*dy
        plot_surface(interfaces_m, xlim, ylim, zlim, view_axis, title = 'Interfaces Along Z (Velocity model)', interpolationF = interpolationF)
        
        return x_vals, y_vals, interfaces_m
    
    def gradient_plot(self, INTERFACE=1, interpolation=1, SIGMA=0, SOBEL=False, CMAP='viridis'):
        from scipy.ndimage import zoom, gaussian_filter, sobel
        import numpy as np
        import matplotlib.pyplot as plt
    
        xlim = np.max(self.spatial_arrs[0])
        ylim = np.max(self.spatial_arrs[1])
        
        #### Get the interfaces
        _, _, interfaces_m = self.plot_velocity_model_surface_3D(
            view_axis=(3, -4, 0.8), interpolationF=interpolation
        )
        interface = interfaces_m[:, :, INTERFACE]
        
        #### Apply smoothing
        interface = gaussian_filter(interface, sigma=SIGMA)
    
        #### Find steepness / Steepness profile
        step = (np.max(self.spatial_arrs[0]) - np.min(self.spatial_arrs[0])) / len(self.spatial_arrs[0])
        if SOBEL:
            dx = sobel(interface, axis=1)  # gradient in x-direction
            dy = sobel(interface, axis=0)  # gradient in y-direction
        else:
            dx, dy = np.gradient(interface, step, step)
    
        interface_gradient = np.hypot(dx, dy)
        theta = np.arctan(interface_gradient)              # in radians
        theta_deg = np.degrees(theta)                      # in degrees
        
        #### Find curvature (second derivatives)
        dxx = np.gradient(dx, step, axis=1)
        dyy = np.gradient(dy, step, axis=0)
        dxy = np.gradient(dx, step, axis=0)
    
        # Gaussian curvature (determinant of Hessian)
        K = dxx * dyy - dxy**2
        # Mean curvature (trace of Hessian)
        H = dxx + dyy
    
        # initialize label map
        #  1 = valley (concave-up),  
        # -1 = ridge (concave-down),  
        #  0 = saddle or flat
        label = np.zeros_like(interface, dtype=int)
        
        # elliptical regions only where K > 0
        elliptical = K > 0
        
        # valley if mean curvature positive
        valley = elliptical & (H > 0)
        label[valley] = 1
        
        # ridge if mean curvature negative
        ridge = elliptical & (H < 0)
        label[ridge] = -1
        
        #### Plot steepness
        fig = plt.figure(figsize=(12, 12))
        fig.suptitle('Steepness (degrees)')
        ax = fig.add_subplot(1, 1, 1)
        pos = ax.imshow(theta_deg.T, aspect='equal', cmap=CMAP)
        ax.invert_yaxis()
        plt.colorbar(pos, ax=ax)
    
        #### Plot curvature 
        fig = plt.figure(figsize=(12, 12))
        fig.suptitle('Curvature ')
        ax = fig.add_subplot(1, 1, 1)
        pos = ax.imshow(K.T, aspect='equal', cmap=CMAP)
        ax.invert_yaxis()
        plt.colorbar(pos, ax=ax)
        
        #### Contour plot of interface
        ny, nx = interface.shape
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y)
        plt.figure(figsize=(12, 12))
        n_lines = 30
        CF = plt.contourf(X, Y, interface,
                          levels=n_lines,
                          cmap=CMAP)
        CS = plt.contour(X, Y, interface,
                         levels=n_lines,
                         colors='gray',
                         linewidths=0.1)
        cbar = plt.colorbar(CF)
        cbar.set_label('Height')
        plt.title('Filled Contour Map')
        plt.xlabel('X index')
        plt.ylabel('Y index')
        plt.axis('equal')
        plt.suptitle('Topography contour')
        plt.show()
        
        # Now return curvature as well
        gauss_curv = K
        mean_curv = H
        return interface, theta_deg, gauss_curv, mean_curv


        
    def calculate_volume(self, vel_reservoir, vel_caprock = 2607):
        """
        Given a 3D matrix with sound velocity values (self.vel_matrix), the function iterates through the matrix to find the volume of the reservoir (vel_reservoir). 

        Parameters
        ----------
        vel_reservoir : TYPE
            Speed of sound in the reservoir. Should be around the water velocity initially (1500), and air velocity after injection (300)
        vel_caprock : TYPE, optional
            Speed of sound in the caprock. The default is 2607.

        Returns
        -------
        TYPE
            DESCRIPTION.
        vx_reservoir : TYPE
            Number of voxels within the reservoir.
        vol_reservoir : TYPE
            Volume inside the reservoir (in m3).
        vx_caprock : TYPE
            Number of voxels within the caprock.
        vol_caprock : TYPE
            Volume of the caprock (in m3).

        """
        print('Volume calculations...')
        if self.vel_matrix is None:
            print('Velocity matrix is not created.')
            return
        dx = self.spatial_arrs[0][1] - self.spatial_arrs[0][0]
        dy = self.spatial_arrs[1][1] - self.spatial_arrs[1][0]
        dz = self.spatial_arrs[2][1] - self.spatial_arrs[2][0]
        self.vx_vol = dx*dy*dz # Volume of each voxel
        
        vx_reservoir = 0 # Number of voxels within the reservoir
        vx_caprock = 0
        for xi in range(len(self.spatial_arrs[0])):
            for yi in range(len(self.spatial_arrs[1])):
                for zi in range(len(self.spatial_arrs[2])):
                    if self.vel_matrix[xi,yi,zi] == vel_reservoir: # Check if the voxel is inside the reservoir
                        vx_reservoir += 1
                    if self.vel_matrix[xi,yi,zi] == vel_caprock: # Check if the voxel is inside the reservoir
                        vx_caprock += 1
                        
        vol_reservoir = vx_reservoir*self.vx_vol
        vol_caprock = vx_caprock*self.vx_vol
        self.nvx_reservoir = vx_reservoir
        print('done\n')
        return self.vx_vol, vx_reservoir, vol_reservoir, vx_caprock, vol_caprock           
        
    def simulate_injection(self, frames, injection_point, vel_reservoir, vel_CO2, vel_caprock):
       print('Beginning injection simulation...\n')
       if self.vel_matrix is None:
           print('Velocity matrix is not created.')
           return
       
       [nx,ny,nz] = np.shape(self.vel_matrix)
       downsampling = round(self.nvx_reservoir/frames)
       nvx_ds = round(self.nvx_reservoir/downsampling) # Number of voxels within the reservoir, downsampled
       [x0,y0,z0] = injection_point
       if z0 == 0:
           z0 = 1 # Setting it to zero risk going out of bounds, repeating -1
       injection_sim = np.repeat(self.vel_matrix[np.newaxis,...], nvx_ds, axis=0)
       
       n = 0        
       i = 0 # Number of changed pixels
       [rows,cols,nz] = [nx,ny,nz]
       visited = set()  # To avoid revisiting the same cell

       while z0<nz:
           start_time = time.time()
           #### Make nearest neighbour searches
           queue = deque([(x0, y0, z0)])  # Queue holds (row, col, distance)
           while queue:
               # if any(item[2] != z0 for item in queue): # Check if there are multiple z0 elements
               #     if  not all(queue[i][2] <= queue[i + 1][2] for i in range(len(queue) - 1)): # Check if its ascending 
               #         queue = deque(sorted(queue, key=lambda item: item[2]))
               queue = deque(sorted(queue, key=lambda item: item[2]))
               row, col, z0 = queue.popleft()
               # Skip if out of bounds or already visited
               if (row, col, z0) in visited or not (0 <= row < rows and 0 <= col < cols and 0 <= z0 <= nz):
                   continue
               visited.add((row, col, z0))
               
               C = (n,row, col, z0)
               
               L = (n,row,col-1,z0) # Left of pixel (birdsview)
               R = (n,row,col+1,z0) # Right
               T = (n,row-1,col,z0) # Top (birdsview)
               B = (n,row+1,col,z0) # Bottom 
               BR = (n,row+1,col+1,z0) # Bottom-right
               BL = (n,row+1,col-1,z0) # Bottom-left
               TR = (n,row-1,col+1,z0) # Top-right
               TL = (n,row-1,col-1,z0) # Top-left
               
               U = C[:3] + (z0-1,) # Up (cross-section view)
               UTR = TR[:3] + (z0-1,) 
               UTL = TL[:3] + (z0-1,) 
               UBR = BR[:3] + (z0-1,) 
               UBL = BL[:3] + (z0-1,) 
               UL = L[:3] + (z0-1,) 
               UR = R[:3] + (z0-1,) 
               UB = B[:3] + (z0-1,)
               UT = T[:3] + (z0-1,) 

               # Check if the current cell matches the target value
               if injection_sim[C] == vel_reservoir:
                   if injection_sim[U] != vel_reservoir: # check if cell abve can store
                       # If no cells above or neghbouring can store, current cell is the gas floor
                       injection_sim[C] = vel_CO2 # Change value (inject) at current 
                       i += 1
                       
                       if i % downsampling == 0:
                           if n < nvx_ds-1:
                               # print('Injection saved')
                               n += 1
                               C,L,R,T,B,U = update_targets(n,row,col,z0)
                               curr_time = time.time()-start_time
                               print('Completion: ' + str(round(1000*i/self.nvx_reservoir)/10) + ' %' + ', z: ' + str(z0) + ', Time remaining: ' + str(round((curr_time/i)*self.nvx_reservoir)) + ', Elapsed time: ' + str(round(curr_time)))
                           else:
                               print('Saving stopped, capacity reached')
                               continue
                           if n>0:
                               injection_sim[n,:,:,:] = injection_sim[n-1,:,:,:] # Copy information from previous matrix to current
               
               # Add neighbors to the queue if they are not caprock
               
               # Check if the air can migrate higher first
               up = False
               if injection_sim[U] == vel_reservoir:
                  queue.append(U[1:])  # Up
                  up = True
               if injection_sim[UT] == vel_reservoir:
                   queue.append(UT[1:])  # Up with top-movement ..
                   up = True
               if injection_sim[UB] == vel_reservoir:
                   queue.append(UB[1:])  # Down
                   up = True
               if injection_sim[UL] == vel_reservoir:
                   queue.append(UL[1:])  # Left
                   up = True
               if injection_sim[UR] == vel_reservoir:
                   queue.append(UR[1:])  # Right
                   up = True
               if injection_sim[UTR] == vel_reservoir:
                   queue.append(UTR[1:])  # Right
                   up = True
               if injection_sim[UTL] == vel_reservoir:
                   queue.append(UTL[1:])  # Right
                   up = True
               if injection_sim[UBR] == vel_reservoir:
                   queue.append(UBR[1:])  # Right
                   up = True
               if injection_sim[UBL] == vel_reservoir:
                   queue.append(UBL[1:])  # Right  
                   up = True                      
                   # queue.append((row, col,z0-1, distance + 1))  # Original code
               if up == False: # Only spread horizontally, if the current depth is the global minimum
                   if injection_sim[T] != vel_caprock:
                       queue.append(T[1:])  # Up
                   if injection_sim[B] != vel_caprock:
                       queue.append(B[1:])  # Down
                   if injection_sim[L] != vel_caprock:
                       queue.append(L[1:])  # Left
                   if injection_sim[R] != vel_caprock:
                       queue.append(R[1:])  # Right
                   if injection_sim[TR] != vel_caprock:
                       queue.append(TR[1:])  # Right
                   if injection_sim[TL] != vel_caprock:
                       queue.append(TL[1:])  # Right
                   if injection_sim[BR] != vel_caprock:
                       queue.append(BR[1:])  # Right
                   if injection_sim[BL] != vel_caprock:
                       queue.append(BL[1:])  # Right

           # When the queue is empty and while loop ends, increment depth
           z0 += 1
           
           C,L,R,T,B,U = update_targets(n,row,col,z0)
       return injection_sim 

    def simulate_injection_save_to_folder(self, folderpath, frames, injection_point, vel_reservoir, vel_CO2, vel_caprock, save_states = False, debug = False):
        """
        The function is a simplistic apporach to simulate air injection into a reservoir, where the air is trapped by an impermable caprock layer.
        Essentially, an injection point is selected from where the injection begins, and the pixel in that location has its sound velocity value changed to represent that air has been injected to that pixel.
        Subsequently, all the neighbouring pixels at the same depth (or higher) of the injected pixels are added to the queue of pixels to be injected into afterwards, which forms the iterative while loop.
        The beginning of while loop starts with a sorting of the queue depending on the depth of the pixel, so that the highermost pixels are always injcted into first (as uplift always pulls the air upwards).
        If no neighbouring pixels are eligible for injection, the depth increments one step, and the next while loop iteration begins injection at a lower/deeper point in the reservoir. 
        
        Parameters
        ----------
        folderpath : TYPE
            Folder to save the snapshots.
        frames : TYPE
            Number of snapshots/moments to be saved for creating a video (or something else). Higher amount of frames saves more data and gives more frequent updates..
        injection_point : TYPE
            The point to start the injection. [480,80] is the [x,y] coordinates. 
            The depth of injection should be within the reservoir, so it depends on the water column. If no water column is selected, injection oint can be, for example, 1..
        vel_reservoir : TYPE
            Speed of sound in the reservoir..
        vel_CO2 : TYPE
            Speed of sound in the CO2 substance, in this case air, which is 300 m/s..
        vel_caprock : TYPE
            Speed of sound in the caprock layer (2600 m/s)..
        
        Returns
        -------
        injection_sim : TYPE
            An updated version of the self.vel_matrix, that contains the me velocity matrix, but now with some values changed due to the air injection.
        
        """
        if self.vel_matrix is None:
            print('Velocity matrix is not created.')
            return
        
        [nx,ny,nz] = np.shape(self.vel_matrix)
        downsampling = round(self.nvx_reservoir/frames)
        [x0,y0,z0] = injection_point
        if z0 == 0:
            z0 = 1 # Setting it to zero risk going out of bounds, repeating -1
        injection_sim = self.vel_matrix
        
        n = 0
        i = 0 # Number of changed pixels
        [rows,cols,nz] = [nx,ny,nz]
        visited = set()  # To avoid revisiting the same cell
     
        while z0<nz:
            start_time = time.time()
            #### Make nearest neighbour searches
            queue = deque([(x0, y0, z0)])  # Queue holds (row, col, distance)
            while queue:
                # if any(item[2] != z0 for item in queue): # Check if there are multiple z0 elements
                #     if  not all(queue[i][2] <= queue[i + 1][2] for i in range(len(queue) - 1)): # Check if its ascending 
                #         queue = deque(sorted(queue, key=lambda item: item[2]))
                queue = deque(sorted(queue, key=lambda item: item[2]))
                row, col, z0 = queue.popleft()
                # Skip if out of bounds or already visited
                if (row, col, z0) in visited or not (0 <= row < rows and 0 <= col < cols and 0 <= z0 <= nz):
                    continue
                visited.add((row, col, z0))
                
                C = (row, col, z0)
                
                L = (row,col-1,z0) # Left of pixel (birdsview)
                R = (row,col+1,z0) # Right
                T = (row-1,col,z0) # Top (birdsview)
                B = (row+1,col,z0) # Bottom 
                BR = (row+1,col+1,z0) # Bottom-right
                BL = (row+1,col-1,z0) # Bottom-left
                TR = (row-1,col+1,z0) # Top-right
                TL = (row-1,col-1,z0) # Top-left
                
                U = C[:2] + (z0-1,) # Up (cross-section view)
                UTR = TR[:2] + (z0-1,) 
                UTL = TL[:2] + (z0-1,) 
                UBR = BR[:2] + (z0-1,) 
                UBL = BL[:2] + (z0-1,) 
                UL = L[:2] + (z0-1,) 
                UR = R[:2] + (z0-1,) 
                UB = B[:2] + (z0-1,)
                UT = T[:2] + (z0-1,) 
     
                #### Check if the current cell matches the target value
                if injection_sim[C] == vel_reservoir:
                    if injection_sim[U] != vel_reservoir: # check if cell abve can store
                        # If no cells above or neghbouring can store, current cell is the gas floor
                        injection_sim[C] = vel_CO2 # Change value (inject) at current 
                        i += 1
                        
                        if i % downsampling == 0:
                            vol = math.floor(i*self.vx_vol*1000*10)/10
                            vol_str = f"_vol_{str(vol).replace('.', ',')}"
                            filename = 'injection_sim_' + str(n) + vol_str
                            if save_states is True:
                                np.savez_compressed(folderpath+filename,injection_sim)
                            if debug is True:
                                # Plot results
                                fig = plt.figure(figsize=(12,12))
                                ax = fig.add_subplot(111)
                                fig_extent = (0,self.extent[0],self.extent[2]*4,0)
                                data = injection_sim[:,ny//2,:]
                                # data = np.fliplr(data)
                                im = ax.imshow(data.T,cmap='Blues',extent = fig_extent)
                                ax.set_xlabel('Distance (m)')
                                ax.set_ylabel('Distance (m)')
                                fig.colorbar(im)
                                # ax.invert_xaxis()
                                title = filename
                                ax.set_title(title)
                                plt.suptitle(title)
                            n += 1
                            C,L,R,T,B,U = update_targets2(row,col,z0)
                            curr_time = time.time()-start_time
                            print('Completion: ' + str(round(1000*i/self.nvx_reservoir)/10) + ' %' + ', z: ' + str(z0) + ', Time total: ' + str(round((curr_time/i)*self.nvx_reservoir)) + ', Elapsed time: ' + str(round(curr_time)))
     
                
                # Add neighbors to the queue if they are not caprock
                
                # Check if the air can migrate higher first
                up = False
                if injection_sim[U] == vel_reservoir:
                   queue.append(U)  # Up
                   up = True
                if injection_sim[UT] == vel_reservoir:
                    queue.append(UT)  # Up with top-movement ..
                    up = True
                if injection_sim[UB] == vel_reservoir:
                    queue.append(UB)  # Down
                    up = True
                if injection_sim[UL] == vel_reservoir:
                    queue.append(UL)  # Left
                    up = True
                if injection_sim[UR] == vel_reservoir:
                    queue.append(UR)  # Right
                    up = True
                if injection_sim[UTR] == vel_reservoir:
                    queue.append(UTR)  # Right
                    up = True
                if injection_sim[UTL] == vel_reservoir:
                    queue.append(UTL)  # Right
                    up = True
                if injection_sim[UBR] == vel_reservoir:
                    queue.append(UBR)  # Right
                    up = True
                if injection_sim[UBL] == vel_reservoir:
                    queue.append(UBL)  # Right  
                    up = True                      
                    # queue.append((row, col,z0-1, distance + 1))  # Original code
                if up == False: # Only spread horizontally, if the current depth is the global minimum
                    if injection_sim[T] != vel_caprock:
                        queue.append(T)  # Up
                    if injection_sim[B] != vel_caprock:
                        queue.append(B)  # Down
                    if injection_sim[L] != vel_caprock:
                        queue.append(L)  # Left
                    if injection_sim[R] != vel_caprock:
                        queue.append(R)  # Right
                    if injection_sim[TR] != vel_caprock:
                        queue.append(TR)  # Right
                    if injection_sim[TL] != vel_caprock:
                        queue.append(TL)  # Right
                    if injection_sim[BR] != vel_caprock:
                        queue.append(BR)  # Right
                    if injection_sim[BL] != vel_caprock:
                        queue.append(BL)  # Right
     
            # When the queue is empty and while loop ends, increment depth
            z0 += 1
            
            C,L,R,T,B,U = update_targets2(row,col,z0)
        return injection_sim

    
    
    def save_vel_obj(self, filename):
        # Save the object to a file
        with open(f'{filename}.pkl', 'wb') as file:
            pickle.dump(self, file)
            
    def load_vel_obj(self, filename):
        try:
            with open(filename, 'rb') as file:
                loaded_obj = pickle.load(file)
                # Overwrite all attributes of the current object
                self.__dict__.clear()  # Clear current attributes
                self.__dict__.update(loaded_obj.__dict__)  # Update with loaded object's attributes
        except FileNotFoundError:
            print("The file was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

def update_targets2(row,col,z0):
    C = (row, col, z0)
    
    L = (row,col-1,z0) # Left of pixel (birdsview)
    R = (row,col+1,z0) # Right
    T = (row-1,col,z0) # Top (birdsview)
    B = (row+1,col,z0) # Bottom 
    
    U = C[:2] + (z0-1,) # Up (cross-section view)
    return C,L,R,T,B,U     
    
def update_targets(n,row,col,z0):
    C = (n,row, col, z0)
    
    L = (n,row,col-1,z0) # Left of pixel (birdsview)
    R = (n,row,col+1,z0) # Right
    T = (n,row-1,col,z0) # Top (birdsview)
    B = (n,row+1,col,z0) # Bottom 
    
    U = C[:3] + (z0-1,) # Up (cross-section view)
    return C,L,R,T,B,U
    
def fill_nans_channel(data):
    
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    known_mask = ~np.isnan(data)
    
    # Points with known values
    known_points = np.array([x[known_mask], y[known_mask]]).T
    known_values = data[known_mask]
    
    # Points with missing values
    missing_points = np.array([x[~known_mask], y[~known_mask]]).T
    
    # Interpolate missing values
    filled_data = data.copy()
    filled_data[~known_mask] = griddata(known_points, known_values, missing_points, method='linear')
    
    # Optional fallback to nearest if some still remain NaN
    nan_mask = np.isnan(filled_data)
    if np.any(nan_mask):
        filled_data[nan_mask] = griddata(known_points, known_values,
                                         np.array([x[nan_mask], y[nan_mask]]).T,
                                         method='nearest')
    return filled_data

def nanmean_filter(data, size=3):
    """Replace NaNs with mean of local non-NaN neighbors."""
    def nanmean(values):
        valid = values[~np.isnan(values)]
        return np.mean(valid) if valid.size > 0 else np.nan
    
    return generic_filter(data, nanmean, size=size, mode='mirror')

def vector_to_view_angles(x, y, z):
    # Normalize the vector
    vec = np.array([x, y, z])
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Vector cannot be zero length")
    x, y, z = vec / norm

    # Elevation: angle from xy-plane upward
    elev = np.degrees(np.arcsin(z))

    # Azimuth: angle from x-axis in xy-plane
    azim = np.degrees(np.arctan2(y, x))

    return elev, azim