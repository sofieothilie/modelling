# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:13:37 2024

@author: kasperah
contributed by: Marijn Beugelsdijk

To import:
import sys
from seismic_class import Seismic


"""
import scipy
import math
import sys
import numpy as np
from scipy.stats import iqr
import segyio
from shutil import copyfile
import matplotlib
from matplotlib.widgets import Slider, RadioButtons
# matplotlib.use('QtAgg')
from matplotlib import pyplot as plt
import time
import multiprocessing as mp
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from scipy.signal import spectrogram
import dill 
import pandas as pd
from collections import deque
import pickle
import os
from pathlib import Path
import cv2
from scipy.ndimage import generic_filter
from scipy.signal import hilbert
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter
from copy import deepcopy
# import numba
# from numba import jit, cuda 
import re
import copy 
from joblib import Parallel, delayed
from tqdm import tqdm
import functools
import datetime
import json
import os
import getpass

# Optional: try to use numba if available
try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False

def log_all_methods_to_object(cls):
    """Class decorator to log all method calls into a string in each object."""
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value) and not attr_name.startswith("__"):
            setattr(cls, attr_name, log_function_call_to_object(attr_value))
    return cls

def log_function_call_to_object(func):
    """Decorator to log a function/method call into self.log_str."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Build log entry
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "class": args[0].__class__.__name__ if args else None,
            "function": func.__name__,
            "args": repr(args[1:]) if args else repr(args),  # skip self
            "kwargs": repr(kwargs)
        }

        # Initialize or reset log_str safely
        if args and hasattr(args[0], "__dict__"):
            if not hasattr(args[0], "log_str") or args[0].log_str is None:
                args[0].log_str = ""
            args[0].log_str += json.dumps(log_entry) + "\n"

        # Call the original function
        return func(*args, **kwargs)

    return wrapper



def deringing_worker(args):
    """
    Worker function to perform de-ringing on a single trace.
    Designed for use with multiprocessing.Pool.
    """
    # Unpack all arguments
    trace, t_samples, f_center, tb_us, fit_window_us, extend_fit_us, fs = args
    num_samples = len(trace)

    # Define the ringing model inside the worker
    def ring_model(t, A, alpha, f, phi, off):
        return A * np.exp(-alpha * t) * np.cos(2 * np.pi * f * t + phi) + off

    # Taper the initial pulse
    n_taper = int(tb_us * 1e-6 * fs)
    taper = np.ones_like(trace)
    if n_taper > 0:
        win = (1 - np.cos(np.linspace(0, np.pi, n_taper))) / 2
        taper[:n_taper] = win
    x_blank = trace * taper

    # Define the window for curve fitting
    i0 = int(fit_window_us[0] * 1e-6 * fs)
    i1 = int(fit_window_us[1] * 1e-6 * fs)

    fit_succeeded = False
    if (0 <= i0 < i1 < num_samples) and (np.max(np.abs(x_blank[i0:i1])) > 1e-9):
        t_fit, x_fit = t_samples[i0:i1], x_blank[i0:i1]
        p0 = [np.max(np.abs(x_fit)), 1e4, f_center, 0, 0]
        bounds = ([0, 0, f_center * 0.9, -np.pi, -np.inf], 
                  [np.inf, np.inf, f_center * 1.1, np.pi, np.inf])
        try:
            popt, _ = curve_fit(ring_model, t_fit, x_fit, p0=p0, maxfev=20000, bounds=bounds)
            fit_succeeded = True
        except (RuntimeError, ValueError):
            popt = p0 
    
    if fit_succeeded:
        i1_ext = min(int(extend_fit_us * 1e-6 * fs), num_samples)
        fit_full = ring_model(t_samples, *popt)
        
        taper_sub = np.zeros_like(trace)
        taper_sub[i0:i1_ext] = 1.0
        nt_taper_end = int(0.05 * (i1_ext - i0))
        if nt_taper_end > 1:
            taper_sub[i1_ext - nt_taper_end:i1_ext] = np.cos(np.linspace(0, np.pi / 2, nt_taper_end))**2
        
        return x_blank - fit_full * taper_sub
    else:
        return x_blank
    
        
def matrix_RMS(matrix, window_size):
    """
    Compute the RMS of a signal using a centered window for 2D or 3D matrices.

    Parameters:
    matrix (numpy array): Input signal (2D or 3D array)
    window_size (int): Number of samples for the RMS window

    Returns:
    numpy array: The transformed signal with the same shape as input
    """
    # Check if the input is 2D or 3D
    if matrix.ndim == 2:
        n_traces, n_samples = matrix.shape
        matrix_rms = np.zeros((n_traces, n_samples))
        for i in range(n_traces):
            trace = matrix[i, :]
            squared_signal = trace ** 2
            moving_avg = uniform_filter1d(squared_signal, size=window_size, mode='reflect')
            rms_signal = np.sqrt(moving_avg)
            matrix_rms[i, :] = rms_signal

    elif matrix.ndim == 3:
        n_slices, n_traces, n_samples = matrix.shape
        matrix_rms = np.zeros((n_slices, n_traces, n_samples))
        for k in range(n_slices):
            for i in range(n_traces):
                trace = matrix[k, i, :]
                squared_signal = trace ** 2
                moving_avg = uniform_filter1d(squared_signal, size=window_size, mode='reflect')
                rms_signal = np.sqrt(moving_avg)
                matrix_rms[k, i, :] = rms_signal

    else:
        raise ValueError("Input matrix must be 2D or 3D.")

    matrix_rms = np.nan_to_num(matrix_rms, nan=0.0)
    return matrix_rms

def trace_RMS(trace, window_size):
    """
    Compute the RMS of a 1D signal using a centered window.
    
    Parameters:
    trace (numpy array): Input signal (1D array)
    window_size (int): Number of samples for the RMS window
    
    Returns:
    numpy array: The transformed signal with the same length as input
    """
    squared_signal = trace ** 2
    moving_avg = uniform_filter1d(squared_signal, size=window_size, mode='reflect')
    rms_signal = np.sqrt(moving_avg)
    return rms_signal

def dB(data):
    return 20*np.log10(abs(np.double(data)))

def dB_inv(data):
    return 10**(data/20)

def resize_matrix(matrix, target_shape):
    """
    Resizes a 2D or 3D matrix to a specified target shape.
    - If reducing: uses averaging (for 3D).
    - If expanding: uses interpolation (for 2D and 3D).

    Args:
        matrix (np.array): The input matrix (2D or 3D).
        target_shape (tuple): The target shape.

    Returns:
        np.array: Resized matrix.
    """
    from scipy.interpolate import interpn
    ndim = matrix.ndim

    if ndim == 2:
        # 2D case
        x, y = matrix.shape
        new_x, new_y = target_shape

        # Grids
        orig_grid_x = np.linspace(0, x - 1, x)
        orig_grid_y = np.linspace(0, y - 1, y)
        new_grid_x = np.linspace(0, x - 1, new_x)
        new_grid_y = np.linspace(0, y - 1, new_y)
        new_xx, new_yy = np.meshgrid(new_grid_x, new_grid_y, indexing="ij")

        # Interpolate
        resized_matrix = interpn((orig_grid_x, orig_grid_y), matrix,
                                 (new_xx, new_yy), method="linear",
                                 bounds_error=False, fill_value=None)
        return resized_matrix

    elif ndim == 3:
        # 3D case
        x, y, z = matrix.shape
        new_x, new_y, new_z = target_shape

        if new_x <= x and new_y <= y and new_z <= z:
            return reduce_matrix_to_shape(matrix, target_shape)
        else:
            orig_grid_x = np.linspace(0, x - 1, x)
            orig_grid_y = np.linspace(0, y - 1, y)
            orig_grid_z = np.linspace(0, z - 1, z)
            new_grid_x = np.linspace(0, x - 1, new_x)
            new_grid_y = np.linspace(0, y - 1, new_y)
            new_grid_z = np.linspace(0, z - 1, new_z)
            new_xx, new_yy, new_zz = np.meshgrid(new_grid_x, new_grid_y, new_grid_z, indexing="ij")

            resized_matrix = interpn((orig_grid_x, orig_grid_y, orig_grid_z), matrix,
                                     (new_xx, new_yy, new_zz), method="linear",
                                     bounds_error=False, fill_value=None)
            return resized_matrix
    else:
        raise ValueError("Input matrix must be 2D or 3D.")


def reduce_matrix_with_averaging(matrix, block_size):
    """
    Reduces a 3D matrix by averaging values within blocks.
    
    Args:
        matrix (np.array): The input 3D matrix (x, y, z).
        block_size (tuple): The block size (dx, dy, dz) to average over, for each dimension.
        
    Returns:
        np.array: The reduced matrix after averaging.
    """
    dx, dy, dz = block_size
    
    # Get the shape of the matrix
    x, y, z = matrix.shape
    
    # Ensure the matrix dimensions are divisible by the block size
    if x % dx != 0 or y % dy != 0 or z % dz != 0:
        raise ValueError("Matrix dimensions must be divisible by the block size in each dimension.")
    
    # Reshape and average over the blocks
    reduced_matrix = matrix.reshape(x//dx, dx, y//dy, dy, z//dz, dz).mean(axis=(1, 3, 5))
    
    return reduced_matrix

def reduce_matrix_to_shape(matrix, target_shape):
    """
    

    Parameters.
    ----------
    matrix : TYPE
        The input 3D matrix (x, y, z).
    target_shape : TYPE
        The target shape (new_x, new_y, new_z) for the reduced matrix.

    Returns
    -------
    reduced_matrix : np.array
        The reduced matrix after averaging.

    """
    # Get the shape of the input matrix
    x, y, z = matrix.shape
    new_x, new_y, new_z = target_shape
    
    # Create the reduced matrix
    reduced_matrix = np.zeros(target_shape)
    
    # Calculate the size of each block
    step_x = x / new_x
    step_y = y / new_y
    step_z = z / new_z
    
    # Iterate over the new matrix and fill in the values by averaging corresponding blocks in the original matrix
    for i in range(new_x):
        for j in range(new_y):
            for k in range(new_z):
                # Determine the start and end indices for each block in the original matrix
                x_start = int(np.floor(i * step_x))
                x_end = min(int(np.ceil((i + 1) * step_x)), x)  # Ensure we don't exceed the matrix dimensions
                y_start = int(np.floor(j * step_y))
                y_end = min(int(np.ceil((j + 1) * step_y)), y)
                z_start = int(np.floor(k * step_z))
                z_end = min(int(np.ceil((k + 1) * step_z)), z)
                
                # Average the values within the block, but avoid empty blocks
                if x_end > x_start and y_end > y_start and z_end > z_start:
                    block = matrix[x_start:x_end, y_start:y_end, z_start:z_end]
                    reduced_matrix[i, j, k] = np.mean(block)
    
    return reduced_matrix

def cosine_taper(angle, cutoff, max_angle):
    if angle <= cutoff:
        return 1  # No taper for angles below the cutoff
    elif angle >= max_angle:
        return 0  # Complete mute for angles above max_angle
    else:
        # Apply cosine taper for angles between cutoff and max_angle
        return 0.5 * (1 + np.cos(np.pi * (angle - cutoff) / (max_angle - cutoff)))
    

def cosine_taper_optimized(angle, cutoff, max_angle):
    taper = np.ones_like(angle)
    # Apply cosine taper for angles between cutoff and max_angle
    taper[(angle > cutoff) & (angle < max_angle)] = 0.5 * (1 + np.cos(np.pi * (angle[(angle > cutoff) & (angle < max_angle)] - cutoff) / (max_angle - cutoff)))
    taper[angle >= max_angle] = 0
    return taper


def plot_cosine_taper():
    # Parameters for cosine taper
    cutoff_angle = 0  # cutoff angle in degrees
    max_angle = 15     # maximum angle in degrees for which to apply the taper
    
    # Generate synthetic angles from 0 to max_angle
    angles = np.linspace(-max_angle, max_angle, 100)  # angles from 0° to max_angle
    # Apply the tapering to each angle
    taper_values = np.array([cosine_taper(angle, cutoff_angle, max_angle) for angle in angles])
    
    # Example amplitudes before tapering (e.g., constant amplitude of 1 for demonstration)
    original_amplitudes = np.ones_like(angles)
    
    # Calculate the tapered amplitudes
    tapered_amplitudes = original_amplitudes * taper_values
    
    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(angles, original_amplitudes, label='Original Amplitude', linestyle='--', color='blue')
    plt.plot(angles, tapered_amplitudes, label='Tapered Amplitude', color='red')
    plt.axvline(cutoff_angle, color='black', linestyle=':', label=f'Cutoff Angle = {cutoff_angle}°')
    plt.axvline(max_angle, color='gray', linestyle=':', label=f'Max Angle = {max_angle}°')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Amplitude')
    plt.title('Cosine Tapering on High-Angle Events')
    plt.legend()
    plt.grid(True)
    plt.show()

def migration_worker_MP(args):
    """
    Function that performs one iteration of Kirchov migration. It assumes:
        - a very small emission angle
        - RMS velocity of zero-offset 

    Parameters
    ----------
    args : TYPE
        DESCRIPTION.

    Returns
    -------
    img : TYPE
        DESCRIPTION.

    """
    data, Nx, Ny, Nz, dx, dy, x_0, y_0, dz, i, x_S, x_R, y_S, y_R, z_R, z_S, aperture, maxAngle, dt, Nt, cutOffAngle, img_vel = args
    img = np.zeros([Nx,Ny,Nz]) # Migrated image
    off = np.sqrt((x_R - x_S)**2 + (y_R - y_S)**2) # Offset
    d_to_rad = 180 / np.pi
    sin_maxAngle = np.sin(maxAngle*np.pi/180) # Constant for angle check
    x_MP = (x_S + x_R) / 2 # x-midpoint
    y_MP = (y_S + y_R) / 2 # y-midpoint
    z_MP = (z_S + z_R) / 2 # z-midpoint
    
    if abs(off) < aperture:
        for i in range(0,Nx): # Iterate of the x-pixels 
            x = dx*i + x_0
            h_x = x_MP - x # Horizontal distance to pixel, x
            for j in range(0,Ny): # Iterate of the y-pixels 
                y = dy*j + y_0
                h_y = y_MP - y # Horizontal distance to pixel, y
                h = np.sqrt(h_x**2 + h_y**2) #  Horizontal distance to pixel from source 
                for k in range(0,Nz):  # Iterate of the z-pixels 
                    z = dz*k # Current depth
                    h_z = z - z_MP # Vertical distance to pixel, z
                    l_s = np.sqrt(h_z**2 + h**2) # Distance to pixel from source
                    sin_theta = h/(l_s + 0.0000001) # Angle for tapering. The small added number is to prevent division by zero.
                    if sin_theta<sin_maxAngle: # Stop if angle is too high (it should have zero amplitude anyways)
                        vel_rms = img_vel[i,j,k] # RMS velocity at this pixel
                        l_r = np.sqrt((z_MP-z)**2 + (x_MP-x)**2 + (y_MP-y)**2) # Distance to pixel from receiver
                        ti = (l_s+l_r)/vel_rms # Two-way traveltime 
                        it = int(ti/dt) # Two-way traveltime, but converted to the index in the trace
                        if(it < Nt):
                            G = (vel_rms**2)*ti # Correction factor due to spherical spreading
                            # Contribution from this specific source to this (i,j,k) pixel
                            theta_d = np.abs(np.arcsin(sin_theta)) * d_to_rad # In degrees
                            img[i,j,k] = cosine_taper(theta_d,cutOffAngle,maxAngle)*data[it]*G + img[i,j,k] 
    return img

def migration_worker(args):
    """
    Function that performs one iteration of Kirchov migration. It assumes:
        - a very small emission angle
        - RMS velocity of zero-offset 

    Parameters
    ----------
    args : TYPE
        DESCRIPTION.

    Returns
    -------
    img : TYPE
        DESCRIPTION.

    """
    data, Nx, Ny, Nz, dx, dy, x_0, y_0, dz, i, x_S, x_R, y_S, y_R, z_R, z_S, aperture, maxAngle, dt, Nt, cutOffAngle, img_vel = args
    img = np.zeros([Nx,Ny,Nz]) # Migrated image
    off = np.sqrt((x_R - x_S)**2 + (y_R - y_S)**2) # Offset
    rad_to_d = 180 / np.pi
    sin_maxAngle = np.sin(maxAngle*np.pi/180) # Constant for angle check
    
    if abs(off) < aperture:
        for i in range(0,Nx): # Iterate of the x-pixels 
            x = dx*i + x_0
            h_x = x_S - x # Horizontal distance to pixel, x
            for j in range(0,Ny): # Iterate of the y-pixels 
                y = dy*j + y_0
                h_y = y_S - y # Horizontal distance to pixel, y
                h = np.sqrt(h_x**2 + h_y**2) #  Horizontal distance to pixel from source 
                for k in range(0,Nz):  # Iterate of the z-pixels 
                    z = dz*k # Current depth
                    h_z = z - z_S # Vertical distance to pixel, z
                    l_s = np.sqrt(h_z**2 + h**2) # Distance to pixel from source
                    sin_theta = h/(l_s + 0.0000001) # Angle for tapering. The small added number is to prevent division by zero.
                    if sin_theta<sin_maxAngle: # Stop if angle is too high (it should have zero amplitude anyways)
                        vel_rms = img_vel[i,j,k] # RMS velocity at this pixel
                        l_r = np.sqrt((z_R-z)**2 + (x_R-x)**2 + (y_R-y)**2) # Distance to pixel from receiver
                        ti = (l_s+l_r)/vel_rms # Two-way traveltime 
                        it = int(ti/dt) # Two-way traveltime, but converted to the index in the trace
                        if(it < Nt):
                            G = (vel_rms**2)*ti # Correction factor due to spherical spreading
                            # Contribution from this specific source to this (i,j,k) pixel
                            theta_d = np.abs(np.arcsin(sin_theta)) * rad_to_d # In degrees
                            img[i,j,k] = cosine_taper(theta_d,cutOffAngle,maxAngle)*data[it]*G + img[i,j,k] 
    return img

def migration_worker_2D(args):
    """
    Function that performs one iteration of Kirchov migration. It assumes:
        - a very small emission angle
        - RMS velocity of zero-offset 

    Parameters
    ----------
    args : TYPE
        DESCRIPTION.

    Returns
    -------
    img : TYPE
        DESCRIPTION.

    """
    data, Nx, Nz, dx, x_0, y_0, dz, i, x_S, x_R, z_R, z_S, aperture, maxAngle, dt, Nt, cutOffAngle, img_vel = args
    img = np.zeros([Nx,Nz]) # Migrated image
    off = np.sqrt((x_R - x_S))**2 # Offset
    rad_to_d = 180 / np.pi
    sin_maxAngle = np.sin(maxAngle*np.pi/180) # Constant for angle check
    
    if abs(off) < aperture:
        for i in range(0,Nx): # Iterate of the x-pixels 
            x = dx*i + x_0
            h_x = x_S - x # Horizontal distance to pixel, x
            h = h_x #  Horizontal distance to pixel from source 
            for k in range(0,Nz):  # Iterate of the z-pixels 
                z = dz*k # Current depth
                h_z = z - z_S # Vertical distance to pixel, z
                l_s = np.sqrt(h_z**2 + h**2) # Distance to pixel from source
                sin_theta = h/(l_s + 0.0000001) # Angle for tapering. The small added number is to prevent division by zero.
                if sin_theta<sin_maxAngle: # Stop if angle is too high (it should have zero amplitude anyways)
                    vel_rms = img_vel[i,k] # RMS velocity at this pixel
                    l_r = np.sqrt((z_R-z)**2 + (x_R-x)**2) # Distance to pixel from receiver
                    ti = (l_s+l_r)/vel_rms # Two-way traveltime 
                    it = int(ti/dt) # Two-way traveltime, but converted to the index in the trace
                    if(it < Nt):
                        G = (vel_rms**2)*ti # Correction factor due to spherical spreading
                        # Contribution from this specific source to this (i,j,k) pixel
                        theta_d = np.abs(np.arcsin(sin_theta)) * rad_to_d # In degrees
                        img[i,k] = cosine_taper(theta_d,cutOffAngle,maxAngle)*data[it]*G + img[i,k] 
    return img

def NMOstacking_worker(args):
    """
    Function that performs one iteration of Kirchov migration. It assumes:
        - a very small emission angle
        - RMS velocity of zero-offset 

    Parameters
    ----------
    args : TYPE
        DESCRIPTION.

    Returns
    -------
    img : TYPE
        DESCRIPTION.

    """
    cmp_gather, Nx, Ny, No, Nt, cmpOffset, OffsetAperture, t, dt, vel_rms, i = args
    cmp_stacked = np.zeros([Nx,Ny,Nt]) # Migrated image
    # cmp_gather_NMO = np.zeros([Nx,Ny,No,Nt])
    # for i in range(Nx):
    for j in range(Ny):
        for o in range(No):
            # Normal moveout correction
            if cmpOffset[o] <= OffsetAperture:
                # Vectorized NMO correction
                t_off = cmpOffset[o] / vel_rms[j, o, :]  # shape (Nt,)
                mask = (2 * t_off) < t             # valid NMO condition
                
                # Compute NMO corrected time where valid, else 0
                t_nmo = np.zeros_like(t)
                t_nmo[mask] = np.sqrt(t[mask]**2 - 4 * t_off[mask]**2)
                
                # Convert to sample indices
                it_nmo = (t_nmo / dt).astype(int)
                
                # Clip to valid range
                it_nmo = np.clip(it_nmo, 0, Nt - 1)
                
                # Accumulate CMP stack
                np.add.at(cmp_stacked[i, j], it_nmo, cmp_gather[i, j, o])
                # for it in range(len(t)):
                #     t0 = t[it]
                #     # Compute the NMO corrected travel time
                #     t_off = (cmpOffset[o] / vel_rms[j,o,it]) # Additional traveltime for the offset
                #     if 2*t_off < t0:
                #         t_nmo = np.sqrt(t0**2 - 4*t_off**2)
                #     else:
                #         t_nmo = 0
                        
                #     # Find the corresponding time sample index for NMO corrected time
                #     it_nmo = int(t_nmo / dt)
                   
                #     # If the NMO-corrected time index is within bounds, apply the correction
                #     if it_nmo < len(t):
                #         cmp_stacked[i,j,it_nmo] += cmp_gather[i,j,o,it]
                #         # cmp_gather_NMO[i,j,o,it_nmo] += cmp_gather[i,j,o,it]
                # # self.cmp_stacked[i,j,:] += self.cmp_gather[i,j,o,:]

    return cmp_stacked

def save_seis_object(seis, filename):
    with open(filename + ".pkl", "wb") as f:
        dill.dump(seis, f)
        
def load_seis_object(seis, filename):
    # Load object
    with open(filename, "rb") as f:
        loaded_seis_obj = dill.load(f)
    return loaded_seis_obj

def find_source_elev(seis,d, threshold_percentile,min_height,max_height):
    # First find the zero-offset trace       
    trace_ZO = -1
    for trace in iter(d):
        # print(1)
        sx = seis.header['sourceX'][trace]
        sy = seis.header['sourceY'][trace]
        gx = seis.header['groupX'][trace]
        gy = seis.header['groupY'][trace]
        if sx==gx and sy==gy:
            trace_ZO = trace
    
    # Check if a zero-offset trace was found
    if trace_ZO==-1:
        print('No zero-offset trace found')
        return 0

    # Find the elveation of this offset
    min_idx = (np.abs(seis.header['samples'] - 2*min_height/(1481))).argmin()
    max_idx = (np.abs(seis.header['samples'] - 2*max_height/(1481))).argmin()
    idx = find_trace_peaks(seis.traces[trace_ZO], threshold_percentile, min_idx, max_idx)
    elev = 1481*seis.header['samples'][idx]/2
    
    return elev
    
def find_trace_peaks(trace, threshold_percentile,min_idx, max_idx):
    threshold = np.percentile(np.abs(trace), threshold_percentile)
    # Find all peaks
    peaks, _ = scipy.signal.find_peaks(trace, height=threshold)  # set height to filter small noise peaks
    
    # Exclude peaks too early in the trace
    # start_idx = 250
    # end_idx = 500
    
    middle_peaks = [p for p in peaks if min_idx < p < max_idx]
    
    # Locate the biggest peak in the middle segment
    if middle_peaks:
        middle_peak_amplitudes = [trace[p] for p in middle_peaks]
        max_idx = np.argmax(middle_peak_amplitudes)
        main_peak_index = middle_peaks[max_idx]
        # print(f"Main middle peak found at index {main_peak_index}, amplitude {trace[main_peak_index]}")
        return main_peak_index
    else:
        print("No peak found in the middle segment.")
        return 0
    
def interpolate_data(data, interpolationF = 1, GaussKernel = 0):
    is_2D = (data.ndim == 2)

    # Expand 2D to 3D if needed
    if is_2D:
        data = data[:, None, :]  # shape (nx, 1, nz)
    
    # Interpolation
    if interpolationF > 1:
        print('Interpolating data...\n')
        (nx, ny, nz) = data.shape
        new_shape = (nx * interpolationF, ny * interpolationF, nz)
        data = resize_matrix(data, new_shape)
    
    # Gaussian smoothing (directional)
    if GaussKernel != 0:
        print('Gaussian smoothing...\n')
        # If scalar, expand to tuple
        if np.isscalar(GaussKernel):
            sigma = (GaussKernel,) * data.ndim
        else:
            # Convert list/tuple to tuple of correct length
            sigma = tuple(GaussKernel)
            if len(sigma) != data.ndim:
                raise ValueError(f"GaussKernel must have {data.ndim} elements for this data.")
        
        data = scipy.ndimage.gaussian_filter(data, sigma)
       
    # Squeeze back to 2D if input was 2D
    if is_2D:
        data = data[:, 0, :]
    
    return data

def find_adaptive_indices_around_max(signal, threshold_ratio=0.5):
    # Find the index of the maximum value
    max_index = np.argmax(signal)
    max_value = signal[max_index]

    # Define a threshold based on the maximum value
    threshold = threshold_ratio * max_value

    # Find the left boundary
    left_index = max_index
    while left_index > 0 and signal[left_index] > threshold:
        left_index -= 1
    if signal[left_index] <= threshold:
        left_index += 1

    # Find the right boundary
    right_index = max_index
    while right_index < len(signal) - 1 and signal[right_index] > threshold:
        right_index += 1
    if signal[right_index] <= threshold:
        right_index -= 1

    # Get the indices around the maximum value
    indices_around_max = np.arange(left_index, right_index + 1)

    return indices_around_max    

def create_bandpass_filter(lowcut, highcut, fs, order=4):
    import scipy.signal as signal
    """
    Create a bandpass filter using a Butterworth filter.
    
    Parameters:
    - lowcut: Lower cutoff frequency (Hz)
    - highcut: Upper cutoff frequency (Hz)
    - fs: Sampling frequency (Hz)
    - order: Order of the filter (default=4)
    
    Returns:
    - b, a: Numerator (b) and denominator (a) of the filter's transfer function
    """
    nyquist = 0.5 * fs  # Nyquist frequency (half of the sampling rate)
    low = lowcut / nyquist  # Normalize the lower cutoff frequency
    high = highcut / nyquist  # Normalize the upper cutoff frequency
    
    # Design a Butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    import scipy.signal as signal
    """
    Apply a bandpass filter to the input signal data.
    
    Parameters:
    - data: The input signal data (1D array)
    - lowcut: Lower cutoff frequency (Hz)
    - highcut: Upper cutoff frequency (Hz)
    - fs: Sampling frequency (Hz)
    - order: Order of the filter (default=4)
    
    Returns:
    - filtered_data: The filtered signal
    """
    b, a = create_bandpass_filter(lowcut, highcut, fs, order)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def nanmean_filter(data, size=3):
    """Replace NaNs with mean of local non-NaN neighbors."""
    def nanmean(values):
        valid = values[~np.isnan(values)]
        return np.mean(valid) if valid.size > 0 else np.nan
    
    return generic_filter(data, nanmean, size=size, mode='mirror')

def nanmean_filter_expand(data, max_size=20):
    """Iteratively replace NaNs using the mean of nearest non-NaN neighbors, expanding the neighborhood until all NaNs are filled."""
    filled = data.copy()
    nan_mask = np.isnan(filled)
    size = 3  # Start with 3x3 neighborhood
    
    def nanmean(values):
        valid = values[~np.isnan(values)]
        return np.mean(valid) if valid.size > 0 else np.nan

    while np.any(nan_mask) and size <= max_size:
        # Apply the local nan-mean filter
        filtered = generic_filter(filled, nanmean, size=size, mode='mirror')

        # Update only the NaN positions
        filled[nan_mask] = filtered[nan_mask]

        # Recalculate the NaN mask
        nan_mask = np.isnan(filled)

        # If NaNs remain, expand the neighborhood
        size += 2  # Keep size odd: 3, 5, 7, ...

    if np.any(nan_mask):
        print(f"Warning: Some NaNs could not be filled even with neighborhood size {size - 2}.")

    return filled

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

def fit_envelope(signal, t):
    # Extract the envelope using the Hilbert transform
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    
    # Define fitting function
    def exp_decay(t, A, alpha):
        return A * np.exp(-alpha * t)

    # Fit only the envelope
    popt, _ = curve_fit(exp_decay, t, envelope, p0=[1, 1])

    # Generate the fitted curve
    fitted_envelope = exp_decay(t, *popt)
    return envelope, fitted_envelope, popt

def plot_surface(interfaces_m, xlim, ylim, zlim, view_axis = (2,-3,1), title = 'Interfaces Along Z ', interpolationF = 1,):
    from scipy.interpolate import RegularGridInterpolator
    
    interfaces_m = interpolate_data(interfaces_m, interpolationF = interpolationF, GaussKernel = 0)
    
    # Plot everything
    n_x = np.shape(interfaces_m)[0]
    n_y = np.shape(interfaces_m)[1]
    dx = np.max(xlim)/n_x  # depth step size
    dy = np.max(ylim)/n_y  # depth step size
    # dz = MAX_DEPTH/np.shape(data)[2]  # depth step size
    max_interfaces = np.shape(interfaces_m)[2]
    
    x_vals = np.arange(n_x)*dx
    y_vals = np.arange(n_y)*dy
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d')
    
    vmin = np.nanmin(interfaces_m)
    vmax = np.nanmax(interfaces_m)
    for n in range(max_interfaces):
        Z_surface = interfaces_m[:, :, n]
        # Prepare interpolator for smooth Z values
        interpolator = RegularGridInterpolator((x_vals, y_vals), Z_surface)
        surf = ax.plot_surface(
        X_grid, Y_grid, Z_surface,
        cmap='ocean', edgecolor='none', alpha=1,linewidth=0.01,
        vmin = vmin, vmax =vmax,
        label=f'Interface {n+1}',
        antialiased=False  # This is key!
        # ccount  = 200,
        # rcount = 500
        )
        
        # # Plot wireframe only in the X-direction
        # ax.plot_wireframe(
        #     X_grid, Y_grid, Z_surface,
        #     rstride=0,  # draw every row (controls X direction lines)
        #     cstride=1,  # draw no columns (hide Y direction lines)
        #     color='gray',
        #     linewidth=1
        # )
    elev, azim = vector_to_view_angles(view_axis[0],view_axis[1],view_axis[2])
    ax.view_init(elev=elev, azim=azim)  # Change these to set viewing directio
    
    # Set the background color of the panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Distance (m)')
    ax.grid(True)
    # Optionally, change the axes background color
    # fig.set_facecolor('white')  # This changes the background color of the plot area
    # ax.set_facecolor('white')  # This changes the background color of the plot area
    ax.set_zlabel('Depth (m)')
    ax.set_box_aspect((xlim, ylim, 2*zlim))  # If using physical units (Z is exaggerated)
    ax.set_title(title)
    ax.invert_zaxis()
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()
    
def processing_pipeline_debug(data_folder, debug_folder = None):
    # Go through folder and add:
    seis = Seismic()
    files = os.listdir(data_folder)    
    seis_i = Seismic()
    for file in files:
        full_path = data_folder + file
        
        # Check if the file exists
        file_path = Path(full_path).resolve()
        if file_path.exists() and os.path.splitext(file)[1] == '.segy':
            print(f"File {file} exists")
            # If the file exists, add it to a seismic file
            seis_i.read_segy(full_path)
            seis_i.scale_data(2000,'ms')
            seis_i_ZO = seis_i.create_ZO()
            seis_i_ZO.plot_all_traces(DB=False,AMP=False)
            save_all_open_plots(debug_folder, name=file, file_format="png", dpi = 200)
            plt.close('all')
        else:
            print(f"File {file} does not exist.")

def save_all_open_plots(folder_path, name='Data', file_format="png", dpi = 200):
    os.makedirs(folder_path, exist_ok=True)

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        # Get figure suptitle if exists
        title_obj = fig._suptitle
        if title_obj is not None:
            title = title_obj.get_text()
            title = re.sub(r'[<>:"/\\|?*]', '_', title)
        else:
            title = f"{name}_figure_{fig_num}"

        filename = f"{title}.{file_format}"
        filepath = os.path.join(folder_path, filename)
        fig.savefig(filepath, dpi = dpi)
        print(f"Saved: {filepath}")

def read_segy_folder_to_array(data_folder, debug = False, file_limit = None, file_step = None):
    
    # Go through folder and add:
    files = os.listdir(data_folder)
    seis_files = list()
    
    # Only keep every 10th file
    if file_step is not None:
        files = files[::file_step]
            
    count = 0
    for file in files:
        seis_i = Seismic()
        if file_limit is not None:
            if count >= file_limit:
                break
        count += 1
        full_path = data_folder + file
        
        # Check if the file exists
        file_path = Path(full_path).resolve()
        if file_path.exists() and os.path.splitext(file)[1] == '.segy':
            print(f"File {file} exists")
            # If the file exists, add it to a seismic file
            seis_i.read_segy(full_path)

            
            # Add to general seis object
            if debug == True:
                seis_i.plot_header()
                plt.close('all')
                
            # Add seismic file to list
            seis_i.filename = file
            seis_files.append(seis_i)
        else:
            print(f"File {file} does not exist.")
    return seis_files

def compute_nrms_binned(seis_1, seis_2, nx = 400, ny = 50, t0 = 0, t1 = 0.3/1000, z0 = 0, z1 = 0.3, off0 = 0, off1 = None, CMP = False, stacking = False, sorting = False, migration = False, vel_mod = None):
    """
    Compute NRMS for two CMP-binned datasets with same dimensions.
    
    Parameters:
        cmp_gather_1, cmp_gather_2: np.ndarray
            Shape: (x_bin, y_bin, num_offsets, nt)
        t0, t1: float
            Time window (seconds)
        dt: float
            Sample interval (seconds)
    Returns:
        nrms_array: np.ndarray
            Shape: (x_bin, y_bin, num_offsets)
    """

    
   
    if CMP is True: 
        # Limit the time sampling
        dt = seis_1.header['samples'][1] - seis_1.header['samples'][0]
        i0 = int(t0 / dt)
        i1 = int(t1 / dt)
        
        # Sort to CMP points
        if sorting is True:
            seis_1.sort_to_bin(nx,ny,num_offsets=20)
            seis_2.sort_to_bin(nx,ny,num_offsets=20)
        else:
            if seis_1.cmp_gather is None:
                seis_1.sort_to_bin(nx,ny,num_offsets=20)
            if seis_2.cmp_gather is None:
                seis_2.sort_to_bin(nx,ny,num_offsets=20)            
            
        # Limit the max offsets used
        offset = seis_1.cmpOffset
        off0_idx = (np.abs(offset - off0)).argmin()
        if off1 is None:
            off1_idx = len(offset)
        else:
            off1_idx = (np.abs(offset - off1)).argmin()
        
        # Collect grid
        cmp_gather_1 = seis_1.cmp_gather
        cmp_gather_2 = seis_2.cmp_gather
        assert cmp_gather_1.shape == cmp_gather_2.shape, "CMP grids must match"
        x_bin, y_bin, num_offsets, nt = cmp_gather_1.shape
        
        
        if stacking is True:
            # Stack data
            seis_1.NMO_stacking(vel_mod, OffsetAperture = off1, norm = False)
            seis_2.NMO_stacking(vel_mod, OffsetAperture = off1, norm = False)
            
            # Stacked data
            nrms_array = np.full((x_bin, y_bin), np.nan)
            for ix in range(x_bin):
                for iy in range(y_bin):
                        A = seis_1.cmp_stacked[ix, iy, i0:i1]
                        B = seis_2.cmp_stacked[ix, iy, i0:i1]
                        
                        if np.all(A == 0) or np.all(B == 0):
                            continue  # skip empty bins
                        
                        rms_1 = np.sqrt(np.mean(A**2))
                        rms_2 = np.sqrt(np.mean(B**2))
                        rms_diff = np.sqrt(np.mean((A - B) ** 2))
                        mean_rms = rms_1 + rms_2
                        
                        if mean_rms > 0:
                            nrms_array[ix, iy] = 200 * rms_diff / mean_rms
        elif stacking is False:
            # Sorted data w/ offset
            nrms_array = np.full((x_bin, y_bin, num_offsets), np.nan)
            for ix in range(x_bin):
                for iy in range(y_bin):
                    for io in range(off0_idx, off1_idx):
                        A = cmp_gather_1[ix, iy, io, i0:i1]
                        B = cmp_gather_2[ix, iy, io, i0:i1]
                        
                        if np.all(A == 0) or np.all(B == 0):
                            continue  # skip empty bins
                        
                        rms_1 = np.sqrt(np.mean(A**2))
                        rms_2 = np.sqrt(np.mean(B**2))
                        rms_diff = np.sqrt(np.mean((A - B) ** 2))
                        mean_rms = rms_1 + rms_2
                        
                        if mean_rms > 0:
                            nrms_array[ix, iy, io] = 200 * rms_diff / mean_rms
        else:
            print('Sorting is on, but nothing happens...\n')
    
    elif migration is True and CMP is False:
        
        # Check if there are migrated data
        if seis_1.mig_img is None:
            print('No mig img in seis_1.\n')
        if seis_2.mig_img is None:
            print('No mig img in seis_2.\n')
        
        # Find depth starting and end-points
        dz = seis_1.mig_img_SA[2][1] - seis_1.mig_img_SA[2][0]
        i0 = int(z0 / dz)
        i1 = int(z1 / dz)
        
        # Generic data
        x_bin, y_bin, nt = seis_1.mig_img.shape
        nrms_array = np.full((x_bin, y_bin), np.nan)
        for ix in range(x_bin):
            for iy in range(y_bin):
                A = seis_1.mig_img[ix, iy, i0:i1]
                B = seis_2.mig_img[ix, iy, i0:i1]
                
                if np.all(A == 0) or np.all(B == 0):
                    continue  # skip empty bins
                
                rms_1 = np.sqrt(np.mean(A**2))
                rms_2 = np.sqrt(np.mean(B**2))
                rms_diff = np.sqrt(np.mean((A - B) ** 2))
                mean_rms = rms_1 + rms_2
                
                if mean_rms > 0:
                    nrms_array[ix, iy] = 200 * rms_diff / mean_rms
                        
    else:
        print('No calculations were performed...\n')
        nrms_array = None
    return nrms_array
    
def make_comparison_birdview_plots(
    seis_0, seis_1,
    center_depth_m=0.22, thickness_m=0.05,
    method='max_rms', map_type='continuous',
    smoothing_sigma=6, cmap='RdBu_r',
    vmin=None, vmax=None,
    normalize=False, plotting=True,
    match_mode="resize",      # "resize" or "overlap"
    shift_mode="manual",      # "manual" or "auto"
    shift=(0, 0),             # Used if shift_mode="manual"
    max_shift=3               # Search range if shift_mode="auto"
):
    BV_0 = seis_0.plot_map_view(
        center_depth_m=center_depth_m, 
        thickness_m=thickness_m,
        method=method,
        map_type=map_type,
        smoothing_sigma=smoothing_sigma,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    BV_1 = seis_1.plot_map_view(
        center_depth_m=center_depth_m, 
        thickness_m=thickness_m,
        method=method,
        map_type=map_type,
        smoothing_sigma=smoothing_sigma,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    if normalize:
        BV_0 = BV_0 / np.mean(BV_0)
        BV_1 = BV_1 / np.mean(BV_1)

    # --- Handle different sizes ---
    if match_mode == "resize":
        from seispy import resize_matrix
        BV_1 = resize_matrix(BV_1, BV_0.shape)

    elif match_mode == "overlap":
        x0_arr, y0_arr = seis_0.mig_img_SA[0], seis_0.mig_img_SA[1]
        x1_arr, y1_arr = seis_1.mig_img_SA[0], seis_1.mig_img_SA[1]

        x_min, x_max = max(x0_arr.min(), x1_arr.min()), min(x0_arr.max(), x1_arr.max())
        y_min, y_max = max(y0_arr.min(), y1_arr.min()), min(y0_arr.max(), y1_arr.max())

        if x_min >= x_max or y_min >= y_max:
            print("No spatial overlap between datasets.")
            return None

        ix0_min, ix0_max = np.searchsorted(x0_arr, [x_min, x_max])
        iy0_min, iy0_max = np.searchsorted(y0_arr, [y_min, y_max])
        ix1_min, ix1_max = np.searchsorted(x1_arr, [x_min, x_max])
        iy1_min, iy1_max = np.searchsorted(y1_arr, [y_min, y_max])

        BV_0 = BV_0[ix0_min:ix0_max, iy0_min:iy0_max]
        BV_1 = BV_1[ix1_min:ix1_max, iy1_min:iy1_max]

        if BV_0.shape != BV_1.shape:
            from seispy import resize_matrix
            BV_1 = resize_matrix(BV_1, BV_0.shape)

    else:
        raise ValueError("match_mode must be 'resize' or 'overlap'")

    # --- Shifting ---
    def apply_shift(arr, dx, dy):
        arr_shifted = np.roll(arr, shift=dx, axis=0)
        arr_shifted = np.roll(arr_shifted, shift=dy, axis=1)
        # blank wrapped edges
        if dx > 0: arr_shifted[:dx, :] = np.nan
        elif dx < 0: arr_shifted[dx:, :] = np.nan
        if dy > 0: arr_shifted[:, :dy] = np.nan
        elif dy < 0: arr_shifted[:, dy:] = np.nan
        return arr_shifted

    def compute_nrms(a, b):
        mask = ~np.isnan(a) & ~np.isnan(b)
        if np.sum(mask) == 0:
            return np.inf
        a, b = a[mask], b[mask]
        rms_diff = np.sqrt(np.mean((a - b) ** 2))
        rms_sum = np.sqrt(np.mean(a ** 2)) + np.sqrt(np.mean(b ** 2))
        return 200 * rms_diff / rms_sum

    if shift_mode == "manual":
        dx, dy = shift
        BV_1 = apply_shift(BV_1, dx, dy)
        best_shift = (dx, dy)

    elif shift_mode == "auto":
        best_nrms = np.inf
        best_shift = (0, 0)
        for dx in range(-max_shift, max_shift + 1):
            for dy in range(-max_shift, max_shift + 1):
                candidate = apply_shift(BV_1, dx, dy)
                nrms = compute_nrms(candidate, BV_0)
                if nrms < best_nrms:
                    best_nrms = nrms
                    best_shift = (dx, dy)
        BV_1 = apply_shift(BV_1, *best_shift)
        print(f"Best shift: {best_shift}, NRMS={best_nrms:.2f}%")

    else:
        raise ValueError("shift_mode must be 'manual' or 'auto'")

    # --- Difference ---
    diff = BV_1 - BV_0

    if plotting:
        plt.figure(figsize=(12,8))
        ax = plt.gca()
        extent = [seis_0.mig_img_SA[0][0], seis_0.mig_img_SA[0][-1],
                  seis_0.mig_img_SA[1][0-1], seis_0.mig_img_SA[1][0]]
        ax.set_facecolor('white')
        im = ax.imshow(diff.T, cmap=cmap, extent=extent, interpolation='bilinear')
        cbar = plt.colorbar(im, shrink=0.7, pad=0.05)
        cbar.set_label(f"{method}", size=12)
        ax.set_xlabel("Distance (m)", size=12)
        ax.set_ylabel("Distance (m)", size=12)
        ax.set_title(f"Map View ({method}, {map_type}, mode={match_mode}, shift={best_shift})", size=16, pad=20)

    return diff


def filter_by_trans_id_and_offset(seis, trans_id=None, max_offset=None):
    """
    Filter a seismic object in place to keep only traces with the given trans_id
    and within a maximum source-receiver offset.

    Parameters
    ----------
    seis : object
        Seismic object with `header` (dict-like) and `data` (ndarray) attributes.
        Must contain at least 'trans_id', 'sourceX', 'sourceY', 'groupX', 'groupY'.
    trans_id : int, optional
        The source index to keep. If None, no filtering by trans_id.
    max_offset : float, optional
        Maximum allowed source-receiver offset. If None, no offset filtering.
    """
    mask = np.ones(len(seis.header['trans_id']), dtype=bool)

    if trans_id is not None:
        mask &= (seis.header['trans_id'] == trans_id)

    if max_offset is not None:
        sx, sy = seis.header['sourceX'], seis.header['sourceY']
        gx, gy = seis.header['groupX'], seis.header['groupY']
        offset = np.sqrt((sx - gx) ** 2 + (sy - gy) ** 2)
        mask &= (offset <= max_offset)

    # Convert mask to indices
    keep_idx = np.where(mask)[0]

    # Invert to get traces to remove
    all_idx = np.arange(len(mask))
    remove_idx = np.setdiff1d(all_idx, keep_idx)

    # Remove unwanted traces
    seis.remove_traces(remove_idx)
    return seis


def _nmo_time_zero_offset(t0, offset, vel):
    """
    NMO correction: given zero-offset time t0, offset h and RMS velocity vel,
    return the time in the offset trace where that zero-offset event would appear:
        t(h) = sqrt( t0^2 + (2*h/vel)^2 )
    Arguments:
        t0: scalar or array (s)
        offset: scalar offset (m) or array matching t0
        vel: scalar RMS velocity (m/s)
    Returns:
        t_h: scalar or array (s)
    """
    return np.sqrt(np.maximum(0.0, t0**2 + (2.0 * offset / vel)**2))


def semblance_scan(cmp_gather, offsets, t, vel_vec, win_len_samples=7, normalize=True, apply_rms_smoothing=False):
    if cmp_gather.shape[0] == len(t) and cmp_gather.shape[1] == len(offsets):
        cmp_gather = cmp_gather.T  # Ensure shape is (n_offsets, nt)

    n_offsets, nt = cmp_gather.shape
    n_vel = len(vel_vec)
    dt = t[1] - t[0]
    half_w = win_len_samples // 2
    max_it = nt - 1

    semblance = np.zeros((n_vel, nt), dtype=np.float32)

    for iv, vel in enumerate(vel_vec):
        for it0, t0 in enumerate(t):
            th = _nmo_time_zero_offset(t0, offsets, vel)
            it_frac = th / dt
            valid_mask = (it_frac >= 0) & (it_frac <= max_it)

            if not np.any(valid_mask):
                continue

            sum_a, sum_a2, count = 0.0, 0.0, 0

            for k in np.where(valid_mask)[0]:
                idx = it_frac[k]
                i_floor = int(np.floor(idx))
                i0 = max(i_floor - half_w, 0)
                i1 = min(i_floor + half_w + 1, nt)

                samples = cmp_gather[k, i0:i1]
                if samples.size == 0:
                    continue

                sum_a += np.sum(samples)
                sum_a2 += np.sum(samples ** 2)
                count += samples.size

            if count == 0:
                continue

            if normalize:
                denom = count * sum_a2
                s = (sum_a ** 2) / denom if denom > 0 else 0.0
            else:
                s = sum_a2

            semblance[iv, it0] = s

    if apply_rms_smoothing:
        semblance = uniform_filter1d(semblance, size=3, axis=1, mode='nearest')

    return semblance, vel_vec, t


def semblance_scan_NJIT(cmp_gather, offsets, t, vel_vec, win_len_samples=7, normalize=True, apply_rms_smoothing=False):
    if cmp_gather.shape[0] == len(t) and cmp_gather.shape[1] == len(offsets):
        cmp_gather = cmp_gather.T

    semblance = _compute_semblance_core(cmp_gather.astype(np.float32), offsets.astype(np.float32),
                                        t.astype(np.float32), vel_vec.astype(np.float32),
                                        win_len_samples, normalize)

    if apply_rms_smoothing:
        semblance = uniform_filter1d(semblance, size=3, axis=1, mode='nearest')

    return semblance, vel_vec, t


if _NUMBA_AVAILABLE:
    @njit(parallel=True)
    def _compute_semblance_core(cmp_gather, offsets, t, vel_vec, win_len_samples, normalize):
        n_offsets, nt = cmp_gather.shape
        n_vel = len(vel_vec)
        dt = t[1] - t[0]
        half_w = win_len_samples // 2
        max_it = nt - 1
    
        semblance = np.zeros((n_vel, nt), dtype=np.float32)
    
        for iv in prange(n_vel):
            vel = vel_vec[iv]
            for it0 in range(nt):
                t0 = t[it0]
                th = np.sqrt(t0**2 + (offsets / vel)**2)
                it_frac = th / dt
    
                sum_a, sum_a2, count = 0.0, 0.0, 0
    
                for k in range(n_offsets):
                    idx = it_frac[k]
                    if idx < 0 or idx > max_it:
                        continue
    
                    i_floor = int(idx)
                    i0 = max(i_floor - half_w, 0)
                    i1 = min(i_floor + half_w + 1, nt)
    
                    for i in range(i0, i1):
                        val = cmp_gather[k, i]
                        sum_a += val
                        sum_a2 += val * val
                        count += 1
    
                if count == 0:
                    continue
    
                if normalize:
                    denom = count * sum_a2
                    s = (sum_a ** 2) / denom if denom > 0 else 0.0
                else:
                    s = sum_a2
    
                semblance[iv, it0] = s
    
        return semblance
else:
    print('Numba is not available!')


def plot_semblance(semblance, vel_vec, t, vmax=None, vmin=None, cmap='hot',
                   pick_velocities=None, figsize=(10, 6)):
    """
    Plot semblance spectrum, optionally with velocity picks overlaid.
    """
    

    plt.figure(figsize=figsize)
    extent = [t[0], t[-1], vel_vec[0], vel_vec[-1]]
    im = plt.imshow(semblance, origin='lower', extent=extent, aspect='auto',
                    cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Semblance velocity spectrum')
    plt.colorbar(im, label='Semblance')

    if pick_velocities is not None:
        plt.plot(t, pick_velocities, 'c-', linewidth=2, label="Picked velocity")
        plt.legend()

    plt.gca().invert_xaxis()  # optional: comment out if you prefer time increasing right
    plt.show()


def pick_velocities_from_semblance(semblance, vel_vec, t, smooth_window=11, method="argmax"):
    """
    Automatic velocity picking from a semblance spectrum.

    Parameters
    ----------
    semblance : ndarray, shape (n_vel, nt)
        Semblance values (velocity x time).
    vel_vec : ndarray, shape (n_vel,)
        Trial velocities (m/s).
    t : ndarray, shape (nt,)
        Time samples (s).
    smooth_window : int
        Window length (in samples) for median smoothing of velocity picks.
    method : str, "argmax" or "weighted"
        - "argmax": pick the velocity with the maximum semblance at each time.
        - "weighted": compute a semblance-weighted average velocity at each time.

    Returns
    -------
    v_picks : ndarray, shape (nt,)
        Picked velocity function vs. time (m/s).
    """
    if method == "argmax":
        # Pick velocity at maximum semblance
        max_idx = np.argmax(semblance, axis=0)  # index of best vel for each time
        v_picks = vel_vec[max_idx]
    elif method == "weighted":
        # Weighted average by semblance values
        weights = semblance / (semblance.sum(axis=0, keepdims=True) + 1e-10)
        v_picks = np.sum(weights * vel_vec[:, None], axis=0)
    else:
        raise ValueError("method must be 'argmax' or 'weighted'")

    # Apply median filter for smoothness
    if smooth_window > 1:
        v_picks = median_filter(v_picks, size=smooth_window, mode="nearest")

    return v_picks

def build_velocity_volume(seis, vel_min=1200, vel_max=4000, n_vel=60, win_len_samples=30, smooth_window=9, method="argmax",apply_rms_smoothing=True, plot_progress=True, apply_dix=True, parallel = True, n_jobs=-1, numba = True):
    """
    Build a 3D velocity model (nx, ny, nt) by scanning semblance
    and picking stacking velocities at every CMP bin.
    """
    nx, ny, n_off, nt = seis.cmp_gather.shape
    t = np.array(seis.header["samples"])
    dt = t[1] - t[0]

    vel_vec = np.linspace(vel_min, vel_max, n_vel)
    
    offsets = np.array(seis.cmpOffset)
    
    def process_cmp(ix, iy):
        cmp_gather = seis.cmp_gather[ix, iy, :, :]
        if np.allclose(cmp_gather, 0.0):
            return np.zeros(nt, dtype=np.float32)

        # semblance spectrum
        if numba:
            semblance, _, _ = semblance_scan_NJIT(
                cmp_gather, offsets, t, vel_vec,
                win_len_samples=win_len_samples,
                apply_rms_smoothing=apply_rms_smoothing
            )
        else:
            semblance, _, _ = semblance_scan(
                cmp_gather, offsets, t, vel_vec,
                win_len_samples=win_len_samples,
                apply_rms_smoothing=apply_rms_smoothing
            )

        # pick velocity function
        v_picks = pick_velocities_from_semblance(
            semblance, vel_vec, t,
            smooth_window=smooth_window,
            method=method
        )

        return v_picks.astype(np.float32)
    
    if parallel is True:
        # Create a list of all CMP indices
        tasks = [(ix, iy) for ix in range(nx) for iy in range(ny)]
        
        # Use tqdm to wrap the iterable
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_cmp)(ix, iy) for ix, iy in tqdm(tasks, disable=not plot_progress, desc="Processing CMPs")
        )
        vel_model = np.array(results, dtype=np.float32).reshape(nx, ny, nt)
    else:
        vel_model = np.zeros((nx, ny, nt), dtype=np.float32)
    
        for ix in range(nx):
            for iy in range(ny):
                cmp_gather = seis.cmp_gather[ix, iy, :, :]
                if np.allclose(cmp_gather, 0.0):
                    continue  # skip empty bins
     
                offsets = np.array(seis.cmpOffset)
     
                # semblance spectrum
                if numba:
                    semblance, _, _ = semblance_scan_NJIT(
                        cmp_gather, offsets, t, vel_vec,
                        win_len_samples=win_len_samples,
                        apply_rms_smoothing=apply_rms_smoothing
                    )
                else:
                    semblance, _, _ = semblance_scan(
                        cmp_gather, offsets, t, vel_vec,
                        win_len_samples=win_len_samples,
                        apply_rms_smoothing=apply_rms_smoothing
                    )
     
                # pick velocity function
                v_picks = pick_velocities_from_semblance(
                    semblance, vel_vec, t,
                    smooth_window=smooth_window,
                    method=method
                )
     
                vel_model[ix, iy, :] = v_picks
     
                if plot_progress and (ix % 5 == 0 and iy % 5 == 0):
                    print(f"Processed CMP bin ({ix},{iy})")
                    
                    
    # Apply Dix conversion if requested
    if apply_dix:
        dz = np.diff(t, prepend=0)  # shape (nt,)
        t1 = dz[:-1]  # shape (nt-1,)
        t2 = dz[1:]   # shape (nt-1,)
    
        # Expand dims for broadcasting over nx, ny
        t1 = t1[None, None, :]
        t2 = t2[None, None, :]
    
        V_rms_sq = vel_model**2  # shape (nx, ny, nt)
    
        numerator = V_rms_sq[:, :, 1:] * t2 - V_rms_sq[:, :, :-1] * t1
        denominator = t2 - t1
        V_int = np.sqrt(np.maximum(numerator / denominator, 0))
    
        # Add first sample
        V_int = np.concatenate([V_int[:, :, 0:1], V_int], axis=2)
        vel_interval = V_int
    else:
        vel_interval = None

    return vel_model, vel_interval, t


def plot_velocity_volume(vel_model, t, slice_axis="x", slice_index=0, time_index=None, cmap="jet"):
    """
    Visualize slices of the 3D velocity model.

    Parameters
    ----------
    vel_model : ndarray, shape (nx, ny, nt)
        Velocity cube.
    t : ndarray, shape (nt,)
        Time axis (s).
    slice_axis : str, "x" or "y" or "time"
        Which slice to plot.
    slice_index : int
        Index of inline/crossline to plot.
    time_index : int
        For slice_axis="time", which time sample to plot.
    """
    nx, ny, nt = vel_model.shape

    if slice_axis == "x":
        panel = vel_model[slice_index, :, :].T  # shape (nt, ny)
        extent = [0, ny, t[-1], t[0]]
        xlabel = "CMP-Y index"; ylabel = "Time (s)"
    elif slice_axis == "y":
        panel = vel_model[:, slice_index, :].T  # shape (nt, nx)
        extent = [0, nx, t[-1], t[0]]
        xlabel = "CMP-X index"; ylabel = "Time (s)"
    elif slice_axis == "time":
        if time_index is None:
            raise ValueError("Provide time_index for time slice")
        panel = vel_model[:, :, time_index]
        extent = [0, vel_model.shape[1], 0, vel_model.shape[0]]
        xlabel = "CMP-Y index"; ylabel = "CMP-X index"
    else:
        raise ValueError("slice_axis must be 'x', 'y', or 'time'")

    plt.figure(figsize=(8, 6))
    im = plt.imshow(panel, cmap=cmap, aspect="auto", extent=extent)
    plt.colorbar(im, label="Velocity (m/s)")
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f"Velocity slice: {slice_axis}={slice_index}" if slice_axis!="time" else f"Time slice: t={t[time_index]:.2f}s")
    plt.show()




@log_all_methods_to_object
class Seismic:
    # Class body for an object of seismic data
    # Contains all headers, traces, and functions to operate on the seismic data
    def __init__(self):
        self.header = {
            "sourceX" : None, # X position of the source
            "sourceY" : None, # Y position of the source
            "groupX" : None, # X position of the receiver
            "groupY" : None, # Y position of the receiver
            "samples" : None, # Time-stamps in milliseconds
            "format" : None,
            "trans_id" : None, # Id of the transducer/trace
            "source_V" : None, # Voltage of the source (mV)
            "RcvElev" : None, # Elevation of the receiver
            "SrcElev" : None # Elevation of the source (group)
            }
        self.trace_headers = None #Trace headers
        self.shot_gathers = None # Will store a dictionary of shot gathers
        self.filename = None
        self.traces = None
        self.cmp_gather = None # CMP sorted data
        self.cmp_gather_NMO = None # CMP sorted data, corrected for offset travel times
        self.cmp_stacked = None # Stacked data at the CMP
        self.scaling_factor = 1 # Default to 1. Set to 2000 for verasonics data
        # self.unit = 1 # Default to 's'
        self.mig_img = None # Migrated image from prestack
        self.mig_img_SA = None # Spatial arrays/coordinates for migrated image
        self.cmp_mig_img = None # Post-stack migration image
        self.cmpX = None # X Coordinates for common midpoints 
        self.cmpY = None # Y coordinates for common midpoints
        self.z_t = None # Time-to-depth conversion
        self.cmpOffset = None # Offset for common midpoints
        self.vel_rms_1 = None # RMS velocities imported from a velocity model (1D array)
        self.vel_rms_2 = None # RMS velocities squared, imported from a velocity model (1D array)
        self.vel = None # Velocity in layer as a function of time
    
# %% File and data management   

    def read_segy(self, filename_in):
        """
        Read the .segy file containing seismic/ultrasound data. Data created in the lab is recorded in the .segy file format.

        Parameters
        ----------
        filename_in : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        print('Import data...')
        # Reads a .segy file
        # Based on tutorial: https://github.com/equinor/segyio?tab=readme-ov-file#tutorial
        with segyio.open(filename_in, strict=False) as segyfile:
            segyfile.mmap()
            # Extract header word for all traces
            self.header['sourceX'] = segyfile.attributes(segyio.TraceField.SourceX)[:]
            self.header['sourceY'] = segyfile.attributes(segyio.TraceField.SourceY)[:]
            self.header['groupX'] = segyfile.attributes(segyio.TraceField.GroupX)[:]
            self.header['groupY'] = segyfile.attributes(segyio.TraceField.GroupY)[:]
            self.header['nsum'] = segyfile.attributes(segyio.TraceField.NSummedTraces)[:]
            self.header['nstack'] = segyfile.attributes(segyio.TraceField.NStackedTraces)[:]
            self.header['samples'] = segyfile.samples
            self.filename = filename_in
            self.header['RcvElev'] = segyfile.attributes(segyio.TraceField.ReceiverGroupElevation)[:]
            self.header['SrcElev'] = segyfile.attributes(segyio.TraceField.SourceSurfaceElevation)[:]
            self.header['trans_id'] = segyfile.attributes(segyio.TraceField.FieldRecord)[:]
            self.header['source_V'] = segyfile.attributes(segyio.TraceField.EnergySourcePoint)[:]
            # self.header['format'] = segyfile.format
            
            # Extract traces
            self.traces = np.zeros([len(segyfile.trace),np.shape(segyfile.trace)], dtype=np.float32)
            for i in range(len(segyfile.trace)):
                self.traces[i,:] = segyfile.trace[i]
        
        self.scale_data(2000,'ms')

        print('Total traces:', str(np.shape(self.traces)[1]))
        print('Number of samples:', str(np.shape(self.traces)[0]))
        print('Receiver x-position:', str(np.round(np.min(self.header['groupX']),2)), ' - ', str(np.round(np.max(self.header['groupX']),2)))
        print('Receiver y-position:', str(np.round(np.min(self.header['groupY']),2)), ' - ', str(np.round(np.max(self.header['groupY']),2)))
        print('Data imported successfully.')
        print('...\n')
        
    def read_segy_folder(self, data_folder, x_start = 0, x_end = None, centered = False, title = False, debug = False):
        import re
        # x-Start: Starting position of the grid
        # x_end: End position of the grid
        # title: If the position is written in the title
        # centered: 
        
        # Go through folder and add:
        files = os.listdir(data_folder)
        
        # Calculate velocity (if relevant)
        if x_end is not None:
            x_shift_it = (x_end-x_start)/len(files) # iterative shift to x-position
            x_pos = x_start # Set current position to zero or a random starting position
        
        seis_i = Seismic()
        for file in files:
            full_path = data_folder + file
            
            # Check if the file exists
            file_path = Path(full_path).resolve()
            if file_path.exists() and os.path.splitext(file)[1] == '.segy':
                print(f"File {file} exists")
                # If the file exists, add it to a seismic file
                seis_i.read_segy(full_path)
                
                # Adjust the x-position from end-position
                if x_end is not None:
                    if title==True: # Find the x_shift (since this was not done in matlab...)
                        match = re.search(r"xpos-([-+]?\d*\.\d+|\d+)", file)
                        if match:
                            x_shift_0 = float(match.group(1))  # Convert to float if needed
                    elif centered==True: # Use the center x-position
                        x_shift_0 = np.unique(seis_i.header['sourceX'])
                    else:
                        x_shift_0 = 0
                    seis_i.shift_trace_positions(-x_shift_0, 0, 0) # Remove existing position

                    # Add position based on velocity
                    x_pos += x_shift_it  # Convert to float if needed
                    seis_i.shift_trace_positions(x_pos, 0, 0)
                
                
                # Add to general seis object
                if debug == True:
                    seis_i.plot_header()
                    plt.close('all')
                self.add_seismicData(seis_i)
            else:
                print(f"File {file} does not exist.")
    
    def reduce_trace_size(self):
        # Converts all the traces from float64 to float 32
        # for i in range(len(self.traces[:,0])):
        # arr = np.random.rand(np.shape(self.traces))
        self.traces = self.traces.astype(np.float32, copy=False)
        
    def filter_trans_by_pos(self, x_lim = None, y_lim = None):
        """
        Keep only traces where both source and receiver positions are inside the specified bounds.
    
        Parameters
        ----------
        x_lim : list or tuple [x_min, x_max]
        y_lim : list or tuple [y_min, y_max]
    
        Returns
        -------
        seis_filtered : Seismic object with filtered traces
        """
        seis_filtered = Seismic()
        
        if x_lim is None:
            x_low = np.min(self.header['groupX'])
            x_high = np.min(self.header['groupX'])
            x_lim = [0,3]
        
        if y_lim is None:
            y_low = np.min(self.header['groupY'])
            y_high = np.min(self.header['groupY'])
            y_lim = [0,1]
            
            
        # Convert headers to numpy arrays
        groupX = np.array(self.header['groupX'])
        groupY = np.array(self.header['groupY'])
        sourceX = np.array(self.header['sourceX'])
        sourceY = np.array(self.header['sourceY'])
    
        # Boolean masks for inside bounds
        mask_group = (groupX >= x_lim[0]) & (groupX <= x_lim[1]) & \
                     (groupY >= y_lim[0]) & (groupY <= y_lim[1])
        mask_source = (sourceX >= x_lim[0]) & (sourceX <= x_lim[1]) & \
                      (sourceY >= y_lim[0]) & (sourceY <= y_lim[1])
    
        # Keep only traces where both source AND receiver are inside
        mask = mask_group & mask_source
    
        # Apply mask to traces
        seis_filtered.traces = self.traces[mask]
    
        # Apply mask to per-trace headers
        for key in ['sourceX', 'sourceY', 'groupX', 'groupY',
                    'RcvElev', 'SrcElev', 'nsum', 'nstack', 'trans_id']:
            seis_filtered.header[key] = self.header[key][mask]
    
        # Copy metadata that isn’t per-trace
        seis_filtered.header['samples'] = self.header['samples']
        seis_filtered.filename = self.filename
    
        print(f"Filtered traces: kept {np.sum(mask)} / {len(self.traces)} traces "
              f"(source AND receiver inside bounds)")
        return seis_filtered
        
    def remove_traces(self,trace_nr):
        # Removes all trace(s) within trace_nr
        self.header['sourceX'] = np.delete(self.header['sourceX'], trace_nr)
        self.header['sourceY'] = np.delete(self.header['sourceY'], trace_nr)
        self.header['groupX'] = np.delete(self.header['groupX'], trace_nr)
        self.header['groupY'] = np.delete(self.header['groupY'], trace_nr)
        self.header['nsum'] = np.delete(self.header['nsum'], trace_nr)
        self.header['nstack'] = np.delete(self.header['nstack'], trace_nr)
        self.header['RcvElev'] = np.delete(self.header['RcvElev'], trace_nr)
        self.header['SrcElev'] = np.delete(self.header['SrcElev'], trace_nr)
        self.header['trans_id'] = np.delete(self.header['trans_id'], trace_nr)       
        
        self.traces = np.delete(self.traces, trace_nr,axis=0)

    def export_segy(self,filename):
        # Create a .segy file from the seismic object. 
        spec = segyio.spec()
        spec.samples = self.header['samples']
        spec.tracecount = np.shape(self.traces)[0]
        spec.format = 1
        
        # Create file
        with segyio.create(filename,spec) as segyfile:
            segyfile.trace = self.traces
            for itr, x in enumerate(segyfile.header):
                # x.update({segyio.TraceField.GroupX: SeismicData.header['groupX']})
                x.update({segyio.TraceField.GroupX: np.int32(np.around(self.header['groupX'][itr]))})
                x.update({segyio.TraceField.GroupY: np.int32(np.around(self.header['groupY'][itr]))})
                x.update({segyio.TraceField.SourceX: np.int32(np.around(self.header['sourceX'][itr]))})
                x.update({segyio.TraceField.SourceY: np.int32(np.around(self.header['sourceY'][itr]))})
                x.update({segyio.TraceField.NSummedTraces: np.int32(np.around(self.header['nsum'][itr]))})
                x.update({segyio.TraceField.NStackedTraces: np.int32(np.around(self.header['nstack'][itr]))})
                x.update({segyio.TraceField.SourceSurfaceElevation: np.int32(np.around(self.header['SrcElev'][itr]))})
                x.update({segyio.TraceField.ReceiverGroupElevation: np.int32(np.around(self.header['RcvElev'][itr]))})
    
    def duplicate_single_source(self, source_idx=None):
        """
        Create a duplicate of the current Seismic object, 
        keeping only traces belonging to a single source (shot gather).
    
        Uses the existing `organize_shots_by_location()` function 
        to identify shot-gather locations.
    
        Parameters
        ----------
        source_idx : int
            Index of the source to keep (0-based index corresponding to 
            keys in `self.shot_gathers_by_loc` and rows in `self.shot_locations`).
    
        Returns
        -------
        Seismic
            A new Seismic object containing only the selected shot gather.
        """
        import numpy as np
        from copy import deepcopy
    
        # --- Ensure shots are organized ---
        if not hasattr(self, "shot_gathers_by_loc") or self.shot_gathers_by_loc is None:
            print("Organizing shots by location...")
            self.organize_shots_by_location()
    
        if not hasattr(self, "shot_gathers_by_loc") or len(self.shot_gathers_by_loc) == 0:
            raise ValueError("No shot gathers found. Run `organize_shots_by_location()` first.")
    
        # --- Validate source index ---
        num_sources = len(self.shot_gathers_by_loc)
        if source_idx is None:
            raise ValueError("You must specify `source_idx` (0-based index of the source).")
        if source_idx < 0 or source_idx >= num_sources:
            raise IndexError(f"source_idx {source_idx} out of range (0–{num_sources - 1}).")
    
        trace_indices = self.shot_gathers_by_loc[source_idx]
        source_loc = self.shot_locations[source_idx]
    
        # --- Create new Seismic object ---
        seis_new = self.__class__()  # preserve subclass type
        seis_new.filename = getattr(self, "filename", None)
    
        # Copy traces and subset header
        seis_new.traces = self.traces[trace_indices, :].copy()
        seis_new.header = {}
        for key, value in self.header.items():
            if isinstance(value, np.ndarray):
                if len(value) == len(self.header["sourceX"]):
                    seis_new.header[key] = value[trace_indices].copy()
                else:
                    seis_new.header[key] = deepcopy(value)
            else:
                seis_new.header[key] = deepcopy(value)
    
        # --- Copy only relevant shot metadata ---
        seis_new.shot_locations = np.array([source_loc])
        seis_new.shot_gathers_by_loc = {0: np.arange(len(trace_indices))}
    
        print(f"Created duplicate with {len(trace_indices)} traces "
              f"from source #{source_idx} at location {source_loc}.")
        return seis_new
    

    def duplicate_single_gather(self, trans_id):
        """
        Create a full duplicate of the current Seismic object, 
        but keep only the traces belonging to one gather (identified by trans_id).
    
        Parameters
        ----------
        trans_id : int or float
            The gather/transducer ID to keep.
    
        Returns
        -------
        Seismic
            A new Seismic object that contains only the selected gather.
        """
        import numpy as np
        from copy import deepcopy
    
        # --- 1. Verify data integrity ---
        if self.traces is None or 'trans_id' not in self.header:
            raise ValueError("Seismic data not loaded or 'trans_id' header missing.")
    
        trans_ids = self.header['trans_id']
        if trans_id not in trans_ids:
            available = np.unique(trans_ids)
            raise ValueError(f"Requested trans_id {trans_id} not found. Available IDs: {available}")
    
        # --- 2. Find matching trace indices ---
        gather_mask = trans_ids == trans_id
        gather_indices = np.where(gather_mask)[0]
    
        # --- 3. Create a new Seismic object and copy over relevant attributes ---
        seis_new = self.__class__()  # preserve subclass type if extended later
    
        # Copy metadata that applies globally
        seis_new.filename = getattr(self, 'filename', None)
        seis_new.header = {}
        seis_new.traces = self.traces[gather_indices, :].copy()
    
        # Copy header fields (selectively for per-trace arrays)
        for key, value in self.header.items():
            if isinstance(value, np.ndarray):
                if len(value) == len(trans_ids):  # per-trace header
                    seis_new.header[key] = value[gather_indices].copy()
                else:
                    seis_new.header[key] = deepcopy(value)
            else:
                seis_new.header[key] = deepcopy(value)
    
        # Copy additional derived attributes if they exist
        for attr in ['shot_gathers', 'shot_locations', 'shot_gathers_by_loc']:
            if hasattr(self, attr):
                setattr(seis_new, attr, deepcopy(getattr(self, attr)))
    
        print(f"Created duplicate with {len(gather_indices)} traces for gather ID {trans_id}.")
        return seis_new
        
# %% Migration

    def prestack_migration_parallel(self, maxDepth = 0.5, xlim = None, ylim = None, res = (100,1,1000), IF = 4, upsample_factor = 1, aperture = None, cutOffAngle = 30, maxAngle = 45, vel_mod = None, n_threads = 20, C = False, vel_model_plot = False):
        """
        Perform prestack migration on the seismic section (data). The important processing happens inside the 'migration_worker' function, that is accessed by parallel workers.
        This function iterates through all of the source-receiver pairs in the seis object (also called traces), and constructs an image (img) for each of these traces based on Kirchov migration, 
        which are then summed up to form the final image.

        Parameters
        ----------
        maxDepth : TYPE, optional
            Maximum depth to create an image to. The default is 0.5.
        res : TYPE, optional
            Resolution of the output image. NB! Numebr of pixels dramatically increases computation time. Ideally, the depth resolution should be higher than the trace sampling. The default is (100,1,1000).
        aperture : TYPE, optional
            Limits the number of traces used to improve computation time. Does not work so well, and not needed. The default is None.
        cutOffAngle : TYPE, optional
            Sets the starting angle to which the signal begins to be damped by a tapering function. The default is 30.
        maxAngle : TYPE, optional
            At this angle, the trace should be reduced to zero. The default is 45.
        vel_mod : TYPE, optional
            Velocity model, which should be an object of the Velocity_model class from the velpy script. The default is None.

        Returns
        -------
        Self.mig_img is created and appended to the seis object, which can be plotted by the support function 'plot_migdata'.

        """
        
        print('Performing prestack migration\n')
        self_copy = copy.deepcopy(self)
        # Upsample the traces
        self_copy = self_copy.filter_trans_by_pos(x_lim = xlim, y_lim=ylim)

        self_copy.upsample_time(upsample_factor = upsample_factor)
        
        data = self_copy.traces
        x_S = self_copy.header['sourceX'] # X-position of the sources
        x_R = self_copy.header['groupX'] # X-position of the receivers
        y_S = self_copy.header['sourceY'] # Y-position of the sources
        y_R = self_copy.header['groupY'] # Y-position of the receivers
        z_S = self_copy.header['SrcElev'] # Elevation of sources
        z_R = self_copy.header['RcvElev'] # Elevation of sources
        
        if xlim is None:
            x_0 = np.min(x_S)
            x_tot = np.max(x_S) - np.min(x_S)
        else:
            x_0 = xlim[0]
            x_tot = xlim[1]-xlim[0]
            # self_copy = self_copy.filter_trans_by_pos(x_lim = [xlim[0],xlim[1]])
        if ylim is None:
            y_0 = np.min(y_S)
            y_tot = np.max(y_S) - np.min(y_S)
        else:
            y_0 = ylim[0]
            y_tot = ylim[1]-ylim[0]
            # self_copy = self_copy.filter_trans_by_pos(y_lim = [ylim[0],ylim[1]])

        if xlim is not None:
            if x_0 < np.min(x_S):
                x_dubb_min = x_0
            else:
                x_dubb_min = np.min(x_S)
            if xlim[1] > np.max(x_S):
                x_dubb_max = xlim[1]
            else:
                x_dubb_max = np.max(x_S)
           
            # Add dummy traces to expand the ROI
            dubb_trace = np.zeros(len(data[0,:]))
            data = np.concatenate((data, dubb_trace[None, :]), axis=0)
            data = np.concatenate((data, dubb_trace[None, :]), axis=0)


            x_S = np.append(x_S, x_dubb_min)
            x_R = np.append(x_R, x_dubb_min)
            y_S = np.append(y_S, y_0)
            y_R = np.append(y_R, y_0)
            z_S = np.append(z_S, 0)
            z_R = np.append(z_R, 0)
           
            x_S = np.append(x_S, x_dubb_max)
            x_R = np.append(x_R, x_dubb_max)
            y_S = np.append(y_S, y_0)
            y_R = np.append(y_R, y_0)
            z_S = np.append(z_S, 0)
            z_R = np.append(z_R, 0)
            
            self_copy.traces = data
            self_copy.header['sourceX'] =  x_S # X-position of the sources
            self_copy.header['groupX']  = x_R # X-position of the receivers
            self_copy.header['sourceY']  = y_S # Y-position of the sources
            self_copy.header['groupY'] = y_R # Y-position of the receivers
            self_copy.header['SrcElev'] = z_S # Elevation of sources
            self_copy.header['RcvElev'] = z_R # Elevation of sources
            
        if ylim is not None:
           
            if y_0 < np.min(y_S):
                y_dubb_min = y_0
            else:
                y_dubb_min = np.min(y_S)
            
            if ylim[1] > np.max(y_S):
                y_dubb_max = ylim[1]
            else:
                y_dubb_max = np.max(y_S)
                
            # Add dummy traces to expand the ROI
            dubb_trace = np.zeros(len(data[0,:]))
            data = np.concatenate((data, dubb_trace[None, :]), axis=0)
            data = np.concatenate((data, dubb_trace[None, :]), axis=0)


            x_S = np.append(x_S, x_0)
            x_R = np.append(x_R, x_0)
            y_S = np.append(y_S, y_dubb_min)
            y_R = np.append(y_R, y_dubb_min)
            z_S = np.append(z_S, 0)
            z_R = np.append(z_R, 0)
           
            x_S = np.append(x_S, x_0)
            x_R = np.append(x_R, x_0)
            y_S = np.append(y_S, y_dubb_max)
            y_R = np.append(y_R, y_dubb_max)
            z_S = np.append(z_S, 0)
            z_R = np.append(z_R, 0)
            
            self_copy.traces = data
            self_copy.header['sourceX'] =  x_S # X-position of the sources
            self_copy.header['groupX']  = x_R # X-position of the receivers
            self_copy.header['sourceY']  = y_S # Y-position of the sources
            self_copy.header['groupY'] = y_R # Y-position of the receivers
            self_copy.header['SrcElev'] = z_S # Elevation of sources
            self_copy.header['RcvElev'] = z_R # Elevation of sources

            
        if aperture == None:
            aperture = np.max(x_S)
        
        
        t = self_copy.header['samples'] # Time-array
        dt = t[1]-t[0]
        if res is None:
            res = self_copy.compute_grid_requirements(vel_mod, f_center=150e3, xlim = xlim, ylim = ylim, max_depth = maxDepth, image_fraction=IF)
            print(f"nx = {res[0]}, ny = {res[1]}, nz = {res[2]}")
        (Nx,Ny,Nz) = res # Pixels in the image
        
        # Extract key information
        Nr = len(data[:,0]) # Number of channels
        Nt = len(t) # Number of samples
        dx = x_tot/Nx
        dy = y_tot/Ny
        dz = maxDepth/Nz
        
        # Assign coordinates to the migrated image for later reference
        x_arr = np.linspace(x_0, x_tot + x_0, Nx)
        y_arr = np.linspace(y_0, y_0 + y_tot, Ny)
        z_arr = np.linspace(0, maxDepth, Nz)
        self_copy.mig_img_SA = [x_arr,y_arr,z_arr]
        self.mig_img_SA = [x_arr,y_arr,z_arr]
        
        if vel_model_plot is True:
            # Define pixel-accurate velocity model
            img_vel = np.zeros([Nx,Ny,Nz])
            for xi in range(Nx):
                x = dx*xi + x_0
                for yi in range(Ny):
                    y = dy*yi + y_0
                    for zi in range(Nz):
                        z = dz*zi
                        img_vel[xi,yi,zi] = vel_mod.retrieveRMSVelAtPoint(x, y, z)
                        
            fig_extent=(self_copy.mig_img_SA[1][0],self_copy.mig_img_SA[1][-1],maxDepth,0)
            plt.figure()
            plt.title('Velocity model at migration area')
            plt.imshow(img_vel[int(Nx/2),:,:].T,aspect='auto', extent=fig_extent)
            plt.xlabel('Y distance (m)')
            plt.ylabel('Z distance (m)')
            
            fig_extent=(self_copy.mig_img_SA[0][0],self_copy.mig_img_SA[0][-1],maxDepth,0)
            plt.figure()
            plt.title('Velocity model at migration area')
            plt.imshow(img_vel[:,int(Ny/2),:].T,aspect='auto', extent=fig_extent)
            plt.xlabel('X distance (m)')
            plt.ylabel('Z distance (m)')
        
        
        # Migrate on parallel workers
        if C == True:
            try:
                import migration2 as migration
                print("Using migration2")
            except ImportError:
                import migration as migration  # fallback
                print("Using migration")

            self_copy.traces = self_copy.traces.astype(np.float64, copy=False)
            img = migration.prestack_migration_par(self_copy, maxDepth = maxDepth, res = res, aperture = aperture, cutOffAngle = cutOffAngle, maxAngle = maxAngle, vel_mod = vel_mod, n_threads = n_threads) # Perform the migration
            self_copy.traces = self_copy.traces.astype(np.float32, copy=False)
        else:
            img = np.zeros([Nx,Ny,Nz]) # Resulting migrated image
            img_vel = np.zeros([Nx,Ny,Nz])
            for xi in range(Nx):
                x = dx*xi + x_0
                for yi in range(Ny):
                    y = dy*yi + y_0
                    for zi in range(Nz):
                        z = dz*zi
                        img_vel[xi,yi,zi] = vel_mod.retrieveRMSVelAtPoint(x, y, z)
            args = [(data[i,:], Nx, Ny, Nz, dx, dy, x_0, y_0, dz,i, x_S[i], x_R[i], y_S[i], y_R[i], z_R[i], z_S[i], aperture, maxAngle, dt, Nt, cutOffAngle, img_vel) for i in range(Nr)]
            total_count = 0
            with mp.Pool() as pool:
                results = pool.imap(migration_worker, args)
                for result in results:
                    img += result
                    total_count += 1
                    print('Count: ', total_count)
            
            print('Done\n')
        
        self.mig_img = img
    

    
    
    def postStack_migration(self, maxDepth = 0.5, res = (100,1,1000), aperture = None, cutOffAngle = 30, maxAngle = 45, vel_mod = None):
        print('Performing zero-offset post-stack migration\n')
        
        # Create zero-offset section from the cmp stcaked data, and convert it to the "normal" format for mgiration processing (see prestack migration method)
        Nr = len(self.cmpX)*len(self.cmpY)
        data = np.zeros([Nr,len(self.cmp_stacked[0,0,:])])
        x_S = np.zeros([Nr])
        y_S = np.zeros([Nr])
        n = 0        

        for i in range(len(self.cmpX)):
            for j in range(len(self.cmpY)):
                data[n,:] = self.cmp_stacked[i,j,:]
                x_S[n] = self.cmpX[i]
                y_S[n] = self.cmpY[j]
                n += 1
        x_R = x_S
        y_R = y_S
        x_0 = np.min(x_S)
        y_0 = np.min(y_S)
        
        if aperture == None:
            aperture = np.max(x_S)
        (Nx,Ny,Nz) = res # Pixels in the image
        
        # Extract key information
        # Nr = len(data[:,0]) # Number of channels
        t = self.header['samples'] # Time-array
        dt = t[1]-t[0]
        Nt = len(t) # Number of samples
        x_tot = np.max(x_S)-np.min(x_S)
        dx = x_tot/Nx
        dz = maxDepth/Nz
        y_tot = np.max(y_S) - np.min(y_S)
        dy = y_tot/Ny
        
        # Define pixel-accurate velocity model
        img_vel = np.zeros([Nx,Ny,Nz])
        for xi in range(Nx):
            x = dx*xi + x_0
            for yi in range(Ny):
                y = dy*yi + y_0
                for zi in range(Nz):
                    z = dz*zi
                    img_vel[xi,yi,zi] = vel_mod.retrieveRMSVelAtPoint(x, y, z)
        ## Migrate
        img = np.zeros([Nx,Ny,Nz])
        args = [(data[i,:], Nx, Ny, Nz, dx, dy, x_0, y_0, dz,i, x_S[i], x_R[i], y_S[i], y_R[i],0,0, aperture, maxAngle, dt, Nt, cutOffAngle, img_vel) for i in range(Nr)]
        total_count = 0
        with mp.Pool() as pool:
            # for i in range(len(Nr_list)):
            results = pool.imap(migration_worker, args)
            for result in results:
                img += result
                total_count += 1
                print('Count: ', total_count)
            
       
        print('Done\n')
        self.cmp_mig_img = img
    
    def plot_post_stack_migration(self,axis = 'x', line= 0, maxDepth = 1,  noiseDB=-np.inf):
        if axis=='x':
            y_min = np.min(self.cmpX)
            y_max = np.max(self.cmpX)
            data = self.cmp_mig_img[:,line,:]
        elif axis=='y':
            y_min = np.min(self.cmpY)
            y_max = np.max(self.cmpY)
            data = self.cmp_mig_img[line,:,:]
        maxAmp = np.max(data)/2 # Max amplitude for plotting to half the maximum 
        t_arr = self.header['samples']
        # Plots the stacked data created in "sort-to-bin" methods
        
        plt.figure()
        plt.title('Post-stack migrated data')
        plt.imshow(data.T,cmap ='BrBG',extent = [y_min,y_max,maxDepth,0], aspect='auto',vmin=-maxAmp,vmax=maxAmp)
        plt.xlabel('Distance (m) along xlines')
        plt.ylabel('Distance (m)')
        plt.colorbar(label='Amplitude')
        plt.show()
      
        
        mig_img_dB = 20*np.log(abs(np.double(data)))        
        mig_img_dB[mig_img_dB<noiseDB] = noiseDB 
        plt.figure()
        plt.title('Post-stack migrated data')
        plt.imshow(mig_img_dB.T,cmap ='RdBu_r',extent = [y_min,y_max,maxDepth,0], aspect='auto')
        plt.xlabel('Distance (m) along xlines')
        plt.ylabel('Distance (m)')
        plt.colorbar(label='Amplitude (dB)')
        plt.show()
        
        mig_img_RMS = matrix_RMS(data,8)        
        plt.figure()
        plt.title('Pre-stack migrated data')
        plt.imshow(mig_img_RMS.T,cmap ='RdBu_r',extent = [y_min,y_max,maxDepth,0], aspect='auto')
        plt.xlabel('Distance (m) along xlines')
        plt.ylabel('Distance (m)')
        plt.colorbar(label='Amplitude (RMS)')
        plt.show()
        

        
    def plot_migdata(self, axis='x',
                 AMPlim=None, RMSlim=100, noiseDB=-300,
                 RMS_window=8, interpolationF=1, GaussKernel=0,
                 interpolation='gaussian'):

        data = self.mig_img
        axis_sel = axis  # current axis selection
    
        # --- Helper to extract slice ---
        def get_slice(idx, axis_sel):
            if data.ndim == 2:
                return data
            if axis_sel == 'y':
                return data[:, idx, :]
            elif axis_sel == 'x':
                return data[idx, :, :]
            elif axis_sel == 'z':
                return data[:, :, idx]
            else:
                raise ValueError("axis must be 'x', 'y' or 'z'")
    
        # --- Helper to compute axis limits ---
        def get_axis_info(axis_sel):
            if axis_sel == 'y':
                y_min, y_max = np.min(self.mig_img_SA[0]), np.max(self.mig_img_SA[0])
                x_min, x_max = np.min(self.mig_img_SA[2]), np.max(self.mig_img_SA[2])
                max_index = data.shape[1] - 1
            elif axis_sel == 'x':
                y_min, y_max = np.min(self.mig_img_SA[1]), np.max(self.mig_img_SA[1])
                x_min, x_max = np.min(self.mig_img_SA[2]), np.max(self.mig_img_SA[2])
                max_index = data.shape[0] - 1
            elif axis_sel == 'z':
                y_min, y_max = np.min(self.mig_img_SA[0]), np.max(self.mig_img_SA[0])
                x_min, x_max = np.min(self.mig_img_SA[1]), np.max(self.mig_img_SA[1])
                max_index = data.shape[2] - 1
            return y_min, y_max, x_min, x_max, max_index
    
        # --- Initial slice ---
        line = 0
        y_min, y_max, x_min, x_max, max_index = get_axis_info(axis_sel)
        slice_data = get_slice(line, axis_sel)
        slice_data = interpolate_data(slice_data, interpolationF, GaussKernel)
        if GaussKernel > 0:
            slice_data = gaussian_filter(slice_data, sigma=GaussKernel)
    
        # Global min/max for scaling sliders
        global_min, global_max = np.min(data), np.max(data)
        if AMPlim is None:
            AMPlim = global_max / 2
    
        # --- Figure setup ---
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.35, bottom=0.25)  # leave more space for widgets
    
        im = ax.imshow(slice_data.T, cmap='seismic',
                       extent=[y_min, y_max, x_max, x_min],
                       vmin=-AMPlim, vmax=AMPlim,
                       interpolation=interpolation,
                       interpolation_stage='data')
        cbar = plt.colorbar(im, ax=ax)
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Distance (m)')
        ax.set_title("Pre-stack migrated data (Amplitude)")
    
        # --- Slider for slice index ---
        if data.ndim == 3:
            ax_slider = plt.axes([0.35, 0.15, 0.55, 0.03])
            slider = Slider(ax_slider, 'Slice', 0, max_index,
                            valinit=line, valstep=1)
    
        # --- Sliders for min/max scale ---
        ax_vmin = plt.axes([0.35, 0.08, 0.55, 0.03])
        ax_vmax = plt.axes([0.35, 0.05, 0.55, 0.03])
        vmin_slider = Slider(ax_vmin, 'vmin', global_min, 0, valinit=-AMPlim)
        vmax_slider = Slider(ax_vmax, 'vmax', 0, global_max, valinit=AMPlim)
    
        # --- Radio buttons for mode ---
        ax_radio = plt.axes([0.05, 0.5, 0.2, 0.15])
        radio = RadioButtons(ax_radio, ('Amplitude', 'dB', 'RMS'))

    
        # --- Update function ---
        def update_display(idx, mode, axis_sel):
            y_min, y_max, x_min, x_max, max_index = get_axis_info(axis_sel)
            new_slice = get_slice(idx, axis_sel)
            new_slice = interpolate_data(new_slice, interpolationF, GaussKernel)
            if GaussKernel > 0:
                new_slice = gaussian_filter(new_slice, sigma=GaussKernel)
    
            im.set_extent([y_min, y_max, x_max, x_min])
    
            if mode == 'Amplitude':
                im.set_data(new_slice.T)
                im.set_cmap('seismic')
                im.set_clim(vmin_slider.val, vmax_slider.val)
                cbar.set_label('Amplitude')
                ax.set_title(f"Pre-stack migrated data (Amplitude) [{axis_sel}-axis]")
    
            elif mode == 'dB':
                mig_img_dB = 20 * np.log10(np.abs(np.double(new_slice)))
                mig_img_dB[mig_img_dB < noiseDB] = noiseDB
                im.set_data(mig_img_dB.T)
                im.set_cmap('seismic')
                im.set_clim(noiseDB, np.max(mig_img_dB))
                cbar.set_label('Amplitude (dB)')
                ax.set_title(f"Pre-stack migrated data (dB) [{axis_sel}-axis]")
    
            elif mode == 'RMS':
                mig_img_RMS = matrix_RMS(new_slice, RMS_window)
                im.set_data(mig_img_RMS.T)
                im.set_cmap('seismic')
                im.set_clim(0, RMSlim)
                cbar.set_label('Amplitude (RMS)')
                ax.set_title(f"Pre-stack migrated data (RMS) [{axis_sel}-axis]")
    
            fig.canvas.draw_idle()
    
        # --- Callbacks ---
        def slider_update(val):
            idx = int(slider.val) if data.ndim == 3 else 0
            update_display(idx, radio.value_selected, axis_sel)
    
        def vmin_update(val):
            im.set_clim(vmin_slider.val, vmax_slider.val)
            fig.canvas.draw_idle()
    
        def vmax_update(val):
            im.set_clim(vmin_slider.val, vmax_slider.val)
            fig.canvas.draw_idle()
    
        def radio_update(label):
            idx = int(slider.val) if data.ndim == 3 else 0
            update_display(idx, label, axis_sel)
    
        if data.ndim == 3:
            slider.on_changed(slider_update)
        vmin_slider.on_changed(vmin_update)
        vmax_slider.on_changed(vmax_update)
        radio.on_clicked(radio_update)
    
        # --- Initial draw ---
        update_display(line, 'Amplitude', axis_sel)
        plt.show()
        
    def plot_map_view(self, center_depth_m, thickness_m, method='average_rms',smoothing_sigma=3.0, map_type='continuous',contrast_percentile=80.0,vmax_percentile=99.0,cmap='RdBu_r', vmin=None, vmax=None, plotting = False):
        
        """
        Generates a final, publication-quality "bird's-eye view" map.
    
        This is a powerful, unified function that can create both continuous
        energy maps and high-contrast anomaly maps using one of three
        calculation methods. It ensures a white background and correct aspect ratio.
    
        Args:
            center_depth_m (float): The center of the vertical slab in meters.
            thickness_m (float): The total thickness of the slab in meters.
            method (str): The calculation method.
                          - 'average_rms', 'max_amplitude', or 'max_rms'.
            smoothing_sigma (float): The strength of the Gaussian smoothing filter.
            map_type (str): The type of map to generate.
                            - 'continuous': A full energy map with a white background.
                            - 'anomaly': A high-contrast, thresholded map with a white background.
            contrast_percentile (float): For 'anomaly' maps, the threshold percentile.
            vmax_percentile (float): Percentile for the max of the color scale.
            cmap (str): The colormap to use.
        """
        
        
        print(f"--- Generating Map View: method='{method}', type='{map_type}' ---")
    
        #### --- 1. Calculate the base map using the chosen method ---
        _, _, z_coords = self.mig_img_SA
        start_idx = np.argmin(np.abs(z_coords - (center_depth_m - thickness_m / 2.0)))
        end_idx = np.argmin(np.abs(z_coords - (center_depth_m + thickness_m / 2.0)))
        slab = self.mig_img[:, :, start_idx:end_idx+1]
        
        if method == 'average_rms':
            base_map = np.sqrt(np.mean(slab**2, axis=2))
        elif method == 'max_amplitude':
            base_map = np.max(np.abs(slab), axis=2)
        elif method == 'max_rms':
            # Ensure matrix_RMS is available in your class
            base_map = np.max(matrix_RMS(slab, window_size=8), axis=2)
        else:
            raise ValueError("Method must be 'average_rms', 'max_amplitude', or 'max_rms'.")
    
        #### --- 2. Apply Smoothing ---
        if smoothing_sigma and smoothing_sigma > 0:
            final_map = gaussian_filter(base_map, sigma=smoothing_sigma)
        else:
            final_map = base_map
    
        #### --- 3. Configure Plot based on Map Type ---
        if map_type == 'continuous':
            if vmin is None:
                vmin = np.min(final_map)
            if vmax is None:
                vmax = np.percentile(final_map, vmax_percentile)
            map_for_plotting = final_map
        
        elif map_type == 'anomaly':
            threshold = np.percentile(final_map, contrast_percentile)
            map_for_plotting = final_map.copy()
            map_for_plotting[map_for_plotting < threshold] = np.nan
            if vmin is None:
                vmin = threshold
            if vmax is None:
                vmax = np.nanpercentile(final_map, vmax_percentile)
        else:
            raise ValueError("map_type must be 'continuous' or 'anomaly'.")
        
        #### --- 4. Generate the Plot ---
        # CORRECTLY calculate height to match the real-world aspect ratio
        if plotting is True:
            plt.figure(figsize=(12,8)) # Add height for title
            ax = plt.gca()
            extent = [self.mig_img_SA[0][0],self.mig_img_SA[0][-1], self.mig_img_SA[1][0-1], self.mig_img_SA[1][0]]
            # CORRECTLY ensure a white background for all cases
            ax.set_facecolor('white') 
            
            im = ax.imshow(map_for_plotting.T, cmap=cmap, vmin=vmin, vmax=vmax,
                           extent=extent, interpolation='bilinear')
            
            cbar = plt.colorbar(im, shrink=0.7, pad=0.05)
            cbar.set_label(f"{method}", size=12)
            
            ax.set_xlabel("Distance (m)", size=12)
            ax.set_ylabel("Distance (m)", size=12)
            ax.set_title(f"Map View ({method} , {map_type})", size=16, pad=20)
            # CORRECTLY use a visible grid color for a white background
            ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.4)
        
        return map_for_plotting
    
    def plot_final_map(self, center_depth_m, thickness_m, 
                         smoothing_sigma=3.0, 
                         contrast_percentile=75,
                         cmap='RdBu_r', aspect_ratio='auto'):
        """
        Generates a final, high-contrast, interpretive "bird's-eye view" map.

        This function is designed to transform the raw depth slice into a clean,
        interpretable map by applying aggressive smoothing and high-contrast
        thresholding, similar to the 1MHz prototype's goal.

        Args:
            center_depth_m (float): The center of the vertical slab in meters.
            thickness_m (float): The total thickness of the slab in meters.
            smoothing_sigma (float): The strength of the Gaussian smoothing filter.
                                     Values from 2.0 to 5.0 are effective for
                                     removing texture and acquisition footprints.
            contrast_percentile (float): The threshold for creating high contrast.
                                         Only data above this percentile of energy
                                         will be shown. E.g., 75 means only the
                                         top 25% of the energy is displayed.
            cmap (str): The colormap to use.
            aspect_ratio (str): The aspect ratio for the plot axes.
        """
        from scipy.ndimage import gaussian_filter
        print("--- Generating Final Interpretive Map ---")

        # --- 1. Calculate the smoothest possible base: Average RMS ---
        _, _, z_coords = self.mig_img_SA
        start_idx = np.argmin(np.abs(z_coords - (center_depth_m - thickness_m / 2.0)))
        end_idx = np.argmin(np.abs(z_coords - (center_depth_m + thickness_m / 2.0)))
        slab = self.mig_img[:, :, start_idx:end_idx+1]
        base_map = np.sqrt(np.mean(slab**2, axis=2))

        # --- 2. Apply Aggressive Smoothing ---
        # This is the key step to remove the texture and footprint.
        print(f"Applying aggressive Gaussian smoothing with sigma={smoothing_sigma}...")
        smoothed_map = gaussian_filter(base_map, sigma=smoothing_sigma)

        # --- 3. Apply High-Contrast Thresholding ---
        # This makes the main features "pop."
        print(f"Applying contrast enhancement. Keeping top {100-contrast_percentile}% of energy.")
        threshold_value = np.percentile(smoothed_map, contrast_percentile)
        
        # Create a copy and set all values below the threshold to zero (or NaN for transparency)
        final_map = smoothed_map.copy()
        final_map[final_map < threshold_value] = np.nan # Use NaN to make it transparent

        # --- 4. Generate the Plot ---
        x_coords, y_coords, _ = self.mig_img_SA
        
        # --- CHANGE 1: Control the Figure Size for a Rectangle ---
        # Instead of a square (10, 10), we'll make a figure that respects the data's aspect ratio.
        x_range = x_coords[-1] - x_coords[0]
        y_range = y_coords[-1] - y_coords[0]
        fig_width = 10 # Set a base width
        fig_height = fig_width * (y_range / x_range) # Calculate height to match data shape
        plt.figure(figsize=(fig_width, fig_height))
        
        ax = plt.gca()
        extent = [x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]]
        
        # --- CHANGE 2: Remove the Gray Background ---
        # We simply delete the ax.set_facecolor('gray') line.
        # The default background will be white.
        
        # --- CHANGE 3: Enforce the Correct Aspect Ratio ---
        # We change the 'aspect' argument in the imshow call.
        im = ax.imshow(final_map.T, cmap=cmap, extent=extent, 
                       aspect='auto', # Use 'auto' to fill the rectangular figure
                       interpolation='bilinear')
        
        cbar = plt.colorbar(im, shrink=0.8, pad=0.05) # Adjusted shrink for rectangle
        cbar.set_label("High-Energy Anomaly Amplitude", size=12)
        
        ax.set_xlabel("Distance (m)", size=12)
        ax.set_ylabel("Distance (m)", size=12)
        ax.set_title(f"Interpretive Map of Reflector Energy", size=16, pad=20)
        ax.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.2)
        
    def plot_mig_aperture(self, f_center = 150e3, axis = 'y', line=None, maxDepth = 0.5, res = None, image_fraction = 1, aperture = None, cutOffAngle = 30, maxAngle = 45, trace_nr = 0, noiseDB = 0, vel_mod = None, DB = True, AMP  = True, RMS = True, interpolation = None):
        """
        Plot the spatial reconstruction of a single source-receiver pair. Useful for understanding how migration works in connection to the velocity model. Same parameters used as for migration, but takes much less time.

        Parameters
        ----------
        axis : TYPE, optional
            DESCRIPTION. The default is 'x'.
        line : TYPE, optional
            DESCRIPTION. The default is 0.
        maxDepth : TYPE, optional
            DESCRIPTION. The default is 0.5.
        res : TYPE, optional
            DESCRIPTION. The default is (100,1,1000).
        aperture : TYPE, optional
            DESCRIPTION. The default is None.
        cutOffAngle : TYPE, optional
            DESCRIPTION. The default is 30.
        maxAngle : TYPE, optional
            DESCRIPTION. The default is 45.
        trace_nr : TYPE, optional
            DESCRIPTION. The default is 0.
        noiseDB : TYPE, optional
            DESCRIPTION. The default is 0.
        vel_mod : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        data = self.traces
        
            
        x_S = self.header['sourceX'] # X-position of the sources
        x_R = self.header['groupX'] # X-position of the receivers
        y_S = self.header['sourceY'] # Y-position of the sources
        y_R = self.header['groupY'] # Y-position of the receivers
        z_S = self.header['SrcElev'] # Elevation of sources
        z_R = self.header['RcvElev'] # Elevation of sources

        x_0 = np.min(x_S)
        y_0 = np.min(y_S)
        if aperture == None:
            aperture = np.max(x_S)
        
       
        
        # Extract key information
        t = self.header['samples'] # Time-array
        dt = t[1]-t[0]
        Nt = len(t) # Number of samples
        x_tot = np.max(x_S) - np.min(x_S)
        y_tot = np.max(y_S) - np.min(y_S)
        
        if res is None:
            res = self.compute_grid_requirements(vel_mod, f_center=f_center, xlim = [x_0,x_tot+x_0], ylim = [y_0,y_tot+y_0], max_depth = maxDepth, image_fraction=image_fraction)
            print(f"nx = {res[0]}, ny = {res[1]}, nz = {res[2]}")
        (Nx,Ny,Nz) = res # Pixels in the image
        
        
        dx = x_tot/Nx
        dy = y_tot/Ny
        dz = maxDepth/Nz
        
                
        # Define pixel-accurate velocity model
        img_vel = np.zeros([Nx,Ny,Nz])
        for xi in range(Nx):
            x = dx*xi + x_0
            for yi in range(Ny):
                y = dy*yi + y_0
                for zi in range(Nz):
                    z = dz*zi
                    img_vel[xi,yi,zi] = vel_mod.retrieveRMSVelAtPoint(x, y, z)

            
            
        ## Migrate
        args = [data[trace_nr,:], Nx, Ny, Nz, dx, dy, x_0, y_0, dz,trace_nr, x_S[trace_nr], x_R[trace_nr], y_S[trace_nr], y_R[trace_nr], z_R[trace_nr], z_S[trace_nr], aperture, maxAngle, dt, Nt, cutOffAngle, img_vel]
        img_i =  migration_worker(args)


        if axis == 'y':
            if line is None:
                line = int(round((x_R[trace_nr] - x_0) / dx))
                print(f'X-position: {x_R[trace_nr]}')
            y_min = np.min(x_R)
            y_max = np.max(x_R)
            data = img_i[line,:,:]
        elif axis == 'x':
            if line is None:
                line = int(round((y_R[trace_nr] - y_0) / dy))
                print(f'Y-position: {y_R[trace_nr]}')
            y_min = np.min(y_R)
            y_max = np.max(y_R)
            data = img_i[:,line,:]

            
            
        # Amplitude plotting
        if AMP is True:
            # if AMPlim is None:
            #     AMPlim = np.max(data)/2 # Max amplitude for plotting to half the maximum 
            plt.figure()
            plt.title('Pre-stack migrated data')
            plt.suptitle("Pre-stack migrated data (amp)")
            plt.imshow(data.T,cmap ='BrBG',extent = [y_min,y_max,maxDepth,0], aspect='auto', interpolation=interpolation)
            plt.xlabel('Distance (m)')
            plt.ylabel('Distance (m)')
            plt.colorbar(label='Amplitude')
            plt.show()

        if DB is True:
            mig_img_dB = 20*np.log(abs(np.double(data)))        
            mig_img_dB[mig_img_dB<noiseDB] = noiseDB 
            plt.figure()
            plt.title('Pre-stack migrated data (dB)')
            plt.suptitle("Pre-stack migrated data (dB)")
            plt.imshow(mig_img_dB.T,cmap ='RdBu_r',extent = [y_min,y_max,maxDepth,0], aspect='auto', interpolation=interpolation)
            plt.xlabel('Distance (m)')
            plt.ylabel('Distance (m)')
            plt.colorbar(label='Amplitude (dB)')
            plt.show()
            
        if RMS is True:
            mig_img_RMS = matrix_RMS(data,8)        
            plt.figure()
            plt.title('Pre-stack migrated data (RMS)')
            plt.suptitle("Pre-stack migrated data (RMS)")
            plt.imshow(mig_img_RMS.T,cmap ='RdBu_r', extent = [y_min,y_max,maxDepth,0], aspect='auto', interpolation=interpolation)
            plt.xlabel('Distance (m)')
            plt.ylabel('Distance (m)')
            plt.colorbar(label='Amplitude (RMS)')
            plt.show()
        return data
    
    def plot_mig_velModel(self,axis='y', line=0, maxDepth = 0.5, res = (1,500,6400), aperture = 1, cutOffAngle = 0, maxAngle = 45, trace_nr = 210, noiseDB = -100, vel_mod = None):
        """
        Plot the velocity model in the area considered by the migration routine. This will confirm if the 'shift_trace_positions' correctly shifted the data to the right location.
        Needs to be considered several times while shifting the traces around to get the best coherence between velocity model and whats seen on the data.
        
        Parameters
        ----------
        axis : TYPE, optional
            DESCRIPTION. The default is 'y'.
        line : TYPE, optional
            DESCRIPTION. The default is 0.
        maxDepth : TYPE, optional
            DESCRIPTION. The default is 0.5.
        res : TYPE, optional
            DESCRIPTION. The default is (1,500,6400).
        aperture : TYPE, optional
            DESCRIPTION. The default is 1.
        cutOffAngle : TYPE, optional
            DESCRIPTION. The default is 0.
        maxAngle : TYPE, optional
            DESCRIPTION. The default is 45.
        trace_nr : TYPE, optional
            DESCRIPTION. The default is 210.
        noiseDB : TYPE, optional
            DESCRIPTION. The default is -100.
        vel_mod : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        x_S = self.header['sourceX']
        y_S = self.header['sourceY']
    
        x_0 = np.min(x_S)
        y_0 = np.min(y_S)
        if aperture == None:
            aperture = np.max(x_S)
        
        (Nx,Ny,Nz) = res # Pixels in the image
        
        # Extract key information
        x_tot = np.max(x_S) - np.min(x_S)
        y_tot = np.max(y_S) - np.min(y_S)
        dx = x_tot/Nx
        dy = y_tot/Ny
        dz = maxDepth/Nz
        
        # Define pixel-accurate velocity model
        img_vel = np.zeros([Nx,Ny,Nz])
        for xi in range(Nx):
            x = dx*xi + x_0
            for yi in range(Ny):
                y = dy*yi + y_0
                for zi in range(Nz):
                    z = dz*zi
                    img_vel[xi,yi,zi] = vel_mod.retrieveRMSVelAtPoint(x, y, z)
    
        plt.figure()
        
        if axis == 'x':
            data = img_vel[:,line,:]
            fig_extent=(np.min(x_S),np.max(x_S),Nz*dz,0)
        elif axis == 'y':
            data = img_vel[line,:,:]
            fig_extent=(np.min(y_S),np.max(y_S),Nz*dz,0)
        plt.imshow(data.T,aspect='auto', extent=fig_extent)
        plt.xlabel('Distance(m)')
        plt.ylabel('Distance(m)')
        plt.show()
# %% Filtering
    def plot_FFT_trace(self,trace_nr, fft_threshold=10):
        signal = self.traces[trace_nr,:]
        time = self.header['samples']
        
        signal_fft = np.fft.fft(signal)
        N = len(signal)
        freq_axis = np.fft.fftfreq(N, d=time[1]-time[0]) # Frequency axis of the FFT
        phase = np.angle(signal_fft)  # Phase of the signal at each frequency
        freq_max_idx = find_peaks(abs(signal_fft), threshold=fft_threshold) # Find frequency maximas with an amplitude more than 10 past its neighbours
        freq_max = freq_axis[freq_max_idx[0]] # Frequency amplitudes
        phase_max = phase[freq_max_idx[0]] # Phase at the most common frequencies
        
        # Signal plotting
        fig,ax = plt.subplots()
        ax.plot(freq_axis,abs(signal_fft))
        ax.set_title('Frequency Components of the Signal')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Magnitude')
        
        # Phase 
        fig2,ax2 = plt.subplots()
        fig2.suptitle('Phase sign x amplitude')
        ax2.plot(freq_axis,abs(signal_fft)*np.sign(phase)) # Multiplies the FFT with the sign of the phase
        ax2.set_title('Frequency Components of the Signal')
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Phase')
        
        # FX plot
        
        # return signal_fft, freq_axis, freq_max, phase, phase_max
    
    def plot_spectrogram(self,trace_nr, nperseg=200, cmap='viridis'):
        """
        Compute and plot the spectrogram of a given signal.
    
        Parameters:
        - signal: 1D numpy array, the amplitude vs. time signal
        - fs: Sampling frequency in Hz
        - nperseg: Length of each segment for STFT (default: 256)
        - cmap: Colormap for plotting (default: 'viridis')
    
        Returns:
        - f: Frequency axis values
        - t: Time axis values
        - Sxx: Spectrogram power
        """
        signal = self.traces[trace_nr,:]
        t = self.header['samples']
        fs = 1/(t[1]-t[0])
        f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=nperseg//2, scaling='density')
    
        plt.figure(figsize=(8, 5))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap=cmap)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.title('Spectrogram')
        plt.colorbar(label='Power [dB]')
        plt.show()
            
    
    def apply_bandpass(self, f0 = 150000, f_low = 0.5, f_high = 1.5):
        for i in range(len(self.traces[:,0])):
            self.traces[i,:] = apply_bandpass_filter(self.traces[i,:], f_low*f0, f_high*f0, 8*f0)

    def noise_reduction_threshold(self,traceNr, sampleStart,sampleEnd):
        max_mute = np.max(self.traces[traceNr,sampleStart:sampleEnd])
        for i in range(np.shape(self.traces)[0]): # traces
            for j in range(np.shape(self.traces)[1]): # samples
                if np.abs(self.traces[i,j]) < max_mute: # Check if below threshold
                    self.traces[i,j] = 0
            
    def remove_saturated_traces(self, start_time):
        print('Finding all traces to remove...\n')
        t_arr = self.header['samples']
        idx = (np.abs(t_arr - start_time)).argmin()
        del_list = np.array([])
        threshold = np.max(self.traces)

        for i in range(np.shape(self.traces)[0]): # traces
            trace_i = self.traces[i,idx:]
            if np.max(abs(trace_i)) >= threshold: # saturation
                del_list = np.append(del_list,i)
        
        print('Removing traces...\n')
        del_list = del_list.astype(int)
        self.remove_traces(del_list)
        return del_list
    
    def apply_gain(self,trace_nr,correction=False, epsilon = 0.003):
        t = self.header['samples']
        signal = self.traces[trace_nr,:]
        envelope, fitted_envelope, popt = fit_envelope(signal, t)
        normalized_signal = signal / (fitted_envelope + epsilon)
        normalized_signal = normalized_signal/np.mean(abs(normalized_signal)) * np.mean(abs(signal))
        
        if correction==True:
            # Apply for all traces
            for i in range(0,np.shape(self.traces)[0]):
                signal = self.traces[i,:]
                normalized_signal = signal / (fitted_envelope + epsilon)
                self.traces[i,:] = normalized_signal/np.mean(abs(normalized_signal)) * np.mean(abs(signal))
        else:
            # --- 5. Plot the results ---
            plt.figure(figsize=(10, 6))
            
            plt.subplot(3, 1, 1)
            plt.plot(t, signal, label='Original Signal')
            plt.title("Original Signal")
            
            plt.subplot(3, 1, 2)
            plt.plot(t, envelope, label='Envelope', color='orange')
            plt.plot(t, fitted_envelope, '--', label='Fitted Envelope', color='red')
            plt.title("Envelope and Fit")
            plt.legend()
            
            plt.subplot(3, 1, 3)
            plt.plot(t, normalized_signal, label='Normalized Signal', color='green')
            plt.title("Normalized Signal")
            plt.xlabel("Time")
            plt.legend()
            
            plt.tight_layout()
            plt.show()        
            
            
    def dampen_saturated_traces(self, start_time, idx_rng = 60, percentile_f = 99):
        """
        

        Parameters
        ----------
        start_time : TYPE
            Start time to begin looking for saturated peaks (eliminates the source signature).
        idx_rng : INT, optional
            Number of samples left and right of the saturated peak to be dampened. The default is 60.

        Returns
        -------
        red_list : TYPE
            List of traces that have been reduced.

        """
        print('Finding all traces to fix...\n')
        t_arr = self.header['samples']
        idx = (np.abs(t_arr - start_time)).argmin()
        
        seis_ZO = self.create_ZO() # Extract the zero-offset data
        traces = self.traces[:,idx:] # Isolate the early part with the source signature
        max_values = np.max(abs(traces), axis=1) # Find the maximum value in each trace, excluding the source
        threshold = np.percentile(max_values, percentile_f) # use the top 1% as an indication of the threshold
        n_samples = self.traces.shape[1]
        del_list = []
        for i in range(np.shape(self.traces)[0]): # traces
            trace_i = self.traces[i,:]
            work_trace = trace_i[idx:]
            saturated_indices = np.where(np.abs(work_trace) >= threshold)[0]
            if len(saturated_indices) > 0:
                # Create a mask for which parts of the trace to zero out
                zero_mask = np.zeros_like(trace_i, dtype=bool)
                for s_idx in saturated_indices:
                    abs_idx = idx + s_idx
                    start = max(abs_idx - idx_rng, 0)
                    end = min(abs_idx + idx_rng, n_samples)
                    zero_mask[start:end] = True
                
                # Apply the mask
                self.traces[i][zero_mask] = 0
                del_list.append(i)
        return np.array(del_list)
                
    def remove_nearzero_traces(self, threshold_to_noise, ref_trace, start_time, end_time):
        # Removes all traces with amplitude below the threshold. 
        # "Signature end" marks the beginning of the threshold check 
        print('Finding all traces to remove...\n')
        differences = np.abs(self.header['samples'] - start_time)
        start_index = np.argmin(differences)     
        differences = np.abs(self.header['samples'] - end_time)
        end_index = np.argmin(differences)  
        # Threshold equal to twice the max noise level in the target region
        threshold = threshold_to_noise*np.max(self.traces[ref_trace,start_index:end_index]) 
        
        del_list = np.array([])
        for i in range(np.shape(self.traces)[0]):
            max_amp = np.max(abs(self.traces[i,start_index:-1]))
            if max_amp < threshold:
                del_list = np.append(del_list,i)
        print('Removing traces...\n')
        del_list = del_list.astype(int)
        self.remove_traces(del_list)

# %% Various
    def create_velocity_model_2D(self, model):
        print('Create velocity model...')
        # Takes in the defined velocity model and interpolates to fit the seismic data
        t = self.header['samples']
        dt = t[1]-t[0]
        self.vel_rms_1 = np.zeros_like(t)
        self.vel_rms_2 = np.zeros_like(t)
        self.z_t = np.zeros_like(t)
        self.vel = np.zeros_like(t)
        
        # Create a velocity matrix
        it_0 = 0
        for li in range(len(model.layers)):
            travel_time = model.travel_time_total[li]
            it = int(travel_time/dt)
            self.vel[it_0:it] = model.layers[li].velocity
            it_0 = it

        # Calculate the rms velocities
        for ti in range(1,len(t)):
            self.vel_rms_2[ti] = t[ti]**(-1)*dt*self.vel[ti]**2 + t[ti]**(-1) * t[ti-1]*self.vel_rms_2[ti-1]

            
        self.vel_rms_2[0] = self.vel_rms_2[1]        
        self.vel_rms_1 = np.sqrt(self.vel_rms_2)
        print('Done.\n')
    
    def create_velocity_model_3D(self, ilines, xlines, model, extent):
        # Creates a velocity model in the seis structure
        # Assumes the model is a [x,y,z] defined matrix
        # Function resamples the x,y points to the CMP, and calculates the rms velocities for each point
        print('Create velocity model...')
        t = self.header['samples']
        dt = t[1]-t[0]
        (nx,ny,nz) = np.shape(model) # Number of pixels in the 
        
        x,y,z = extent # Length of model in each dimension
        # dx = x/nx # Pixel increments 
        # dy = y/ny
        # dz = z/nz
        
        # Reformat model into a ilines,xlines model
        model_new = resize_matrix(model, (ilines,xlines,len(t)))
        dz_new = z/len(t)
        # model_new = reduce_matrix_to_shape(model, (ilines,xlines,len(t)))
        tau = np.zeros_like(model_new) # Travel-time to each depth point
        self.vel_rms_1 = np.zeros_like(model_new)
        self.vel_rms_2 = np.zeros_like(model_new)
        self.z_t = np.zeros_like(model_new)
        self.vel = np.zeros_like(model_new)
        
        # Calculate two-way travel-times for each iline,xline point
        for i in range(ilines):
            for x in range(xlines):
                travel_time = 0
                for zi in range(len(t)):
                    # d = np.sqrt((i*dx)**2 + (x*dy)**2 + (dz*zi)**2)
                    travel_time += 2*dz_new/model_new[i,x,zi] # Two-way travel-time to pint iline,xline,t
                    tau[i,x,zi] = travel_time  
        
        # Assign velocity information to the seis object, at the correct traveltimes found in tau
        for i in range(ilines):
            for x in range(xlines):
                it_0 = 0
                for zi in range(len(t)):
                    it = int(tau[i,x,zi]/dt) # Index in time array for this travel-time
                    self.vel[i,x,it_0:it] = model_new[i,x,zi]
                    it_0 = it
        

        # Calculate the rms velocities
        for i in range(ilines):
            for x in range(xlines):
                for ti in range(1,len(t)):
                    self.vel_rms_2[i,x,ti] = t[ti]**(-1)*dt*self.vel[i,x,ti]**2 + t[ti]**(-1) * t[ti-1]*self.vel_rms_2[i,x,ti-1]
       
            
        self.vel_rms_2[:,:,0] = self.vel_rms_2[:,:,1]        
        self.vel_rms_1 = np.sqrt(self.vel_rms_2)
        print('Done.\n')
        return model_new,tau
        
    def plot_2D_velocity_model(self):
        t = self.header['samples']
        
        plt.figure()
        plt.plot(t*1000,self.vel_rms_1)
        plt.title('RMS velocity')
        plt.xlabel('Time (ms)')
        
        plt.figure()
        plt.plot(t*1000,self.vel)
        plt.title('Layer velocity')
        plt.xlabel('Time (ms)')
        
        ## Create velocity model in 2D
        v_rms_m = np.tile(self.vel_rms_1, (100, 1))
        plt.figure()
        plt.imshow(v_rms_m.T,aspect='auto')
    

    
    def plot_3D_velocity_model(self,iline,xline):
        t = self.header['samples']
        x_min = 0
        x_max = len(self.vel_rms_1[0,:,0])
        
        plt.figure()
        plt.imshow(self.vel_rms_1[iline,:,:].T,aspect='auto', extent=(x_min,x_max,max(t*1000),min(t*1000)), cmap='inferno', vmin = 1400)
        plt.title('Iline: ' + str(iline))
        plt.xlabel('Xlines')
        plt.ylabel('Time (ms)')
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.001)

        x_max = len(self.vel_rms_1[:,0,0])
        plt.figure()
        plt.imshow(self.vel_rms_1[:,xline,:].T,aspect='auto',  extent=(x_min,x_max,max(t*1000),min(t*1000)), cmap='inferno', vmin = 1400)
        plt.title('Xline: ' + str(xline))
        plt.xlabel('Ilines')
        plt.ylabel('Time (ms)')
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.001)

        plt.figure()
        plt.plot(t*1000,self.vel_rms_1[iline,xline,:])
        plt.title('RMS velocity')
        plt.xlabel('Time (ms)')
        plt.show(block=False)
        plt.pause(0.001)

        plt.figure()
        plt.plot(t*1000,self.vel[iline,xline,:])
        plt.title('Layer velocity')
        plt.xlabel('Time (ms)')
        plt.show(block=False)
        plt.pause(0.001)
        
        
        # # Create a 2D birds view representation
        # vel_matrix_2D = np.sum(self.vel,axis=2)
        # plt.figure()
        # plt.imshow(vel_matrix_2D,aspect='auto')
        # plt.show()
        

    def seismic_semblance_wrapper(self, ix, iy, vel_min=1000, vel_max=5000, n_vel=80, win_len_ms=20, plot=True):
        """
        Example method to add to Seismic class.
        Computes semblance for CMP at bin indices (ix, iy).
        Assumes self.cmp_gather has shape (nx, ny, n_offsets, nt) and self.cmpOffset array exists (m),
        and self.header['samples'] is the time vector in seconds.
        """
        # extract gather
        cmp = self.cmp_gather[ix, iy, :, :]   # shape (n_offsets, nt)
        offsets = np.array(self.cmpOffset)    # m
        t = np.array(self.header['samples'])  # seconds
    
        # velocity vector
        vel_vec = np.linspace(vel_min, vel_max, n_vel)
    
        # window length in samples
        dt = t[1] - t[0]
        win_len_samples = max(1, int(round(win_len_ms / 1000.0 / dt)))
    
        sembl, vel_vec, t = semblance_scan(cmp, offsets, t, vel_vec, win_len_samples=win_len_samples, apply_rms_smoothing=True)
    
        if plot:
            plot_semblance(semblance=sembl, vel_vec=vel_vec, t=t)
    
        return sembl, vel_vec, t
    

    def plot_header(self):
        """
        Create a bird-view plot of where the source-receiver pairs are located in space.

        Returns
        -------
        None.

        """
        # Scatter plot sources and receivers color-coded on their number
        if self.header['groupX'] is None:
            sys.exit('Insufficient header information. Read the segy file first.')
        plt.figure()
        plt.scatter(self.header['sourceX'], self.header['sourceY'], c=self.header['nsum'], edgecolor='blue')
        plt.scatter(self.header['groupX'], self.header['groupY'], c=self.header['nstack'], edgecolor='none')
        plt.xlabel('X')
        plt.ylabel('Y')

    def plot_trace(self, trace_nr):
        # Plots on specific trace (trace_nr)
        plt.figure()
        plt.plot(self.header['samples']*1000,self.traces[trace_nr, :])
        plt.xlabel('Time (ms)')
        plt.ylabel('Ampltiude')
        plt.title('Trace: ' + str(trace_nr))
        plt.show()

    
        
    
    def plot_all_traces(self, DB = True, RMS = True, AMP = True, noiseDB=-np.inf, interpolationF = 1, GaussKernel = 0, window_size = 8, start_trace = 0, end_trace = 0, block_size = 3000, trans_id = None):
        """
        Function plots all traces in the seis file. 
        Parameters:
        noiseDB: noise level in DB. Used for plotting.
        
        """
        def select_trans_id(self, target_id):
            """
            Return a new Seismic object with only traces matching a given trans_id.
            """
            seis_filtered = Seismic()
        
            # Boolean mask of traces matching the chosen trans_id
            mask = (self.header['trans_id'] == target_id)
        
            # Apply mask to traces
            seis_filtered.traces = self.traces[mask]
        
            # Copy all per-trace headers with the mask applied
            for key in ['sourceX', 'sourceY', 'groupX', 'groupY',
                        'RcvElev', 'SrcElev', 'nsum', 'nstack', 'trans_id']:
                seis_filtered.header[key] = self.header[key][mask]
        
            # Copy metadata that isn’t per-trace
            seis_filtered.header['samples'] = self.header['samples']
            seis_filtered.filename = self.filename
        
            print(f"Selected {np.sum(mask)} traces with trans_id = {target_id}")
            return seis_filtered

        Nt = len(self.header['samples']) # Number of samples
        dt = self.header['samples'][1]-self.header['samples'][0] # Sapling time
        t_arr = np.linspace(0,dt*Nt,Nt) # Remove amplificaiton factor, and convert to ms
        noise = dB_inv(noiseDB)
        
        # Filter to one specific transducer
        if not trans_id is None:
            seis_filtered = select_trans_id(self, trans_id)
            data = seis_filtered.traces
            
            if seis_filtered.traces.size == 0:
                print(f"No traces found for trans_id = {trans_id}")
                return seis_filtered  # or skip plotting
        else:
            data = self.traces
            
        # Interpolate data
        data = interpolate_data(data, interpolationF, GaussKernel)        
        
        n_traces = data.shape[0]    # assuming shape (n_traces, n_samples)
        if end_trace == 0:
            end_trace = n_traces
        end_block = int(np.ceil(end_trace / block_size))
        start_block = int(np.ceil(start_trace / block_size))
        
        # Plot blocks
        for i in range(start_block,end_block):
            start_idx = i * block_size
            end_idx = min((i+1) * block_size, n_traces)
        
            # Subset the matrix
            data_block = data[start_idx:end_idx, :]
            data_block_dB = dB(data_block)
            data_RMS_block = matrix_RMS(data_block, window_size)
            
            
            if DB == True:
                ## Energy data
                energy_block = np.zeros(np.shape(data_block))
                for trc in range(len(data_block[:,0])):
                    for k in range(8,len(data_block[0,:])):
                        energy_block[trc,k] = np.sum(abs(data_block[trc,k-8:k]))
                energy_block = 20*np.log(abs(np.double(energy_block)))
                # max_val = np.max(trace_data)
                # noise_level = find_noise_level(signal)
            
            
            
                # Plot dB and Energy
                fig = plt.figure(figsize=(10,5))
                fig.add_subplot(1,2,1)
                plt.imshow(data_block_dB.T, aspect='auto', 
                           extent=(start_idx+1, end_idx, 1000*t_arr[-1], 0), 
                           cmap='seismic')
                plt.colorbar(label='Amplitude (dB)')
                plt.xlabel('Traces')
                
                fig.add_subplot(1,2,2)
                plt.imshow(energy_block.T, aspect='auto', 
                           extent=(start_idx+1, end_idx, 1000*t_arr[-1], 0), 
                           cmap='seismic')
                plt.colorbar(label='Energy (dB)')
                plt.xlabel('Traces')
                
                plt.suptitle(f'Block {i+1}/{end_block-start_block}')
                plt.show()
            
            if AMP == True:
                # Plot raw data
                
                threshold = np.percentile(np.abs(data_block), 95)
                plt.figure(figsize=(8,5))
                plt.imshow(data_block.T, aspect='auto', 
                           extent=(start_idx+1, end_idx, 1000*t_arr[-1], 0), 
                           cmap='seismic', vmax=threshold, vmin=noise)
                plt.xlabel('Traces')
                plt.ylabel('Time (ms)')
                plt.title(f'Amplitude - Block {i+1}/{end_block-start_block}')
                plt.show()
            
            if RMS == True:
                # Plot RMS
                threshold = np.percentile(np.abs(data_RMS_block), 95)
                plt.figure(figsize=(8,5))
                plt.imshow(data_RMS_block.T, aspect='auto', 
                           extent=(start_idx+1, end_idx, t_arr[-1], 0), 
                           cmap='seismic', vmax=threshold, vmin=noise)
                plt.xlabel('Traces')
                plt.ylabel('Time (ms)')
                plt.title(f'RMS - Block {i+1}/{end_block-start_block}')
                plt.show()
            
            plt.pause(0.1)  # pause for 0.1 seconds (or any small delay you want)

            
        # ## Plotting
        # data_dB[data_dB<noiseDB] = noiseDB        # hist_data = plt.hist(trace_data.flatten(),50)
        # energy_data[energy_data<noiseDB] = noiseDB        # hist_data = plt.hist(trace_data.flatten(),50)
        # fig = plt.figure()
        # fig.add_subplot(1,2,1)
        # plt.imshow(data_dB.T, aspect='auto', extent=(1,len(data)+1,t_arr[-1],0), cmap ='RdBu_r') #  norm = colors.LogNorm
        # plt.colorbar(label='Amplitude (dB)')
        # plt.xlabel('Traces')
        # fig.add_subplot(1,2,2)
        # plt.imshow(energy_data.T, aspect='auto', extent=(1,len(data)+1,t_arr[-1],0), cmap ='RdBu_r') #  norm = colors.LogNorm
        # plt.colorbar(label='Energy (dB)')
        # plt.xlabel('Traces')
        
        # # Raw data 
        # noise = dB_inv(noiseDB)
        # threshold = np.percentile(np.abs(data), 95)
        # plt.figure()
        # plt.imshow(data.T,aspect='auto', extent=(1,len(data)+1,t_arr[-1],0), cmap ='RdBu_r', vmax = threshold, vmin = noise)
        # plt.xlabel('Traces')
        # plt.ylabel('Time (ms)')
        # plt.title('Amplitude')
        # plt.show()
        
        # # RMS
        # noise = dB_inv(noiseDB)
        # data_RMS = matrix_RMS(data, window_size)
        # threshold = np.percentile(np.abs(data_RMS), 95)
        # plt.figure()
        # plt.imshow(data_RMS.T,aspect='auto', extent=(1,len(data_RMS)+1,t_arr[-1],0), cmap ='RdBu_r', vmax = threshold, vmin = noise)
        # plt.xlabel('Traces')
        # plt.ylabel('Time (ms)')
        # plt.title('RMS')
        # plt.show()
        

    def wiggle_plot(self, data, traces_arr):
        """
        Create a "wiggle-plot" of a selection of the traces in the seis object, for visualization.

        Parameters
        ----------
        data : TYPE
            A bit unecessary, but the data within self.traces.
        traces_arr : TYPE
            Array describing which traces in self.traces to be plotted.

        Returns
        -------
        None.

        """
        # Plots the traces in the traces_arr in a wiggle plot
        fig,ax = plt.subplots()
        Nt = len(self.header['samples']) # Number of samples
        dt = self.header['samples'][1]-self.header['samples'][0] # Sapling time
        t_arr = np.linspace(0,dt*Nt,Nt)*1000 # Remove amplificaiton factor, and convert to ms
        for i in range(len(traces_arr)):
            trace_i = traces_arr[i]
            y = t_arr
            offset = 2*np.max(abs(data))*i
            x = data[trace_i,:] + offset
            ax.plot(x,y,'k-')
            ax.fill_betweenx(y,offset,x,where=(x>offset),color='k')

        plt.gca().invert_yaxis()
        plt.xticks([1,2,3,4,5])
        xticks_ = [2*np.max(abs(data))*i for i in range(len(traces_arr))]
        xtick_labels = [str(traces_arr[i]) for i in range(len(traces_arr))]
        plt.xticks(xticks_,xtick_labels)
        plt.xlabel('Trace nr')
        plt.ylabel('Time (ms)')
        plt.show(block=False)
        plt.pause(0.001)

    def write_segy_importedFunction(self, filename_in):
        # This function is imported from the tutorial page, and doesnt work.
        # Its kept here purely as a reference tool.
        # https://segyio.readthedocs.io/en/latest/segyio.html?highlight=format#trace-header-and-attributes
        if len(sys.argv) < 7:
            sys.exit("Usage: {} [file] [samples] [first iline] [last iline] [first xline] [last xline]".format(sys.argv[0]))

        spec = segyio.spec()
        filename_in = sys.argv[1]

        # to create a file from nothing, we need to tell segyio about the structure of
        # the file, i.e. its inline numbers, crossline numbers, etc. You can also add
        # more structural information, but offsets etc. have sensible defautls. This is
        # the absolute minimal specification for a N-by-M volume
        spec.sorting = 2
        spec.format = 1
        spec.samples = range(int(sys.argv[2]))
        spec.ilines = range(*map(int, sys.argv[3:5]))
        spec.xlines = range(*map(int, sys.argv[5:7]))

        with segyio.create(filename_in, spec) as f:
            # one inline consists of 50 traces
            # which in turn consists of 2000 samples
            start = 0.0
            step = 0.00001
            # fill a trace with predictable values: left-of-comma is the inline
            # number. Immediately right of comma is the crossline number
            # the rightmost digits is the index of the sample in that trace meaning
            # looking up an inline's i's jth crosslines' k should be roughly equal
            # to i.j0k
            trace = np.arange(start = start,
                              stop  = start + step * len(spec.samples),
                              step  = step,
                              dtype = np.single)

            # Write the file trace-by-trace and update headers with iline, xline
            # and offset
            tr = 0
            for il in spec.ilines:
                for xl in spec.xlines:
                    f.header[tr] = {
                        segyio.su.offset : 1,
                        segyio.su.iline  : il,
                        segyio.su.xline  : xl
                    }
                    f.trace[tr] = trace + (xl / 100.0) + il
                    tr += 1

            f.bin.update(
                tsort=segyio.TraceSortingFormat.INLINE_SORTING
            )
        
    def add_seismicData(self,seismic_new):
        # This function adds another seismic object to "self"
        if np.size(self.traces) == 1: # In case the self object is empty
            self.header['sourceX'] = seismic_new.header['sourceX']
            self.header['sourceY'] = seismic_new.header['sourceY']
            self.header['groupX'] = seismic_new.header['groupX']
            self.header['groupY'] = seismic_new.header['groupY']
            # self.header['format'] = seismic_new.header['format']
            self.header['samples'] = seismic_new.header['samples']
            self.header['nsum'] = seismic_new.header['nsum']
            self.header['nstack'] = seismic_new.header['nstack']
            self.header['RcvElev'] = seismic_new.header['RcvElev']
            self.header['SrcElev'] = seismic_new.header['SrcElev']
            self.traces = seismic_new.traces
            self.header['trans_id'] = seismic_new.header['trans_id']
        else:
            self.header['sourceX'] = np.append(self.header['sourceX'],seismic_new.header['sourceX'])
            self.header['sourceY'] = np.append(self.header['sourceY'],seismic_new.header['sourceY'])
            self.header['groupX'] = np.append(self.header['groupX'],seismic_new.header['groupX'])
            self.header['groupY'] = np.append(self.header['groupY'],seismic_new.header['groupY'])
            self.header['RcvElev'] = np.append(self.header['RcvElev'],seismic_new.header['RcvElev'])
            self.header['SrcElev'] = np.append(self.header['SrcElev'],seismic_new.header['SrcElev'])
            self.header['nsum'] = np.append(self.header['nsum'],seismic_new.header['nsum'])
            self.header['nstack'] = np.append(self.header['nstack'],seismic_new.header['nstack'])
            self.header['trans_id'] = np.append(self.header['trans_id'],seismic_new.header['trans_id'])
            self.traces = np.append(self.traces,seismic_new.traces,axis=0)
            
        # Erase migrated and stacked data (no longer relevant)
        [self.__setattr__(k, None) for k in self.__dict__ if k not in {'header', 'traces', 'filename', 'scaling_factor', 'vel_rms_1', 'vel_rms_2', 'vel'}]

        pass
   
    def expand_spatial_ROI(self, xmin=None, xmax=None, ymin=None, ymax=None):
        """
        Expand the ROI of the seismic dataset by adding ghost traces (zero amplitude)
        at the bounding box defined by [xmin, xmax] and [ymin, ymax].
        Ghost traces are zero-offset (source = receiver).
    
        Parameters
        ----------
        xmin, xmax : float or None
            Minimum and maximum X-coordinates. If None, use current dataset min/max.
        ymin, ymax : float or None
            Minimum and maximum Y-coordinates. If None, use current dataset min/max.
        """
        n_samples = self.traces.shape[1]  # number of time samples per trace
        ghost_trace = np.zeros(n_samples)
    
        # Use dataset min/max if arguments are None
        if xmin is None: xmin = np.min(self.header['sourceX'])
        if xmax is None: xmax = np.max(self.header['sourceX'])
        if ymin is None: ymin = np.min(self.header['sourceY'])
        if ymax is None: ymax = np.max(self.header['sourceY'])
    
        # Define bounding box corners
        corners = [
            (xmin, ymin),  # bottom-left
            (xmax, ymin),  # bottom-right
            (xmin, ymax),  # top-left
            (xmax, ymax)   # top-right
        ]
    
        for (x, y) in corners:
            # Append zero-amplitude trace
            self.traces = np.vstack([self.traces, ghost_trace])
    
            # Append headers with zero-offset geometry
            self.header['sourceX'] = np.append(self.header['sourceX'], x)
            self.header['sourceY'] = np.append(self.header['sourceY'], y)
            self.header['groupX'] = np.append(self.header['groupX'], x)
            self.header['groupY'] = np.append(self.header['groupY'], y)
    
            # Safe defaults for other keys
            for key in ['RcvElev', 'SrcElev', 'nsum', 'nstack', 'trans_id']:
                if key in self.header:
                    if key in ['nsum', 'nstack']:
                        self.header[key] = np.append(self.header[key], 0)
                    elif key == 'trans_id':
                        self.header[key] = np.append(self.header[key], -1)  # mark ghost
                    else:
                        self.header[key] = np.append(self.header[key], self.header[key][0])
    
        # Clear migrated/stacked data
        [self.__setattr__(k, None) for k in self.__dict__ if k not in 
         {'header', 'traces', 'filename', 'scaling_factor', 'vel_rms_1', 'vel_rms_2', 'vel'}]
    
        print(f"Added {len(corners)} ghost traces (zero-offset) to expand ROI "
              f"to X=[{xmin}, {xmax}], Y=[{ymin}, {ymax}].")
    
    def find_trace_elevation(self, threshold_percentile,min_height,max_height, water_col):
        n_traces = np.shape(self.traces)[0]
        d = deque()
        sx_ref = self.header['sourceX'][0] # Reference position
        sy_ref = self.header['sourceY'][0] # Reference position
        for trace in range(n_traces):
            sx = self.header['sourceX'][trace]
            sy = self.header['sourceY'][trace]
            if sx==sx_ref and sy==sy_ref: # If the source has not changed, add to queue
                d.append(trace)
            else: # If the source has changed, 
                # Change source elevation for all within the group
                elev = find_source_elev(self,d,threshold_percentile,min_height,max_height)
                # Correct for the elevation
                self.header['SrcElev'][d] = elev
                self.header['RcvElev'][d] = elev # Assume the receiver has the same elevation. Fine for small offsets
                # Clear the queue
                d.clear()
                sx_ref = self.header['sourceX'][trace] # Reference position
                sy_ref = self.header['sourceY'][trace] # Reference position
                d.append(trace)
        # When finished, change source elevation for all traces remaining in the group
        elev = find_source_elev(self,d,threshold_percentile,min_height,max_height)
        self.header['SrcElev'][d] = elev
        d.clear()
            
    def height_adjustment(self, water_col):
        # Removes the height of the transducer header, and adds it to the trace instead
        # water_col: water_column in the velocity model
        t = self.header['samples']
        dt = t[1]-t[0]
        vel = 1481 # Assumes the correction is done in water
        
        for i in range(len(self.traces)):
            z_R = water_col - self.header['RcvElev'][i] # z-position of receiver
            z_S = water_col - self.header['SrcElev'][i] # z-position of soucre
            if z_R is None:
                z_R = 0
            if z_S is None:
                z_S = 0
            
            total_shift = (z_R + z_S)
            
            # Shift trace depending on the elevation midpoint
            shift_t = int((total_shift/vel)/dt)
            # add_zeros = np.zeros(abs(shift_t))
            if shift_t>0: # Add zeros to compensate for reduced height
                # temp_trace = np.concatenate((add_zeros, self.traces[i,:]))
                # temp_trace = np.delete(temp_trace,range(len(self.traces[i,:])-shift_t,len(self.traces[i,:])))
                # Prepend zeros and trim the end to maintain shape
                new_arr = np.concatenate([np.zeros(shift_t), self.traces[i,:]])[:len(self.traces[i,:])]
                self.traces[i,:] = new_arr
            elif shift_t<0: # Remove first numbers to compensate for increased height
                # temp_trace = np.delete(self.traces[i,:],range(abs(shift_t)))
                # temp_trace = np.concatenate((temp_trace,add_zeros))
                new_arr = np.concatenate([self.traces[i,shift_t:], np.zeros(shift_t)])
                self.traces[i,:] = new_arr
            
            # Reset at the end
            self.header['RcvElev'][i] = 0
            self.header['SrcElev'][i] = 0
            

    def sort_to_bin(self, x_bin, y_bin, num_offsets = 10):
        """
        Sorts the data into a pre-defined grid of x_bin by y_bin pixels, where each pixels contain one trace containing many overlapping traces.

        Parameters
        ----------
        x_bin : int
            Number of pixels/ilines/midpoints in x-direction.
        y_bin : int
            Number of pixels/xlines/midpoints in y-direction.
        num_offsets : int, optional
            Number of offset bins. The default is 10.

        Returns
        -------
        None.

        """
        print('Sorting data to binning grid...\n')
        # Organize binning and positional arrays
        self.cmpX = np.linspace(np.min(self.header['groupX']),np.max(self.header['groupX']),x_bin,  dtype=np.float32)
        self.cmpY = np.linspace(np.min(self.header['groupY']),np.max(self.header['groupY']),y_bin,  dtype=np.float32)
        self.cmpOffset = np.linspace(0,0.5*np.sqrt(
            (np.max(self.header['groupY']) - np.min(self.header['groupY']))**2 + 
            (np.max(self.header['groupX']) - np.min(self.header['groupX']))**2),num_offsets)
        self.fold = np.zeros([x_bin, y_bin])
        self.cmp_gather = np.zeros([x_bin, y_bin,num_offsets, len(self.traces[0])])
        
        # Sort to midpoints
        for i in range(1,len(self.traces),1):
            print('\r',np.round(100*i/(len(self.traces)),1), '% complete', end='')
            
            # Extract useful information from trace i
            x_S = self.header['sourceX'][i] # x-position of source
            x_R = self.header['groupX'][i] # x-position of receiver
            y_S = self.header['sourceY'][i] # y-position of source
            y_R = self.header['groupY'][i] # y-position of receiver
            
            # Find midpoints
            mid_point_x = (x_S + x_R)/2
            mid_point_y = (y_S + y_R)/2
            offset = np.sqrt(np.power(mid_point_x-x_S,2) + np.power(mid_point_y-y_S,2)) # Offset to the midpoint

            # Find position in binning grid the midpoints are nearest to
            # array = np.asarray(x_bin_pos)
            id_x = (np.abs(self.cmpX - mid_point_x)).argmin() 
            # array = np.asarray(y_bin_pos)
            id_y = (np.abs(self.cmpY - mid_point_y)).argmin() 
            id_offset = (np.abs(self.cmpOffset - offset)).argmin() 
            
            # Add trace to the closest point on the binning grid
            self.cmp_gather[id_x,id_y,id_offset,:] += self.traces[i,:]
            self.fold[id_x,id_y] = self.fold[id_x,id_y] + 1
            
        print('\ndone\n')
    
            
            
    
    def NMO_stacking(self, vel_mod, OffsetAperture = None, norm = True, parallel = False):
        # Stacks the offset to the CMP, while applying an aperture defining the maximum offset
        print('Stacking data after NMO correction...\n')
        if self.cmp_gather is None:
            sys.exit('Need to sort to CMP points before stacking!\n')
        
        if vel_mod is None:
            sys.exit('Upload a velocity model first.')
            
        if OffsetAperture is None:
            OffsetAperture = np.max(self.cmpOffset)
        
        # Organize binning and positional arrays
        t = self.header['samples']
        dt = t[1]-t[0]
                
        Nx = len(self.cmpX)
        Ny = len(self.cmpY)
        No = len(self.cmpOffset)
        Nt = len(t)
        
        # Calculate RMS values
        vel_rms = np.zeros([Nx, Ny, No, Nt])
        for i in range(Nx):
            for j in range(Ny):
                for o in range(No):
                    if self.cmpOffset[o] <= OffsetAperture:
                        # for it in range(len(t)):
                            # t0 = t[it]
                            # vel_rms[i,j,o,it] = vel_mod.retrieveRMSVelAtPoint(self.cmpX[i], self.cmpY[j], vel_mod._find_depth_for_travel_time(self.cmpX[i],self.cmpY[j], t0, 0.5*t0*2600))
                        t_i, vRMS = vel_mod.retrieveRMSVelAtTime(self.cmpX[i], self.cmpY[j])
                        t_new = np.linspace(t_i[0], t_i[-1], Nt)
                        V_rms_new = np.interp(t_new, t_i, vRMS)
                        vel_rms[i,j,o,:] = V_rms_new
                            
        # Sort to midpoints
        cmp_stacked = np.zeros([Nx,Ny,Nt],  dtype=np.float32)
        cmp_gather_NMO = np.zeros([Nx,Ny,No,Nt],  dtype=np.float32)
        
        if parallel is True:
            args = [(self.cmp_gather, Nx, Ny, No, Nt, self.cmpOffset, OffsetAperture, t, dt, vel_rms[i,:,:,:], i) for i in range(Nx)]
            total_count = 0
            with mp.Pool() as pool:
                results = pool.imap(NMOstacking_worker, args)
                for result in results:
                    # cmp_stacked_i, cmp_gather_NMO = result
                    cmp_stacked += result
                    total_count += 1
                    print('Count: ', total_count)
        else:
            total_count = 0
            for i in range(Nx):
                args = (self.cmp_gather, Nx, Ny, No, Nt, self.cmpOffset, OffsetAperture, t, dt, vel_rms[i,:,:,:], i)
                # cmp_stacked_i, cmp_gather_NMO = NMOstacking_worker(args)
                cmp_stacked += NMOstacking_worker(args)
                total_count += 1
                print('Count: ', total_count)
        
        self.cmp_stacked = cmp_stacked
        
        if norm == True:
            self.cmp_stacked[i,j,:] = self.cmp_stacked[i,j,:]/np.sum(abs(self.cmp_stacked[i,j,:]))
        
        print('\ndone\n')
        
        
    
    def plot_fold(self, fold = True, scatter = False):
        # Scatter plot midpoints color-coded on their number
        if self.header['groupX'] is None:
            sys.exit('Insufficient header information. Read the segy file first.')
        if scatter is True:
            plt.figure()
            for i in range(0,len(self.cmpX)):
                for j in range(0,len(self.cmpY)):
                    # cmpX = self.cmpX[i]
                    # cmpY = self.cmpY[i]
                    # plt.scatter(self.cmpX, self.cmpY, c=self.header['nsum'], edgecolor='blue')
                    plt.scatter(self.cmpX[i], self.cmpY[j], edgecolor='none')
            plt.xlim([np.min(self.cmpX),np.max(self.cmpX)])
            plt.ylim([np.min(self.cmpY),np.max(self.cmpY)])
            plt.figure()
            plt.xlabel('Ilines')
            plt.ylabel('Xlines')
        if fold is True:
            plt.imshow(self.fold.T,aspect='auto')
        plt.show(block=False)

    def plot_stacked_data(self, line, axis = 'x', DB = False, AMP = False, RMS = False, AMPlim = None, noiseDB=-np.inf, interpolationF = 1, GaussKernel = 0, threshold_percentile=99):
        """
        Plot function.

        Parameters
        ----------
        iline : TYPE
            DESCRIPTION.
        xline : TYPE
            DESCRIPTION.
        noiseDB : TYPE, optional
            DESCRIPTION. The default is -np.inf.
        interpolationF : TYPE, optional
            DESCRIPTION. The default is 1.
        GaussKernel : TYPE, optional
            DESCRIPTION. The default is 0, which means no filtering. Set to (0.5,0.5,0) for standard smoothening.

        Returns
        -------
        None.

        """
        # x_min = np.min(self.header['groupX'])
        # x_max = np.max(self.header['groupX'])
        # y_min = np.min(self.header['groupY'])
        # y_max = np.max(self.header['groupY'])
        t_arr = self.header['samples']*1000 # To plot in ms
        data = self.cmp_stacked
        
        if axis=='y':
            y_min = np.min(self.header['groupX'])
            y_max = np.max(self.header['groupX'])
            data = data[:,line,:]
        elif axis == 'x':
            y_min = np.min(self.header['groupY'])
            y_max = np.max(self.header['groupY'])
            data = data[line,:,:]
            
        data = interpolate_data(data, interpolationF, GaussKernel) 
        
        # Plots the stacked data created in "sort-to-bin" methods
        if AMPlim is None:
            AMPlim = np.max(data)
        
        
        fig_size = (12,8)
        if AMP is True:
            plt.figure(figsize=(fig_size))
            plt.title('Stacked data. Iline: ' + str(line))
            plt.imshow(data.T,cmap ='RdBu_r',extent = [y_min,y_max,t_arr[-1],t_arr[0]], aspect='auto',vmin=-AMPlim,vmax=AMPlim)
            plt.xlabel('Distance (m) along xlines')
            plt.ylabel('Time (ms)')
            plt.colorbar(label='Amplitude')
            plt.show(block=False)
            plt.pause(0.001)
        if DB is True:
            stacked_DB = 20*np.log(abs(np.double(data)))        
            stacked_DB[stacked_DB<noiseDB] = noiseDB 
            plt.figure(figsize=fig_size)
            plt.title('Stacked data. Iline: ' + str(line))
            plt.imshow(stacked_DB.T,cmap ='RdBu_r', extent = [y_min,y_max,t_arr[-1],t_arr[0]], aspect='auto')
            plt.xlabel('Distance (m) along xlines')
            plt.ylabel('Time (ms)')
            plt.colorbar(label='Amplitude (dB)')
            plt.show(block=False)
            plt.pause(0.001)
        if RMS is True:
            img_RMS = matrix_RMS(data,8)
            threshold = np.percentile(np.abs(img_RMS), threshold_percentile)
            noise_RMS = dB_inv(noiseDB)  
            plt.figure(figsize=fig_size)
            plt.title('Stacked data. Iline: ' + str(line))
            plt.imshow(img_RMS.T,cmap ='RdBu_r',extent = [y_min,y_max,t_arr[-1],t_arr[0]], aspect='auto',vmin = noise_RMS, vmax=threshold)
            plt.xlabel('Distance (m) along xlines')
            plt.ylabel('Distance (m)')
            plt.colorbar(label='Amplitude (RMS)')
            plt.show()
    
        # if AMP is True:
        #     plt.figure(figsize=fig_size)
        #     plt.title('Stacked data. Xline: ' + str(xline))
        #     plt.imshow(data[:,int(xline),:].T,cmap ='RdBu_r', extent = [x_min,x_max,t_arr[-1],t_arr[0]], aspect='auto',vmin=-AMPlim,vmax=AMPlim)
        #     plt.xlabel('Distance (m) along ilines')
        #     plt.ylabel('Time (ms)')
        #     plt.colorbar(label='Amplitude')
        #     plt.show(block=False)
        #     plt.pause(0.001)

        # if DB is True:
        #     plt.figure(figsize=fig_size)
        #     plt.title('Stacked data. Xline: ' + str(xline))
        #     plt.imshow(stacked_DB[:,int(xline),:].T,cmap ='RdBu_r', extent = [x_min,x_max,t_arr[-1],t_arr[0]], aspect='auto')
        #     plt.xlabel('Distance (m) along ilines')
        #     plt.ylabel('Time (ms)')
        #     plt.colorbar(label='Amplitude (dB)')
        #     plt.show(block=False)
        #     plt.pause(0.001)
        # if RMS is True:
        #     img_RMS = matrix_RMS(data[:,int(xline),:],8)  
        #     threshold = np.percentile(np.abs(img_RMS), threshold_percentile)
        #     noise_RMS = dB_inv(noiseDB)         
        #     # mig_img_RMS[mig_img_RMS<noise_RMS] = noise_RMS  
        #     plt.figure(figsize=fig_size)
        #     plt.title('Stacked data. Xline: ' + str(xline))
        #     plt.imshow(img_RMS.T,cmap ='RdBu_r',extent = [x_min,x_max,t_arr[-1],t_arr[0]], aspect='auto',vmin = noise_RMS, vmax=threshold)
        #     plt.xlabel('Distance (m) along xlines')
        #     plt.ylabel('Distance (m)')
        #     plt.colorbar(label='Amplitude (RMS)')
        #     plt.show()

    def plot_birdsView(self,data, z_axis, z_low, z_high):
        # Assumes 3D data
        (ilines,xlines,depth) = np.shape(data)
        low_idx = np.argmin(np.abs(z_axis-z_low*0.001))
        high_idx = np.argmin(np.abs(z_axis-z_high*0.001))
        data_DB = 20*np.log(abs(np.double(data)))      
        noiseDB = -800
        data_DB[data_DB<noiseDB] = noiseDB 
        
        data_sum = np.zeros_like(data[:,:,0])
        for i in range(ilines):
            for j in range(xlines):
                data_sum[i,j] = np.sum(np.abs(data[i,j,:]))
                
        data_sum_DB = np.zeros_like(data[:,:,0])
        for i in range(ilines):
            for j in range(xlines):
                data_sum_DB[i,j] = np.sum(np.abs(data_DB[i,j,:]))
                
                
        plt.figure()
        plt.imshow(data_sum)
        plt.colorbar(label='Amplitude')
        
        plt.figure()
        plt.imshow(data_sum_DB)
        plt.colorbar(label='Amplitude (dB)')

    def plot_cmp_gather(self,iline,xline,DB = False, noiseDB=-np.inf, NMO = False):
        # Plots the offsets for a specific gather
        offset_min = np.min(self.cmpOffset)
        offset_max = np.max(self.cmpOffset)
        t_arr = self.header['samples']
        
        
        if DB is True:
            # Convert to decibel
            gather = 20*np.log(abs(np.double(self.cmp_gather)))
            gather[gather<noiseDB] = noiseDB 
        else:
            gather = self.cmp_gather
        
        # Plots the stacked data created in "sort-to-bin" methods
        plt.figure()
        plt.title('Uncorrected gather.' + 'iline: ' + str(iline) + ', xline: ' + str(xline))
        plt.imshow(gather[int(iline),int(xline),:,:].T,cmap ='RdBu_r',extent = [offset_min,offset_max,t_arr[-1],t_arr[0]], aspect='auto',vmin=np.min(gather[iline,xline,:,:]),vmax=np.max(gather[iline,xline,:,:]))
        plt.xlabel('Offset')
        if NMO is True:
            # Repeat for NMO corrected data
            # Convert to decibel
            gather = 20*np.log(abs(np.double(self.cmp_gather_NMO)))
            gather[gather<noiseDB] = noiseDB 
            
            # Plots the stacked data created in "sort-to-bin" methods
            plt.figure()
            plt.title('NMO corrected gather.' + 'iline: ' + str(iline) + ', xline: ' + str(xline))
            plt.imshow(gather[int(iline),int(xline),:,:].T,cmap ='RdBu_r',extent = [offset_min,offset_max,t_arr[-1],t_arr[0]], aspect='auto',vmin=np.min(gather[iline,xline,:,:]),vmax=np.max(gather[iline,xline,:,:]))
            plt.xlabel('Offset')
        
    def scale_data(self, scaling_factor, time_unit): # Scales all data by dividing by a factor
        """
        Scale the data to physical values. When data is exported from the lab, they are upscaled by a factor of 2000 (for stupid reasons), which must be corrected when importing the data. 

        Parameters
        ----------
        scaling_factor : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        print('Scaling data...')
        
        
        if time_unit=='ms':
            t_scale = 1000 # Divide by an additional factor of a 1000 to compensate for the "ms" unit
        elif time_unit =='s':
            t_scale = 1
        
        # Scale the data
        self.header['sourceX'] = self.header['sourceX']/scaling_factor
        self.header['sourceY'] = self.header['sourceY']/scaling_factor
        self.header['groupX'] = self.header['groupX']/scaling_factor
        self.header['groupY'] = self.header['groupY']/scaling_factor
        self.header['RcvElev'] = self.header['RcvElev']/scaling_factor
        self.header['SrcElev'] = self.header['SrcElev']/scaling_factor
        self.header['samples'] = self.header['samples']/(scaling_factor*t_scale)
        self.traces = self.traces/scaling_factor
        self.scaling_factor = 1 # After scaling, this should be corrected to 1
        
        print('Receiver x-position:', str(np.round(np.min(self.header['groupX']),2)), ' - ', str(np.round(np.max(self.header['groupX']),2)))
        print('Receiver y-position:', str(np.round(np.min(self.header['groupY']),2)), ' - ', str(np.round(np.max(self.header['groupY']),2)))
        print('...\n')
        return self
    
    def shift_trace_positions(self, x, y, z):
        """
        Shift the entire collection of traces by the (x,y,z) vector desribed by the arguments. Needed to align the data to the velocity model.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        z : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        print('Shifting coordinates of the traces...')
        
        # Function that shifts the x,y,z positions of all headers by the input arguments
        self.header['sourceX'] = self.header['sourceX'] + x
        self.header['sourceY'] = self.header['sourceY'] + y
        self.header['groupX'] = self.header['groupX'] + x
        self.header['groupY'] = self.header['groupY'] + y
        self.header['RcvElev'] = self.header['RcvElev'] + z
        self.header['SrcElev'] = self.header['SrcElev'] + z
        
        print('New receiver x-position:', str(np.round(np.min(self.header['groupX']),2)), ' - ', str(np.round(np.max(self.header['groupX']),2)))
        print('New receiver y-position:', str(np.round(np.min(self.header['groupY']),2)), ' - ', str(np.round(np.max(self.header['groupY']),2)))
        print('...\n')
        
        
    def mute_traces(self, start_time):
        print('Mutes all traces...\n')
        Nt = len(self.header['samples']) # Number of samples
        dt = self.header['samples'][1]-self.header['samples'][0] # Sapling time
        t_arr = np.linspace(0,dt*Nt,Nt) # Remove amplificaiton factor, and convert to ms
        idx = (np.abs(t_arr - start_time)).argmin()
        self.traces[:,0:idx] = 0
        return self

    def create_ZO(self):
        seis_ZO = Seismic()
        import time
        start_time = time.time()
        for i in range(len(self.traces)):
            x_S = self.header['sourceX'][i]
            y_S = self.header['sourceY'][i]
            x_R = self.header['groupX'][i]
            y_R = self.header['groupY'][i]
            offset = np.sqrt(np.power(x_S-x_R,2) + np.power(y_S-y_R,2)) # Offset to the midpoint
            if offset == 0:
                # Add data from i to new object
                if np.size(seis_ZO.traces) == 1: # In case the self object is empty
                    seis_ZO.header['sourceX'] = self.header['sourceX'][i]
                    seis_ZO.header['sourceY'] = self.header['sourceY'][i]
                    seis_ZO.header['groupX'] = self.header['groupX'][i]
                    seis_ZO.header['groupY'] = self.header['groupY'][i]
                    seis_ZO.header['RcvElev'] = self.header['RcvElev'][i]
                    seis_ZO.header['SrcElev'] = self.header['SrcElev'][i]
                    seis_ZO.header['nsum'] = self.header['nsum'][i]
                    seis_ZO.header['nstack'] = self.header['nstack'][i]
                    seis_ZO.header['samples'] = self.header['samples']
                    # seis_ZO.traces = np.zeros([1,len(self.traces[0,:])])
                    seis_ZO.traces = np.reshape(self.traces[i],[1,len(self.traces[i])])
                else:
                    seis_ZO.header['sourceX'] = np.append(seis_ZO.header['sourceX'],self.header['sourceX'][i])
                    seis_ZO.header['sourceY'] = np.append(seis_ZO.header['sourceY'],self.header['sourceY'][i])
                    seis_ZO.header['groupX'] = np.append(seis_ZO.header['groupX'],self.header['groupX'][i])
                    seis_ZO.header['groupY'] = np.append(seis_ZO.header['groupY'],self.header['groupY'][i])
                    seis_ZO.header['RcvElev'] = np.append(seis_ZO.header['RcvElev'],self.header['RcvElev'][i])
                    seis_ZO.header['SrcElev'] = np.append(seis_ZO.header['SrcElev'],self.header['SrcElev'][i])
                    seis_ZO.header['nsum'] = np.append(seis_ZO.header['nsum'],self.header['nsum'][i])
                    seis_ZO.header['nstack'] = np.append(seis_ZO.header['nstack'],self.header['nstack'][i])
                    seis_ZO.traces = np.append(seis_ZO.traces,np.reshape(self.traces[i],[1,len(self.traces[i])]),axis=0)
                
            elapsed_time = time.time() - start_time
            time_per_i = (elapsed_time/(i+1))
            remaining_time_sec = round((len(self.traces)-i) * time_per_i)
            remaining_time_min = np.floor(remaining_time_sec/60)
            print('\r',np.round(100*i/(len(self.traces)),1), '% complete, time remaining: ',str(remaining_time_min), 'minutes, ',str(remaining_time_sec-remaining_time_min*60), 'seconds', end='')
        print('done\n')
        return seis_ZO
    
    def remove_ZO(self):
        """
        Remove zero-offset traces (where source and receiver coincide).
        Returns a new Seismic object with only non-zero-offset traces.
        """
        seis_nZO = Seismic()
        
        # Compute offsets for all traces at once (vectorized)
        dx = self.header['sourceX'] - self.header['groupX']
        dy = self.header['sourceY'] - self.header['groupY']
        offsets = np.sqrt(dx**2 + dy**2)
        
        # Boolean mask for traces with non-zero offset
        mask = offsets != 0
        
        # Apply mask to headers (vectorized copy)
        for key in ['sourceX', 'sourceY', 'groupX', 'groupY',
                    'RcvElev', 'SrcElev', 'nsum', 'nstack', 'trans_id']:
            seis_nZO.header[key] = self.header[key][mask]
        
        # Keep "samples" and filename (not per-trace, so just copy directly)
        seis_nZO.header['samples'] = self.header['samples']
        seis_nZO.filename = self.filename
        
        # Slice traces directly
        seis_nZO.traces = self.traces[mask]
        
        print(f"Removed {np.sum(~mask)} zero-offset traces out of {len(self.traces)}")
        return seis_nZO

    

    def prestack_migration_2D(self, axis = 'x', maxDepth = 0.5, res = (100,1000), aperture = None, cutOffAngle = 30, maxAngle = 45, vel_mod = None, vel_model_plot = False):
        print('Performing prestack migration\n')
        # Create zero-offset section
        data = self.traces
        z_S = self.header['SrcElev'] # Elevation of sources
        z_R = self.header['RcvElev'] # Elevation of sources
        if axis=='y': # Use the ilines for imaging
            x_S = self.header['sourceX']
            x_R = self.header['groupX']
            y = np.mean(self.header['sourceY'])
        elif axis=='x': # Swap to using the xlines
            x_S = self.header['sourceY']
            x_R = self.header['groupY']
            y = np.mean(self.header['sourceX'])
        x_0 = np.min(x_S)
        if aperture == None:
            aperture = np.max(x_S)
    
        (Nx,Nz) = res # Pixels in the image
        
        # Extract key information
        Nr = len(data[:,0]) # Number of channels
        t = self.header['samples'] # Time-array
        dt = t[1]-t[0]
        Nt = len(t) # Number of samples
        x_tot = np.max(x_S)-np.min(x_S)
        dx = x_tot/Nx
        dz = maxDepth/Nz
        
        # Assign coordinates to the migrated image for later reference
        x_arr = np.linspace(np.min(x_S), np.max(x_S), Nx)
        z_arr = np.linspace(0, maxDepth, Nz)
        self.mig_img_SA = [x_arr,z_arr]
        
        # Define pixel-accurate velocity model
        img_vel = np.zeros([Nx,Nz])
        for xi in range(Nx):
            x = dx*xi + x_0
            for zi in range(Nz):
                z = dz*zi
                if axis=='y':
                    img_vel[xi,zi] = vel_mod.retrieveRMSVelAtPoint(x, y, z)
                elif axis=='x':
                    img_vel[xi,zi] = vel_mod.retrieveRMSVelAtPoint(y, x, z)
        
        if vel_model_plot is True:        
            fig_extent=(np.min(x_S),np.max(x_S),maxDepth,0)
            plt.figure()
            plt.title('Velocity model at migration area')
            plt.imshow(img_vel.T,aspect='auto', extent=fig_extent)
            plt.xlabel('Y distance (m)')
            plt.ylabel('Z distance (m)')
            
            fig_extent=(np.min(x_S),np.max(x_S),maxDepth,0)
            plt.figure()
            plt.title('Velocity model at migration area')
            plt.imshow(img_vel.T,aspect='auto', extent=fig_extent)
            plt.xlabel('X distance (m)')
            plt.ylabel('Z distance (m)')
                    
        # Migrate on parallel workers
        img = np.zeros([Nx,Nz]) # Migrated image
        args = [(data[i,:], Nx, Nz, dx, x_0, y, dz, i, x_S[i], x_R[i], z_R[i], z_S[i], aperture, maxAngle, dt, Nt, cutOffAngle, img_vel) for i in range(Nr)]
        total_count = 0
        with mp.Pool() as pool:
            results = pool.imap(migration_worker_2D, args)
            for result in results:
                img += result
                total_count += 1
                print('Count: ', total_count)
        
        print('Done\n')
        self.mig_img = img
        return img
    
    def create_mig_img_pos(self,res = (10,10,100), maxDepth = 0.3):
        x_S = self.header['sourceX'] # X-position of the sources
        y_S = self.header['sourceY'] # Y-position of the sources
        (Nx,Ny,Nz) = res # Pixels in the image
        # Assign coordinates to the migrated image for later reference
        x_arr = np.linspace(np.min(x_S), np.max(x_S), Nx)
        y_arr = np.linspace(np.min(y_S), np.max(y_S), Ny)
        z_arr = np.linspace(0, maxDepth, Nz)
        self.mig_img_SA = [x_arr,y_arr,z_arr]
    
    
    def compute_grid_requirements(self, vel_model, f_center, xlim = None, ylim = None, max_depth = None, image_fraction=4):
        """
        Compute grid requirements for seismic/ultrasonic FD modeling and imaging,
        with CFL stability warning.
    
        """
        # Find minimum velocity in the model
        v_min = np.min(vel_model.vel_matrix)
        
        # Wavelength
        wavelength = v_min / f_center
     
        # Image grid spacing and sizes
        if xlim is None:
            roi_x = np.max(self.header['groupX']) - np.min(self.header['groupX'])
        else:
            roi_x = xlim[1]-xlim[0]
        if ylim is None:
            roi_y = np.max(self.header['groupY']) - np.min(self.header['groupY'])
        else:
            roi_y = ylim[1]-ylim[0]
        # Grid spacing in space
        dx = wavelength / image_fraction
        dy = dx
        dz = dx  # Usually same fraction in depth
        
        # Grid counts
        nx = math.ceil(roi_x / dx)
        ny = math.ceil(roi_y / dy)
        nz = 8*math.ceil(max_depth / dz)
    
        return (nx,ny,nz)

    
    
    def upsample_time(self, upsample_factor=2, kind='linear'):
        
        """
        Upsample seismic data in time.
        
        Parameters
        ----------
        data : 2D array (nt x nx)
        t_old : 1D array of original time samples
        upsample_factor : int, factor to reduce Δt
        kind : str, interpolation method
        
        Returns
        -------
        data_upsampled : 2D array (nt_new x nx)
        t_new : 1D array of new time samples
        """
        from scipy.interpolate import interp1d

        data = self.traces
        t_old = self.header['samples']
        nr, nt = data.shape
        dt_old = t_old[1] - t_old[0]
        dt_new = dt_old / upsample_factor
        t_new = np.arange(0, t_old[-1] + dt_new, dt_new)
        nt_new = len(t_new)
        
        data_upsampled = np.zeros((nr, nt_new))
        
        for i in range(nr):
            f = interp1d(t_old, data[i,:], kind=kind, fill_value="extrapolate")
            data_upsampled[i,:] = f(t_new)
        
        self.traces = data_upsampled
        self.header['samples'] = t_new
        # return data_upsampled, t_new     


    def get_trace_at_xy(self, x_target, y_target):
        """
        Retrieve the migrated trace closest to a given (x, y) position.
        
        Parameters
        ----------
        seis : Seismic
            Seismic object containing migrated image (mig_img) and spatial axes (mig_img_SA).
        x_target : float
            Target X coordinate (in same units as seis.mig_img_SA[0]).
        y_target : float
            Target Y coordinate (in same units as seis.mig_img_SA[1]).
        
        Returns
        -------
        trace : np.ndarray
            1D array representing the migrated trace at the closest (x, y) location.
        (ix, iy) : tuple
            Indices of the selected trace in the migrated image.
        (x_sel, y_sel) : tuple
            Actual spatial coordinates of the selected trace.
        """
    
        # Extract spatial axes from migrated image
        x_axis, y_axis, _ = self.mig_img_SA
    
        # Find nearest indices
        ix = np.abs(x_axis - x_target).argmin()
        iy = np.abs(y_axis - y_target).argmin()
    
        # Extract the corresponding trace
        trace = self.mig_img[ix, iy, :]
    
        # Return both the trace and metadata
        x_sel = x_axis[ix]
        y_sel = y_axis[iy]
    
        return trace, (ix, iy), (x_sel, y_sel)
    
    
    def make_surface_plot(self, vel_model, max_interfaces = 2, peak_prominance = 5, peak_distance = 5, max_depth = 0.3, xy_lims = ([0,3],[0,1]), search_margin = 0.05, view_axis = (1,1,1), interpolationF = 1, SPIKE = None, debugging = False):    
        x_vals, y_vals, interfaces_m = vel_model.plot_velocity_model_surface_3D(view_axis, interpolationF = 1)
        
        #### Extract a sub-scetion of the velocity model
        x_min = xy_lims[0][0]  
        x_max = xy_lims[0][1]  
        dx = x_vals[1]-x_vals[0]
        
        # Compute corresponding indices
        i_start = int(x_min / dx)
        i_end   = int(x_max / dx)
        
        # Update matrices
        interfaces_m = interfaces_m[i_start:i_end+1, :, :]  # keep all Y and Z
        x_vals = x_vals[i_start:i_end+1]
        y_vals = y_vals[i_start:i_end+1]  # unchanged


        #### Spike confguration
        if SPIKE is None:
            SPIKE = np.inf
        # Downsample velocity model surface 
        # Takes in data in the form of x,y,z - resamples to a new dimension
        interfaces_m_model = interfaces_m
        (n_x, n_y) = np.shape(self.mig_img[:,:,0])
        resized_channels = []
        
        for i in range(np.shape(interfaces_m_model)[2]):
            resized = cv2.resize(interfaces_m_model[:, :, i], (n_y, n_x), interpolation=cv2.INTER_LINEAR)
            resized_channels.append(resized)
        
        # Stack back into 3D array
        resized = np.stack(resized_channels, axis=-1)
        interfaces_m_ds = resized
        
        
        # Find peaks in the data
        mig_img = self.mig_img # Could be other data
        # Get RMS data
        data = matrix_RMS(mig_img, window_size = 64)
        
        # Parameters
        MAX_INTERFACES = max_interfaces
        PEAK_PROMINANCE = peak_prominance  # adjust depending on your signal strength
        PEAK_DISTANCE = peak_distance      # minimum distance between peaks in samples
        MAX_DEPTH = max_depth
        SEARCH_MARGIN = search_margin # Search distance from the model to look for answer in the data
        dz = MAX_DEPTH/np.shape(data)[2]
        SEARCH_MARGIN_idx = int(SEARCH_MARGIN/dz) # Search distance from the model to look for answer in the data

        # Initialize array to store interface depths
        interfaces_data = np.full((n_x, n_y, MAX_INTERFACES), np.nan)
        interfaces_amp = np.full((n_x, n_y, MAX_INTERFACES), np.nan)
        interfaces_sum = np.full((n_x, n_y, MAX_INTERFACES), np.nan)
       

        for j in range(n_y):
            if j==10 and debugging:
                plt.figure()
                plt.imshow(data[:,j,:].T,aspect='auto',vmax = 0.2*np.max(data[:,j,:]))
                plt.title(f'Slice nr {j}')
            for i in range(n_x):
                # ax_2.clear()      # Clear the image figure
                # img_2 = ax_2.plot(data[i,j,:])
                # plt.show()
                for m in range(MAX_INTERFACES):
                    # Find index corresponding to the interface in the model
                    model_surface = interfaces_m_ds[i,j,m]
                    idx = int(model_surface/dz)

                    if idx - SEARCH_MARGIN_idx < 0:
                        trace = data[i, j, 0:idx + SEARCH_MARGIN_idx]
                        local_maxima_idx_add = 0
                    elif idx + SEARCH_MARGIN_idx > np.shape(data)[2]:
                        trace = data[i, j, idx - SEARCH_MARGIN_idx:]
                        local_maxima_idx_add = idx-SEARCH_MARGIN_idx # Dubugging 
                    else:
                        trace = data[i, j, idx - SEARCH_MARGIN_idx:idx + SEARCH_MARGIN_idx]
                        local_maxima_idx_add = idx-SEARCH_MARGIN_idx # Dubugging 
                    # peaks, _ = find_peaks(trace, prominence=PEAK_PROMINANCE, distance=PEAK_DISTANCE)
                    interfaces_sum[i,j,m] = np.sum(abs(trace))
                    
                    avg = np.average(np.abs(trace))
                    SPIKE_CHECK = False
                    count = 0
                    while not SPIKE_CHECK:
                        local_maxima_idx = np.argmax(np.abs(trace))
                        local_maxima = trace[local_maxima_idx]
                        # Check if value is above expectation (SPIKE)
                        if local_maxima<SPIKE:
                            SPIKE_CHECK = True
                        else:
                            data[i,j,local_maxima_idx_add + local_maxima_idx] = 0
                            # plt.plot(i, local_maxima_idx + local_maxima_idx_add, 'go')  # 'ro' = red circle marker

                        if local_maxima <= avg: # Stop loop if no maxima is found
                            SPIKE_CHECK = True
                        count+=1
                        if count>1000:
                            raise ValueError("Loop got stuck")

                    if local_maxima>avg+PEAK_PROMINANCE:
                        if j==10 and debugging:
                            plt.plot(i, idx, 'yo')  # 'ro' = red circle marker
                            plt.plot(i, local_maxima_idx + local_maxima_idx_add, 'ro')  # 'ro' = red circle marker
                        interfaces_data[i,j,m] = idx + local_maxima_idx # Find the max value
                        interfaces_amp[i,j,m] = local_maxima # Find the max value
                    else:
                        interfaces_amp[i,j,m] = 0 # Find the max value
        interfaces_m = interfaces_data * dz  # convert to meters
        plt.show()

        # Interpolate missing numbers
        for i in range(MAX_INTERFACES):
            interfaces_m[:,:,i] = nanmean_filter_expand(interfaces_m[:,:,i],interpolationF)
            

        # # Plot everything
        xlim = np.max(vel_model.spatial_arrs[0])
        ylim = np.max(vel_model.spatial_arrs[1])
        zlim = MAX_DEPTH
        plot_surface(interfaces_m, xlim, ylim, zlim, view_axis, title = 'Interfaces Along Z', interpolationF = interpolationF)
        plot_surface(interfaces_amp, xlim, ylim, zlim, view_axis, title = 'Interfaces Along Z (Amplitude)', interpolationF = interpolationF)
        plot_surface(interfaces_sum, xlim, ylim, zlim, view_axis, title = 'Interfaces Along Z (Summed)', interpolationF = interpolationF)

        return interfaces_m, interfaces_amp, interfaces_m_model, interfaces_sum
    
    def make_surface_plot_old(self, vel_model, max_interfaces = 2, peak_prominance = 5, peak_distance = 5, max_depth = 0.3, search_margin = 0.05, view_axis = (1,1,1), interpolationF = 1):    
        x_vals, y_vals, interfaces_m = vel_model.plot_velocity_model_surface_3D(view_axis, interpolationF = 1)
        
        # Downsample velocity model surface 
        # Takes in data in the form of x,y,z - resamples to a new dimension
        interfaces_m_model = interfaces_m
        (n_x, n_y) = np.shape(self.mig_img[:,:,0])
        resized_channels = []
        
        for i in range(np.shape(interfaces_m_model)[2]):
            resized = cv2.resize(interfaces_m_model[:, :, i], (n_y, n_x), interpolation=cv2.INTER_LINEAR)
            resized_channels.append(resized)
        
        # Stack back into 3D array
        resized = np.stack(resized_channels, axis=-1)
        interfaces_m_ds = resized
        
        
        # Find peaks in the data
        mig_img = self.mig_img # Could be other data
        # Get RMS data
        data = matrix_RMS(mig_img, window_size = 64)
        
        # Parameters
        MAX_INTERFACES = max_interfaces
        PEAK_PROMINANCE = peak_prominance  # adjust depending on your signal strength
        PEAK_DISTANCE = peak_distance      # minimum distance between peaks in samples
        MAX_DEPTH = max_depth
        SEARCH_MARGIN = search_margin # Search distance from the model to look for answer in the data
        dz = MAX_DEPTH/np.shape(data)[2]
        SEARCH_MARGIN_idx = int(SEARCH_MARGIN/dz) # Search distance from the model to look for answer in the data

        # Initialize array to store interface depths
        interfaces_data = np.full((n_x, n_y, MAX_INTERFACES), np.nan)
        interfaces_amp = np.full((n_x, n_y, MAX_INTERFACES), np.nan)
        
        for i in range(n_x):
            for j in range(n_y):
                for m in range(MAX_INTERFACES):
                    # Find index corresponding to the interface in the model
                    model_surface = interfaces_m_ds[i,j,m]
                    idx = int(model_surface/dz)
                    
                    if j==12: # Debugging
                        pass
                    if idx - SEARCH_MARGIN_idx < 0:
                        trace = data[i, j, 0:idx + SEARCH_MARGIN_idx]
                    elif idx + SEARCH_MARGIN_idx > np.shape(data)[2]:
                        trace = data[i, j, idx - SEARCH_MARGIN_idx:]
                    else:
                        trace = data[i, j, idx - SEARCH_MARGIN_idx:idx + SEARCH_MARGIN_idx]
                    # peaks, _ = find_peaks(trace, prominence=PEAK_PROMINANCE, distance=PEAK_DISTANCE)
                    
                    local_maxima_idx = np.argmax(np.abs(trace))
                    local_maxima = trace[local_maxima_idx]
                    avg = np.average(np.abs(trace))
                    if local_maxima>avg+PEAK_PROMINANCE:
                        interfaces_data[i,j,m] = idx + local_maxima_idx # Find the max value
                        interfaces_amp[i,j,m] = local_maxima # Find the max value
        
        interfaces_m = interfaces_data * dz  # convert to meters
        
        # # Interpolate missing numbers
        # for i in range(MAX_INTERFACES):
        #     interfaces_m[:,:,i] = nanmean_filter(interfaces_m[:,:,i],3)
            
        # # Interpolate missing numbers
        # for i in range(MAX_INTERFACES):
        #     interfaces_amp[:,:,i] = nanmean_filter_expand(interfaces_amp[:,:,i], max_size=20)
        

        # # Plot everything
        xlim = np.max(vel_model.spatial_arrs[0])
        ylim = np.max(vel_model.spatial_arrs[1])
        zlim = MAX_DEPTH
        plot_surface(interfaces_m, xlim, ylim, zlim, view_axis, title = 'Interfaces Along Z', interpolationF = interpolationF)
        plot_surface(interfaces_amp, xlim, ylim, zlim, view_axis, title = 'Interfaces Along Z (Amplitude)', interpolationF = interpolationF)

        return interfaces_m, interfaces_amp, interfaces_m_model
        


    def save_seis_obj(self, filename):
        # Save the object to a file
        with open(f'{filename}.pkl', 'wb') as file:
            pickle.dump(self, file)
    
    def load_seis_obj(self, filename):
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
            

   
    def plot_shot_gather_by_id(self, trans_id, **kwargs):
        """
        Selects and plots a single shot gather based on its transducer ID.
    
        This method identifies all traces that share a common 'trans_id' from the
        SEGY headers and plots them using the 'plot_all_traces' method.
    
        Args:
            trans_id (int): The specific transducer ID to find and plot. This value
                            comes from the 'FieldRecord' field in the SEGY file.
            **kwargs: Arbitrary keyword arguments to be passed directly to the
                      'plot_all_traces' method (e.g., DB=True, AMP=False,
                      RMS=True, block_size=1000).
    
        Raises:
            ValueError: If the required seismic data is not loaded or if the
                        specified transducer ID cannot be found.
        """
        # --- 1. Validate that the necessary data has been loaded ---
        if self.traces is None or self.header.get('trans_id') is None:
            raise ValueError("Seismic data not loaded or 'trans_id' header not found. "
                             "Please load a SEGY file using .read_segy() first.")
    
        # --- 2. Find all trace indices that match the target transducer ID ---
        # Use np.where for an efficient search
        trans_id_array = self.header['trans_id']
        matching_indices = np.where(trans_id_array == trans_id)[0]
    
        # --- 3. Handle the case where the transducer ID is not found ---
        if len(matching_indices) == 0:
            unique_ids = np.unique(trans_id_array)
            raise ValueError(f"Transducer ID '{trans_id}' not found in the data.\n"
                             f"Available IDs are: {unique_ids}")
    
        print(f"Found {len(matching_indices)} traces for Transducer ID '{trans_id}'.")
    
        # --- 4. Create a new, temporary Seismic object containing ONLY the shot gather ---
        shot_gather_obj = Seismic()
        
        # Copy traces and the time sample array
        shot_gather_obj.traces = self.traces[matching_indices, :]
        shot_gather_obj.header['samples'] = self.header['samples']
    
        # Filter all other header arrays to include only the matching traces
        for key, value in self.header.items():
            if isinstance(value, np.ndarray) and key != 'samples':
                shot_gather_obj.header[key] = value[matching_indices]
    
        # --- 5. Call the plot_all_traces method on the new object to generate the plot ---
        print("Generating plot using the 'plot_all_traces' method...")
        shot_gather_obj.plot_all_traces(**kwargs)
        
        
    def organize_shot_gathers(self):
        """
        Identifies unique shot gathers, prints a full summary, and stores the
        trace indices for each gather in the 'self.shot_gathers' attribute.
        """
        # --- 1. Validate that data is loaded ---
        if self.header.get('trans_id') is None:
            print("Error: 'trans_id' header not found. Please load data first.")
            return
    
        # --- 2. Find unique shot IDs and count the traces in each ---
        all_trans_ids = self.header['trans_id']
        unique_ids, counts = np.unique(all_trans_ids, return_counts=True)
        
        num_gathers = len(unique_ids)
    
        if num_gathers == 0:
            print("No shot gathers were found in the data.")
            self.shot_gathers = {} # Ensure the attribute is an empty dict
            return
    
        # --- 3. Print the overall summary statement ---
        if np.all(counts == counts[0]):
            traces_per_gather = counts[0]
            print(f"{num_gathers} shot-gather groups found, with {traces_per_gather} traces in each gather.")
        else:
            print(f"{num_gathers} shot-gather groups found, with varying numbers of traces.")
        
        print("---------------------------------------------") # Separator for clarity
    
        # --- 4. NEW: Print the detailed list ---
        print("--- Detailed Shot Gather List ---")
        for tid, count in zip(unique_ids, counts):
            print(f"  ID: {tid:<5} | Traces: {count}")
        print("---------------------------------------------")
    
        # --- 5. Create and store the shot gather dictionary ---
        gathers_dict = {}
        for uid in unique_ids:
            indices = np.where(all_trans_ids == uid)[0]
            gathers_dict[uid] = indices
        
        self.shot_gathers = gathers_dict
        print("\nShot gather information has been calculated and stored in 'self.shot_gathers'.")
    
    def plot_shot_gather_by_id(self, trans_id, **kwargs):
        """
        Selects and plots a single shot gather based on its transducer ID.
    
        This method identifies all traces that share a common 'trans_id' from the
        SEGY headers and plots them using the 'plot_all_traces' method.
    
        Args:
            trans_id (int): The specific transducer ID to find and plot. This value
                            comes from the 'FieldRecord' field in the SEGY file.
            **kwargs: Arbitrary keyword arguments to be passed directly to the
                      'plot_all_traces' method (e.g., DB=True, AMP=False,
                      RMS=True, block_size=1000).
    
        Raises:
            ValueError: If the required seismic data is not loaded or if the
                        specified transducer ID cannot be found.
        """
        # --- 1. Validate that the necessary data has been loaded ---
        if self.traces is None or self.header.get('trans_id') is None:
            raise ValueError("Seismic data not loaded or 'trans_id' header not found. "
                             "Please load a SEGY file using .read_segy() first.")
    
        # --- 2. Find all trace indices that match the target transducer ID ---
        # Use np.where for an efficient search
        trans_id_array = self.header['trans_id']
        matching_indices = np.where(trans_id_array == trans_id)[0]
    
        # --- 3. Handle the case where the transducer ID is not found ---
        if len(matching_indices) == 0:
            unique_ids = np.unique(trans_id_array)
            raise ValueError(f"Transducer ID '{trans_id}' not found in the data.\n"
                             f"Available IDs are: {unique_ids}")
    
        print(f"Found {len(matching_indices)} traces for Transducer ID '{trans_id}'.")
    
        # --- 4. Create a new, temporary Seismic object containing ONLY the shot gather ---
        shot_gather_obj = Seismic()
        
        # Copy traces and the time sample array
        shot_gather_obj.traces = self.traces[matching_indices, :]
        shot_gather_obj.header['samples'] = self.header['samples']
    
        # Filter all other header arrays to include only the matching traces
        for key, value in self.header.items():
            if isinstance(value, np.ndarray) and key != 'samples':
                shot_gather_obj.header[key] = value[matching_indices]
    
        # --- 5. Call the plot_all_traces method on the new object to generate the plot ---
        print("Generating plot using the 'plot_all_traces' method...")
        shot_gather_obj.plot_all_traces(**kwargs)
        
        
    def organize_shot_gathers(self):
        """
        Identifies unique shot gathers, prints a full summary, and stores the
        trace indices for each gather in the 'self.shot_gathers' attribute.
        """
        # --- 1. Validate that data is loaded ---
        if self.header.get('trans_id') is None:
            print("Error: 'trans_id' header not found. Please load data first.")
            return
    
        # --- 2. Find unique shot IDs and count the traces in each ---
        all_trans_ids = self.header['trans_id']
        unique_ids, counts = np.unique(all_trans_ids, return_counts=True)
        
        num_gathers = len(unique_ids)
    
        if num_gathers == 0:
            print("No shot gathers were found in the data.")
            self.shot_gathers = {} # Ensure the attribute is an empty dict
            return
    
        # --- 3. Print the overall summary statement ---
        if np.all(counts == counts[0]):
            traces_per_gather = counts[0]
            print(f"{num_gathers} shot-gather groups found, with {traces_per_gather} traces in each gather.")
        else:
            print(f"{num_gathers} shot-gather groups found, with varying numbers of traces.")
        
        print("---------------------------------------------") # Separator for clarity
    
        # --- 4. NEW: Print the detailed list ---
        print("--- Detailed Shot Gather List ---")
        for tid, count in zip(unique_ids, counts):
            print(f"  ID: {tid:<5} | Traces: {count}")
        print("---------------------------------------------")
    
        # --- 5. Create and store the shot gather dictionary ---
        gathers_dict = {}
        for uid in unique_ids:
            indices = np.where(all_trans_ids == uid)[0]
            gathers_dict[uid] = indices
        
        self.shot_gathers = gathers_dict
        print("\nShot gather information has been calculated and stored in 'self.shot_gathers'.")
        
        
    def organize_shots_by_location(self):
        """
        Analyzes the source headers to identify all unique shot locations.

        This is the core organizational step for location-based processing.
        It populates two new class attributes:
        1. self.shot_locations: A NumPy array where each row is a unique
           [sourceX, sourceY] coordinate pair.
        2. self.shot_gathers_by_loc: A dictionary where the key is the
           shot_index (0, 1, 2...) and the value is a NumPy array of
           trace indices belonging to that shot.

        This method is essential for preparing data for 3D imaging and for
        the `plot_shot_gather_by_loc` function.
        """
        # --- 1. Validate that the necessary header data has been loaded ---
        if self.header.get('sourceX') is None or self.traces is None:
            raise ValueError("Header data or traces not found. Please load a SEGY file first.")

        print("Organizing shots by unique (sourceX, sourceY) locations...")
        coords = np.vstack((self.header['sourceX'], self.header['sourceY'])).T
        unique_locs, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
        
        self.shot_locations = unique_locs
        self.shot_gathers_by_loc = {}
        
        num_gathers = len(unique_locs)
        for i in range(num_gathers):
            self.shot_gathers_by_loc[i] = np.where(inverse_indices == i)[0]
            
        print(f"-> Found {num_gathers} unique shot-gather locations.")
        print("-> Shot map has been created and stored in 'self.shot_locations' and 'self.shot_gathers_by_loc'.")
        
        
        
    def plot_shot_gather_by_loc(self, shot_index, sort_receivers_by='groupY', line_index=None, **kwargs):
        """
        Selects a shot gather by its global index and plots the traces that
        were recorded by receivers on a specified survey line.

        Args:
            shot_index (int): The global index of the shot to plot, from the
                              list created by organize_shots_by_location().
            sort_receivers_by (str, optional): Header to sort the final set of
                                               receiver traces. Defaults to 'groupY'.
            line_index (int, optional): If provided, the function will
                                                   filter the gather to show only the
                                                   traces recorded by receivers on this
                                                   specific line. If None (default),
                                                   it shows all traces from all lines.
            **kwargs: Keyword arguments for plot_all_traces (e.g., AMP=True).
        """
        # --- 1. Validate that the organization step has been run ---
        if self.shot_gathers_by_loc is None or self.shot_locations is None:
            raise RuntimeError("Shot gathers have not been organized. "
                               "Please run the .organize_shots_by_location() method first.")

        # --- 2. Select the Shot Gather based on the global shot_index ---
        num_gathers = len(self.shot_locations)
        if not 0 <= shot_index < num_gathers:
            raise ValueError(f"Invalid shot_index '{shot_index}'. "
                             f"Please provide an index between 0 and {num_gathers - 1}.")
        
        shot_coords = self.shot_locations[shot_index]
        print(f"-> Selecting global shot_index {shot_index} at location: {shot_coords}")
        
        # Get all trace indices belonging to this shot
        all_trace_indices_for_shot = self.shot_gathers_by_loc[shot_index]
        
        # --- 3. Filter the Traces by Receiver Line (if requested) ---
        if line_index is not None:
            print(f"--- Filtering receivers to display only those on line index: {line_index} ---")
            
            # Find the X-coordinate for the target receiver line
            unique_receiver_x_coords = np.unique(self.header['groupX']) # Find all receiver lines
            if not 0 <= line_index < len(unique_receiver_x_coords):
                raise ValueError(f"Invalid line_index. Please choose between 0 and {len(unique_receiver_x_coords) - 1}.")
            target_receiver_x = unique_receiver_x_coords[line_index]
            print(f"-> Corresponding receiver X-coordinate: {target_receiver_x}")

            # Get the 'groupX' values for all traces in our selected shot
            receiver_x_for_shot = self.header['groupX'][all_trace_indices_for_shot]

            # Find which of these traces lie on the target line
            on_line_mask = np.isclose(receiver_x_for_shot, target_receiver_x)
            
            # The final set of traces are the ones that pass the mask
            final_trace_indices = all_trace_indices_for_shot[on_line_mask]
            
            if len(final_trace_indices) == 0:
                print("(!) Warning: No receivers for this shot were found on the specified display line.")
                return # Exit if there's nothing to plot
        else:
            # If no line is specified, use all traces from the shot
            print("-> Displaying receivers from all available lines.")
            final_trace_indices = all_trace_indices_for_shot

        # --- 4. Create a temporary object and plot the final set of traces ---
        print(f"-> Plotting {len(final_trace_indices)} receiver traces.")
        
        shot_gather_obj = Seismic()
        shot_gather_obj.traces = self.traces[final_trace_indices, :]
        shot_gather_obj.header['samples'] = self.header['samples']
        for key, value in self.header.items():
            if isinstance(value, np.ndarray) and key != 'samples':
                shot_gather_obj.header[key] = value[final_trace_indices]

        if sort_receivers_by and len(shot_gather_obj.traces) > 0:
            if sort_receivers_by not in shot_gather_obj.header:
                raise ValueError(f"Sorting header '{sort_receivers_by}' not found.")
            print(f"-> Sorting receiver traces by '{sort_receivers_by}' for plotting.")
            sort_indices = np.argsort(shot_gather_obj.header[sort_receivers_by])
            shot_gather_obj.traces = shot_gather_obj.traces[sort_indices, :]
            for key in shot_gather_obj.header:
                if isinstance(shot_gather_obj.header[key], np.ndarray) and key != 'samples':
                    shot_gather_obj.header[key] = shot_gather_obj.header[key][sort_indices]

        print("-> Generating plot...")
        shot_gather_obj.plot_all_traces(**kwargs)
        
    def plot_line_gather(self, line_index, sort_shots_by='sourceY', **kwargs):
        """
        Extracts the nearest-to-zero-offset trace from each shot along a line
        of constant X-coordinate and plots them together as a 2D image.

        Args:
            line_index (int): The index of the survey line to plot, based on
                              the unique sorted X-coordinates.
            sort_shots_by (str, optional): The header to sort the shots along the
                                           line for correct visualization. Defaults
                                           to 'sourceY'.
            **kwargs: Keyword arguments passed to plot_all_traces (e.g., AMP=True).
        """
        if self.shot_gathers_by_loc is None:
            raise RuntimeError("Shot gathers have not been organized. "
                               "Please run the .organize_shots_by_location() method first.")
        
        print(f"--- Creating gather for line index: {line_index} ---")
        
        # 1. Find all unique X coordinates to identify the lines
        unique_x_coords = np.unique(self.shot_locations[:, 0])
        if not 0 <= line_index < len(unique_x_coords):
            raise ValueError(f"Invalid line_index. Please choose between 0 and {len(unique_x_coords) - 1}.")
        
        target_x = unique_x_coords[line_index]
        print(f"-> Corresponding X-coordinate: {target_x}")

        # 2. Find all shot indices that belong to this line
        shot_indices_for_line = [i for i, loc in enumerate(self.shot_locations) if loc[0] == target_x]
        
        # 3. For each shot on the line, find its single nearest-to-zero-offset trace
        zo_trace_indices = []
        for shot_idx in shot_indices_for_line:
            trace_indices_for_shot = self.shot_gathers_by_loc[shot_idx]
            
            # Calculate offset for all receivers in this one shot
            sx = self.header['sourceX'][trace_indices_for_shot]
            sy = self.header['sourceY'][trace_indices_for_shot]
            gx = self.header['groupX'][trace_indices_for_shot]
            gy = self.header['groupY'][trace_indices_for_shot]
            offsets = np.sqrt((sx - gx)**2 + (sy - gy)**2)
            
            # Find the index of the trace with the minimum offset
            min_offset_local_idx = np.argmin(offsets)
            
            # Convert back to the global trace index and add to our list
            zo_trace_indices.append(trace_indices_for_shot[min_offset_local_idx])

        print(f"-> Found {len(zo_trace_indices)} zero-offset traces for this line.")

        # 4. Create a new temporary Seismic object for the line gather
        line_gather_obj = Seismic()
        line_gather_obj.traces = self.traces[zo_trace_indices, :]
        line_gather_obj.header['samples'] = self.header['samples']
        for key, value in self.header.items():
            if isinstance(value, np.ndarray) and key != 'samples':
                line_gather_obj.header[key] = value[zo_trace_indices]

        # 5. Sort the line gather for plotting (e.g., by sourceY)
        print(f"-> Sorting line by '{sort_shots_by}' for plotting.")
        sort_indices = np.argsort(line_gather_obj.header[sort_shots_by])
        line_gather_obj.traces = line_gather_obj.traces[sort_indices, :]
        # Headers don't need re-sorting for plot_all_traces simple x-axis

        # 6. Plot the final result
        print("-> Generating plot...")
        line_gather_obj.plot_all_traces(**kwargs)       
            
    def inspect_object_state(self, object_name="Seismic Object"):
        """
        A diagnostic tool to print the current shape of the trace data and the
        length of every array in the header dictionary. This is used to find
        data inconsistencies.
        """
        print(f"\n--- State Inspection for: {object_name} ---")
        if self.traces is not None:
            print(f"Traces Matrix Shape: {self.traces.shape} (expected: traces, samples)")
        else:
            print("Traces Matrix: Not loaded")

        print("\nHeader Array Lengths:")
        if self.header:
            max_key_len = max(len(key) for key in self.header.keys() if self.header.get(key) is not None)
            for key, value in self.header.items():
                if isinstance(value, np.ndarray):
                    print(f"  - self.header['{key}']:".ljust(max_key_len + 18) + f" len = {len(value)}")
                elif value is not None:
                    print(f"  - self.header['{key}']:".ljust(max_key_len + 18) + " Not an array")
        else:
            print("  - Header is empty.")
        print("-------------------------------------------\n")
        
    def apply_bandpass_filter(self, lowcut, highcut, order=4, plot_response=False):
        """Applies a zero-phase Butterworth bandpass filter to all seismic traces."""
        if self.traces is None or self.header.get('samples') is None:
            raise ValueError("Traces or header samples not found. Load data first.")
        if len(self.header['samples']) < 2:
            raise ValueError("Cannot determine sampling rate from header.")
        if lowcut >= highcut:
            raise ValueError(f"lowcut frequency ({lowcut}) must be less than highcut frequency ({highcut}).")
            
        dt = self.header['samples'][1] - self.header['samples'][0]
        fs = 1.0 / dt
        print(f"Applying bandpass filter with fs={fs:.2f} Hz, lowcut={lowcut} Hz, highcut={highcut} Hz...")
    
        nyq = 0.5 * fs
        if lowcut >= nyq or highcut >= nyq:
            raise ValueError(f"Cutoff frequencies must be below Nyquist ({nyq:.2f} Hz).")
        
        sos = scipy.signal.butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    
        if plot_response:
            w, h = scipy.signal.sosfreqz(sos, worN=8000, fs=fs)
            plt.figure(figsize=(10, 6))
            plt.plot(w, np.abs(h), 'b', label='Filter Response')
            plt.axvline(lowcut, color='g', linestyle='--', label=f'Lowcut: {lowcut} Hz')
            plt.axvline(highcut, color='r', linestyle='--', label=f'Highcut: {highcut} Hz')
            plt.title(f"Butterworth Bandpass Filter Response (Order {order})")
            plt.xlabel("Frequency [Hz]"); plt.ylabel("Gain")
            plt.grid(True, which='both', linestyle='--'); plt.legend(); plt.ylim(0, 1.1)
            plt.xlim(0, min(highcut * 2, nyq))
            plt.show()
    
        print("Filtering traces... (this may take a moment)")
        self.traces = scipy.signal.sosfiltfilt(sos, self.traces, axis=1)
        print("Filtering complete.\n")
            
    def apply_deringing(self, f_center,
                        tb_us=8,
                        fit_window_us=(50, 200),
                        extend_fit_us=300,
                        show_example_trace=-1):
        """
        Removes source ringing by modeling and subtracting an exponentially decaying sinusoid.
        
        Args:
            f_center (float): The center frequency of the ringing to model (in Hz).
            tb_us (int, optional): The length of the cosine taper at the start of the
                                   trace to blank the initial pulse (in microseconds).
            fit_window_us (tuple, optional): The (start_time, end_time) in microseconds
                                             to use for fitting the ringing model.
            extend_fit_us (int, optional): The time in microseconds to which the
                                           subtraction of the model is extended.
            show_example_trace (int, optional): If set to a valid trace index, a
                                                 detailed QC plot is generated.
        """
        # --- 1. VALIDATION AND SETUP ---
        if self.traces is None or self.header.get('samples') is None:
            raise ValueError("Traces or header samples not found. Load data first.")
            
        dt = self.header['samples'][1] - self.header['samples'][0]
        fs = 1.0 / dt
        num_traces, num_samples = self.traces.shape
    
        original_trace_for_plot = None
        if 0 <= show_example_trace < num_traces:
            original_trace_for_plot = self.traces[show_example_trace, :].copy()
    
        # --- 2. DEFINE THE RINGING MODEL (HELPER FUNCTION) ---
        def ring_model(t, A, alpha, f, phi, off):
            return A * np.exp(-alpha * t) * np.cos(2 * np.pi * f * t + phi) + off
    
        print("\n--- Applying De-ringing Process ---")
        failed_fits = 0
    
        # --- 3. MAIN PROCESSING LOOP FOR ALL TRACES ---
        for i in range(num_traces):
            print(f'\rProcessing trace {i+1}/{num_traces}', end='')
            x, t = self.traces[i, :], self.header['samples']
    
            # Taper the initial pulse to avoid it affecting the fit
            n_taper = int(tb_us * 1e-6 * fs)
            taper = np.ones_like(x)
            if n_taper > 0:
                win = (1 - np.cos(np.linspace(0, np.pi, n_taper))) / 2
                taper[:n_taper] = win
            x_blank = x * taper
    
            # Define the window for curve fitting
            i0 = int(fit_window_us[0] * 1e-6 * fs)
            i1 = int(fit_window_us[1] * 1e-6 * fs)
    
            fit_succeeded = False
            # Check if the window is valid and has signal
            if (0 <= i0 < i1 < num_samples) and (np.max(np.abs(x_blank[i0:i1])) > 1e-9):
                t_fit, x_fit = t[i0:i1], x_blank[i0:i1]
                p0 = [np.max(np.abs(x_fit)), 1e4, f_center, 0, 0] # Initial guess
                bounds = ([0, 0, f_center * 0.9, -np.pi, -np.inf], 
                          [np.inf, np.inf, f_center * 1.1, np.pi, np.inf]) # Constrain the fit
    
                try:
                    popt, _ = curve_fit(ring_model, t_fit, x_fit, p0=p0, maxfev=20000, bounds=bounds)
                    fit_succeeded = True
                except (RuntimeError, ValueError):
                    popt = p0 # Use initial guess if fit fails
                    failed_fits += 1
            
            # If the fit was successful, subtract the modeled ring
            if fit_succeeded:
                i1_ext = min(int(extend_fit_us * 1e-6 * fs), num_samples)
                fit_full = ring_model(t, *popt)
                
                # Create a tapered window for subtraction
                taper_sub = np.zeros_like(x)
                taper_sub[i0:i1_ext] = 1.0
                nt_taper_end = int(0.05 * (i1_ext - i0)) # Taper the end of the subtraction
                if nt_taper_end > 1 and (i1_ext - nt_taper_end) < i1_ext:
                    taper_sub[i1_ext - nt_taper_end:i1_ext] = np.cos(np.linspace(0, np.pi / 2, nt_taper_end))**2
                
                self.traces[i, :] = x_blank - fit_full * taper_sub
            else:
                self.traces[i, :] = x_blank # If fit failed, just keep the tapered trace
    
        print(f"\nDe-ringing complete. {failed_fits} trace(s) failed to converge on a fit.")
    
        # --- 4. DETAILED QC PLOT (IF REQUESTED) ---
        if original_trace_for_plot is not None:
            print(f"Generating QC plot for trace {show_example_trace}...")
            # This is your provided code block, slightly adapted for clarity
            
            _t = self.header['samples']
            _i0 = int(fit_window_us[0] * 1e-6 * fs)
            _i1 = int(fit_window_us[1] * 1e-6 * fs)
            
            if not (0 <= _i0 < _i1 < num_samples and np.max(np.abs(original_trace_for_plot[_i0:_i1])) > 1e-9):
                print("Cannot generate QC plot: The selected trace has invalid data in the fit window.")
                return
    
            _t_fit, _x_fit = _t[_i0:_i1], original_trace_for_plot[_i0:_i1]
            _p0 = [np.max(np.abs(_x_fit)), 1e4, f_center, 0, 0]
            _bounds = ([0, 0, f_center * 0.9, -np.pi, -np.inf], 
                       [np.inf, np.inf, f_center * 1.1, np.pi, np.inf])
            try:
                _popt, _ = curve_fit(ring_model, _t_fit, _x_fit, p0=_p0, bounds=_bounds)
            except RuntimeError:
                _popt = _p0
    
            _fit_full = ring_model(_t, *_popt)
            _i1_ext = min(int(extend_fit_us * 1e-6 * fs), num_samples)
            _taper_sub = np.zeros_like(_t)
            _taper_sub[_i0:_i1_ext] = 1
            _nt = int(0.05 * (_i1_ext - _i0))
            if _nt > 1: _taper_sub[_i1_ext - _nt:_i1_ext] = np.cos(np.linspace(0, np.pi/2, _nt))**2
    
            t_us = self.header['samples'] * 1e6
            fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
            
            axes[0].plot(t_us, original_trace_for_plot, 'k-', label='Original Trace')
            axes[0].plot(t_us, _fit_full, 'b--', alpha=0.7, label='Modeled Ring')
            axes[0].plot(t_us, _fit_full * _taper_sub, 'r-', lw=2, label='Ring to be Subtracted')
            axes[0].axvspan(fit_window_us[0], fit_window_us[1], color='orange', alpha=0.2, label='Fit Window')
            axes[0].set_title(f'De-Ringing Analysis - Trace {show_example_trace}')
            axes[0].set_ylabel('Amplitude'); axes[0].legend(); axes[0].grid(True)
            
            axes[1].plot(t_us, original_trace_for_plot, color='gray', alpha=0.6, label='Original')
            axes[1].plot(t_us, self.traces[show_example_trace, :], 'g-', label='Final Deringed Trace')
            axes[1].set_title('Before vs. After Comparison')
            axes[1].set_xlabel('Time (µs)'); axes[1].set_ylabel('Amplitude'); axes[1].legend(); axes[1].grid(True)
            
            plt.xlim(0, extend_fit_us * 1.5)
            plt.tight_layout()
            plt.show()
    
    def find_closest_neighbor_by_loc(self, source_x, source_y, search_axis='groupY', direction='positive'):
        """Finds the index of the closest neighboring receiver on the same line as the source."""
        if self.traces is None: raise ValueError("Seismic data not loaded.")
        if search_axis not in ['groupX', 'groupY']: raise ValueError("search_axis must be 'groupX' or 'groupY'.")
    
        sx_arr, sy_arr = self.header['sourceX'], self.header['sourceY']
        source_indices = np.where((np.isclose(sx_arr, source_x)) & (np.isclose(sy_arr, source_y)))[0]
        if len(source_indices) == 0: return None
    
        line_axis = 'groupX' if search_axis == 'groupY' else 'groupY'
        source_line_coord = source_x if line_axis == 'groupX' else source_y
        
        on_line_mask = np.isclose(self.header[line_axis][source_indices], source_line_coord)
        on_line_indices = source_indices[on_line_mask]
        if len(on_line_indices) == 0: return None
            
        receiver_search_coords = self.header[search_axis][on_line_indices]
        source_search_coord = source_y if search_axis == 'groupY' else source_x
        displacements = receiver_search_coords - source_search_coord
    
        if direction == 'positive': candidate_mask = displacements > 1e-6
        elif direction == 'negative': candidate_mask = displacements < -1e-6
        else: raise ValueError("Direction must be 'positive' or 'negative'.")
        
        candidate_indices_local = np.where(candidate_mask)[0]
        if len(candidate_indices_local) == 0: return None
            
        closest_in_candidates_idx = np.argmin(np.abs(displacements[candidate_mask]))
        closest_on_line_idx = candidate_indices_local[closest_in_candidates_idx]
        return on_line_indices[closest_on_line_idx]
        
    def find_all_nearest_neighbors(self, search_axis='groupY'):
        """
        Finds the indices of all unique nearest neighbor traces (positive and negative)
        for every unique source location in the dataset.
    
        Returns:
            list: A sorted list of unique integer trace indices corresponding to all
                  the nearest neighbors found.
        """
        if self.traces is None: raise ValueError("Seismic data not loaded.")
    
        source_coords = np.vstack((self.header['sourceX'], self.header['sourceY'])).T
        unique_sources = np.unique(source_coords, axis=0)
        
        neighbor_indices = set() # Use a set to automatically handle duplicates
        print(f"Finding all nearest neighbors for {len(unique_sources)} unique source locations...")
        
        for sx, sy in unique_sources:
            # Find positive neighbor
            idx_pos = self.find_closest_neighbor_by_loc(sx, sy, search_axis=search_axis, direction='positive')
            if idx_pos is not None:
                neighbor_indices.add(idx_pos)
                
            # Find negative neighbor
            idx_neg = self.find_closest_neighbor_by_loc(sx, sy, search_axis=search_axis, direction='negative')
            if idx_neg is not None:
                neighbor_indices.add(idx_neg)
                
        return sorted(list(neighbor_indices))
    
    def auto_pick_first_arrival(self, search_window_us, plot_picks=False, trace_indices=None, method='amplitude'):
        """
        Picks an event using either max amplitude or the peak of the Hilbert envelope.
    
        The 'envelope' method is more robust for complex wavelets as it picks the
        center of energy, preventing cycle skipping. The QC plot will adapt to
        show the data used for picking (raw traces for 'amplitude', envelope for 'envelope').
    
        Args:
            search_window_us (tuple): The (start, end) time in microseconds to search.
            plot_picks (bool, optional): If True, a QC plot is generated.
            trace_indices (array-like, optional): A subset of trace indices to process.
            method (str, optional): 'amplitude' or 'envelope'. Defaults to 'amplitude'.
        """
        if self.traces is None: raise ValueError("Seismic data not loaded.")
        
        fs = 1.0 / (self.header['samples'][1] - self.header['samples'][0])
        num_traces_total, num_samples = self.traces.shape
        
        indices_to_process = np.arange(num_traces_total) if trace_indices is None else np.array(trace_indices, dtype=int)
    
        start_idx = int(search_window_us[0] * 1e-6 * fs)
        end_idx = int(search_window_us[1] * 1e-6 * fs)
    
        if not (0 <= start_idx < end_idx <= num_samples):
            raise ValueError(f"Search window {search_window_us} µs is invalid.")
    
        picks_in_samples = np.full(num_traces_total, np.nan)
        
        # --- Core Picking Logic ---
        trace_subset = self.traces[indices_to_process, start_idx:end_idx]
        if method == 'amplitude':
            print("Picking using method: Maximum Amplitude")
            peak_local_indices = np.argmax(np.abs(trace_subset), axis=1)
        elif method == 'envelope':
            print("Picking using method: Peak of Envelope (Hilbert)")
            envelope_subset = np.abs(scipy.signal.hilbert(trace_subset, axis=1))
            peak_local_indices = np.argmax(envelope_subset, axis=1)
        else:
            raise ValueError("Method must be 'amplitude' or 'envelope'.")
            
        picks_in_samples[indices_to_process] = start_idx + peak_local_indices
        picks_in_seconds = picks_in_samples / fs
        
        # --- NEW: Adaptive QC Plotting ---
        if plot_picks:
            plot_start_us = search_window_us[0] - 10
            plot_end_us = search_window_us[1] + 10
            plt.figure(figsize=(12, 8))
            
            # Decide which data to display in the background
            if method == 'envelope':
                print("Generating Envelope QC Plot...")
                # Calculate envelope for the full traces for a better visual
                plot_data = np.abs(scipy.signal.hilbert(self.traces[indices_to_process, :], axis=1))
                vmax = np.percentile(plot_data, 98)
                cmap = 'viridis' # Use a sequential colormap for all-positive data
                plot_title = 'Auto-Picker Quality Control (Envelope Method)'
                cbar_label = 'Envelope Value'
            else: # Default to amplitude
                print("Generating Amplitude QC Plot...")
                plot_data = self.traces[indices_to_process, :]
                vmax = np.percentile(np.abs(plot_data), 98)
                cmap = 'RdBu_r' # Use a diverging colormap for bipolar data
                plot_title = 'Auto-Picker Quality Control (Amplitude Method)'
                cbar_label = 'Amplitude'
    
            # Display the chosen data
            plt.imshow(plot_data.T, aspect='auto', cmap=cmap, vmax=vmax, vmin=0 if method == 'envelope' else -vmax,
                       extent=[0, len(indices_to_process), self.header['samples'][-1] * 1e6, 0])
            
            # Common plotting elements
            plt.axhspan(search_window_us[0], search_window_us[1], color='white', alpha=0.2, zorder=1, label='Search Window')
            pick_times_us = picks_in_seconds[indices_to_process] * 1e6
            plt.scatter(np.arange(len(indices_to_process)), pick_times_us, 
                        c='yellow', s=20, marker='o', edgecolors='black', linewidths=0.5, label='Picks', zorder=2)
                        
            plt.title(plot_title); plt.xlabel('Trace Number'); plt.ylabel('Time (µs)')
            plt.legend(); plt.ylim(plot_end_us, plot_start_us)
            plt.colorbar(label=cbar_label); plt.show()
            
        return picks_in_seconds
    
    def normalize_by_picked_horizon(self, picks_in_seconds, normalization_window_us, 
                                      target_rms=None, plot_statistics=False, 
                                      trace_indices=None, apply_to='subset'):
        """
        Normalizes trace amplitudes using a dynamic window centered on a picked
        horizon, with options to control the scope of the calculation and application.
        """
        if self.traces is None: raise ValueError("Seismic data not loaded.")
        fs = 1.0 / (self.header['samples'][1] - self.header['samples'][0])
        num_traces_total, num_samples = self.traces.shape
        
        if trace_indices is None:
            indices_for_calc = np.arange(num_traces_total)
        else:
            indices_for_calc = np.array(trace_indices, dtype=int)
            
        print(f"Calculating RMS from a subset of {len(indices_for_calc)} traces...")
        
        rms_values_subset = []
        for i in indices_for_calc:
            pick_time = picks_in_seconds[i]
            if np.isnan(pick_time): continue
            pick_idx = int(pick_time * fs)
            win_start = max(0, pick_idx + int(normalization_window_us[0] * 1e-6 * fs))
            win_end = min(num_samples, pick_idx + int(normalization_window_us[1] * 1e-6 * fs))
            if win_start >= win_end: continue
            windowed_data = self.traces[i, win_start:win_end]
            rms_values_subset.append(np.sqrt(np.mean(windowed_data**2)))
    
        if not rms_values_subset:
            print("Warning: No valid RMS values found. Normalization skipped.")
            return np.nan
    
        mean_rms_before = np.mean(rms_values_subset)
        _target_rms = mean_rms_before if target_rms is None else target_rms
        print(f"Target RMS for normalization is: {_target_rms:.6f}")
        
        if apply_to == 'all':
            global_scaling_factor = _target_rms / mean_rms_before if mean_rms_before > 1e-9 else 1.0
            print(f"Applying a global scaling factor of {global_scaling_factor:.4f} to ALL traces.")
            self.traces *= global_scaling_factor
        elif apply_to == 'subset':
            print(f"Applying scaling factors to the subset of {len(indices_for_calc)} traces.")
            scaling_factors = _target_rms / (np.array(rms_values_subset) + 1e-9)
            self.traces[indices_for_calc, :] *= scaling_factors[:, np.newaxis]
        else:
            raise ValueError("apply_to must be 'subset' or 'all'.")
    
        if plot_statistics:
            rms_values_after = []
            for i in indices_for_calc:
                pick_time = picks_in_seconds[i]
                if np.isnan(pick_time): continue
                pick_idx = int(pick_time * fs)
                win_start = max(0, pick_idx + int(normalization_window_us[0] * 1e-6 * fs))
                win_end = min(num_samples, pick_idx + int(normalization_window_us[1] * 1e-6 * fs))
                if win_start >= win_end: continue
                rms_values_after.append(np.sqrt(np.mean(self.traces[i, win_start:win_end]**2)))
    
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            axes[0].hist(rms_values_subset, bins=50, alpha=0.7, label='Before')
            axes[0].hist(rms_values_after, bins=50, alpha=0.7, label='After')
            axes[0].axvline(mean_rms_before, color='blue', linestyle='--', label=f'Mean Before: {mean_rms_before:.4f}')
            axes[0].axvline(_target_rms, color='red', linestyle='--', label=f'Target RMS: {_target_rms:.4f}')
            axes[0].set_title('RMS Amplitude Distribution'); axes[0].set_xlabel('RMS in Dynamic Window'); axes[0].legend(); axes[0].grid(True)
            
            if apply_to == 'all':
                 axes[1].hist([global_scaling_factor], bins=1, color='green', rwidth=0.1)
                 axes[1].set_title(f'Global Scaling Factor Applied: {global_scaling_factor:.4f}')
            else: # subset
                scaling_factors = _target_rms / (np.array(rms_values_subset) + 1e-9)
                axes[1].hist(scaling_factors, bins=50, color='green')
                axes[1].set_title('Distribution of Scaling Factors')
            axes[1].set_xlabel('Scaling Factor'); axes[1].grid(True)
            plt.suptitle("Picked Horizon Normalization QC"); plt.tight_layout(); plt.show()
        
        print("Normalization complete.\n")
        return _target_rms
    
    
    def apply_deconvolution(self, operator_len_ms, pre_whitening_percent=0.1, 
                          design_window_ms=None, plot_example_trace=-1):
        """
        Applies spiking deconvolution, with an optional design window for robustness.
    
        This method designs a Wiener filter to compress the wavelet. For best results,
        the filter should be designed on a clean, isolated wavelet using the
        'design_window_ms' parameter.
    
        Args:
            operator_len_ms (float): The length of the deconvolution filter in milliseconds.
                                     Should be about the length of the target wavelet.
            pre_whitening_percent (float, optional): A stabilization factor. Defaults to 0.1.
            design_window_ms (tuple, optional): A (start_ms, end_ms) tuple. If provided,
                                                 the filter is designed ONLY from this portion
                                                 of the trace. This is highly recommended.
            plot_example_trace (int, optional): Index of a trace for QC plotting.
        """
        if self.traces is None:
            raise ValueError("Traces not loaded. Please read a SEGY file first.")
        
        print("\n--- Applying Spiking Deconvolution with Design Window ---")
        dt = self.header['samples'][1] - self.header['samples'][0]
        fs = 1.0 / dt
        operator_len_samples = int((operator_len_ms / 1000) * fs)
    
        if operator_len_samples <= 0:
            raise ValueError("Operator length is too short for the given sampling rate.")
        print(f"Using operator length: {operator_len_ms} ms ({operator_len_samples} samples)")
        if design_window_ms:
            print(f"Using design window from {design_window_ms[0]} ms to {design_window_ms[1]} ms.")
    
        num_traces, num_samples = self.traces.shape
        deconvolved_traces = np.zeros_like(self.traces)
    
        original_trace_for_plot = None
        if 0 <= plot_example_trace < num_traces:
            original_trace_for_plot = self.traces[plot_example_trace, :].copy()
    
        for i in range(num_traces):
            if i % 100 == 0:
                print(f'\rProcessing trace {i+1}/{num_traces}', end='')
            
            trace = self.traces[i, :]
            
            # --- DESIGN WINDOW LOGIC ---
            # Use a specific part of the trace to design the filter if a window is given.
            if design_window_ms is not None:
                start_sample = int((design_window_ms[0] / 1000) * fs)
                end_sample = int((design_window_ms[1] / 1000) * fs)
                
                if 0 <= start_sample < end_sample <= num_samples:
                    design_trace = trace[start_sample:end_sample]
                else:
                    design_trace = trace # Fallback if window is invalid
            else:
                design_trace = trace # Default to using the whole trace
            
            # Autocorrelation is now calculated ONLY from the clean design_trace
            autocorr = np.correlate(design_trace, design_trace, mode='full')
            center_point = len(design_trace) - 1
            autocorr = autocorr[center_point : center_point + operator_len_samples]
            
            # Design the inverse filter from the autocorrelation
            r = scipy.linalg.toeplitz(autocorr)
            r[0, 0] *= (1.0 + pre_whitening_percent / 100.0)
            g = np.zeros(operator_len_samples); g[0] = 1.0
    
            try:
                inverse_filter = np.linalg.solve(r, g)
            except np.linalg.LinAlgError:
                print(f"\nWarning: Could not design filter for trace {i+1}. Skipping.")
                inverse_filter = g
    
            # Apply the well-designed filter to the ORIGINAL, FULL-LENGTH trace
            deconvolved_traces[i, :] = np.convolve(trace, inverse_filter, mode='same')
    
        print(f'\rProcessing trace {num_traces}/{num_traces}... Done.')
        self.traces = deconvolved_traces
        
        # QC plotting remains the same
        if original_trace_for_plot is not None:
            print(f"Plotting QC example for trace {plot_example_trace}...")
            t_arr = self.header['samples'] * 1e3
            fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
            axes[0].plot(t_arr, original_trace_for_plot, label='Before Deconvolution')
            axes[0].set_title(f'Trace {plot_example_trace} - Before Deconvolution')
            axes[0].grid(True); axes[0].legend(); axes[0].set_ylabel('Amplitude')
            axes[1].plot(t_arr, self.traces[plot_example_trace, :], label='After Deconvolution', color='red')
            axes[1].set_title(f'Trace {plot_example_trace} - After Deconvolution')
            axes[1].grid(True); axes[1].legend(); axes[1].set_xlabel('Time (ms)'); axes[1].set_ylabel('Amplitude')
            plt.tight_layout()
            plt.show()
            
            
    
    def apply_tvsw(self, window_length_ms, overlap_percent=50, smoothing_factor_hz=10000, plot_example_trace=None):
        """
        Applies Time-Varying Spectral Whitening (TVSW) to the stacked data.
        This enhances temporal resolution by balancing the spectrum in sliding windows.
    
        Args:
            window_length_ms (float): The length of the sliding window in milliseconds.
                                      This is the most critical tuning parameter.
            overlap_percent (int, optional): The percentage of overlap between windows. Defaults to 50.
            smoothing_factor_hz (float, optional): The width of the moving average filter
                                                   used to smooth the spectrum, in Hz.
            plot_example_trace (tuple, optional): A tuple (iline, xline) of a trace
                                                  to use for generating a QC plot.
        """
        if self.cmp_stacked is None:
            raise ValueError("Stacked data (cmp_stacked) not found. Run NMO stacking first.")
    
        print("\n--- Applying Time-Varying Spectral Whitening ---")
        
        # --- 1. Parameter Conversion ---
        dt = self.header['samples'][1] - self.header['samples'][0]
        fs = 1.0 / dt
        window_len_samples = int((window_length_ms / 1000) * fs)
        if window_len_samples % 2 != 0: window_len_samples += 1 # Ensure even length for FFT
        
        smoothing_len_samples = int(smoothing_factor_hz / (fs / window_len_samples))
        step_samples = int(window_len_samples * (1 - overlap_percent / 100))
    
        print(f"Window: {window_length_ms} ms ({window_len_samples} samples), "
              f"Smoothing: {smoothing_factor_hz} Hz ({smoothing_len_samples} samples)")
    
        # --- 2. Get Input Data and Prepare Output Array ---
        input_data = self.cmp_stacked
        output_data = np.zeros_like(input_data)
        num_ilines, num_xlines, num_samples = input_data.shape
    
        # --- 3. Main Loop over all Traces ---
        for i in range(num_ilines):
            for j in range(num_xlines):
                trace = input_data[i, j, :]
                
                # --- 4. Sliding Window Processing for a single trace ---
                whitened_trace = np.zeros(num_samples)
                win_sum = np.zeros(num_samples)
                hanning_win = np.hanning(window_len_samples)
    
                for k_start in range(0, num_samples - window_len_samples, step_samples):
                    k_end = k_start + window_len_samples
                    
                    # Extract and window the segment
                    segment = trace[k_start:k_end] * hanning_win
                    
                    # To frequency domain
                    spec = np.fft.fft(segment)
                    amp_spec = np.abs(spec)
                    
                    # Smooth the amplitude spectrum to get the general shape
                    # Add epsilon for stability to prevent division by zero
                    smoothed_amp_spec = uniform_filter1d(amp_spec, size=smoothing_len_samples) + 1e-9
                    
                    # Whitening operator is the inverse of the spectral shape
                    whitening_op = 1.0 / smoothed_amp_spec
                    
                    # Apply whitening and convert back to time domain
                    whitened_spec = spec * whitening_op
                    whitened_segment = np.fft.ifft(whitened_spec).real
                    
                    # Overlap-add back to the full trace
                    whitened_trace[k_start:k_end] += whitened_segment * hanning_win
                    win_sum[k_start:k_end] += hanning_win**2
    
                # Normalize by the window sum to handle overlaps correctly
                whitened_trace[win_sum > 0] /= win_sum[win_sum > 0]
                output_data[i, j, :] = whitened_trace
        
        # --- 5. Update the object's data ---
        original_stacked_for_plot = self.cmp_stacked if plot_example_trace else None
        self.cmp_stacked = output_data
        print("TVSW complete.")
    
        # --- 6. QC Plotting ---
        if plot_example_trace is not None:
            print(f"Generating QC plot for trace: {plot_example_trace}")
            iline, xline = plot_example_trace
            
            # Get data for plots
            trace_before = original_stacked_for_plot[iline, xline, :]
            trace_after = self.cmp_stacked[iline, xline, :]
            t_axis = self.header['samples'] * 1000
            
            freq_axis = np.fft.fftfreq(num_samples, d=dt)
            spec_before = np.abs(np.fft.fft(trace_before))
            spec_after = np.abs(np.fft.fft(trace_after))
    
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Time domain plots
            axes[0, 0].plot(t_axis, trace_before)
            axes[0, 0].set_title(f'Trace Before TVSW')
            axes[0, 0].set_ylabel('Amplitude'); axes[0, 0].grid(True)
            
            axes[0, 1].plot(t_axis, trace_after, color='C1')
            axes[0, 1].set_title(f'Trace After TVSW')
            axes[0, 1].grid(True)
    
            # Frequency domain plots
            axes[1, 0].plot(freq_axis, spec_before)
            axes[1, 0].set_title('Spectrum Before TVSW')
            axes[1, 0].set_xlabel('Frequency (Hz)'); axes[1, 0].set_ylabel('Magnitude')
            axes[1, 0].set_xlim(0, fs / 2)
    
            axes[1, 1].plot(freq_axis, spec_after, color='C1')
            axes[1, 1].set_title('Spectrum After TVSW')
            axes[1, 1].set_xlabel('Frequency (Hz)')
            axes[1, 1].set_xlim(0, fs / 2)
            
            plt.tight_layout()
            plt.show()
            
    def apply_deringing_parallel(self, f_center,
                                 tb_us=8,
                                 fit_window_us=(50, 200),
                                 extend_fit_us=300,
                                 n_threads=20):
        """
        [PARALLEL VERSION] Removes source ringing by modeling and subtracting an
        exponentially decaying sinusoid using multiple CPU cores.
        """
        if self.traces is None:
            raise ValueError("Traces not found. Load data first.")
            
        dt = self.header['samples'][1] - self.header['samples'][0]
        fs = 1.0 / dt
        num_traces, _ = self.traces.shape
    
        if n_threads is None:
            n_threads = mp.cpu_count()
        
        print(f"\n--- Applying Parallel De-ringing using {n_threads} cores ---")
    
        # Prepare a list of arguments for each trace
        args = [(self.traces[i, :], self.header['samples'], f_center, tb_us,
                 fit_window_us, extend_fit_us, fs) for i in range(num_traces)]
    
        # Create a pool of workers and map the job.
        # pool.map distributes the work and blocks until all results are ready.
        with mp.Pool(processes=n_threads) as pool:
            results = pool.map(deringing_worker, args)
    
        # The results are a list of processed traces. Reassemble them into the traces matrix.
        self.traces = np.array(results)
        print("Parallel de-ringing complete.\n")
            
    

    

    def extract_wavelet(self, center_depth_m, thickness_m, 
                        iline_step=10, xline_step=10, window_samples=40):
        """
        Extracts a stable, average wavelet from the main reflector in the 3D migrated volume.

        Args:
            center_depth_m (float): The approximate center depth of the target reflector.
            thickness_m (float): The thickness of the window to search for the reflector peak.
            iline_step (int): The spacing of traces to analyze in the inline direction.
            xline_step (int): The spacing of traces to analyze in the crossline direction.
            window_samples (int): The number of samples to extract around the peak (+/-).

        Returns:
            numpy.ndarray: The stacked, average wavelet.
        """
        if self.mig_img is None:
            raise ValueError("Migrated data (`mig_img`) not found.")
        print("--- Extracting Average Wavelet from 3D Volume ---")
        
        # --- 1. Define the search slab and analysis grid ---
        _, _, z_coords = self.mig_img_SA
        dz = z_coords[1] - z_coords[0]
        start_idx = np.argmin(np.abs(z_coords - (center_depth_m - thickness_m / 2.0)))
        end_idx = np.argmin(np.abs(z_coords - (center_depth_m + thickness_m / 2.0)))
        slab = self.mig_img[:, :, start_idx:end_idx]
        
        num_ilines, num_xlines, _ = self.mig_img.shape
        wavelet_stack = []

        # --- 2. Loop through a sparse grid of traces ---
        for il in range(0, num_ilines, iline_step):
            for xl in range(0, num_xlines, xline_step):
                trace_slab = slab[il, xl, :]
                if np.sum(np.abs(trace_slab)) < 1e-9: # Skip empty traces
                    continue
                
                # --- 3. Find peak, window, and stack ---
                peak_idx_local = np.argmax(np.abs(trace_slab))
                peak_idx_global = start_idx + peak_idx_local
                
                win_start = peak_idx_global - window_samples
                win_end = peak_idx_global + window_samples
                
                if win_start >= 0 and win_end < len(z_coords):
                    full_trace = self.mig_img[il, xl, :]
                    wavelet_segment = full_trace[win_start:win_end]
                    wavelet_stack.append(wavelet_segment)

        if not wavelet_stack:
            raise RuntimeError("Could not extract any wavelets. Check depth and thickness parameters.")

        # --- 4. Average the stack and return ---
        average_wavelet = np.mean(np.array(wavelet_stack), axis=0)
        print(f"Successfully stacked {len(wavelet_stack)} wavelets.")
        return average_wavelet

    def apply_wavelet_shaping(self, source_wavelet, target_wavelet, filter_len=40, pre_whitening=0.1):
        """
        Designs a Wiener filter to shape a source wavelet into a target wavelet
        and applies it to the entire 3D migrated data volume.

        Args:
            source_wavelet (numpy.ndarray): The wavelet extracted from the data.
            target_wavelet (numpy.ndarray): The desired output wavelet (e.g., a Ricker).
            filter_len (int): The length of the shaping filter in samples.
            pre_whitening (float): A stabilization factor (percentage) for the filter design.
        """
        from scipy.signal import convolve
        from scipy.linalg import solve_toeplitz
        print("--- Designing and Applying Wavelet Shaping Filter ---")

        # --- 1. Design the Wiener Shaping Filter ---
        # Autocorrelation of the source wavelet
        r_xx = np.correlate(source_wavelet, source_wavelet, mode='full')
        r_xx = r_xx[len(source_wavelet)-1 : len(source_wavelet)-1 + filter_len]
        r_xx[0] *= (1.0 + pre_whitening / 100.0) # Add pre-whitening for stability

        # Cross-correlation of target and source
        r_yx = np.correlate(target_wavelet, source_wavelet, mode='full')
        center_lag = len(source_wavelet) - 1 - int(filter_len / 2)
        r_yx = r_yx[center_lag : center_lag + filter_len]

        # Solve the Toeplitz system to get the filter
        shaping_filter = solve_toeplitz(r_xx, r_yx)

        # --- 2. Apply the filter to the 3D data ---
        print("Applying filter to 3D data volume... (this may take a moment)")
        # We apply the 1D filter along the depth axis (axis=2)
        original_data = self.mig_img
        sharpened_data = np.apply_along_axis(
            lambda trace: convolve(trace, shaping_filter, mode='same'),
            axis=2,
            arr=original_data
        )
        
        self.mig_img = sharpened_data
        print("Wavelet shaping complete. `self.mig_img` has been updated.")
        return shaping_filter
                
        
        # REPLACE the old apply_wavelet_shaping with this one in seispy.py

    def design_and_apply_shaping(self, source_wavelet, target_wavelet, filter_len=40, pre_whitening=0.1, return_filter_only=False):
        """
        Designs a Wiener filter and applies it, returning the sharpened data.
        Critically, it also calculates a single amplitude scaling factor based on the RMS
        of the input and output data to ensure energy preservation.
        """
        from scipy.signal import convolve
        from scipy.linalg import solve_toeplitz
        print("--- Designing Wiener Shaping Filter ---")

        # --- 1. Design the Wiener Shaping Filter ---
        r_xx = np.correlate(source_wavelet, source_wavelet, mode='full')
        r_xx = r_xx[len(source_wavelet)-1 : len(source_wavelet)-1 + filter_len]
        r_xx[0] *= (1.0 + pre_whitening / 100.0)
        r_yx = np.correlate(target_wavelet, source_wavelet, mode='full')
        center_lag = len(source_wavelet) - 1 - int(filter_len / 2)
        r_yx = r_yx[center_lag : center_lag + filter_len]
        shaping_filter = solve_toeplitz(r_xx, r_yx)
        
        if return_filter_only:
            return shaping_filter

        # --- 2. Apply the filter to the 3D data ---
        print("Applying filter to 3D data volume...")
        original_data = self.mig_img
        sharpened_data = np.apply_along_axis(
            lambda trace: convolve(trace, shaping_filter, mode='same'),
            axis=2,
            arr=original_data
        )
        
        # --- 3. Calculate the single, critical scaling factor ---
        # This factor ensures the overall energy is preserved.
        rms_original = np.sqrt(np.mean(original_data**2))
        rms_sharpened = np.sqrt(np.mean(sharpened_data**2))
        
        # Avoid division by zero if the data is all zeros
        scaling_factor = rms_original / rms_sharpened if rms_sharpened > 1e-9 else 1.0

        print(f"Calculated a global scaling factor of: {scaling_factor:.4f}")
        
        # --- 4. Apply the scaling factor ---
        sharpened_data *= scaling_factor
        
        # --- 5. Update the object's data ---
        self.mig_img = sharpened_data
        print("Wavelet shaping and scaling complete. `self.mig_img` has been updated.")
        
        # Return both for QC and potential reuse
        return shaping_filter, scaling_factor
        
    
    # PASTE THIS NEW FUNCTION INTO your Seismic CLASS

    

    
# %% NRMS methods
    
    def compute_nrms_within_shot_locations(self, t0=0.0, t1=0.2/1000, window=None,
                                           normalize=True, shot_index=None, plotting=True):
        """
        Compute NRMS between receivers within each shot gather, grouped by (sourceX, sourceY) location.
    
        Uses `seis.shot_gathers_by_loc` and `seis.shot_locations`
        created by `seis.organize_shots_by_location()`.
    
        Parameters
        ----------
        seis : Seismic
            Seismic object with loaded traces and headers.
        t0, t1 : float
            Time window (seconds) for NRMS calculation.
        window : int or None
            Optional RMS smoothing window size (samples).
        normalize : bool
            Normalize traces by RMS amplitude before NRMS computation.
        shot_index : int or None
            If specified, compute NRMS for this single shot location (QC mode).
            If None, compute NRMS for all shot locations.
    
        Returns
        -------
        nrms_results : dict or np.ndarray
            - If `shot_index` is None: dict {shot_index: NRMS_matrix}
            - If `shot_index` is set: NRMS_matrix (2D array)
        """
    
        # --- Ensure shots are organized ---
        if not hasattr(self, "shot_gathers_by_loc") or not self.shot_gathers_by_loc:
            self.organize_shots_by_location()
    
        dt = self.header['samples'][1] - self.header['samples'][0]
        i0, i1 = int(t0 / dt), int(t1 / dt)
    
        nrms_results = {}
    
        # --- Select shots to process ---
        if shot_index is not None:
            if shot_index not in self.shot_gathers_by_loc:
                raise ValueError(f"Shot index {shot_index} not found in seis.shot_gathers_by_loc.")
            shot_dict = {shot_index: self.shot_gathers_by_loc[shot_index]}
            print(f"Computing NRMS for single shot location index {shot_index}")
        else:
            shot_dict = self.shot_gathers_by_loc
            print(f"Computing NRMS for all {len(shot_dict)} shot locations...")
    
        # --- Loop over shots ---
        for sid, indices in shot_dict.items():
            gather = self.traces[indices, i0:i1]  # shape: (n_receivers, n_samples)
            if gather.size == 0:
                continue
    
            if window:
                gather = matrix_RMS(gather, window)
    
            n_traces = gather.shape[0]
            nrms_matrix = np.zeros((n_traces, n_traces))
    
            for i in range(n_traces):
                for j in range(i + 1, n_traces):
                    A, B = gather[i], gather[j]
                    if normalize:
                        A /= np.sqrt(np.mean(A**2) + 1e-12)
                        B /= np.sqrt(np.mean(B**2) + 1e-12)
                    rms_diff = np.sqrt(np.mean((A - B)**2))
                    rms_sum = np.sqrt(np.mean(A**2)) + np.sqrt(np.mean(B**2))
                    nrms_val = 200 * rms_diff / (rms_sum + 1e-12)
                    nrms_matrix[i, j] = nrms_val
                    nrms_matrix[j, i] = nrms_val
    
            nrms_results[sid] = nrms_matrix
        
        if plotting:
            plt.figure(figsize=(6,5))
            plt.imshow(nrms_matrix, cmap='magma', origin='lower')
            title = "NRMS between receivers"
            if self is not None and shot_index is not None and hasattr(self, "shot_locations"):
                loc = self.shot_locations[shot_index]
                title += f" - Shot {shot_index} @ ({loc[0]:.1f}, {loc[1]:.1f})"
            plt.title(title)
            plt.xlabel("Receiver index")
            plt.ylabel("Receiver index")
            plt.colorbar(label="NRMS (%)")
            plt.tight_layout()
            plt.show()
        # Return a single result if QC mode
        if shot_index is not None:
            return nrms_results[shot_index]
        else:
            print(f"Completed NRMS computation for {len(nrms_results)} shot locations.")
            return nrms_results
                
        
        
    def compute_nrms_vs_distance(self, t0=0.0, t1=0.2/1000, window=None,
                                 normalize=True, shot_index=None,
                                 receiver_key=('groupX', 'groupY')):
        """
        Compute NRMS between receivers within each shot gather and average by receiver spacing.
        Returns either {shot_index: (distances, mean_nrms)} or (distances, mean_nrms).
        """
        if not hasattr(self, "shot_gathers_by_loc") or not self.shot_gathers_by_loc:
            self.organize_shots_by_location()
    
        dt = self.header['samples'][1] - self.header['samples'][0]
        i0, i1 = int(t0 / dt), int(t1 / dt)
        distance_nrms = {}
    
        if shot_index is not None:
            if shot_index not in self.shot_gathers_by_loc:
                raise ValueError(f"Shot index {shot_index} not found.")
            shots_to_process = {shot_index: self.shot_gathers_by_loc[shot_index]}
        else:
            shots_to_process = self.shot_gathers_by_loc
    
        for sid, indices in shots_to_process.items():
            gather = self.traces[indices, i0:i1]
            if gather.size == 0:
                continue
            if window:
                gather = matrix_RMS(gather, window)
    
            rec_x = self.header[receiver_key[0]][indices]
            rec_y = self.header[receiver_key[1]][indices]
            rec_coords = np.vstack((rec_x, rec_y)).T
    
            n_traces = len(indices)
            distances, nrms_vals = [], []
    
            for i in range(n_traces):
                for j in range(i + 1, n_traces):
                    dx, dy = rec_coords[i] - rec_coords[j]
                    dist = np.sqrt(dx**2 + dy**2)
                    A, B = gather[i], gather[j]
                    if normalize:
                        A /= np.sqrt(np.mean(A**2) + 1e-12)
                        B /= np.sqrt(np.mean(B**2) + 1e-12)
                    rms_diff = np.sqrt(np.mean((A - B)**2))
                    rms_sum = np.sqrt(np.mean(A**2)) + np.sqrt(np.mean(B**2))
                    nrms = 200 * rms_diff / (rms_sum + 1e-12)
                    distances.append(dist)
                    nrms_vals.append(nrms)
    
            distances = np.array(distances)
            nrms_vals = np.array(nrms_vals)
            bins = np.unique(distances)
            mean_nrms = np.array([np.mean(nrms_vals[distances == b]) for b in bins])
            distance_nrms[sid] = (bins, mean_nrms)
            
        if shot_index is not None:
            self.distance_nrms = distance_nrms[shot_index]
        else:
            self.distance_nrms = distance_nrms
        return distance_nrms[shot_index] if shot_index is not None else distance_nrms
    
    
    def plot_nrms_vs_distance(self, data, shot_index=None, all_shots=False,
                              regression=False, poly_order=1, color='red',
                              nrms_threshold=141):
        """
        Plot NRMS vs distance for either a single shot or all shot-gathers,
        with optional regression line. Filters out NRMS values above a threshold.
    
        Parameters
        ----------
        data : tuple or dict
            - If all_shots=False: (distances, mean_nrms) for one shot.
            - If all_shots=True: {shot_index: (distances, mean_nrms)} for multiple shots.
        seis : Seismic or None, optional
            Optional, for showing shot coordinates in the legend/title.
        shot_index : int or None, optional
            Shot index (used only if all_shots=False).
        all_shots : bool, default=False
            If True, plot all shots together. If False, plot a single shot.
        regression : bool, default=False
            If True, fit and overlay a regression line.
        poly_order : int, default=1
            Polynomial order for regression (1 = linear).
        color : str, default='red'
            Color of regression line.
        nrms_threshold : float, default=141
            Remove all datapoints with NRMS larger than this value.
        """
        if hasattr(self, "distance_nrms"):
           data =  self.distance_nrms
        if all_shots:
            plt.figure(figsize=(7,6))
            all_distances, all_nrms = [], []
            for sid, (distances, mean_nrms) in data.items():
                mask = mean_nrms <= nrms_threshold
                distances = np.array(distances)[mask]
                mean_nrms = np.array(mean_nrms)[mask]
                plt.scatter(distances, mean_nrms, alpha=0.01, s=5, color='black')
                all_distances.extend(distances)
                all_nrms.extend(mean_nrms)
            plt.title("NRMS vs Receiver Distance for All Shot Gathers")
            grid_alpha = 0.4
            if regression and len(all_distances) > 1:
                coeffs = np.polyfit(all_distances, all_nrms, poly_order)
                poly = np.poly1d(coeffs)
                xfit = np.linspace(min(all_distances), max(all_distances), 200)
                plt.plot(xfit, poly(xfit), color=color, lw=2,
                         label=f"Poly{poly_order} fit")
                plt.legend()
        else:
            distances, mean_nrms = data
            distances = np.array(distances)
            mean_nrms = np.array(mean_nrms)
            mask = mean_nrms <= nrms_threshold
            distances = distances[mask]
            mean_nrms = mean_nrms[mask]
    
            plt.figure(figsize=(6,5))
            plt.scatter(distances, mean_nrms, s=15)
            title = "NRMS vs Receiver Distance"
            if shot_index is not None and self is not None and hasattr(self, "shot_locations"):
                loc = self.shot_locations[shot_index]
                title += f" - Shot {shot_index} @ ({loc[0]:.1f}, {loc[1]:.1f})"
            plt.title(title)
            grid_alpha = 0.5
            if regression and len(distances) > 1:
                coeffs = np.polyfit(distances, mean_nrms, poly_order)
                poly = np.poly1d(coeffs)
                xfit = np.linspace(min(distances), max(distances), 200)
                plt.plot(xfit, poly(xfit), color=color, lw=2,
                         label=f"Poly{poly_order} fit")
                plt.legend()
    
        plt.xlabel("Receiver distance (m)")
        plt.ylabel("NRMS (%)")
        plt.grid(True, ls='--', alpha=grid_alpha)
        plt.tight_layout()
        plt.show()
    

        
        
                
        
        
        
        
        