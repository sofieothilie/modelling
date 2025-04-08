#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import SymLogNorm
from concurrent.futures import ProcessPoolExecutor
from mpl_toolkits.mplot3d import Axes3D

M = 100
N = 100

data_folder = "./wave_data"

def plot_data(file_path, output_path):
    data = np.fromfile(file_path, dtype=np.float32).reshape(N,M)
    shape = (10, 10 * N / M) if M > N else (10 * M / N, 10)#correct ratio
    plt.figure(figsize=shape)
    plt.imshow(data, cmap='jet', origin='lower')
    plt.colorbar()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(output_path)
    plt.close()



def plot_data_3d(file_path, output_path):
    data = np.fromfile(file_path, dtype=np.float64).reshape(N, M)
    
    x = np.linspace(0, 1, M)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    norm = SymLogNorm(linthresh=1e-3)
    ax.plot_surface(X, Y, data, cmap='jet', edgecolor='none', norm=norm)

    ax.view_init(elev=40, azim=-80)
    ax.invert_yaxis()
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Wave Amplitude")
    ax.set_zlim(-0.01,0.01)
    
    plt.savefig(output_path)
    plt.close()


def plot_data_2d(file_path, output_path):
    data = pd.read_csv(file_path, sep=' ', dtype=np.float64)
    plt.imshow(data, norm=SymLogNorm(linthresh=1e-3, vmin=-1, vmax=1), cmap='seismic')
    plt.colorbar()
    plt.savefig(output_path)
    plt.close()


def plot_data_1d(file_path, output_path):
    # Load the data from the file and reshape it into a 2D array of shape (N, M)
    data = pd.read_csv(file_path, sep=' ', dtype=np.float64)

    # Extract the slice at y = 50 (row 50)
    y_index = 50  # Change this to any row index you want to plot
    slice_data = data.values[y_index, :]  # Select the entire row corresponding to y = 50

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the 1D slice
    ax.plot(slice_data, color='blue')

    # Apply symlog scale to the y-axis for symmetric log
    #ax.set_yscale('symlog', linthresh=1e-4)  # Set the threshold for linear region near 0

    ax.set_xlabel("X")  # This corresponds to the x-axis (column index)
    ax.set_ylabel("Value")  # This corresponds to the data values
    ax.set_title(f"1D Slice at y = {y_index}")

    ax.set_ylim((-1,1))
    ax.axvline(x=100, color='red', linestyle='--', label='Boundary limit')
    
    # Save the plot to the output path
    plt.savefig(output_path)
    plt.close()



def process_file(data_file):
    output_file = data_file.replace(".dat", ".png").replace("wave_data", "wave_images")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plot_data_2d(data_file, output_file)
    print(f"Plot saved to {output_file}")

def main():
    if not os.path.isdir(data_folder):
        print(f"Error: Data folder {data_folder} does not exist.")
        return
    
    os.makedirs("images", exist_ok=True)
    
    data_files = [os.path.join(root, file) for root, _, files in os.walk(data_folder) for file in files if file.endswith(".dat")]
    
    with ProcessPoolExecutor() as executor:
        executor.map(process_file, data_files)
    
    print("All plots have been generated.")

if __name__ == "__main__":
    main()
