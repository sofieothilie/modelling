#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import SymLogNorm
from concurrent.futures import ProcessPoolExecutor
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from math import ceil

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

def scale(x):
    global dh
    return  x *dh*1e2

def plot_data_2d(file_path, output_path):

        
    global Nx 
    global Ny 
    global Nz 

    global dh
    global ppw
    # print(dh)
    step = ppw / 2

    data_raw = np.fromfile(file_path, dtype=np.float32)
    n = data_raw.size
    # find factor pairs of n
    pairs = []
    for r in range(1, int(np.sqrt(n)) + 1):
        if n % r == 0:
            c = n // r
            pairs.append((r, c))
            if r != c:
                pairs.append((c, r))
    if not pairs:
        # fallback: reshape to a single row
        rows, cols = 1, n
    else:
        # prefer pair closest to expected aspect ratio Nz/Ny (rows ~ Nz/step, cols ~ Ny/step)
        desired_ratio = (Nz / step) / (Ny / step) if (Ny > 0 and step > 0) else None
        if desired_ratio:
            rows, cols = min(pairs, key=lambda rc: abs((rc[0] / rc[1]) - desired_ratio))
        else:
            rows, cols = max(pairs)  # take the pair with largest rows
    data = data_raw.reshape(rows, cols)

    # Keep only the region of x from 1000 to 1500
    # data = data.iloc[:, 1000:1500]

    # print(data.min())


        
    Lx =  step*scale(data.shape[1])
    Lz = step*scale( data.shape[0])

    plt.imshow(data, norm=SymLogNorm(linthresh=1e-5, vmin=-1, vmax=1), cmap='seismic' , extent=[0, Lx, Lz, 0])
    plt.xlabel("Y (cm)")
    plt.ylabel("Z (cm)")

    # plt.imshow(data,vmin=-0.1, vmax=0.1, cmap='seismic')
    plt.colorbar()
    
    

    
    plt.text(
        1.23, -0.1,
        file_path,
        ha='center', va='top', transform=plt.gca().transAxes,
        fontsize=8, color='gray'
    )
    iteration = int(file_path.split("/")[-1].split(".")[0])
    
    # Time stamp as text box
    time_in_microseconds = snapshots* dt * iteration * 1e6
    if time_in_microseconds < 1e2:
        time_text = f"{time_in_microseconds:.3f} μs"
    else:
        time_in_milliseconds = time_in_microseconds / 1e3
        time_text = f"{time_in_milliseconds:.3f} ms"
    
    plt.gcf().text(
        0.7, 0.92,  # x, y (top right of the plot)
        time_text.replace("e-0", "e-").replace("e+0", "e+"),
        ha='center', va='top',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    
    # Draw an empty rectangle with padding from each border
    
    plt.savefig(output_path)
    plt.close()


def plot_data_1d(file_path, output_path):
    # Load the data from the file and reshape it into a 2D array of shape (N, M)
    data = pd.read_csv(file_path, sep=' ', dtype=np.float64)

    # Extract the slice at y = 50 (row 50)
    y_index = 22  # Change this to any row index you want to plot
    slice_data = data.values[y_index, :]  # Select the entire row corresponding to y = 50

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the 1D slice
    ax.plot(slice_data, color='blue')

    # Apply symlog scale to the y-axis for symmetric log
    ax.set_yscale('symlog', linthresh=1e-5)  # Set the threshold for linear region near 0

    ax.set_xlabel("X")  # This corresponds to the x-axis (column index)
    ax.set_ylabel("Value")  # This corresponds to the data values
    ax.set_title(f"1D Slice at y = {y_index}")

    ax.set_ylim((-1,1))
    # ax.axvline(x=100, color='red', linestyle='--', label='Boundary limit')
    
    # Save the plot to the output path
    plt.savefig(output_path)
    plt.close()



def process_file(data_file):
    output_file = data_file.replace(".dat", ".png").replace("wave_data", "wave_images")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plot_data_2d(data_file, output_file)
    return output_file

def main():
    if not os.path.isdir(data_folder):
        print(f"Error: Data folder {data_folder} does not exist.")
        return
    global ppw
    ppw = float(sys.argv[1])
    global snapshots
    snapshots =  int(sys.argv[2])
    global dt
    
    global dh
    freq = 150e3 #1e6
    dh = 1500.0 / freq / ppw 

    global Nx 
    global Ny 
    global Nz 
    Nx = int(float(sys.argv[3]) / dh)
    Ny = int(float(sys.argv[4]) / dh)
    Nz = int(float(sys.argv[5]) / dh)

    dt = 0.9 * dh / (2270 * np.sqrt(3))
    
    os.makedirs("images", exist_ok=True)
    
    data_files = [os.path.join(root, file) for root, _, files in os.walk(data_folder) for file in files if file.endswith(".dat")]
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, data_files), total=len(data_files), desc="Plotting plots"))    
    print("All plots have been generated.")

if __name__ == "__main__":
    main()
