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
    data = pd.read_csv(file_path, sep=' ', dtype=np.float64)
    global padding

    
    x0 = scale(-padding)
    y0 = scale(-padding)
        
    Lx =  scale(data.shape[1]) +x0
    Lz = scale( data.shape[0]) +y0
    plt.imshow(data, norm=SymLogNorm(linthresh=1e-3, vmin=-1, vmax=1), cmap='seismic' , extent=[x0, Lx, Lz, y0])
    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")

    # plt.imshow(data,vmin=-0.1, vmax=0.1, cmap='seismic')
    plt.colorbar()
    
    
    global dt
    rect = plt.Rectangle(
        (0,0),  # Bottom-left corner
        scale((data.shape[1] - 2 * padding)),  # Width
        scale((data.shape[0] - 2 * padding)),  # Height
        linewidth=1, edgecolor='green', facecolor='none', linestyle='dotted'
    )
    

    
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

    plt.gca().add_patch(rect)
    
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
    
    ppw = float(sys.argv[1])
    global snapshots
    snapshots =  int(sys.argv[2])
    global dt
    global padding 
    padding = int(sys.argv[3])
    
    global dh 
    dh = 1500.0 / 1e6 / ppw 

    dt = 0.9 * dh / (2270 * np.sqrt(3))
    
    os.makedirs("images", exist_ok=True)
    
    data_files = [os.path.join(root, file) for root, _, files in os.walk(data_folder) for file in files if file.endswith(".dat")]
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, data_files), total=len(data_files), desc="Plotting plots"))    
    print("All plots have been generated.")

if __name__ == "__main__":
    main()
