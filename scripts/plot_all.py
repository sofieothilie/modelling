#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from mpl_toolkits.mplot3d import Axes3D

import launch_sim

M = launch_sim.res[1]
N = launch_sim.res[0]

data_folder = "./wave_data"

def plot_data(file_path, output_path):
    data = np.fromfile(file_path, dtype=np.float64).reshape(N,M)
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
    
    ax.plot_surface(X, Y, data, cmap='jet', edgecolor='none')
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Wave Amplitude")
    ax.set_zlim(-1,1)
    
    plt.savefig(output_path)
    plt.close()


def process_file(data_file):
    output_file = data_file.replace(".dat", ".png").replace("wave_data", "wave_images")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plot_data_3d(data_file, output_file)
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
