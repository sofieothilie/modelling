#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor


import launch_sim

M = launch_sim.res[1]
N = launch_sim.res[0]

data_folder = "./data"

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

def process_file(data_file):
    output_file = data_file.replace(".dat", ".png").replace("data", "images")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plot_data(data_file, output_file)
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
