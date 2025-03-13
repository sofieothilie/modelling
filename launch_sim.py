import numpy as np
import scipy.io
import modeling
from scipy.io import loadmat

res = (10,20,10)
dt = 1e-9
n_steps = 1000
snapshot_freq = 20
sensor_height = 0.2

def main():
    mat_file = './zzz.mat' # Desktop
    data = scipy.io.loadmat(mat_file)
    modelThickness = 0.2
    caprockDepth = data['zzz']
    [nx,ny] = np.shape(caprockDepth)

    print(np.shape(caprockDepth))



    signature_mat = loadmat('filterOff_transducer_to_transducer.mat')
    waveform = signature_mat['data1'].squeeze()  # Replace with actual variable name


    signature = waveform[300:500].astype(np.float64)
    signature /= np.max(np.abs(signature)) 

    fs = int(8e6)#sampling frequency

    modeling.launch_model(caprockDepth, signature, res, dt, n_steps, snapshot_freq, sensor_height, fs)

if __name__ == "__main__":
    main()
