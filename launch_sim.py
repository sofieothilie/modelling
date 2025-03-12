import numpy as np
import scipy.io
import modeling
from scipy.io import loadmat

mat_file = './zzz.mat' # Desktop
data = scipy.io.loadmat(mat_file)
modelThickness = 0.2
caprockDepth = data['zzz']
[nx,ny] = np.shape(caprockDepth)

print(np.shape(caprockDepth))

res = (10,93,17)
dt = 1e-9
n_steps = 1000
snapshot_freq = 20
sensor_height = 0.2

signature_mat = loadmat('filterOff_transducer_to_transducer.mat')
waveform = signature_mat['data1'].squeeze()  # Replace with actual variable name


signature = waveform[300:500].astype(np.float64)
signature /= np.max(np.abs(signature)) 

fs = int(8e6)#sampling frequency

modeling.launch_model(caprockDepth, signature, res, dt, n_steps, snapshot_freq, sensor_height, fs)