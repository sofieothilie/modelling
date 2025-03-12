import numpy as np
import scipy.io
import modeling

mat_file = './zzz.mat' # Desktop
data = scipy.io.loadmat(mat_file)
modelThickness = 0.2
caprockDepth = data['zzz']
[nx,ny] = np.shape(caprockDepth)

print(np.shape(caprockDepth))

res = (10,100,100)
dt = 1e-9
n_steps = 5000
snapshot_freq = 20
sensor_height = 0.2

modeling.launch_model(caprockDepth, res, dt, n_steps, snapshot_freq, sensor_height)