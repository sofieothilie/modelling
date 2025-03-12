import numpy as np
import scipy.io
import modeling

mat_file = './zzz.mat' # Desktop
data = scipy.io.loadmat(mat_file)
modelThickness = 0.2
caprockDepth = data['zzz'] + modelThickness
[nx,ny] = np.shape(caprockDepth)

print(np.shape(caprockDepth))

res = (10,100,100)

modeling.launch_model(caprockDepth, res, 1e-9, 5000, 20, 0.2)