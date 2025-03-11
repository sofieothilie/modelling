import numpy as np
import scipy.io
import modeling

mat_file = '/home/guillaume/epfl/ntnu/bachelor/CO2-modeling/zzz.mat' # Desktop
data = scipy.io.loadmat(mat_file)
modelThickness = 0.2
caprockDepth = data['zzz'] + modelThickness
[nx,ny] = np.shape(caprockDepth)

print(np.shape(caprockDepth))

res = (100,100,100)

modeling.launch_model(caprockDepth, res, 0.001, 4000, 40, 0.2)