from transforms3d.euler import euler2mat, euler2quat
from transforms3d.quaternions import mat2quat
import numpy as np
import pdb

hip_joint = np.deg2rad([60, 90, 30])
rmat = euler2mat(*hip_joint)

quat1 = mat2quat(rmat)
quat2 = euler2quat(*hip_joint)

pdb.set_trace()

sixdrr = rmat[:,:2].flatten()



