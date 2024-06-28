import scipy.io as sio
import matplotlib.pyplot as plt
import trimesh
import numpy as np


t = np.load("data/processed/Sphere_r_6cm.efield")



plt.imshow(t[:, :, 50, 0, 0].real)
plt.colorbar()
plt.show()