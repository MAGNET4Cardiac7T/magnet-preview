# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:19:45 2020

@author: M.Terekhov

"""

import numpy as np
import scipy.io as sio
import sys
print('Converting CST SAR ASCII file  to MATLAB file')
filename = sys.argv[1]
print('Reading file : ',filename)
# Read file and skip 2 header lines 
a=np.loadtxt(filename,skiprows=2)
a_size=a.shape;
print('Found array size:',a_size);
if a_size[1]!=4:
    print('..data is inconsistent, 4 columns is expected')
    sys.exit()
matrix=[0,0,0]
for i in range(3):  # check size of array 
    matrix[i]=np.size(np.unique(a[:,i]))
    
sar1d=a[:,3] 
sar3d=np.reshape(sar1d,matrix,order='F')    
sar3d=sar3d[::-1,:,:]  
matfile=sys.argv[2]
print('Writing to file : ',matfile)
sio.savemat(matfile,{'sar3d':sar3d});
print('Done !')
