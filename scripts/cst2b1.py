
"""
Created on Fri May  8 00:02:02 2020
Computing B1+  and B1- from the set of CST ASCII files 

@author: M.Terekhov
"""

import numpy as np
import sys 
import os
import fnmatch
import scipy.io as sio

def getListOfFiles(dirName):
    # create a list of files to read 

    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        allFiles.append(fullPath)
   
    FilteredFiles=fnmatch.filter(allFiles,'*.txt')
                
    return FilteredFiles  

def b1_compute(fname):
    #Compute B1+ and B1- 3D arrays
    print('Reading from ',fname)
    data=np.loadtxt(fname,skiprows=2) # skip 2 first lines ( header)
    a_size=data.shape;
    print('Found array size:',a_size);
    if a_size[1]!=9:
        print('..data is inconsistent, 9 columns is expected')
        sys.exit()
    matrix=[0,0,0]
    fov=[0,0,0]
    a_lims=np.zeros([6,1])
    
    # find the size of 3D arrays by unique coordinates 
    for i in range(3):
        matrix[i]=np.size(np.unique(data[:,i]))
    
    # Get limits of X,Y,Z-axes 
    a_lims[0]=np.min(data[:,0])
    a_lims[1]=np.max(data[:,0])
    a_lims[2]=np.min(data[:,1])
    a_lims[3]=np.max(data[:,1])
    a_lims[4]=np.min(data[:,2])
    a_lims[5]=np.max(data[:,2])
    
    #Calculate FOV
    fov[0]=a_lims[1]-a_lims[0]
    fov[1]=a_lims[3]-a_lims[2]
    fov[2]=a_lims[5]-a_lims[4]

    
    # Calculate B1+ and B1+ from H-fields
    B1p=0.5*(-data[:,3]-1j*data[:,6]+1j*(data[:,4]+1j*data[:,7]) ); # B1 plus 1D
    B1m=np.conj(0.5*(-data[:,3]-1j*data[:,6]-1j*(data[:,4]+1j*data[:,7]) )); # B1 minus 1D
    B1p3d=np.reshape(B1p,matrix,order='F') # Reshape to 3D
    B1p3d=B1p3d[::-1,:,:] # Flip   X-direction 
    B1m3d=np.reshape(B1m,matrix,order='F')  # Reshape to 3D
    B1m3d=B1m3d[::-1,:,:] # Flip X-direction     
    return B1p3d,B1m3d,fov,a_lims


print('Converting CST H-field ASCII to B1+ and B1- and save in MAT file')
filenames = getListOfFiles(sys.argv[1])
matfile=sys.argv[2]
nc=len(filenames)
# Compute first arrays to check size of arrays to be produced  
temp_b1p,temp_b1m,fov,alims = b1_compute(filenames[0])
xyz_iso=np.round(np.array(np.shape(temp_b1p))/2)
matsize=np.shape(temp_b1p)
#Allocate 4D arrays to store data 
B1p4d=np.zeros(temp_b1p.shape+(nc,),dtype='complex')
B1m4d=np.zeros(temp_b1p.shape+(nc,),dtype='complex')
B1p4d[:,:,:,0]=temp_b1p
B1m4d[:,:,:,0]=temp_b1m
# Go through files 
for i in range(1,len(filenames)):
    temp_b1p,temp_b1m,fov,alims=b1_compute(filenames[i])
    B1p4d[:,:,:,i]=temp_b1p
    B1m4d[:,:,:,i]=temp_b1m
print('Writing to file : ',matfile)
# Save generated 4D arrays into MATLAB file 
sio.savemat(matfile,{'Nc':nc,'matrix':matsize[0:3],'fov':fov,'xyz_limits':alims,'xyz_iso':xyz_iso[0:3],'B1p3d':B1p4d,'B1m3d':B1m4d,'files':filenames});
print('Done !')


 