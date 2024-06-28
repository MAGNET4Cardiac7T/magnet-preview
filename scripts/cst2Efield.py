
"""
Created on Fri May  8 00:02:02 2020
Computing E-filed  from the set of CST ASCII files 

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

def e_compute(fname):
    #Form 3D arrays of Ex,Ey,Ez fields and store in the single 4D matrix 
    print('Reading from ',fname)
    data=np.loadtxt(fname,skiprows=2)
    a_size=data.shape;
    print('Found array size:',a_size);
    if a_size[1]!=9:
        print('..data is inconsistent, 9 columns is expected')
        sys.exit()
    matrix=[0,0,0]
    fov=[0,0,0]
    a_lims=np.zeros([6,1])
    for i in range(3):
        matrix[i]=np.size(np.unique(data[:,i]))
    

    # Matrix size for E-field (3D array size +3)
    ematrix=matrix.copy()
    ematrix.append(3)
    
    # Allocate 4D matrix to store Ex,Ey,Ez
    E3d=np.zeros(ematrix,dtype='complex')
    
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
    
    # Extract Ex,Ey,Ez as complex 
    Ex=-data[:,3]-1j*data[:,6] 
    Ey=data[:,4]+1j*data[:,7] 
    Ez=data[:,5]+1j*data[:,8]
    
    # Reshape to 3D
    E3d[:,:,:,0]=np.reshape(Ex,matrix,order='F') # 
    E3d[:,:,:,1]=np.reshape(Ey,matrix,order='F') # 
    E3d[:,:,:,2]=np.reshape(Ez,matrix,order='F') # 
    E3d=E3d[::-1,:,:,:] # Flip up to down    
    return E3d,fov,a_lims


print('Converting CST E-field ASCII to 3D matrices and save  in MAT file')
filenames = getListOfFiles(sys.argv[1]) 
matfile=sys.argv[2]

nc=len(filenames) # Number of channels 
temp_e,fov,a_lims = e_compute(filenames[0]) # Get E-field  from the first file
xyz_iso=np.round(np.array(np.shape(temp_e))/2) # Isocenter of matrix 
xyz_iso=xyz_iso[0:3]

ematrix=np.shape(temp_e) # Size of extracted arrays ( 3D size x 3 components   )

#Allocate 5D array to store 3 components of E-field of  all channels  
Efield=np.zeros(ematrix+(nc,),dtype='complex')
# Additional arrays
Exyz=np.zeros(ematrix) # Array of combined E-field  
Eabs=np.zeros(ematrix[0:3]) # Array of E^2 fi

# Go through  files and get E-field in each channel 
Efield[:,:,:,:,0]=temp_e
for i in range(1,len(filenames)):
    temp_e,fov,a_lims=e_compute(filenames[i])
    Efield[:,:,:,:,i]=temp_e
    
Exyz=np.sum(Efield,4)  # combined fields Ex, Ey, Ez 
Eabs=np.abs(np.sum(Exyz*Exyz,3))  # Ex^2+Ey^2+Ez^2  
print('Writing to file : ',matfile)
# Save generated 4D arrays into MATLAB file 
sio.savemat(matfile,{'Nc':nc,'matrix':ematrix[0:3],'fov':fov,'xyz_iso':xyz_iso,'xyz_limits':a_lims,'E3d':Efield,'Eabs':Eabs,'files':filenames});
print('Done !')


 