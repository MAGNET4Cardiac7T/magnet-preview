# -*- coding: utf-8 -*-
"""
Created on Fri Oct 4 19:19:23 2020
Reading 4D complex B1-data from MATLAB file
@author: M.Terekhov
"""

import numpy as np
import mat73 as matio


def read_b1_from_mat(filename):
    print('Reading from MAT file : ',filename)
    mat_content=matio.loadmat(filename)
    keys_list=sorted(mat_content.keys())
    loaded_b1_data=mat_content[keys_list[-1]]
    return loaded_b1_data

def pTX3c(b1,C):
    s=b1.shape
    p=np.ones(s[0:3])
    C=np.reshape(C,(1,C.size))
    pm=np.reshape(p,(p.size,1))
    pmult=np.reshape(pm@C,(s[0],s[1],s[2],C.size))
    b1mult=np.multiply(b1,pmult)
    return np.sum(b1mult,3)

def pTX3a(b1,*argv):
    s=b1.shape
    if len(argv)>0:
        C=argv[0]
    else:
        C=np.ones([1,s[3]])
    if C.size!=s[3]:
        print('Size of array and vector does not match')
    p=np.ones(s[0:3])
    C=np.reshape(C,(1,C.size))
    pm=np.reshape(p,(p.size,1))
    pmult=np.reshape(pm@C,(s[0],s[1],s[2],C.size))
    b1mult=np.multiply(b1,pmult)
    return abs(np.sum(b1mult,3))





