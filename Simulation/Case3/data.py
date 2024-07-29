import numpy as np
import matplotlib.pyplot as plt
import csaps
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
import torch             # torch基础库
import torch.nn as nn    # torch神经网络库
import torch.nn.functional as F    # torch神经网络库 
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image 
import imageio 
import os, gc
import random 
import pynvml
import multiprocessing
import itertools
import subprocess
import gc
from functools import partial


def mean_onedim_fun1(x, a = 0.3, b = 1.6):
    ksize = 6
    i1, i2, i3, i4, i5 = np.meshgrid(np.arange(ksize), np.arange(ksize), np.arange(ksize), np.arange(ksize), np.arange(ksize), indexing='ij')
    im = np.max(np.array([i1,i2,i3,i4,i5]),axis=0)
    reswei = np.sum( np.power((a), im) * np.cos(np.multiply(np.power(b,i1),2*math.pi* x[0])) * np.cos(np.multiply(np.power(b,i2), math.pi*x[1])) * np.cos(np.multiply(np.power(b,i3), (2/3)*math.pi*x[2]))* np.cos(np.multiply(np.power(b,i4), 0.5*math.pi*x[3]))* np.cos(np.multiply(np.power(b,i5), (2/5)*math.pi*x[4])) )
    return reswei 




def generate_onedim_fun1(n,m): 

    x_dim = 5 
    x1 = np.random.uniform(low=0, high=1, size = n*m).reshape(n, m) 
    x2 = np.random.uniform(low=0, high=1, size = n*m).reshape(n, m)
    x = np.empty((n, m, x_dim))
    x[:,:,0] = np.cos(x1*2*math.pi)
    x[:,:,1] = np.sin(x1*2*math.pi)
    x[:,:,2] = x1
    x[:,:,3] = np.sin(x2*2*math.pi)
    x[:,:,4] = x2
    
    x = x.reshape(n, m, x_dim) 



    y=[[]] 
    for i in range(n): 
        yi = np.apply_along_axis(mean_onedim_fun1, 1, x[i]) 
        a = np.random.normal(0, 1, size = 20) * 0.1 
        a2 = np.random.normal(0, 1, size = 20) * 0.1 
        b = np.random.normal(0, 1, size = 20) * 0.1 
        b2 = np.random.normal(0, 1, size = 20) * 0.1 
        c = np.random.normal(0, 1, size = 20) * 0.1 
        c2 = np.random.normal(0, 1, size = 20) * 0.1 
        d = np.random.normal(0, 1, size = 20) * 0.1 
        d2 = np.random.normal(0, 1, size = 20) * 0.1 
        e = np.random.normal(0, 1, size = 20) * 0.1 
        e2 = np.random.normal(0, 1, size = 20) * 0.1 
        for k in range(20): 
            yi = yi + (math.sqrt(3)/math.sqrt(x_dim)) * (a[k] * np.sin((k+1) * math.pi * x[i,:,0]) 
                       + b[k] * np.sin((k+1) * math.pi * x[i,:,1]) 
                       + c[k] * np.sin((k+1) * math.pi * x[i,:,2]) 
                       + d[k] * np.sin((k+1) * math.pi * x[i,:,3]) 
                       + e[k] * np.sin((k+1) * math.pi * x[i,:,4]) 
                       + a2[k] * np.cos((k+1) * math.pi * x[i,:,0]) 
                       + b2[k] * np.cos((k+1) * math.pi * x[i,:,1]) 
                       + c2[k] * np.cos((k+1) * math.pi * x[i,:,2]) 
                       + d2[k] * np.cos((k+1) * math.pi * x[i,:,3]) 
                       + e2[k] * np.cos((k+1) * math.pi * x[i,:,4])) /(k+1)   #np.cos(x[i]) + c * 2*x[i]+ d * np.cos(2*x[i]) 
        y = np.append(y,yi) 
        print(i) 
    y = y.reshape((n,m)) 
    y = y + np.random.normal(loc=0,scale=1,size=(n,m)) * 0.01
    return x,y  




def savedata(seed):

    seed2 = ((seed+50) * 20000331)%2**31
    torch.manual_seed(seed2) 
    np.random.seed(seed2) 
    random.seed(seed2) 
    torch.cuda.manual_seed_all(seed2) 

    print(seed)
    print("start")
    n_train = 3000 
    m_train = 200 

    n_vaild = math.ceil(n_train*0.25) 
    m_valid = m_train 

    x_test = np.random.uniform(low=0, high=1, size=50000).reshape(-1,5)


    x_dim = 5 
    x1 = np.random.uniform(low=0, high=1, size = 10000)
    x2 = np.random.uniform(low=0, high=1, size = 10000)
    x_test = np.empty((10000, x_dim))
    x_test[:,0] = np.cos(x1*2*math.pi)
    x_test[:,1] = np.sin(x1*2*math.pi)
    x_test[:,2] = x1
    x_test[:,3] = np.sin(x2*2*math.pi)
    x_test[:,4] = x2

    y_test = np.apply_along_axis(mean_onedim_fun1, 1, x_test)
    print(seed)
    print("test")

    x,y = generate_onedim_fun1(n=n_train,m= m_train)  

    print(seed)
    print("train")

    x_valid,y_valid = generate_onedim_fun1(n=n_vaild,m=m_valid) 

    print(seed)
    print("valid")


    a = {"x":x,"y":y, "x_valid":x_valid, "y_valid":y_valid, "x_test":x_test, "y_test":y_test}
    np.save("./Simulation/Case3/data/data"+str(seed)+".npy", a)




seedlist = list(range(50)) # This is just a number, from which the real seed is generated

multiprocessing.set_start_method('forkserver', force=True)
if __name__ == '__main__':
    seeds = list(seedlist)
    nproc = 50 
    with multiprocessing.Pool(processes = nproc ) as pool: 
        pool.map(savedata, seedlist) 




