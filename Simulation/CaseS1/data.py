import numpy as np
import matplotlib.pyplot as plt
import csaps
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
import torch           
import torch.nn as nn   
import torch.nn.functional as F   
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image 
import imageio 
from early_stopping import EarlyStopping
import os, gc
import random 
import pynvml
import multiprocessing
import itertools
import subprocess
import gc
from functools import partial
from pandarallel import pandarallel




def mean_onedim_fun1(x):
    return ( x[0]*x[0]*x[1]*x[1]*x[1] + math.log(1+x[2]) + math.sqrt(1+x[3]*x[4]) + math.exp(x[4]/2))




def generate_onedim_fun1(n,m): 
    x_dim = 5 
    x = np.random.uniform(low=0, high=1, size = n*m*x_dim) 
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
            yi = yi +  (math.sqrt(3)/math.sqrt(x_dim)) * (a[k] * np.sin((k+1) * math.pi * x[i,:,0]) 
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
    n_train = 500
    m_train = 500 

    n_vaild = math.ceil(n_train*0.25) 
    m_valid = m_train 

    x_test = np.random.uniform(low=0, high=1, size=50000).reshape(-1,5)
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
    np.save("./20240425/hie/case2/data/data"+str(seed)+".npy", a)




seedlist = list(range(50)) # This is just a number, from which the real seed is generated


multiprocessing.set_start_method('forkserver', force=True)
if __name__ == '__main__':
    seeds = list(seedlist)
    nproc = 50 
    with multiprocessing.Pool(processes = nproc ) as pool: 
        pool.map(savedata, seedlist) 


