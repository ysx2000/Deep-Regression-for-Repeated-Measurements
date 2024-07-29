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
import os, gc
import random 
import pynvml
import multiprocessing
import itertools
import subprocess
import gc
from functools import partial
import time


def kernel(x,y): 
    d = np.sqrt(np.sum((x-y)**2)) 
    return np.exp(-d) 
bankernel0 = 1 



def RKHS_train(n_train, m_train, seed, datapath, lambdac_set = np.geomspace(1e-20, 1e-2, 50)):
    print(str(seed)+"kaishi")
    n_train = n_train
    m_train = m_train
    n_vaild = math.ceil(n_train*0.25)
    m_valid = m_train
    aa = np.load(datapath+str(seed)+".npy",allow_pickle=True).item()

    x_test = aa["x_test"]
    y_test = aa["y_test"]

    x = aa["x"]
    y = aa["y"]
    x = x[:n_train,:m_train]
    y = y[:n_train,:m_train]

    x_valid = aa["x_valid"]
    y_valid = aa["y_valid"]
    x_valid = x_valid[:n_vaild,:m_valid]
    y_valid = y_valid[:n_vaild,:m_valid]
    
    x_dim = x.shape[2]
    x1 = x.reshape(-1,x_dim)
    y1 = y.reshape(-1)
    x_valid = x_valid.reshape(-1,x_dim)
    y_valid = y_valid.reshape(-1)
    # print(x_test.shape[0])

    print("data is realy")

    T1 = time.time()
    valid_loss = np.ndarray((lambdac_set.shape[0],2))
    Phi = np.ndarray((x1.shape[0],x1.shape[0]))
    for i in range(x1.shape[0]):
        for j in range(x1.shape[0]):
            if i<j:
                Phi[i,j] = kernel(x1[i,],x1[j,]) 
            if i==j:
                Phi[i,j] = bankernel0/2
            if i>j:
                Phi[i,j] = 0
    Phi = Phi + Phi.T

    Phi_valid = np.ndarray((x1.shape[0],x_valid.shape[0]))
    for i in range(x1.shape[0]):
        for j in range(x_valid.shape[0]):
            Phi_valid[i,j] = kernel(x1[i,],x_valid[j,])

    Phi_test = np.ndarray((x1.shape[0],x_test.shape[0]))
    for i in range(x1.shape[0]):
        for j in range(x_test.shape[0]):
            Phi_test[i,j] = kernel(x1[i,],x_test[j,])


    valid_loss = np.ndarray((lambdac_set.shape[0],2))

    for ijj in range(lambdac_set.shape[0]):
        lambdac = lambdac_set[ijj]
        print(lambdac)

        coef = np.linalg.solve(Phi+x1.shape[0]*lambdac*np.diag(np.ones(x1.shape[0])), y1)
        valid_loss[ijj,0] = np.mean(((coef@Phi_valid) - y_valid)**2)
        valid_loss[ijj,1] = lambdac

    test_error = [] 
    indopth = valid_loss[:,0].argmin()
    lambdaopt = np.array(valid_loss)[indopth,1]
    print("optimal lambda:"+ str(lambdaopt))
    coef = np.linalg.solve(Phi+x1.shape[0]*lambdaopt*np.diag(np.ones(x1.shape[0])), y1)
    test_error_result = np.mean(((coef@Phi_test) - y_test)**2)

    T2 =time.time()
    print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
    return valid_loss,test_error_result




n_vector = [100,300]
m_vector = [10,30]
ind_matrix = [[n_vector[i], m_vector[j]] for i in range(len(n_vector)) for j in range(len(m_vector))]


n_repeat = 50
if __name__ == '__main__': 
    for ij in range(len(ind_matrix)-1):  ##################################
        i = ind_matrix[ij][0] 
        j = ind_matrix[ij][1] 
        print(i,j) 

        nproc = 50
        multiprocessing.set_start_method('forkserver', force=True) 
        n_list = [i for _ in range(n_repeat)] 
        m_list = [j for _ in range(n_repeat)] 
        datapath = ["./Simulation/Case1/data/data"  for _ in range(n_repeat)] 
        seeds = list(range(n_repeat))
        params = zip(n_list, m_list, seeds, datapath) 
        with multiprocessing.Pool(processes = nproc ) as pool: 
            nnres = pool.starmap(RKHS_train, params) 
            nnres = np.stack(nnres, axis=0) 
        np.save("./Simulation/Case1/res/rkhsreg"+str(i)+"m"+str(j)+".npy", nnres)
        print(i,j) 
        print("is ok") 
        
