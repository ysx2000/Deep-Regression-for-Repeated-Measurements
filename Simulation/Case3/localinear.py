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
import time



def local_linear_estimator(x1, y1, x0, bandwidth): # x,y are training data, x0 is the new data
    x_dim = x1.shape[1] 
    dx_aug = np.hstack((np.ones((x1.shape[0], 1)), x1-x0)) 
    distances = np.linalg.norm(x1-x0, axis=1) 
    dh = distances**2/(bandwidth * x_dim)
    # weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    weights = np.where(dh<1,1-dh,0)
    W = np.diag(weights)
    # XWX = dx_aug.T @ W @ dx_aug + 0.00001*np.diag(np.ones(x_dim+1))
    # prediction = np.linalg.inv(XWX)[0:1,]@ dx_aug.T @ W @ y1
    prediction = np.linalg.solve(dx_aug.T @ W @ dx_aug + 0.00001*np.diag(np.ones(x_dim+1)) , dx_aug.T @ (weights*y1))[0]
    return prediction 
   

def local_linear_train(n_train, m_train, seed, datapath, bandwidth_set=np.geomspace(10**(-8), 1, 20)):
    print(str(seed)+"kaishi")
    n_train = n_train
    m_train = m_train
    n_vaild = math.ceil(n_train*0.25)
    m_valid = m_train
    # ./720victory/holder/wei3d/data/data
    aa = np.load(datapath+str(seed)+".npy",allow_pickle=True).item()

    x_test = aa["x_test"]
    y_test = aa["y_test"]

    # x_test = x_test[1:1000]
    # y_test = y_test[1:1000]

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
    valid_loss = np.ndarray((bandwidth_set.shape[0],2))
    for ijj in range(bandwidth_set.shape[0]):
        bandwidth = list(bandwidth_set)[ijj]
    # for bandwidth in bandwidth_set:
        for i in range(x_valid.shape[0]):
            valid_loss_s = []
            x0 = x_valid[i,]
            res = local_linear_estimator(x1, y1, x0, bandwidth)
            valid_loss_s.append((res-y_valid[i])**2)
        # for i2 in range(100):
        #     for i3 in range(100):
        # valid_loss.append([np.mean(valid_loss),bandwidth]) 
        valid_loss[ijj,0]=np.mean(valid_loss_s)
        valid_loss[ijj,1]=bandwidth

    test_error = [] 
    indopth = valid_loss[:,0].argmin()
    hopt = np.array(valid_loss)[indopth,1]
    print("optimal bandwidth:"+ str(hopt))
    for i in range(x_test.shape[0]):
            # i = i2*100+i3
        if(i%1000 == 0):
            print("test data:"+str(i))
        x0 = x_test[i,]
        res = local_linear_estimator(x1, y1, x0, hopt)
        test_error.append((res-y_test[i])**2)
    test_error_result = np.mean(test_error)

    T2 =time.time()
    print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
    return valid_loss,test_error_result




n_vector = [100,300]
m_vector = [10,30]
ind_matrix = [[n_vector[i], m_vector[j]] for j in range(len(m_vector)) for i in range(len(n_vector))]



n_repeat = 50
if __name__ == '__main__': 
    for ij in range(len(ind_matrix)):  ##################################
        i = ind_matrix[ij][0] 
        j = ind_matrix[ij][1] 
        print(i,j)
        # datapath = "./240409/highdimension/jdp6/data/data" 
        # lambdac_set = np.geomspace(1e-6, 1, 15)

        nproc = 50
        multiprocessing.set_start_method('forkserver', force=True) 
        n_list = [i for _ in range(n_repeat)] 
        m_list = [j for _ in range(n_repeat)] 
        datapath = ["./Simulation/Case3/data/data"  for _ in range(n_repeat)] 
        seeds = list(range(n_repeat))
        params = zip(n_list, m_list, seeds, datapath) 
            # res = RKHS_train(i,j, seed=seed, datapath=datapath)
        with multiprocessing.Pool(processes = nproc ) as pool: 
            nnres = pool.starmap(local_linear_train, params) 
            nnres = np.stack(nnres, axis=0) 
        np.save("./Simulation/Case3/res/locallinear"+str(i)+"m"+str(j)+".npy", nnres)
        print(i,j) 
        print("is ok") 
        
