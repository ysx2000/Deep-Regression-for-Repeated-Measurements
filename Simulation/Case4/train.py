import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" ### Determine the GPU to be used. without commenting out this line of code, only nocuda = 9 can be selected later.
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
import time

def get_gpu_memory(device_id):
    try:
        output = subprocess.check_output(["nvidia-smi", "--id={}".format(device_id), "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"])
        memory_used, memory_total = map(int, output.decode("utf-8").strip().split("\n")[0].split(","))
        return memory_used, memory_total
    except Exception as e:
        print(e)
        return None, None

def get_free_gpu():
    device_ids = list(range(torch.cuda.device_count()))
    memory_usages = []
    for device_id in device_ids:
        memory_used, memory_total = get_gpu_memory(device_id)
        if memory_used is not None and memory_total is not None:
            memory_free = memory_total - memory_used
            memory_usages.append((device_id, memory_free))
        print(memory_total,memory_usages)
    if len(memory_usages) > 0:
        best_device_id = sorted(memory_usages, key=lambda x: x[1])[len(device_ids)-1][0]
        # print(sorted(memory_usages, key=lambda x: x[1]))
        device = torch.device(f"cuda:{best_device_id}")
        return device
    else:
        return None


class Args:
    def __init__(self, batch_size=10, lr =0.001, nepoch = 200, patience = 10, wide = 100, depth = 5, n_train=1, m_train=1) -> None:
        self.batch_size = batch_size
        self.lr = lr
        self.nepoch = nepoch 
        self.patience = patience 
        self.wide = wide 
        self.depth = depth 
        self.biaoji = "wide" + str(wide) + "depth" + str(depth) + "n" + str(n_train) + "m" + str(m_train)
        self.n_train = n_train
        self.m_train = m_train


class EarlyStopping():
    def __init__(self, save_path, args, verbose=False, delta=0):
        self.save_path = save_path 
        self.patience = args.patience 
        self.verbose = verbose 
        self.counter = 0 
        self.best_score = None 
        self.early_stop = False 
        self.val_loss_min = np.Inf 
        self.delta = delta 

    def __call__(self, model, train_loss, valid_loss, test_error, args, seed):

        score = -valid_loss 

        if self.best_score is None: 
            self.best_score = score 
            self.save_checkpoint(model, train_loss, valid_loss, test_error, args, seed) 
        elif score < self.best_score + self.delta: 
            self.counter += 1 
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}') 
            if self.counter >= self.patience: 
                self.early_stop = True 
        else:
            self.best_score = score
            self.save_checkpoint(model, train_loss, valid_loss, test_error, args, seed)
            self.counter = 0

    def save_checkpoint(self, model, train_loss, valid_loss, test_error, args, seed):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ...') 
        # path = os.path.join(self.save_path, 'best'+ args.biaoji +'network.pth') 
        torch.save(model.state_dict(), os.path.join(self.save_path, 'best' + str(seed) + args.biaoji +'network.pth') )
        torch.save(train_loss, os.path.join(self.save_path, 'best'+ str(seed) + args.biaoji +'train_loss.pth')) 
        torch.save(valid_loss, os.path.join(self.save_path, 'best'+ str(seed) + args.biaoji +'valid_loss.pth')) 
        torch.save(test_error, os.path.join(self.save_path, 'best'+ str(seed) + args.biaoji +'test_loss.pth')) 

        self.val_loss_min = valid_loss
    


class Dataset_repeatedmeasurement(Dataset): 
    def __init__(self, x, y) -> None:  
        super().__init__()
        self.x = x 
        self.y = y 


    def __len__(self) -> int: 
        return len(self.x) 
    
    def __getitem__(self, index): 
        return {
            "x" : self.x[index], 
            "y" : self.y[index]
        }




class happynet(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, n_layer): 
        super().__init__()
        if n_layer == 3: 
            self.net = nn.Sequential(
                nn.Linear(n_feature, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_output), 
            )
        elif n_layer == 2: 
            self.net = nn.Sequential(
                nn.Linear(n_feature, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_output), 
            )    
        elif n_layer == 4: 
            self.net = nn.Sequential(
                nn.Linear(n_feature, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_output), 
            )
        elif n_layer == 5: 
            self.net = nn.Sequential(
                nn.Linear(n_feature, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),  
                nn.ReLU(),
                nn.Linear(n_hidden, n_output),
            )
        elif n_layer == 6: 
            self.net = nn.Sequential(
                nn.Linear(n_feature, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),  
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_output),
            )
        elif n_layer == 7: 
            self.net = nn.Sequential(
                nn.Linear(n_feature, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),  
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_output),
            )
        elif n_layer == 8: 
            self.net = nn.Sequential(
                nn.Linear(n_feature, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),  
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_output),
            )
        elif n_layer == 9: 
            self.net = nn.Sequential(
                nn.Linear(n_feature, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),  
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_output),
            )
        elif n_layer == 10: 
            self.net = nn.Sequential(
                nn.Linear(n_feature, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),  
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_output),
            )
        else: 
            print("Error! the depth is not in 3-10")
    
    def forward(self, x):
        k = self.net(x)
        return k





def GPUstrain(x, y, x_valid, y_valid, x_test, y_test,args,seed,nocuda): # batch_size, nepoch=200): 

    x_dim = x.shape[2]

    if nocuda == 0:
        device = torch.device("cuda:0")
    if nocuda == 1:
        device = torch.device("cuda:1")
    if nocuda == 100:
        device = get_free_gpu()
    if nocuda == -1:
        device = torch.device("cpu")
    if nocuda == 9:
        device = torch.device("cuda")


    net = happynet(n_feature=x_dim, n_hidden=args.wide, n_output=1, n_layer=args.depth).to(device)
    nepoch = args.nepoch
    
    optimizer=torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.90, 0.999), eps=1e-8, weight_decay=0., amsgrad=False,) 
    loss_func=nn.MSELoss() 
    train_epochs_loss = [] 
    valid_epochs_loss = [] 
    test_epochs_error = [] 

    x = x.reshape(-1,x_dim)
    y = y.reshape(-1)

    train_dataset = Dataset_repeatedmeasurement(x,y)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    x=torch.from_numpy(x).float() 
    y=torch.from_numpy(y).float() 
    x_valid=torch.from_numpy(x_valid).float().to(device) 
    y_valid=torch.from_numpy(y_valid).float().to(device) 

    x_test = torch.from_numpy(x_test).float().view(-1,1,x_dim).to(device)
    y_test = torch.from_numpy(y_test).float().view(-1).to(device) 


    save_path = "./Simulation/resultsv" 
    early_stopping = EarlyStopping(save_path,args=args)

    for epoch in range(nepoch): 
        net.train()
        train_epoch_loss = []



        # =========================train=========================
        for idx, traindata in enumerate(train_dataloader):

            x_train = traindata["x"]
            y_train = traindata["y"]

            x_train=torch.Tensor(x_train).float().view(-1,1,x_dim).to(device) 
            y_train=torch.Tensor(y_train).float().to(device) 
            outputs=net(x_train) 
            loss=loss_func(outputs.view(-1),y_train.view(-1).float())
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            
            
            train_epoch_loss.append(loss.item())

        del outputs, loss

        train_epochs_loss.append(np.average(train_epoch_loss))
        # =========================valid=========================
        with torch.no_grad():
            net.eval() 
            valid_predict=net(x_valid.view(-1,1,x_dim))
            valid_y_pre=valid_predict.view(-1).detach()
            valid_y_pre=torch.Tensor(valid_y_pre).float()
            loss_valid=loss_func(valid_y_pre, y_valid.view(-1).float())
            valid_epochs_loss.append(loss_valid.item())

            test_predict=net(x_test)
            test_y_pre=test_predict.view(-1).detach()
            test_y_pre=torch.Tensor(test_y_pre).float()
            error_test=loss_func(test_y_pre, y_test)
            test_epochs_error.append(error_test.item())

        print("epoch = {}, training loss = {}, validation loss = {}, test error = {}".format(epoch, np.average(train_epoch_loss), loss_valid, error_test))

        if epoch > 10 or args.n_train*args.m_train > 200:
            early_stopping(net, np.average(train_epoch_loss), loss_valid, error_test,args,seed)
            if early_stopping.early_stop: 
                print("Early stopping")
                break #跳出迭代，结束训练


        del valid_predict, valid_y_pre, loss_valid
        del test_predict, test_y_pre, error_test
    gc.collect()

    return net, train_epochs_loss, valid_epochs_loss, test_epochs_error



    

def onedim(n_train, m_train, seed, datapath, nocuda):

    seed2 = ((seed+50) * 20000331 )% 2**31
    torch.manual_seed(seed2) 
    np.random.seed(seed2) 
    random.seed(seed2) 
    torch.cuda.manual_seed_all(seed2) 

    n_train = n_train
    m_train = m_train
    n_vaild = math.ceil(n_train*0.25)
    m_valid = m_train
    # ./720victory/holder/wei3d/data/data
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

    print("data is realy")

    x_dim = x.shape[2]
    if n_train*m_train < 128:
       batch_size= min(n_train*m_train, 32)
       lr = 0.0005
    elif n_train*m_train < 1024:
        batch_size= 64
        lr = 0.0005
    elif n_train*m_train < 4096:
        batch_size= 128
        lr = 0.001
    elif n_train*m_train < 8192:
        batch_size= 256
        lr = 0.001
    elif n_train*m_train < 16384:
        batch_size= 512
        lr = 0.002
    else:
        batch_size= 1024
        lr = 0.002

    # mean_onedim_fun1_vec = np.vectorize(mean_onedim_fun1)
    w = math.ceil(math.pow(n_train*m_train, 0.214) * 5)

    args = Args(lr=lr, wide=50, depth = 2, batch_size= batch_size, n_train=n_train, m_train=m_train)
    GPUstrain(x=x,y=y,x_valid = x_valid,y_valid=y_valid,x_test=x_test, y_test=y_test, args=args,seed = seed2, nocuda = nocuda)

    a = torch.load('./Simulation/resultsv/best'+ str(seed2) + args.biaoji +'valid_loss.pth')
    b = torch.load('./Simulation/resultsv/best'+ str(seed2) + args.biaoji +'test_loss.pth')
    a = np.expand_dims(a.cpu(), 0)
    b = np.expand_dims(b.cpu(), 0)
    c0 = np.r_[a,b,0]

    args = Args(lr=lr, wide=100, depth = 3, batch_size= batch_size, n_train=n_train, m_train=m_train)
    GPUstrain(x=x,y=y,x_valid = x_valid,y_valid=y_valid,x_test=x_test, y_test=y_test, args=args,seed = seed2, nocuda = nocuda)
    a = torch.load('./Simulation/resultsv/best'+ str(seed2) + args.biaoji +'valid_loss.pth')
    b = torch.load('./Simulation/resultsv/best'+ str(seed2) + args.biaoji +'test_loss.pth')
    a = np.expand_dims(a.cpu(), 0)
    b = np.expand_dims(b.cpu(), 0)
    c1 = np.r_[a,b,1]


    args = Args(lr=lr, wide=200, depth = 4, batch_size= batch_size, n_train=n_train, m_train=m_train)
    GPUstrain(x=x,y=y,x_valid = x_valid,y_valid=y_valid,x_test=x_test, y_test=y_test, args=args,seed = seed2, nocuda = nocuda)
    a = torch.load('./Simulation/resultsv/best'+ str(seed2) + args.biaoji +'valid_loss.pth')
    b = torch.load('./Simulation/resultsv/best'+ str(seed2) + args.biaoji +'test_loss.pth')
    a = np.expand_dims(a.cpu(), 0)
    b = np.expand_dims(b.cpu(), 0)
    c2 = np.r_[a,b,2]

    args = Args(lr=lr, wide=400, depth = 5, batch_size= batch_size, n_train=n_train, m_train=m_train)
    GPUstrain(x=x,y=y,x_valid = x_valid,y_valid=y_valid,x_test=x_test, y_test=y_test, args=args,seed = seed2, nocuda = nocuda)
    a = torch.load('./Simulation/resultsv/best'+ str(seed2) + args.biaoji +'valid_loss.pth')
    b = torch.load('./Simulation/resultsv/best'+ str(seed2) + args.biaoji +'test_loss.pth')
    a = np.expand_dims(a.cpu(), 0)
    b = np.expand_dims(b.cpu(), 0)
    c3 = np.r_[a,b,3]

    args = Args(lr=lr, wide=600, depth = 6, batch_size= batch_size, n_train=n_train, m_train=m_train)
    GPUstrain(x=x,y=y,x_valid = x_valid,y_valid=y_valid,x_test=x_test, y_test=y_test, args=args,seed = seed2, nocuda = nocuda)
    a = torch.load('./Simulation/resultsv/best'+ str(seed2) + args.biaoji +'valid_loss.pth')
    b = torch.load('./Simulation/resultsv/best'+ str(seed2) + args.biaoji +'test_loss.pth')
    a = np.expand_dims(a.cpu(), 0)
    b = np.expand_dims(b.cpu(), 0)
    c4 = np.r_[a,b,4]
    
    args = Args(lr=lr, wide=800, depth = 6, batch_size= batch_size, n_train=n_train, m_train=m_train)
    GPUstrain(x=x,y=y,x_valid = x_valid,y_valid=y_valid,x_test=x_test, y_test=y_test, args=args,seed = seed2, nocuda = nocuda)
    a = torch.load('./Simulation/resultsv/best'+ str(seed2) + args.biaoji +'valid_loss.pth')
    b = torch.load('./Simulation/resultsv/best'+ str(seed2) + args.biaoji +'test_loss.pth')
    a = np.expand_dims(a.cpu(), 0)
    b = np.expand_dims(b.cpu(), 0)
    c5 = np.r_[a,b,5]

    p = np.r_[np.expand_dims(c0, 0),np.expand_dims(c1, 0),np.expand_dims(c2, 0),np.expand_dims(c3, 0),np.expand_dims(c4, 0),np.expand_dims(c5, 0)]#

    return np.concatenate((p[0],p[1],p[2],p[3],p[4],p[5]))




n_vector = [50,100,200,300,400] 
m_vector = [1,2,3,5,8,10,12,15,20,25,30,40,50,60,80] 
ind_matrix = [[n_vector[i], m_vector[j]] for i in range(len(n_vector)) for j in range(len(m_vector))]


n_repeat = 50
if __name__ == '__main__': 
    for ij in range(len(ind_matrix)): 
        i = ind_matrix[ij][0] 
        j = ind_matrix[ij][1] 
        nocuda = 9 ## comment code " os.environ["CUDA_VISIBLE_DEVICES"] = "0" ", then you can let nocuda = 0 or 1
        nproc = 5
        multiprocessing.set_start_method('forkserver', force=True) 

        n_list = [i for _ in range(n_repeat)] 
        m_list = [j for _ in range(n_repeat)] 
        datapath = ["./Simulation/Case4/data/data" for _ in range(n_repeat)] 
        nocuda = [nocuda for _ in range(n_repeat)] 
        seeds = list(range(n_repeat))
        params = zip(n_list, m_list, seeds, datapath, nocuda) 
        with multiprocessing.Pool(processes = nproc ) as pool: 
            nnres = pool.starmap(onedim, params) 
            nnres = np.stack(nnres, axis=0) 
        np.save("./Simulation/Case4/res/res"+str(i)+"m"+str(j)+".npy", nnres) 
        print(i,j) 
        print("is ok") 