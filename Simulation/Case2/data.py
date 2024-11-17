import numpy as np
import math
import torch
import random 
import multiprocessing



def mean_fun1(x, a = 0.3, b = 1.6):
    """
    Calculate the true mean function in Case 2. 
    $$f(x) = \sum_{l=1}^5 \sum_{ k\geq 1} a^{k} \cos(2\pi l^{-1} b^{k_l} \xx_l)$$

    Parameters:
    x: observation points
    a,b: parameters in the mean function. Default: a = 0.3, b = 1.6
    
    Returns: f(x)
    """
    ksize = 20
    i1 = np.arange(ksize)
    reswei = np.sum( np.power((a), i1) * np.cos(np.multiply(np.power(b,i1),2*math.pi* x[0])) + np.power((a), i1) * np.cos(np.multiply(np.power(b,i1), math.pi*x[1])) + np.power((a), i1) * np.cos(np.multiply(np.power(b,i1), (2/3)*math.pi*x[2])) + np.power((a), i1) * np.cos(np.multiply(np.power(b,i1), (2/4)*math.pi*x[3]))+ np.power((a), i1) * np.cos(np.multiply(np.power(b,i1), (2/5)*math.pi*x[4])) )
    return reswei


def generate_fun1(n,m): 
    """
    Generate data for with added clusetered dependent random noise.
    
    Parameters:
    n (int): The number of samples to generate.
    m (int): The number of observations per sample.
    
    Returns:
    tuple: A tuple (x, y) where:
        - x is a numpy array containing the input data.
        - y is a numpy array containing the generated output data with noise.
    """
    x_dim = 5 
    x = np.random.uniform(low=0, high=1, size = n*m*x_dim) 
    x = x.reshape(n, m, x_dim) 
    y=[[]] 
    for i in range(n):
        yi = np.apply_along_axis(mean_fun1, 1, x[i])
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
    """
    Generate and save training, validation, and test data for a machine learning model.
    
    Parameters:
    seed (int): The seed for random number generation to ensure reproducibility.
    
    This function generates random data for training, validation, and testing using specified dimensions 
    and saves them to a `.npy` file. 
    """

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
    y_test = np.apply_along_axis(mean_fun1, 1, x_test)
    print(seed)
    print("test")

    x,y = generate_fun1(n=n_train,m= m_train)  

    print(seed)
    print("train")

    x_valid,y_valid = generate_fun1(n=n_vaild,m=m_valid) 

    print(seed)
    print("valid")


    a = {"x":x,"y":y, "x_valid":x_valid, "y_valid":y_valid, "x_test":x_test, "y_test":y_test}
    np.save("./Simulation/Case2/data/data"+str(seed)+".npy", a)



seedlist = list(range(50)) # This is just a number, from which the real seed is generated

multiprocessing.set_start_method('forkserver', force=True)
if __name__ == '__main__':
    seeds = list(seedlist)
    nproc = 50 
    with multiprocessing.Pool(processes = nproc ) as pool: 
        pool.map(savedata, seedlist) 


