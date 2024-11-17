import numpy as np
import math
import multiprocessing
import time



def local_linear_estimator(x1, y1, x0, bandwidth): 
    """
    Perform a local linear regression to estimate the value at a new data point x0.
    
    Parameters:
    x1: Training data x.
    y1: Training data y.
    x0: New data point for prediction.
    bandwidth: Bandwidth parameter controlling the weighting decay with distance.
    
    Returns:The predicted value at x0 based on local linear regression.
    """
    x_dim = x1.shape[1] 
    dx_aug = np.hstack((np.ones((x1.shape[0], 1)), x1-x0)) 
    distances = np.linalg.norm(x1-x0, axis=1) 
    dh = distances**2/(bandwidth * x_dim)
    weights = np.where(dh<1,1-dh,0)
    W = np.diag(weights)
    prediction = np.linalg.solve(dx_aug.T @ W @ dx_aug + 0.00001*np.diag(np.ones(x_dim+1)) , dx_aug.T @ (weights*y1))[0]
    return prediction 
   

def local_linear_train(n_train, m_train, seed, datapath, bandwidth_set=np.geomspace(0.015, 1, 8)):
    """
    Train a local linear estimator using a training dataset, select the tuning parameter on validation data and evaluate on test data.
    
    Parameters:
    n_train: Number of training samples.
    m_train: Number of observations per training sample.
    seed: Seed for data loading, used to identify the data file.
    datapath: Path to the data file.
    bandwidth_set: Array of bandwidth values to test for the local linear estimator.
    
    Returns:
    tuple: A tuple containing:
        - valid_loss: Array with validation losses and corresponding bandwidths.
        - test_error_result: Mean squared error on the test data using the optimal bandwidth.
    """
    print(str(seed)+"start")
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

    print("data is realy")

    T1 = time.time()
    valid_loss = np.ndarray((bandwidth_set.shape[0],2))
    for ijj in range(bandwidth_set.shape[0]):
        bandwidth = list(bandwidth_set)[ijj]
        for i in range(x_valid.shape[0]):
            valid_loss_s = []
            x0 = x_valid[i,]
            res = local_linear_estimator(x1, y1, x0, bandwidth)
            valid_loss_s.append((res-y_valid[i])**2)
        valid_loss[ijj,0]=np.mean(valid_loss_s)
        valid_loss[ijj,1]=bandwidth

    test_error = [] 
    indopth = valid_loss[:,0].argmin()
    hopt = np.array(valid_loss)[indopth,1]
    print("optimal bandwidth:"+ str(hopt))
    for i in range(x_test.shape[0]):
        if(i%1000 == 0):
            print("test data:"+str(i))
        x0 = x_test[i,]
        res = local_linear_estimator(x1, y1, x0, hopt)
        test_error.append((res-y_test[i])**2)
    test_error_result = np.mean(test_error)

    T2 =time.time()
    print('time:%s' % ((T2 - T1)*1000))
    return valid_loss,test_error_result





n_vector = [100,300]
m_vector = [10,30]
ind_matrix = [[n_vector[i], m_vector[j]] for j in range(len(m_vector)) for i in range(len(n_vector))]


n_repeat = 50
if __name__ == '__main__': 
    for ij in range(len(ind_matrix)): 
        i = ind_matrix[ij][0] 
        j = ind_matrix[ij][1] 
        print(i,j)

        nproc = 50
        multiprocessing.set_start_method('forkserver', force=True) 
        n_list = [i for _ in range(n_repeat)] 
        m_list = [j for _ in range(n_repeat)] 
        datapath = ["./Simulation/Case4/20d/data/data"  for _ in range(n_repeat)] 
        seeds = list(range(n_repeat))
        params = zip(n_list, m_list, seeds, datapath) 
        with multiprocessing.Pool(processes = nproc ) as pool: 
            nnres = pool.starmap(local_linear_train, params) 
            nnres = np.array(nnres, dtype=object)
            nnres = np.stack(nnres, axis=0) 
        np.save("./Simulation/Case4/20d/res/locallinear"+str(i)+"m"+str(j)+".npy", nnres)
        print(i,j) 
        print("is ok") 
        

