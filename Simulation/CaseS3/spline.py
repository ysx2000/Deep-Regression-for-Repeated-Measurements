import numpy as np
import math
import multiprocessing
import time
from scipy.interpolate import BSpline


def concat_phi(phi1,phi2,phi3,phi4,phi5):
    """
    Concatenate and multiply elements of five input matrices to create a higher-dimensional matrix.
    It is used to caculate the tensor spline.
    """
    size1,size2 = phi1.shape
    ret = np.ndarray((size1,size2**5))
    for i in range(size1):
        for j1 in range(size2):
            for j2 in range(size2):
                for j3 in range(size2):
                    for j4 in range(size2):
                        for j5 in range(size2):
                            ret[i,j1*size2**4+j2*size2**3+j3*size2**2+j4*size2+j5] = phi1[i,j1]*phi2[i,j2]*phi3[i,j3]*phi4[i,j4]*phi5[i,j5]
    return ret

def regression_spline_train(n_train, m_train, seed, datapath, splinenumber = np.arange(2,7)):
    """
    Train a regression spline model using B-splines, optimizing over the number of splines 
    to fit the data and selecting the best model through validation.
    
    Parameters:
    n_train: Sample size.
    m_train: Sampling frequency.
    seed: Seed for data loading, used to identify the data file.
    datapath: Path to the data file.
    splinenumber: Array of possible numbers of splines to test.
    
    Returns:
    tuple: A tuple containing:
        - valid_loss: Array with validation losses and corresponding spline numbers.
        - test_error_result: Mean squared error on the test data using the optimal spline number.
    """
    print(str(seed)+"kaishi")
    o = 3
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

    valid_loss = np.ndarray((splinenumber.shape[0],2))

    T1 = time.time()

    for kkindex in range(splinenumber.shape[0]):
        k = splinenumber[kkindex]
        print("splinenumber:"+str(k))
        T3 = time.time()
        t = np.linspace(0,1,k,endpoint=True)
        t = np.r_[[0]*o,t,[1]*o]
        tp = np.linspace(-1,1,k,endpoint=True)
        tp = np.r_[[-1]*o,tp,[1]*o]
        Phi1 = BSpline.design_matrix(x1[:,0].reshape(-1),tp,o).toarray()
        Phi2 = BSpline.design_matrix(x1[:,1].reshape(-1),tp,o).toarray()
        Phi3 = BSpline.design_matrix(x1[:,2].reshape(-1),t,o).toarray()
        Phi4 = BSpline.design_matrix(x1[:,3].reshape(-1),tp,o).toarray()
        Phi5 = BSpline.design_matrix(x1[:,4].reshape(-1),t,o).toarray()

        Mat = concat_phi(Phi1,Phi2,Phi3,Phi4,Phi5)

        ########
        indexused = np.where((Mat != 0).any(axis=0))[0]
        Mat_working = Mat[:, indexused]

        try:
            coef = np.linalg.solve(np.matmul(np.transpose(Mat_working),Mat_working) + np.eye(indexused.shape[0])*(10**(-6)),np.matmul(np.transpose(Mat_working),y1))

            ## validation
            Phi1_validation = BSpline.design_matrix(x_valid[:,0].reshape(-1),tp,o).toarray()
            Phi2_validation = BSpline.design_matrix(x_valid[:,1].reshape(-1),tp,o).toarray()
            Phi3_validation = BSpline.design_matrix(x_valid[:,2].reshape(-1),t,o).toarray()
            Phi4_validation = BSpline.design_matrix(x_valid[:,3].reshape(-1),tp,o).toarray()
            Phi5_validation = BSpline.design_matrix(x_valid[:,4].reshape(-1),t,o).toarray()

            Mat_validation = concat_phi(Phi1_validation,Phi2_validation,Phi3_validation,Phi4_validation,Phi5_validation)
            Mat_validation_working = Mat_validation[:, indexused]

            Y_fit = np.matmul(Mat_validation_working, coef)
            valid_loss[kkindex,0] = np.mean(np.square(y_valid-Y_fit))
            valid_loss[kkindex,1] = k 

        except: 
            pass
        T4 = time.time()
        print('time: %s ms' % ((T4 - T3)*1000))
        ## test
    indopth = valid_loss[:,0].argmin()
    k = round(np.array(valid_loss)[indopth,1])
    t = np.linspace(0,1,k,endpoint=True)
    t = np.r_[[0]*o,t,[1]*o]
    tp = np.linspace(-1,1,k,endpoint=True)
    tp = np.r_[[-1]*o,tp,[1]*o]
    Phi1_test = BSpline.design_matrix(x_test[:,0].reshape(-1),tp,o).toarray()
    Phi2_test = BSpline.design_matrix(x_test[:,1].reshape(-1),tp,o).toarray()
    Phi3_test = BSpline.design_matrix(x_test[:,2].reshape(-1),t,o).toarray()
    Phi4_test = BSpline.design_matrix(x_test[:,3].reshape(-1),tp,o).toarray()
    Phi5_test = BSpline.design_matrix(x_test[:,4].reshape(-1),t,o).toarray() 
    # Phi n*k
    Phi1 = BSpline.design_matrix(x1[:,0].reshape(-1),tp,o).toarray()
    Phi2 = BSpline.design_matrix(x1[:,1].reshape(-1),tp,o).toarray()
    Phi3 = BSpline.design_matrix(x1[:,2].reshape(-1),t,o).toarray()
    Phi4 = BSpline.design_matrix(x1[:,3].reshape(-1),tp,o).toarray()
    Phi5 = BSpline.design_matrix(x1[:,4].reshape(-1),t,o).toarray()

    Mat = concat_phi(Phi1,Phi2,Phi3,Phi4,Phi5)
    indexused = np.where((Mat != 0).any(axis=0))[0]
    Mat_working = Mat[:, indexused]
    

    coef = np.linalg.solve(np.matmul(np.transpose(Mat_working),Mat_working) + np.eye(indexused.shape[0])*(10**(-6)),np.matmul(np.transpose(Mat_working),y1))

    Mat_test = concat_phi(Phi1_test,Phi2_test,Phi3_test,Phi4_test,Phi5_test)
    Mat_test_working = Mat_test[:, indexused]
    Y_fit = np.matmul(Mat_test_working,coef)
    test_error_result = np.mean(np.square(y_test-Y_fit))

    T2 =time.time()
    print("splinenumber:"+str(k)+'time: %s ms' % ((T2 - T1)*1000))
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

        nproc = 10
        multiprocessing.set_start_method('forkserver', force=True) 
        n_list = [i for _ in range(n_repeat)] 
        m_list = [j for _ in range(n_repeat)] 
        datapath = ["./Simulation/CaseS3/data/data"  for _ in range(n_repeat)] 
        seeds = list(range(n_repeat))
        params = zip(n_list, m_list, seeds, datapath) 
        with multiprocessing.Pool(processes = nproc ) as pool: 
            nnres = pool.starmap(regression_spline_train, params) 
            nnres = np.array(nnres, dtype=object)
            nnres = np.stack(nnres, axis=0) 
        np.save("./Simulation/CaseS3/res/regsp"+str(i)+"m"+str(j)+".npy", nnres)
        print(i,j) 
        print("is ok") 