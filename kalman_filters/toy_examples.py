import numpy as np

def gen_nlin_toy_example(a_0, K):
    N = a_0.shape[0]
    a = np.zeros((N, K, 1))
    y = np.zeros((N, K, 1))
    
    a_noise = np.random.normal(loc=0, scale=1, size=(N,K-1,1))
    y_noise = np.random.normal(loc=0, scale=1, size=(N,K,1))
    
    a[:, 0, :] = a_0
    
    for k in range(1, K):
        a[:, k, :] = np.sin(a[:, k-1, :]) + a_noise[:, k-1, :]
    
    y = np.tanh(a) + y_noise
    
    return a,y


def gen_lin_toy_example(a_0, K, F, Q, G, R):
    N, D = a_0.shape
    H, D = G.shape
    
    a = np.zeros((N, K, D))
    y = np.zeros((N, K, H))
    
    a_noise = np.random.multivariate_normal(mean=np.zeros((D,)),
                                            cov=Q,
                                            size=(N,K-1))
    
    y_noise = np.random.multivariate_normal(mean=np.zeros((H,)),
                                            cov=R,
                                            size=(N,K))
    
    a[:, 0, :] = a_0
    
    for k in range(1,K):
        a[:, k, :] = a[:, k-1, :]@(F.T) + a_noise[:, k-1, :]
        
    y = np.einsum('hd, nkd -> nkh', G, a) + y_noise
    
    return a, y       
