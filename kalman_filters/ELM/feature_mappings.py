import numpy as np

def normal_features(D, L, params):
    loc = params[0]
    scale = params[1]
    size = (L, D)
    W_h = np.random.normal(loc=loc, scale=scale, size=size)
    b_h = np.random.normal(loc=loc, scale=scale, size=(1,L))
    return W_h, b_h

def sigmoid(X):
    return 1/(1 + np.exp(-X))

if __name__ == '__main__':
    N = 100
    D = 10
    L = 8
    params=[0,1]
    X = np.random.normal(loc=0, scale=1, size=(N,D))
    print(random_features(X=X, L=L, params=params).shape)
