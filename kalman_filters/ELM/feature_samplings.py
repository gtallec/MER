import numpy as np

def normal_features(D, L, params):
    loc = params[0]
    scale = params[1]
    size = (L, D)
    W_h = np.random.normal(loc=loc, scale=scale, size=size)
    b_h = np.random.normal(loc=loc, scale=scale, size=(L,1))
    return W_h, b_h   

def test_features(D,L, params):
    W_h = np.zeros((L,D))
    np.fill_diagonal(W_h, val = 1)
    b_h = np.zeros((L,1))
    return W_h, b_h

if __name__ == '__main__':
    N = 100
    D = 10
    L = 8
    params=[0,1]
