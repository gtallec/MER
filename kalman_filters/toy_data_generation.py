import numpy as np

def gen_toy_example(a_0, N, K):
    """generate N K-sized trajectories of the dynamical system defined as follow :
    
    a_k = sin(a_(k-1)) + v_k
    y_k = tanh(a_k) + w_k
    
    where w_k and v_k follows normal law
    
    
    Input : 
    a_0 the initial state of each trajectory, shape : (N, D), where D is the dimension of the state space
    
    Output :
    h_states is the N hidden state trajectories, shape : (N, K, D)
    obs is the N observation trajectories, shape : (N, K, D)
    
    Here we are dealing with scalars so that : D = 1
    """
    D = 1
    h_states = np.zeros((N,K,D))
    obs = np.zeros((N,K,D))
    
    
    h_states_noise = np.random.normal(loc=0, scale=1, size=(N,K-1,D))
    obs_noise = np.random.normal(loc=0, scale=1, size=(N,K,D))

    h_states[:,0,:] = a_0
    
    for k in range(1,K):
        h_states[:,k,:] = np.sin(h_states[:,k-1,:]) + h_states_noise[:,k-1,:]
        
    obs = np.tanh(h_states) + obs_noise
    
    return h_states, obs
