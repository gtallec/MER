import ELM
import numpy as np

class DeepKalmanFilter:

    def __init__(self, params_path, STS_transition_net, STO_transition_net, verbose):

        self.params_path = params_path
        self.verbose = verbose

        self.R = None
        self.Q = None

        STS_feature_sampling = STS_transition_net[0]
        STS_sampling_dim = STS_transition_net[1]
        STS_sampling_params = STS_transition_net[2]
        STS_activation = STS_transition_net[3]
        STS_d_activation = STS_transition_net[4]

        if verbose:
            print('--INITIAL STS PARAMETERS--')
            print('STS_feature_sampling : ', STS_feature_sampling)
            print('STS_sampling_dim : ', STS_sampling_dim)
            print('STS_sampling_params : ', STS_sampling_params)
            print('STS_activation : ', STS_activation)
            print('STS_d_activation : ', STS_d_activation, '\n')

        self.F_net = ELM.ELMNet(feature_sampling = STS_feature_sampling,
                                sampling_dim = STS_sampling_dim,
                                sampling_params = STS_sampling_params,
                                activation = STS_activation,
                                d_activation = STS_d_activation,
                                verbose = False)

        STO_feature_sampling = STO_transition_net[0]
        STO_sampling_dim = STO_transition_net[1]
        STO_sampling_params = STO_transition_net[2]
        STO_activation = STO_transition_net[3]
        STO_d_activation = STO_transition_net[4]

        self.G_net = ELM.ELMNet(feature_sampling = STO_feature_sampling,
                                sampling_dim = STO_sampling_dim,
                                sampling_params = STO_sampling_params,
                                activation = STO_activation,
                                d_activation = STO_d_activation,
                                verbose = False)
        if verbose:
            print('--INITIAL STO PARAMETERS--')
            print('STO_feature_sampling : ', STO_feature_sampling)
            print('STO_sampling_dim : ', STO_sampling_dim)
            print('STO_sampling_params : ', STO_sampling_params)
            print('STO_activation : ', STO_activation)
            print('STO_d_activation : ', STO_d_activation , '\n')

    def fit(self, A, Y, lmbda):
        """ Compute the parameters ruling the HMM transitions.
        Parameters :
        A : hidden states time serie 
        N : song/ M : timestamp/ D : hidden feature dimension
        For n an integer :
        A[n] = [a1.T, ..., aM.T] has size MxD.

        y : observation time serie.
        N : song/ M : timestamp/ H : observation feature dimension
        """
        N, M, D = A.shape
        a = A[:, :M-1, :].reshape(N*(M-1), D)
        b = A[:, 1:, :].reshape(N*(M-1), D)

        self.F_net.train(X = a, T = b , C = 1/lmbda)
        self.Q = np.cov(self.F_net.predict(X = a) - b,
                        rowvar=False)

        if self.Q.shape == ():
            self.Q = self.Q.reshape((1,1))

        N, M, H = Y.shape
        A = A.reshape((N*M, D))
        Y = Y.reshape((N*M, H))

        self.G_net.train(X = A, T = Y, C = lmbda)
        self.R = np.cov(self.G_net.predict(X = A) - Y,
                        rowvar=False)

        if self.R.shape == ():
            self.R = self.R.reshape((1,1))

        if verbose:
            print('--TRAINED PARAMETERS--')
            print('self.F_net.beta.shape : ', self.F_net.beta.shape)
            print('self.Q.shape : ', self.Q.shape)
            print('self.G_net.beta.shape : ', self.G_net.beta.shape)
            print('self.R.shape : ', self.R.shape, '\n')
            

if __name__ == '__main__':
    #Simulation parameters
    N = 10 #Number of trajectories
    M = 10 #Size of a trajectory
    D = 5 #Dimension of hidden state space
    L = 1000 #Number of hidden neurons
    H = 2 #Dimension of observation space

    #Simulated Data
    A = np.random.normal(loc = 0, scale = 1, size = (N, M, D))
    Y = np.random.normal(loc = 0, scale = 1, size = (N, M, H))
    #STS parameters
    STS_feature_sampling = ELM.normal_features
    STS_sampling_dim = (D, L)
    STS_sampling_params = [0,1]
    STS_activation = np.tanh
    STS_d_activation = lambda X : 1 - np.tanh(X)**2
    STS_transition_net = (STS_feature_sampling,
                          STS_sampling_dim,
                          STS_sampling_params,
                          STS_activation,
                          STS_d_activation)
    #STO parameters
    STO_feature_sampling = ELM.normal_features
    STO_sampling_dim = (D, L)
    STO_sampling_params = [0,1]
    STO_activation = np.tanh
    STO_d_activation = lambda X : 1 - np.tanh(X)**2
    STO_transition_net = (STO_feature_sampling,
                          STO_sampling_dim,
                          STO_sampling_params,
                          STO_activation,
                          STO_d_activation)   
    
    #meta parameters
    params_path = 'coucou'
    verbose = True

    #hyper parameter
    lmbda = 1e-2

    deep_kalman_filter = DeepKalmanFilter(params_path = params_path,
                                          STS_transition_net = STS_transition_net,
                                          STO_transition_net = STO_transition_net,
                                          verbose = verbose)

    deep_kalman_filter.fit(A = A,
                           Y = Y,
                           lmbda = lmbda)
