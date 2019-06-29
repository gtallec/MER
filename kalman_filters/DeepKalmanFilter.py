import numpy as np
from ELM.ELMNetwork import ELMNetwork as ELMNet
from tqdm import tqdm

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
        STS_verbose = STS_transition_net[5]

        if verbose:
            print('--INITIAL STS PARAMETERS--')
            print('STS_feature_sampling : ', STS_feature_sampling)
            print('STS_sampling_dim : ', STS_sampling_dim)
            print('STS_sampling_params : ', STS_sampling_params)
            print('STS_activation : ', STS_activation)
            print('STS_d_activation : ', STS_d_activation)
            print('STS_verbose : ', STS_verbose, '\n')

        self.F_net = ELMNet(feature_sampling = STS_feature_sampling,
                                sampling_dim = STS_sampling_dim,
                                sampling_params = STS_sampling_params,
                                activation = STS_activation,
                                d_activation = STS_d_activation,
                                verbose = STS_verbose)

        STO_feature_sampling = STO_transition_net[0]
        STO_sampling_dim = STO_transition_net[1]
        STO_sampling_params = STO_transition_net[2]
        STO_activation = STO_transition_net[3]
        STO_d_activation = STO_transition_net[4]
        STO_verbose = STO_transition_net[5]

        self.G_net = ELMNet(feature_sampling = STO_feature_sampling,
                                sampling_dim = STO_sampling_dim,
                                sampling_params = STO_sampling_params,
                                activation = STO_activation,
                                d_activation = STO_d_activation,
                                verbose = STO_verbose)
        if verbose:
            print('--INITIAL STO PARAMETERS--')
            print('STO_feature_sampling : ', STO_feature_sampling)
            print('STO_sampling_dim : ', STO_sampling_dim)
            print('STO_sampling_params : ', STO_sampling_params)
            print('STO_activation : ', STO_activation)
            print('STO_d_activation : ', STO_d_activation)
            print('STO_verbose : ', STO_verbose, '\n')

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

        if self.verbose:
            print('--TRAINED PARAMETERS--')
            print('self.F_net.beta.shape : ', self.F_net.beta.shape)
            print('self.Q.shape : ', self.Q.shape)
            print('self.G_net.beta.shape : ', self.G_net.beta.shape)
            print('self.R.shape : ', self.R.shape, '\n')

    def predict(self, Y, a_0, P_0):
        """Output the Kalman-predicted distribution trajectory

        shape(Y): N, K, H
        shape(a_0) : N, D
        shape(p_0) : N, D, D
        """
        N, K, H = Y.shape
        N, D = a_0.shape

        A = np.zeros((N,K,D))
        P = np.zeros((N,K,D,D))

        A[:, 0, :] = a_0
        P[:, 0, :] = P_0

        for k in tqdm(range(1,K)):

            #a_{k|k-1} = f_{NN}(a_{k-1})
            a_k_km1 = self.F_net.predict(A[:, k-1, :])

            #F = d_{a}f_{NN}|a = a_{k-1}
            F = self.F_net.d_res(A[:, k-1, :])

            #P_{k|k-1} = FP_{k-1}F.T + Q
            P_k_km1 = (np.einsum('nik, nkl, njl -> nij',
                                 F,
                                 P[:, k-1, :, :],
                                 F)
                       + np.tile(self.Q[np.newaxis, :, :],
                                 reps=(N,1,1)))
            
            #G = d_{a}g_{NN}|a = a_{k|k-1}
            G = self.G_net.d_res(a_k_km1)
            
            #GP_{k|k-1}G.T + R
            inter_1 = (np.einsum('nik, nkl, njl -> nij',
                                 G,
                                 P_k_km1,
                                 G)
                       + np.tile(self.R[np.newaxis, :, :],
                                 reps=(N,1,1)))
            
            #G.T((GP_{k|k-1}G.T + R)^{-1})
            inter_2 = np.einsum('nki, nkj -> nij',
                                G,
                                np.linalg.inv(inter_1))

            #K = P_{k|k-1}G.T((GP_{k|k-1}G.T + R)^{-1})
            K = np.einsum('nik, nkj -> nij', P_k_km1, inter_2)

            #a_{k} = a_{k|k-1} + K[y_{k} - g_{NN}(a_{k|k-1})]
            A[:, k, :] = a_k_km1 + np.einsum('nh, ndh -> nd',
                                             (Y[:, k, :]
                                              -
                                              self.G_net.predict(a_k_km1)),
                                             K)
            #P_{k} = P_{k|k-1} - KGP_{k|k-1}
            P[:, k, :, :] = (P_k_km1
                            -
                            np.einsum('nik, nkl, nlj -> nij',
                                      K,
                                      G,
                                      P_k_km1))
        return A, P






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
