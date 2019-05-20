import ELM
import numpy as np

class ExtendedKalmanFilter:

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
                                sampling_dim = STO_feature_sampling,
                                sampling_params = STO_sampling_params,
                                activation = STO_activation,
                                d_activation = STO_d_activation,
                                verbose = False)

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

            self.F_net.train(X = a, T = b , C = lmbda)
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
            

if __name__ == '__main__':
    print('coucou')
    ELM.ELMNet(None,
               None,
               None,
               None,
               None)
