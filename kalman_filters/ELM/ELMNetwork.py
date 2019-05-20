from .feature_samplings import normal_features
import numpy as np

class ELMNetwork:

    def __init__(self, feature_sampling, sampling_dim, sampling_params, activation, d_activation, verbose = True):

        D,L = sampling_dim
        sampling_params = sampling_params
        W_h, b_h = feature_sampling(D, L, sampling_params)

        self.W_h = W_h
        self.b_h = b_h

        self.activation = activation
        self.d_activation = d_activation

        self.beta = None
        self.verbose = verbose

    def train(self, X, T, C):
        """Computes the optimal output weight parameter beta, for the neural network with hidden layer generated from feature_mapping :
        X : the input matrix of size (N, D)
        Y : the output matrix of size (N, H)
        C : the regularization parameter for the norm of beta

        """

        #Compute the hidden layer output of size (N,L)
        H = self.activation(X@(self.W_h).T + self.b_h)
        N, L = H.shape

        self.beta = (np.linalg.inv(np.identity(L)/C + (H.T)@H)
                     @
                     (H.T)@T).T

        if self.verbose:
            print('H.shape', H.shape)
            print('beta.shape', self.beta.shape)

    def d_res(self, X):
        """
        Given a simulation of N trajectories, input X is a batch of N actions a_{k-1} and is therefore of size (N,D) where D is the dimension of the action space.
        """
        #phi'(A_f*a_{k-1} + b_f)
        d_activation_X = self.d_activation(X@(self.W_h).T + self.b_h)
        N, L = d_activation_X.shape
        #diag(phi'(A_f*a_{k-1} + b_f))
        diag_d_activation_X = (
            np.einsum('inj, ij -> nij', np.tile(A = d_activation_X[np.newaxis, :, :], reps = (L,1,1)), np.identity(L)))

        #W_fdiag(phi'(A_f*a_{k-1} + b_f))A_f
        return np.einsum('ik,nkl,lj -> nij', self.beta, diag_d_activation_X, self.W_h) 


    def predict(self, X):
        return (X@(self.W_h.T) + self.b_h)@self.beta.T

if __name__ == '__main__':
    N = 20
    N_train = int(np.floor((9/10)*N))
    D = 10
    H = 2
    L = 1000
    params = [0,1]
    elm_network = ELMNetwork(feature_sampling = normal_features,
                             sampling_dim = (D,L),
                             sampling_params = params,
                             activation = lambda x : x,
                             d_activation = lambda X : 1 - np.tanh(X)**2)

    X = np.random.normal(loc=0, scale=1, size=(N, D))
    A = np.random.normal(loc=0, scale=1, size=(H, D))
    b = np.random.normal(loc=0, scale=1, size=(1,H))

    Y = X@A.T + b
    X_train, Y_train = X[:N_train], Y[:N_train] 
    X_test, Y_test = X[N_train:], Y[N_train:]
    C = 1000
    elm_network.train(X_train, Y_train, C)




