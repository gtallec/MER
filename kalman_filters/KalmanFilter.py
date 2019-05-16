import numpy as np


class KalmanFilter:

    def __init__(self, params_path, verbose):

        self.params_path = params_path
        self.verbose = verbose

        self.F = None
        self.Q = None
        self.G = None
        self.R = None


    def save_params(self):
        np.savez(params_path,
                 F=self.F,
                 Q=self.Q,
                 G=self.G,
                 R=self.R)

    def load_params(self):
        params = np.load(params_path)
        self.F = params['F']
        self.Q = params['Q']
        self.G = params['G']
        self.R = params['R']

    def fit(self, A, Y, lmbda):
        """ Compute the parameters ruling the HMM transitions.

        Parameters :
        A_t : hidden states time serie 
        N : song/ M : timestamp/ D : hidden feature dimension
        For n an integer :
        A[n] = [a1.T, ..., aM.T] has size MxD.

        y : observation time serie.
        N : song/ M : timestamp/ H : observation feature dimension

        """

        N, M, D  = A.shape

        a = A[:, :M-1, :]
        b = A[:, 1:, :] 
        #F
        inter_1 = (np.einsum('nki,nkj -> ij',
                             a,
                             a)
                   +
                   lmbda*np.identity(D))

        inter_2 = np.einsum('nki, nkj -> ij',
                             a,
                             b)

        self.F = np.linalg.inv(inter_1)@inter_2

        #Q
        inter_1 = ((b
                   -
                   np.einsum('nik, jk -> nij', a, self.F))
                   .reshape(N*(M-1), D)
        self.Q = np.cov(inter_1, rowvar=False)

        N, M, H = Y.shape
        Y = Y.reshape(N*M,H)
        A = A.reshape(N*M,D)

        #G
        inter_1 = (Y.T@Y + lmbda*np.identity(H))
        inter_2 = Y.T@A
        self.G = np.linalg.inv(inter_1)@inter_2

        #R
        self.R = np.cov(Y-A@(self.G.T), rowvar=False)

        if self.verbose:
            print('F :', self.F)
            print('F.shape :', self.F.shape)
            print('Q :', self.Q)
            print('Q.shape :', self.Q.shape)
            print('G :', self.G)
            print('G.shape', self.G.shape)
            print('R :', self.R)
            print('R.shape :', self.R.shape)

    def predict(Y, a_0, P_0):
        """Output the Kalman-predicted distribution trajectory

        shape(Y): N, K, H
        shape(a_0) : N, D
        shape(p_0) : N, D, D
        """
        N,K,H, = Y.shape
        a_k = a_0
        P_k = P_0

        A = np.zeros((N,K,D))
        P = np.zeros((N,K,D,D))

        for k in range(K):
            a_tr_k = a_k@self.F

            P_tr_k = (np.einsum('ik, nkl, jl -> nij',
                                self.F,
                                P_k,
                                self.F)
                      + self.Q)

            inter_1 = (np.einsum('ik, nkl, jl -> nij',
                                 self.G,
                                 P_tr_k,
                                 self.G)
                       + self.R)
            inter_2 = np.einsum('ki, nkj -> nij',
                                self.G,
                                np.linalg.inv(inter_1))

            K = np.einsum('nik, nkj -> nij', P_tr_k, inter_2)
            a_k = a_tr_k + np.einsum(' , -> ',
                                     K,
                                     Y[:,k,:] - a_tr_k@(self.G.T)



        




if __name__ == '__main__':

    kalman_filter = KalmanFilter(params_path = 'coucou',
                                 verbose = True)

    kalman_filter.fit(A = 5*np.ones((1,3,7)),
                      Y = 3*np.ones((1,3,5)),
                      lmbda = 3)

