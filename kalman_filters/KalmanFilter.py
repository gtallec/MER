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
        A : hidden states time serie 
        N : song/ M : timestamp/ D : hidden feature dimension
        For n an integer :
        A[n] = [a1.T, ..., aM.T] has size MxD.

        y : observation time serie.
        N : song/ M : timestamp/ H : observation feature dimension

        """

        N, M, D  = A.shape

        a = A[:, :M-1, :].reshape((N*(M-1), D))
        b = A[:, 1:, :].reshape((N*(M-1), D))
        #F
        inter_1 = ((a.T)@a
                   +
                   lmbda*np.identity(D))

        inter_2 = b.T@a

        self.F = (inter_2
                  @
                  np.linalg.inv(inter_1))

        #Q
        self.Q = np.cov(b - a@(self.F.T), rowvar=False)
        
        if self.Q.shape == ():
            self.Q = self.Q.reshape((1,1))

        N, M, H = Y.shape
        Y = Y.reshape(N*M, H)
        A = A.reshape(N*M, D)

        #G
        inter_1 = (A.T@A + lmbda*np.identity(D))
        inter_2 = Y.T@A
        self.G = (inter_2
                  @
                  np.linalg.inv(inter_1))

        #R
        self.R = np.cov(Y-A@(self.G.T), rowvar=False)
        if self.R.shape == ():
            self.R = self.R.reshape((1,1))

        if self.verbose:
            print('F :', self.F)
            print('F.shape :', self.F.shape)
            print('Q :', self.Q)
            print('Q.shape :', self.Q.shape)
            print('G :', self.G)
            print('G.shape', self.G.shape)
            print('R :', self.R)
            print('R.shape :', self.R.shape)

    def predict(self, Y, a_0, P_0):
        """Output the Kalman-predicted distribution trajectory

        shape(Y): N, K, H
        shape(a_0) : N, D
        shape(p_0) : N, D, D
        """
        N, K, H = Y.shape
        N, D = a_0.shape

        A = np.zeros((N, K, D))
        P = np.zeros((N, K, D, D))

        A[:, 0, :] = a_0
        P[:, 0, :, :] = P_0

        for k in range(1,K):

            #a_(k|k-1) = Fa_(k-1) 
            a_k_km1 = A[:, k-1, :]@((self.F).T)

            #P_(k|k-1) = FP_(k-1)F.T + Q
            P_k_km1 = (np.einsum('ik, nkl, jl -> nij',
                                self.F,
                                P[:, k-1, :],
                                self.F)
                      + np.tile(self.Q[np.newaxis, :, :],
                                reps=(N,1,1)))

            #GP_(k|k-1)G.T + R
            inter_1 = (np.einsum('ik, nkl, jl -> nij',
                                 self.G,
                                 P_k_km1,
                                 self.G)
                       + np.tile(self.R[np.newaxis, :, :],
                                 reps=(N,1,1)))

            #G.T((GP_(k|k-1)G.T + R)^{-1})
            inter_2 = np.einsum('ki, nkj -> nij',
                                self.G,
                                np.linalg.inv(inter_1))

            #K = P_(k|k-1)G.T((GP_(k|k-1)G.T + R)^{-1})
            K = np.einsum('nik, nkj -> nij', P_k_km1, inter_2)


            A[:, k, :] = a_k_km1 + np.einsum('nh, njh -> nj',
                                             (Y[:, k, :]
                                              - 
                                              a_k_km1@(self.G.T)),
                                             K)
            P[:, k, :] = (P_k_km1 
                          - 
                          np.einsum('nik, kl, nlj -> nij',
                                    K,
                                    self.G,
                                    P_k_km1))
        return A,P


if __name__ == '__main__':

    kalman_filter = KalmanFilter(params_path = 'coucou',
                                 verbose = True)

    kalman_filter.fit(A = 5*np.ones((1,3,7)),
                      Y = 3*np.ones((1,3,5)),
                      lmbda = 3)

