import numpy as np
from tqdm import tqdm
import time

class KalmanFilter:

    def __init__(self, params_path, verbose):

        self.params_path = params_path
        self.verbose = verbose

        self.F = None
        self.Q = None
        self.G = None
        self.R = None


    def save_params(self):
        np.savez(self.params_path + 'params.npz',
                 F=self.F,
                 Q=self.Q,
                 G=self.G,
                 R=self.R)

    def load_params(self):
        params = np.load(self.params_path + 'params.npz')
        self.F = params['F']
        self.Q = params['Q']
        self.G = params['G']
        self.R = params['R']

    def fit(self, A, a, b, Y, lmbda):
        #TODO:Rewrite description with good shapes and all
        """ Compute the parameters ruling the HMM transitions.

        Parameters :
        A : hidden states time serie 
        N : song/ M : timestamp/ D : hidden feature dimension
        For n an integer :
        A[n] = [a1.T, ..., aM.T] has size MxD.

        y : observation time serie.
        N : song/ M : timestamp/ H : observation feature dimension

        """

        D = A.shape[1]
        H = Y.shape[1]
        #F
        inter_1 = ((a.T)@a
                   +
                   lmbda*np.identity(D))

        inter_2 = b.T@a

        self.F = (inter_2
                  @
                  np.linalg.inv(inter_1))

        #Q
        self.Q = np.atleast_2d(np.cov(b - a@(self.F.T),
                                      rowvar=False,
                                      ddof=0))
        
        #G
        inter_1 = (A.T@A + lmbda*np.identity(D))
        inter_2 = Y.T@A
        self.G = (inter_2
                  @
                  np.linalg.inv(inter_1))

        #R
        self.R = np.atleast_2d(np.cov(Y-A@(self.G.T),
                                      rowvar=False,
                                      ddof=0))

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

        for k in tqdm(range(1,K)):

            t1 = time.time()
            #a_(k|k-1) = Fa_(k-1) 
            a_k_km1 = A[:, k-1, :]@((self.F).T)
            t2 = time.time()
            
            #P_(k|k-1) = FP_(k-1)F.T + Q
            P_k_km1 = ((self.F
                        @
                        P[:, k-1, :, :]
                        @
                        self.F.T)
                       + 
                       np.tile(self.Q[np.newaxis, :, :],
                               reps=(N,1,1)))
            t3 = time.time()

            #GP_(k|k-1)G.T + R
            inter_1 = ((self.G
                          @
                          P_k_km1
                          @
                          self.G.T)
                         +
                         np.tile(self.R[np.newaxis, :, :],
                                 reps=(N,1,1)))
            t4 = time.time()


            #G.T((GP_(k|k-1)G.T + R)^{-1})
            inter_2 = self.G.T@np.linalg.inv(inter_1)
            t5 = time.time()


            #K = P_(k|k-1)G.T((GP_(k|k-1)G.T + R)^{-1})
            K = P_k_km1@inter_2
            t6 = time.time()

            A[:, k, :] = (a_k_km1 
                          +
                          np.einsum('nh, njh -> nj',
                                    Y[:, k, :] - a_k_km1@self.G.T,
                                    K))
            t7 = time.time()
            P[:, k, :, :] = (P_k_km1 - K@(self.G)@P_k_km1)
            t8 = time.time()

            if self.verbose:
                print('delta1 :', t2 - t1)
                print('delta2 :', t3 - t2)
                print('delta3 :', t4 - t3)
                print('delta4 :', t5 - t4)
                print('delta5 :', t6 - t5)
                print('delta6 :', t7 - t6)
                print('delta7 :', t8 - t7)

        return A, P


if __name__ == '__main__':

    kalman_filter = KalmanFilter(params_path = 'coucou',
                                 verbose = True)

    kalman_filter.fit(A = 5*np.ones((1,3,7)),
                      Y = 3*np.ones((1,3,5)),
                      lmbda = 3)

