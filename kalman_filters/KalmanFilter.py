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

    def fit(self, A, B, y, lmbda):
        """ Compute the parameters ruling the HMM transitions.

        Parameters :
        A,B : hidden states time serie 
        N : song/ M : time_stamp/ D : feature dimension
        For n an integer :
        A[n] = [a1.T, ..., a(M-1).T] has size (M-1)xD.
        B[n] = [a2.T, ..., aM.T] has size (M-1)xD.
        y : observation time serie.

        """
        N, M_min_one, D  = A.shape

        inter_1 = (np.einsum('nki,nkj -> ij',
                             A,
                             A)
                   +
                   N*lmbda*np.identity(D))

        inter_2 = np.einsum('nki, nkj -> ij',
                             A,
                             B)

        self.F = np.linalg.inv(inter_1)@inter_2
        
        if self.verbose:
            print('F :', self.F)
            print('F.shape :', self.F.shape)
            print('inter_1 :', inter_1)
            print('inter_2 :', inter_2)



if __name__ == '__main__':

    kalman_filter = KalmanFilter(params_path = 'coucou',
                                 verbose = True)

    kalman_filter.fit(A = 5*np.identity(2).reshape((1,2,2)),
                      B = 2*np.identity(2).reshape((1,2,2)),
                      y = None,
                      lmbda = 3)

