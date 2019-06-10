import numpy as np

def make_grid(G):
    """
    Return the coordinates of the central points of a regular two dimensional grid of [-1,1]x[-1,1] with G gridsteps
    """
    delta = 2 / G
    x = (np.arange(1,G + 1) - 1 / 2) * delta - 1
    x_grid, y_grid = np.meshgrid(x, x)
    print('x_grid : ', x_grid)
    print('y_grid : ', y_grid)
    return np.stack((x_grid, y_grid), axis = 2) 

def density_estimation(annotations, G, K):
    """
    TODO
    annotations should be a (U_{s}, 2) list
    """
    mass_estimation = np.mean(K(annotations, make_grid(G)),
                              axis=0)

    return mass_estimation/np.sum(mass_estimation)

def gaussian_density(annotations, grid, h_a, h_v, verbose=True):

    U_s = annotations.shape[0]
    G = grid.shape[0]
    n_det_cov = np.sqrt(h_a*h_v)
    inv_sqrt_cov = np.diag([1/np.sqrt(h_a), 1/np.sqrt(h_v)])
    grid_annotations = np.tile(annotations[:,
                                           np.newaxis,
                                           np.newaxis,
                                           :],
                               reps=(1,G,G,1))

    return ((1/(2*np.pi*n_det_cov))
            *
            np.exp(-1/2
                   *
                   np.linalg.norm(np.einsum('ik, nxyk -> nxyi',
                                            inv_sqrt_cov,
                                            grid_annotations - grid),
                                  axis=3)
                   )
            )



if __name__ == '__main__':
    G = 2
    h_a = 1
    h_v = 1
    annotations = np.arange(4).reshape(2,2)
    K = lambda annotations, grid : gaussian_density(annotations,
                                                    grid,
                                                    h_a,
                                                    h_v)
    print(density_estimation(annotations, G, K))

    


    
