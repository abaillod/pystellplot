import numpy as np

def principal_curvature(surface, dimension='2D'):
    """Evaluate the surface principal curvature, as defined in https://en.wikipedia.org/wiki/Principal_curvature"""

    L = np.sum(surface.gammadash1dash1()*surface.unitnormal(), axis=2)
    M = np.sum(surface.gammadash1dash2()*surface.unitnormal(), axis=2)
    V = np.sum(surface.gammadash2dash2()*surface.unitnormal(), axis=2)

    nphi = surface.quadpoints_phi.size
    ntheta = surface.quadpoints_theta.size
    matrix = np.zeros((ntheta,nphi,2,2))
    matrix[:,:,0,0]=L.transpose()
    matrix[:,:,1,0]=M.transpose()
    matrix[:,:,0,1]=M.transpose()
    matrix[:,:,1,1]=V.transpose()

    eigvals = np.linalg.eigvals(matrix)


    if dimension=='1D':
        s = L.shape
        if s[0]!=s[1]:
            raise ValueError('Need the same number of quadpoints in phi and theta!')

        k1 = np.diag(eigvals[:,:,0])
        k2 = np.diag(eigvals[:,:,1])

    elif dimension=='2D':
        k1 = np.squeeze(eigvals[:,:,0])
        k2 = np.squeeze(eigvals[:,:,1])
        
    else:
        raise ValueError('Invalid dimension')

    return k1, k2

def gaussian_curvature(surface, dimension=None):
    k1, k2 = principal_curvature(surface, dimension)
    return k1*k2

def mean_curvature(surface, dimension=None):
    k1, k2 = principal_curvature(surface, dimension)
    return 0.2*(k1+k2)


