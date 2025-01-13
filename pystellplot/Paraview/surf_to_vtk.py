from pyevtk.hl import gridToVTK
from simsopt.field import BiotSavart
from simsopt.geo import Surface
import numpy as np


def surf_to_vtk(filename:str, bs:BiotSavart, surf:Surface, close_poloidal:bool=True, close_toroidal:bool=True, normalized:bool=True):
    """
    Similar to simsopt.geo.surface.to_vtk, but now include
    the value of B.n on the surface
    """

    g = surf.gamma()
    n = surf.unitnormal()
    bs.set_points( g.reshape((-1, 3)) )
    ntheta = surf.quadpoints_theta.size
    nphi = surf.quadpoints_phi.size
    B = np.sum(bs.B().reshape((nphi, ntheta, 3)) * n, axis=2)

    if normalized:
        modB = np.linalg.norm(bs.B().reshape((nphi, ntheta, 3)), axis=2)
        B = B / modB

    # Close the torus
    if close_poloidal:
        g = np.concatenate((g, g[:, :1, :]), axis=1)
        n = np.concatenate((n, n[:, :1, :]), axis=1)
        B = np.concatenate((B, B[:, :1]), axis=1)

    if close_toroidal:
        dphi = surf.quadpoints_phi[1] - surf.quadpoints_phi[0]
        if 1 - surf.quadpoints_phi[-1] < 1.1 * dphi:
            g = np.concatenate((g, g[:1, :, :]), axis=0)
            n = np.concatenate((n, n[:1, :, :]), axis=0)
            B = np.concatenate((B, B[:1, :]), axis=0)

    ntor = g.shape[0]
    npol = g.shape[1]
    x = g[:, :, 0].reshape((1, ntor, npol)).copy()
    y = g[:, :, 1].reshape((1, ntor, npol)).copy()
    z = g[:, :, 2].reshape((1, ntor, npol)).copy()
    B = B.reshape((1, ntor, npol))

    contig = np.ascontiguousarray
    pointData = {
        "Bdotn": (contig(B[...]))
    }
    gridToVTK(str(filename), x, y, z, pointData=pointData)
    #gridToVTK(str(filename), x, y, z)

