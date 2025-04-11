from pyevtk.hl import gridToVTK
from simsopt.field import BiotSavart
from simsopt.geo import Surface
from ..Geometry.compute_surface import principal_curvature
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
    modB = np.linalg.norm(bs.B().reshape((nphi, ntheta, 3)), axis=2)
    B = np.sum(bs.B().reshape((nphi, ntheta, 3)) * n, axis=2)

    k1, k2 = principal_curvature(surf)
    
    k1 = k1.transpose()
    k2 = k2.transpose()

    if normalized:
        B = B / modB

    # Close the torus
    if close_poloidal:
        g = np.concatenate((g, g[:, :1, :]), axis=1)
        n = np.concatenate((n, n[:, :1, :]), axis=1)
        B = np.concatenate((B, B[:, :1]), axis=1)
        modB = np.concatenate((modB, modB[:, :1]), axis=1)
        k1 = np.concatenate((k1, k1[:, :1]), axis=1)
        k2 = np.concatenate((k2, k2[:, :1]), axis=1)

    if close_toroidal:
        dphi = surf.quadpoints_phi[1] - surf.quadpoints_phi[0]
        if 1 - surf.quadpoints_phi[-1] < 1.1 * dphi:
            g = np.concatenate((g, g[:1, :, :]), axis=0)
            n = np.concatenate((n, n[:1, :, :]), axis=0)
            B = np.concatenate((B, B[:1, :]), axis=0)
            modB = np.concatenate((modB, modB[:1, :]), axis=0)
            k1 = np.concatenate((k1, k1[:1, :]), axis=0)
            k2 = np.concatenate((k2, k2[:1, :]), axis=0)

    ntor = g.shape[0]
    npol = g.shape[1]
    x = g[:, :, 0].reshape((1, ntor, npol)).copy()
    y = g[:, :, 1].reshape((1, ntor, npol)).copy()
    z = g[:, :, 2].reshape((1, ntor, npol)).copy()

    B = B.reshape((1, ntor, npol))
    modB = modB.reshape((1, ntor, npol))
    k1 = k1.reshape((1, ntor, npol))
    k2 = k2.reshape((1, ntor, npol))


    kg = k1*k2
    kb = 0.5*(k1+k2)

    contig = np.ascontiguousarray
    pointData = {
        "Bdotn": (contig(B[...])),
        "modB": (contig(modB[...])),
        "k1": (contig(k1[...])),
        "k2": (contig(k2[...])),
        "GaussianCurvature": (contig(kg[...])),
        "MeanCurvature": (contig(kb[...]))
    }
    gridToVTK(str(filename), x, y, z, pointData=pointData)
    #gridToVTK(str(filename), x, y, z)

