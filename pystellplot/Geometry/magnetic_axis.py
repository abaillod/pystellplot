from simsopt._core import load
from simsopt.field import compute_fieldlines, compute_toroidal_transits
from simsopt.geo import CurveXYZFourier
import numpy as np
import scipy

import matplotlib.pyplot as plt

def get_magnetic_axis_boozer_surface(bsurf, verbose=False, tmax=50):
    surface = bsurf.surface
    bs = bsurf.biotsavart

    g = surface.cross_section(0)
    r = np.sqrt(g[:,0]**2+g[:,1]**2)
    z = g[:,2]

    R0 = np.mean(r)
    Z0 = np.mean(z)

    return get_magnetic_axis(R0, Z0, bs, verbose=verbose, tmax=tmax)

def get_magnetic_axis_surface(surface, bs, verbose=False, tmax=50):
    g = surface.cross_section(0)
    r = np.sqrt(g[:,0]**2+g[:,1]**2)
    z = g[:,2]

    R0 = np.mean(r)
    Z0 = np.mean(z)

    return get_magnetic_axis(R0, Z0, bs, verbose=verbose, tmax=tmax)


def get_magnetic_axis(R0, Z0, bs, verbose=False, tmax=50):
    def nu(x):
        r = x[0]
        z = x[1]
        res_tys, res_phi_hits = compute_fieldlines(bs,[r],[z],tmax,tol=1e-12,phis=[0])
        
        ntor = compute_toroidal_transits(res_tys, flux=False)
        if ntor<1:
            return 10

        if res_phi_hits[0].shape[0]==0:
            return 10
        
        # Check that we made a full transit
        ind = np.where(res_phi_hits[0][:,1]==0)[0]
        if ind.size==0:
            if verbose:
                print("No transit...")
            return 10
        else:
            res = res_phi_hits[0][ind,:]
            x = res[0,2]
            y = res[0,3]
            z1 = res[0,4]
            r1 = np.sqrt(x**2+y**2)
            
            nu = np.sqrt((r-r1)**2 + (z-z1)**2)
            if verbose:
                print(f"{r}, {z}, {r1}, {z1}, {nu}")
            return nu
        
    x0 = [R0,Z0]
    res = scipy.optimize.minimize(nu, x0)

    # Construct curve
    qpts = np.linspace(0,1,128)
    _, res_phi_hits = compute_fieldlines(bs,[res.x[0]],[res.x[1]],tmax,tol=1e-12,phis=2*np.pi*qpts)

    c = CurveXYZFourier(order=10, quadpoints=qpts)
    c.least_squares_fit(res_phi_hits[0][:128,2:])

    return c