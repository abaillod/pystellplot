from simsopt._core import load
from simsopt.field import compute_fieldlines, compute_toroidal_transits
from simsopt.geo import CurveXYZFourier, CurveRZFourier
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from scipy.optimize import fsolve

import matplotlib.pyplot as plt

def find_magnetic_axis(biotsavart, r0, z0, nfp, order):
    n = r0.size
    # if n % 2 == 0:
    #     n+=1

    length = 2*np.pi/nfp
    points = np.linspace(0, length, n, endpoint=False).reshape((n, 1))
    oneton = np.asarray(range(0, n)).reshape((n, 1))
    fak = 2*np.pi / length
    dists = fak * cdist(points, points, lambda a, b: a-b)
    np.fill_diagonal(dists, 1e-10)  # to shut up the warning
    if n % 2 == 0:
        D = 0.5 \
            * np.power(-1, cdist(oneton, -oneton)) \
            / np.tan(0.5 * dists)
    else:
        D = 0.5 \
            * np.power(-1, cdist(oneton, -oneton)) \
            / np.sin(0.5 * dists)

    np.fill_diagonal(D, 0)
    D *= fak
    phi = points

    def build_residual(rz):
        inshape = rz.shape
        rz = rz.reshape((2*n, 1))
        r = rz[:n ]
        z = rz[n:]
        xyz = np.hstack((r*np.cos(phi), r*np.sin(phi), z))
        biotsavart.set_points(xyz)
        B = biotsavart.B()
        Bx = B[:, 0].reshape((n, 1))
        By = B[:, 1].reshape((n, 1))
        Bz = B[:, 2].reshape((n, 1))
        Br = np.cos(phi)*Bx + np.sin(phi)*By
        Bphi = np.cos(phi)*By - np.sin(phi)*Bx
        residual_r = D @ r - r * Br / Bphi
        residual_z = D @ z - r * Bz / Bphi
        return np.vstack((residual_r, residual_z)).reshape(inshape)

    def build_jacobian(rz):
        rz = rz.reshape((2*n, 1))
        r = rz[:n ]
        z = rz[n:]
        xyz = np.hstack((r*np.cos(phi), r*np.sin(phi), z))
        biotsavart.set_points(xyz)
        GradB = biotsavart.dB_by_dX()
        B = biotsavart.B()
        Bx = B[:, 0].reshape((n, 1))
        By = B[:, 1].reshape((n, 1))
        Bz = B[:, 2].reshape((n, 1))
        dxBx = GradB[:, 0, 0].reshape((n, 1))
        dyBx = GradB[:, 1, 0].reshape((n, 1))
        dzBx = GradB[:, 2, 0].reshape((n, 1))
        dxBy = GradB[:, 0, 1].reshape((n, 1))
        dyBy = GradB[:, 1, 1].reshape((n, 1))
        dzBy = GradB[:, 2, 1].reshape((n, 1))
        dxBz = GradB[:, 0, 2].reshape((n, 1))
        dyBz = GradB[:, 1, 2].reshape((n, 1))
        dzBz = GradB[:, 2, 2].reshape((n, 1))
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        Br = cosphi*Bx + sinphi*By
        Bphi = cosphi*By - sinphi*Bx
        drBr = cosphi*cosphi * dxBx + cosphi*sinphi*dyBx + sinphi*cosphi*dxBy + sinphi*sinphi*dyBy
        dzBr = cosphi*dzBx + sinphi*dzBy
        drBphi = cosphi*cosphi*dxBy + cosphi*sinphi*dyBy - sinphi*cosphi*dxBx - sinphi*sinphi*dyBx
        dzBphi = cosphi*dzBy - sinphi*dzBx
        drBz = cosphi * dxBz + sinphi*dyBz
        # residual_r = D @ r - r * Br / Bphi
        # residual_z = D @ z - r * Bz / Bphi
        dr_resr = D + np.diag((-Br/Bphi - r*drBr/Bphi + r*Br*drBphi/Bphi**2).reshape((n,)))
        dz_resr = np.diag((-r*dzBr/Bphi + r*Br*dzBphi/Bphi**2).reshape((n,)))
        dr_resz = np.diag((-Bz/Bphi - r*drBz/Bphi + r*Bz*drBphi/Bphi**2).reshape((n,)))
        dz_resz = D + np.diag((-r*dzBz/Bphi + r*Bz*dzBphi/Bphi**2).reshape((n,)))
        return np.block([[dr_resr, dz_resr], [dr_resz, dz_resz]])

    x0 = np.vstack((r0, z0))

    soln = fsolve(build_residual, x0, fprime=build_jacobian, xtol=1e-13)

    res = build_residual(soln)
    norm_res = np.sqrt(np.sum(res**2))
    ma_success = norm_res < 1e-10
    #print(norm_res)

    xyz = np.hstack((soln[:n, None]*np.cos(phi), soln[:n, None]*np.sin(phi), soln[n:, None]))
    quadpoints = np.linspace(0, 1/nfp, n, endpoint=False)
    ma_fp = CurveRZFourier(quadpoints, order, nfp, True)
    ma_fp.least_squares_fit(xyz)

    quadpoints = np.linspace(0, nfp, nfp*n, endpoint=False)
    ma_ft = CurveRZFourier(quadpoints, order, nfp, True)
    ma_ft.x = ma_fp.x

    return ma_fp, ma_ft, ma_success


def get_magnetic_axis_boozer_surface(bsurf, order=10, verbose=False, tmax=50):
    surface = bsurf.surface
    bs = bsurf.biotsavart

    g = surface.cross_section(0)
    r = np.sqrt(g[:,0]**2+g[:,1]**2)
    z = g[:,2]

    R0 = np.mean(r)
    Z0 = np.mean(z)

    return get_magnetic_axis(R0, Z0, bs, order=order, verbose=verbose, tmax=tmax)

def get_magnetic_axis_surface(surface, bs, order=10, verbose=False, tmax=50):
    g = surface.cross_section(0)
    r = np.sqrt(g[:,0]**2+g[:,1]**2)
    z = g[:,2]

    R0 = np.mean(r)
    Z0 = np.mean(z)

    return get_magnetic_axis(R0, Z0, bs, order=order, verbose=verbose, tmax=tmax)


def get_magnetic_axis(R0, Z0, bs, order=10, verbose=False, tmax=50):
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

    c = CurveXYZFourier(order=order, quadpoints=qpts)
    c.least_squares_fit(res_phi_hits[0][:128,2:])

    return c
