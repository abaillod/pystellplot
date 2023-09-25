import matplotlib.pyplot as plt
import numpy as np

def plot_poincare( bs, R0, Z0, tmax=1e3, phi_arr=[0], tol=1e-8  ):

    res_tys, res_phi_hits = compute_fieldlines(bs, R0, Z0, tmax=1e4, phis=phi_arr, tol=1e-8, stopping_criteria=[])
    plot_poincare_data(
        res_phi_hits, 
        phi_arr, 
        f'./cnt_coils_poincare.png', 
        dpi=150, 
        surf=initial_cssc_surface, 
        mark_lost=False, 
        aspect='equal'
    )
