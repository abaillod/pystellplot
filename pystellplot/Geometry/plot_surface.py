import numpy as np
import matplotlib.pyplot as plt


def plot_cross_section(surfaces, phi_array, ax=None):
    if ax is None:    
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot()
        
    for surf in surfaces:
        for phi_slice in phi_array:
            cs = surf.cross_section(phi_slice*np.pi)

            rs = np.sqrt(cs[:,0]**2 + cs[:,1]**2)
            rs = np.append(rs, rs[0])
            zs = cs[:,2]
            zs = np.append(zs, zs[0])

            plt.plot(rs, zs, label=fr'$\phi$ = {phi_slice:.2f}π')

    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.tick_params(axis='both', which='major')
    plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout()