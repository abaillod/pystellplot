import numpy as np
import matplotlib.pyplot as plt
from simsopt.geo import SurfaceRZFourier

__all__ = ["plot_surfaces"]

def plot_vmec_surfaces(wout_file, phi=0, ns=10, nt=512, ax=None, show=True):
    """plot_surfaces
    Plot VMEC surfaces on a poloidal plane

    EXAMPLE
    -------
        # Create figure with multiple panels
        fig = plt.figure(figsize=(10,6))
        gs = fig.add_gridspec(2,3)
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[1, 2])
        axs = [ax1,ax2,ax3,ax4,ax5]
       
        # Plot at different toroidal angles
        fname = 'wout.nc'
        phi_arr = np.linspace(0,np.pi,5,endpoint=False) 
        for ax, phi in zip(axs, phi_arr): 
            plot_surfaces(fname,phi=phi,ax=ax,show=False)
       
        # Show figure
        plt.show()
    
    INPUTS
    ------
        wout_file: path to VMEC output file
        phi: Toroidal angle - default is 0
        ns: Number of surfaces to plot - default is 10
        nt: Number of poloidal points - default is 512
        ax: An instance of matplotlib.axes.Axes. If none is
            provided, a new figure is generated.

    REQUIREMENTS
    ------------
        numpy
        matplotlib
        simsopt - this routine takes advantage of the SurfaceRZFourier
         class implemented in simsopt.
    """

    if ax is None:
        fig, ax = plt.subplots()

    surfaces = [SurfaceRZFourier.from_wout(wout_file, s=s) for s in np.linspace(0.0,1,ns)**2]
    theta = np.linspace(0,2*np.pi,nt)
    for s in surfaces:
        R = np.zeros((nt,))
        Z = np.zeros((nt,))
        mpol = s.mpol
        ntor = s.ntor
        for mm in range(0,mpol+1):
            for nn in range(-ntor,ntor+1):
                if mm==0 and nn<0: continue
                rc = s.rc[mm,ntor+nn]
                zs = s.zs[mm,ntor+nn]
                R += rc * np.cos(mm*theta - nn*s.nfp*phi)
                Z += zs * np.sin(mm*theta - nn*s.nfp*phi)
    
        ax.scatter(R,Z,s=1)

    ax.set_title('phi = {:.2f}'.format(phi))
    ax.set_aspect('equal')

    if show:
        plt.show()
