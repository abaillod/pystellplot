from pyevtk.hl import polyLinesToVTK
from simsopt.field.coil import ScaledCurrent
import numpy as np

def coils_to_vtk( coils, filename ):
    """
    Similar to simsopt.geo.curves_to_vtk, but here includes information
    about the current carried by the coils
    """

    def wrap(data):
        return np.concatenate([data, [data[0]]])
    
    curves = [c.curve for c in coils]
    x = np.concatenate([wrap(c.gamma()[:, 0]) for c in curves])
    y = np.concatenate([wrap(c.gamma()[:, 1]) for c in curves])
    z = np.concatenate([wrap(c.gamma()[:, 2]) for c in curves])
    ppl = np.asarray([c.gamma().shape[0]+1 for c in curves])

    currents = np.array([c.current.get_value() for c in coils])

    data = np.concatenate([cur*np.ones((ppl[ii], )) for ii, cur in enumerate(currents)])
    polyLinesToVTK(str(filename), x, y, z, pointsPerLine=ppl, pointData={'Current': data})

