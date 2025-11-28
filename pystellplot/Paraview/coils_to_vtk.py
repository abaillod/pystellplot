from pyevtk.hl import polyLinesToVTK, gridToVTK
from simsopt.field.coil import ScaledCurrent
import numpy as np
import pyvista as pv

def coils_to_vtk( coils, filename, more_data=None ):
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
    pointData = {'Current': data}
    if more_data is not None:
        for key, value in more_data.items():
            more = np.concatenate([value[ii]*np.ones((ppl[ii], )) for ii in range(len(curves))])
            pointData[key] = more

    polyLinesToVTK(str(filename), x, y, z, pointsPerLine=ppl, pointData=pointData)



def frame_to_vtk( curve, tangent, normal, binormal, filename ):
    """
    Write a frame to a VTK file.
    """
    gamma = curve.gamma()
    gamma_closed = np.vstack([gamma, gamma[0]])  # close the loop
    normal = np.vstack([normal, normal[0]])
    binormal = np.vstack([binormal, binormal[0]])
    tangent = np.vstack([tangent, tangent[0]])


    n_points = gamma_closed.shape[0]
    poly = pv.PolyData()
    poly.points = gamma_closed

    lines = np.hstack([[n_points], np.arange(n_points)])
    poly.lines = lines
    
    poly["Normals"] = normal  # Assign normals as point data
    poly["Binormals"] = binormal  # Assign binormals as point data
    poly["Tangents"] = tangent  # Assign tangents as point data
    
    # Save to file
    if not filename.endswith('.vtp'):
        filename += '.vtp'

    poly.save(filename)

def tape_to_vtk( curve, t, n, b, filename, w=4e-3 ):
	gamma = curve.gamma()
	x = gamma[:,0]
	y = gamma[:,1]
	z = gamma[:,2]
	x = np.concatenate([x, [x[0]]])
	y = np.concatenate([y, [y[0]]])
	z = np.concatenate([z, [z[0]]])
	vx = b[:,0]
	vy = b[:,1]
	vz = b[:,2]
	vx = np.concatenate([vx, [vx[0]]])
	vy = np.concatenate([vy, [vy[0]]])
	vz = np.concatenate([vz, [vz[0]]])

	width = np.linspace(0,w,100)
	x = x[:,None]+width[None,:]*vx[:,None]
	y = y[:,None]+width[None,:]*vy[:,None]
	z = z[:,None]+width[None,:]*vz[:,None]

	x = x.reshape((1, len(vx), len(width))).copy()
	y = y.reshape((1, len(vx), len(width))).copy()
	z = z.reshape((1, len(vx), len(width))).copy()

	data = np.ones((np.shape(x)[1],np.shape(x)[2]))

	data = {"twist": (data)[:, :, None]}
	gridToVTK(str(filename), np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(z), pointData=data)