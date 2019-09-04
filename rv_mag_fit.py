import matplotlib.pylab as plt
from scipy.optimize import least_squares
import numpy as np
import mag_cal

plt.figure()

guess = [8,-23,-5,
         35,50,50,
         0,0,0]

def obj_func(X, mag):
    """
    Ellipsoid fit of form:
    (x-x0)**2/a**2 + (y-y0)**2/b**2 + (z-z0)**2/c**2 + d*x*y + e*x*z + f*z*y - 1
    """

    c = 50
    d, e, f = [0]*3
    x0, y0, z0, a, b = X
    x = mag[:,0]
    y = mag[:,1]
    z = mag[:,2]

    vals = (x-x0)**2/a**2 + (y-y0)**2/b**2 + (z-z0)**2/c**2 + d*x*y + e*x*z + f*z*y - 1
    return vals

soln = least_squares(obj_func, [8,-23,-5,
         35,50], args=(data['mag'][:,1:],))

X = soln.x.tolist() + [50, 0,0,0]

el = mag_cal.pts_on_ellipsoid2(*X)

axes = mag_cal.plot_3d_cube(el, axis_equal=True)

mag_cal.plot_3d_cube(data['mag'][:,1:], axes=axes)

# convert these coefficients to st a,b,c,...
x0,y0,z0,ax,ay,az = X[:6]
a = ax**(-2)
g = -x0/ax**2
b = ay**(-2)
h = -y0/ay**2
c = az**(-2)
i = -z0/az**2
d,e,f = [0,0,0]

A4 = np.array([[a,d,e,g],
               [d,b,f,h],
               [e,f,c,i],
               [g,h,i,-1]])
A4 /= z0**2/az**2 + y0**2/ay**2 + x0**2/ax**2
A4[3,3] = -1
A3 = A4[:3, :3]
offset = np.vstack([x0,y0,z0])
T = np.eye(4)
T[3, :3] = offset[:,0]
B4 = T.dot(A4).dot(T.T)
B3 = B4[:3, :3]/-B4[3,3]
ev, rotM = np.linalg.eig(B3)
gain = np.sqrt(1/ev)

plt.figure()
data['mag_cal'] = data['mag'].copy()
data['mag_cal'][:,1:] -= offset.T
data['mag_cal'][:,1:] /= [ax,ay,az]
mag_cal.plot_3d_cube(data['mag_cal'][:,1:], axis_equal=True)
