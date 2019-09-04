
from scipy.optimize import minimize, fmin, least_squares
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pylab as plt
from numpy import sin, cos, sqrt, inf, pi
from scipy.optimize import least_squares, lsq_linear

"""
Perhaps here:
https://teslabs.com/articles/magnetometer-calibration/

for more detailed calibration approach

https://www.mathworks.com/matlabcentral/fileexchange/23377-ellipsoid-fitting
"""
def __ellipsoid_fit(self, s):
        ''' Estimate ellipsoid parameters from a set of points.

            Parameters
            ----------
            s : array_like
              The samples (M,N) where M=3 (x,y,z) and N=number of samples.

            Returns
            -------
            M, n, d : array_like, array_like, float
              The ellipsoid parameters M, n, d.

            References
            ----------
            .. [1] Qingde Li; Griffiths, J.G., "Least squares ellipsoid specific
               fitting," in Geometric Modeling and Processing, 2004.
               Proceedings, vol., no., pp.335-340, 2004
        '''

        # D (samples)
        D = np.array([s[0]**2., s[1]**2., s[2]**2.,
                      2.*s[1]*s[2], 2.*s[0]*s[2], 2.*s[0]*s[1],
                      2.*s[0], 2.*s[1], 2.*s[2], np.ones_like(s[0])])

        # S, S_11, S_12, S_21, S_22 (eq. 11)
        S = np.dot(D, D.T)
        S_11 = S[:6,:6]
        S_12 = S[:6,6:]
        S_21 = S[6:,:6]
        S_22 = S[6:,6:]

        # C (Eq. 8, k=4)
        C = np.array([[-1,  1,  1,  0,  0,  0],
                      [ 1, -1,  1,  0,  0,  0],
                      [ 1,  1, -1,  0,  0,  0],
                      [ 0,  0,  0, -4,  0,  0],
                      [ 0,  0,  0,  0, -4,  0],
                      [ 0,  0,  0,  0,  0, -4]])

        # v_1 (eq. 15, solution)
        E = np.dot(linalg.inv(C),
                   S_11 - np.dot(S_12, np.dot(linalg.inv(S_22), S_21)))

        E_w, E_v = np.linalg.eig(E)

        v_1 = E_v[:, np.argmax(E_w)]
        if v_1[0] < 0: v_1 = -v_1

        # v_2 (eq. 13, solution)
        v_2 = np.dot(np.dot(-np.linalg.inv(S_22), S_21), v_1)

        # quadric-form parameters
        M = np.array([[v_1[0], v_1[3], v_1[4]],
                      [v_1[3], v_1[1], v_1[5]],
                      [v_1[4], v_1[5], v_1[2]]])
        n = np.array([[v_2[0]],
                      [v_2[1]],
                      [v_2[2]]])
        d = v_2[3]

        return M, n, d


def plot_mag(mag):
    plt.clf()
    plt.plot(mag[:,1], mag[:, 2], '.', label='xy')
    plt.plot(mag[:,1], mag[:, 3], '.', label='xz')
    plt.plot(mag[:,2], mag[:, 3], '.', label='yz')
    plt.axis('equal')
    plt.grid('on')
    plt.legend(loc=0)


def basic_comp(mag):
    # https://appelsiini.net/2018/calibrate-magnetometer/
    result = mag.copy()
    offset = (mag[:,1:].max(axis=0) + mag[:,1:].min(axis=0)) / 2
    print(offset)
    result[:, 1:] = mag[:, 1:] - offset

    avg_delta = (mag[:,1:].max(axis=0) - mag[:,1:].min(axis=0)) / 2
    print(avg_delta)
    scale = avg_delta.sum() / 3 / avg_delta
    print(scale)
    for n in range(mag.shape[0]):
        result[n, 1:] = scale * result[n, 1:]
    return result


def plot3d(mag):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mag[:,1], mag[:,2], mag[:,3])


def obj_func(X, mag):
    """
    Ellipsoid fit of form:
    (x-x0)**2/a**2 + (y-y0)**2/b**2 + (z-z0)**2/c**2 + d*x*y + e*x*z + f*z*y - 1
    """

    x0, y0, z0, a, b, c, d, e, f = X
    x = mag[:,0]
    y = mag[:,1]
    z = mag[:,2]

    vals = (x-x0)**2/a**2 + (y-y0)**2/b**2 + (z-z0)**2/c**2 + d*x*y + e*x*z + f*z*y - 1
    return vals

def my_attempt():
    x = least_squares(obj_func, [8,-28,0,35,50,50,0,0,0], args=(data['mag'][:,1:],))


def st_method(mag, adj_rotM=True, plots=True):
    """
    https://www.st.com/resource/en/design_tip/dm00286302.pdf

    Related reference:
    http://www.secs.oakland.edu/~li4/papers/conf/Cui_PLAN2018.pdf
    """
    x = mag[:,[0]]
    y = mag[:,[1]]
    z = mag[:,[2]]
    D = np.hstack([x**2, y**2, z**2, 2*x*y, 2*x*z, 2*y*z, 2*x, 2*y, 2*z])
    v = np.linalg.inv(D.T.dot(D)).dot(D.T.dot(np.ones((mag.shape[0], 1))))
    # A = [ v(1) v(4) v(5) v(7); v(4) v(2) v(6) v(8);v(5) v(6) v(3) v(9);v(7) v(8) v(9) -1 ];
    #        a    d    e    g
    #        d    b    f    h
    #        e    f    c    i
    #        g    f    h    -1
    a,b,c,d,e,f,g,h,i = v[:,0]
    A4 = np.array([[a,d,e,g],
                   [d,b,f,h],
                   [e,f,c,i],
                   [g,h,i,-1]])
    A3 = A4[:3, :3]
    vghi = np.vstack([g,h,i])
    offset = -np.linalg.inv(A3).dot(vghi)
    T = np.eye(4)
    T[3, :3] = offset[:,0]
    B4 = T.dot(A4).dot(T.T)
    B3 = B4[:3, :3]/-B4[3,3]
    ev, rotM = np.linalg.eig(B3)
    gain = np.sqrt(1/ev)

    if adj_rotM:
        # Order eigen vectors:
        i = np.abs(rotM).argmax(axis=0)
        rotM = rotM[:, i]
        gain = gain[i]
        for i in range(3):
            if rotM[i,i] < 0:
                rotM[:, i] = -rotM[:, i]
    if plots:
        plt.figure()
        plot_3d_cube(mag, '.')
        mag_comp = (mag - offset.flatten()).dot(rotM)*(1/gain)*gain.mean()
        plot_3d_cube(mag_comp, 'o')
    return offset, gain, rotM


def st_method2(mag, adj_rotM=True, plots=True):
    """
    https://www.st.com/resource/en/design_tip/dm00286302.pdf

    Related reference:
    http://www.secs.oakland.edu/~li4/papers/conf/Cui_PLAN2018.pdf
    """
    x = mag[:,[0]]
    y = mag[:,[1]]
    z = mag[:,[2]]
    D = np.hstack([x**2 + y**2 - 2*z**2,
                   x**2 - 2*y**2 + z**2,
                   4*x*y, 2*x*z, 2*y*z, 2*x, 2*y, 2*z, np.ones((len(x),1))])
    E = np.hstack([x**2 + y**2 + z**2])
    u = np.linalg.inv(D.T.dot(D)).dot(D.T.dot(E))
    print(u)
    S3 = np.array([[3,1,1],
                   [3,1,-2],
                   [3,-2,1]])
    S = np.vstack([np.hstack([S3, np.zeros((3,7))]),
                   np.hstack([np.zeros((7,3)), np.eye(7)])])
    S[3,3] = 2
    print(S)
    v = S.dot(np.vstack([-1/3, u]))
    print(v)
    v = -v[:-1, [0]]/v[-1, 0]
    print(v)
    # A = [ v(1) v(4) v(5) v(7); v(4) v(2) v(6) v(8);v(5) v(6) v(3) v(9);v(7) v(8) v(9) -1 ];
    #        a    d    e    g
    #        d    b    f    h
    #        e    f    c    i
    #        g    f    h    -1
    a,b,c,d,e,f,g,h,i = v[:,0]
    A4 = np.array([[a,d,e,g],
                   [d,b,f,h],
                   [e,f,c,i],
                   [g,h,i,-1]])
    A3 = A4[:3, :3]
    vghi = np.vstack([g,h,i])
    offset = -np.linalg.inv(A3).dot(vghi).flatten()
    T = np.eye(4)
    T[3, :3] = offset
    B4 = T.dot(A4).dot(T.T)
    B3 = B4[:3, :3]/-B4[3,3]
    ev, rotM = np.linalg.eig(B3)
    gain = np.sqrt(1/ev)

    if adj_rotM:
        # Order eigen vectors:
        i = np.abs(rotM).argmax(axis=0).argsort()
        rotM = rotM[:, i]
        gain = gain[i]
        for i in range(3):
            if rotM[i,i] < 0:
                rotM[:, i] = -rotM[:, i]
    if plots:
        plt.figure()
        axes = plot_3d_cube(mag, '.', axis_equal=True)
        #mag_comp = (mag - offset.flatten()).dot(rotM)*(1/gain)*gain.mean()
        mag_comp = compensate_ellipsoid(mag, offset, gain, rotM)*gain.mean()
        plot_3d_cube(mag_comp, 'o', axis_equal=True, axes=axes)
    return offset, gain, rotM, v, axes


def compensate_ellipsoid(mag, offset, gain, rotM):
    assert(offset.shape == (3,)) #, "Offset shape should be (3,)")
    assert(gain.shape == (3,)) #, "Gain shape should be (3,)")
    assert(rotM.shape == (3,3)) #, "rotM shape should be (3,3)")
    mag_comp = (mag - offset).dot(rotM)*(1/gain)
    return mag_comp


def constrainted_ellipsoid(mag, adj_rotM=True, plots=True):
    """
    https://www.st.com/resource/en/design_tip/dm00286302.pdf

    Related reference:
    http://www.secs.oakland.edu/~li4/papers/conf/Cui_PLAN2018.pdf

    (x-x0)**2/a**2 + (y-y0)**2/b**2 + (z-z0)**2/c**2 = 1

    """
    x = mag[:,[0]]
    y = mag[:,[1]]
    z = mag[:,[2]]
    D = np.hstack([x**2, y**2, z**2, 2*x*y, 2*x*z, 2*y*z, 2*x, 2*y, 2*z])
    v0 = np.linalg.inv(D.T.dot(D)).dot(D.T.dot(np.ones((mag.shape[0], 1))))
    v = lsq_linear(D, np.ones((mag.shape[0],)),
                   bounds=([1e-6]*3 + [-inf,-inf,-inf,-inf,-inf,-inf,],
                           [inf,inf,inf,inf,inf,inf,inf,inf,inf,]))
    print(v0)
    print(v)
    # A = [ v(1) v(4) v(5) v(7); v(4) v(2) v(6) v(8);v(5) v(6) v(3) v(9);v(7) v(8) v(9) -1 ];
    #        a    d    e    g
    #        d    b    f    h
    #        e    f    c    i
    #        g    f    h    -1
    a,b,c,d,e,f,g,h,i = v.x
    #a,b,c,d,e,f,g,h,i = v0[:,0]
    A4 = np.array([[a,d,e,g],
                   [d,b,f,h],
                   [e,f,c,i],
                   [g,h,i,-1]])
    A3 = A4[:3, :3]
    vghi = np.vstack([g,h,i])
    offset = -np.linalg.inv(A3).dot(vghi)
    T = np.eye(4)
    T[3, :3] = offset[:,0]
    B4 = T.dot(A4).dot(T.T)
    B3 = B4[:3, :3]/-B4[3,3]
    ev, rotM = np.linalg.eig(B3)
    print(f"ev: {ev}; rotM: {rotM}")
    gain = np.sqrt(1/ev)

    if adj_rotM:
        # Order eigen vectors:
        i = np.abs(rotM).argmax(axis=0).argsort()
        rotM = rotM[:, i]
        gain = gain[i]
        for i in range(3):
            if rotM[i,i] < 0:
                rotM[:, i] = -rotM[:, i]
    if plots:
        plt.figure()
        plot_3d_cube(mag, '.')
        #mag_comp = (mag - offset.flatten()).dot(rotM)*(1/gain)*gain.mean()
        #plot_3d_cube(mag_comp, 'o')
        ell = pts_on_ellipsoid(a,b,c,d,e,f,g,h,i)
        plot_3d_cube(ell, 'o')
    return offset, gain, rotM


def non_rot_ellipsoid(mag, adj_rotM=True, plots=True):
    """
    https://www.st.com/resource/en/design_tip/dm00286302.pdf

    Related reference:
    http://www.secs.oakland.edu/~li4/papers/conf/Cui_PLAN2018.pdf
    """
    x = mag[:,[0]]
    y = mag[:,[1]]
    z = mag[:,[2]]
    D = np.hstack([x**2, y**2, z**2, 2*x, 2*y, 2*z])
    v = np.linalg.inv(D.T.dot(D)).dot(D.T.dot(np.ones((mag.shape[0], 1))))
    print(v)
    a,b,c,g,h,i = v[:,0]
    offset = np.vstack([g/a, h/b, i/c])
    G = 1 + g**2/a + h**2/b + i**2/c
    gain = np.sqrt(np.vstack([a/G, b/G, c/G]))

    if plots:
        plt.figure()
        plot_3d_cube(mag, '.')
        mag_comp = (mag - offset.flatten())*(1/gain)*gain.mean()
        plot_3d_cube(mag_comp, 'o')
    return offset, gain


def non_rot_ellipsoid2(mag, adj_rotM=True, plots=True):
    """
    ellipsoid = (x-x0)**2/a**2 + (y-y0)**2/b**2 + (z-z0)**2/c**2 = 1
    ellipsoid = (x-x0)**2/a + (y-y0)**2/b + (z-z0)**2/c = 1

    z**2/c**2 - 2*z*z0/c**2 + z0**2/c**2 + y**2/b**2 - 2*y*y0/b**2 + y0**2/b**2 + x**2/a**2 - 2*x*x0/a**2 + x0**2/a**2 = 1
    x**2/a**2 + y0**2/b**2 + z**2/c**2 - 2*x*x0/a**2 - 2*y*y0/b**2 - 2*z*z0/c**2 + x0**2/a** + y0**2/b**2 + z0**2/c**2 = 1
    """
    x = mag[:,[0]]
    y = mag[:,[1]]
    z = mag[:,[2]]
    D = np.hstack([x**2, y**2, z**2, -2*x, -2*y, -2*z])
    v = np.linalg.inv(D.T.dot(D)).dot(D.T.dot(np.ones((mag.shape[0], 1))))
    print(v)
    a = (1/v[0,0])
    b = (1/v[1,0])
    c = (1/v[2,0])
    x0 = v[3,0]*a
    y0 = v[4,0]*b
    z0 = v[5,0]*c
    print(a,b,c,x0,y0,z0)
    offset = np.vstack([g/a, h/b, i/c])
    G = 1 + g**2/a + h**2/b + i**2/c
    gain = np.sqrt(np.vstack([a/G, b/G, c/G]))

    if plots:
        plt.figure()
        plot_3d_cube(mag, '.')
        mag_comp = (mag - offset.flatten())*(1/gain)*gain.mean()
        plot_3d_cube(mag_comp, 'o')
    return offset, gain


def sphere_fit(mag, plots=True):
    """
    https://www.st.com/resource/en/design_tip/dm00286302.pdf

    Related reference:
    http://www.secs.oakland.edu/~li4/papers/conf/Cui_PLAN2018.pdf
    """
    x = mag[:,[0]]
    y = mag[:,[1]]
    z = mag[:,[2]]
    D = np.hstack([x**2 + y**2 + z**2, 2*x, 2*y, 2*z])
    v = np.linalg.inv(D.T.dot(D)).dot(D.T.dot(np.ones((mag.shape[0], 1))))
    print(v)
    a,g,h,i = v[:,0]
    offset = np.vstack([g/a, h/a, i/a])
    G = 1 + g**2/a + h**2/a + i**2/a
    gain = np.sqrt(np.vstack([a/G, a/G, a/G]))

    if plots:
        plt.figure()
        plot_3d_cube(mag, '.')
        mag_comp = (mag - offset.flatten())  #/gain.flatten()*gain.mean()
        plot_3d_cube(mag_comp, 'o')
    return offset, gain


def sphere_fit2(mag, plots=True):
    """
    https://www.st.com/resource/en/design_tip/dm00286302.pdf

    Related reference:
    http://www.secs.oakland.edu/~li4/papers/conf/Cui_PLAN2018.pdf
    """
    x = mag[:,[0]]
    y = mag[:,[1]]
    z = mag[:,[2]]
    # (x-x0)**2 + (y-y0)**2 + (z-z0)**2 = r**2
    # x**2 -2*x*x0 + x0**2 + y**2 -2*y*y0 + y0**2 + z**2 -2*z*z0 + z0**2 = r**2

    D = np.hstack([-2*x,-2*y,2*z,np.ones((len(x),1))])
    b = -(x**2 + y**2 + z**2)
    v = np.linalg.inv(D.T.dot(D)).dot(D.T.dot(b))
    print(v)
    x0,y0,z0,a = v[:,0]
    # a = x0**2 + y0**2 +z0**2 - r**2
    # r = sqrt(x0**2 + y0**2 + z0**2 - a)
    r = np.sqrt(x0**2 + y0**2 + z0**2 -a)
    print(f"r={r}")
    offset = np.vstack([x0, y0, z0])
    gain = np.sqrt(np.vstack([r, r, r]))

    if plots:
        plt.figure()
        axes = plot_3d_cube(mag, '.', axis_equal=True)
        mag_comp = (mag - offset.flatten())
        plot_3d_cube(mag_comp, 'o', axis_equal=True, axes=axes)
    return offset, gain


def plot_3d_cube(mag, mrk='.', axis_equal=False, axes=None):
    #plt.figure()
    if axes is None:
        ax1 = plt.subplot(221)
    else:
        ax1, ax2, ax3 = axes
    ax1.plot(mag[:,0], mag[:,1], mrk)
    ax1.grid(True)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    if axis_equal:
        ax1.axis('square')

    if axis_equal:
        if axes is None:
            ax2 = plt.subplot(222)
    else:
        if axes is None:
            ax2 = plt.subplot(222, sharey=ax1)
    ax2.plot(mag[:,2], mag[:,1], mrk)
    ax2.grid(True)
    ax2.set_xlabel('z')
    ax2.set_ylabel('y')
    if axis_equal:
        ax2.axis('square')

    if axis_equal:
        if axes is None:
            ax3 = plt.subplot(223)
    else:
        if axes is None:
            ax3 = plt.subplot(223, sharex=ax1)
    ax3.plot(mag[:,0], mag[:,2], mrk)
    ax3.grid(True)
    ax3.set_xlabel('x')
    ax3.set_ylabel('z')
    if axis_equal:
        ax3.axis('square')
    axes = (ax1,ax2,ax3)
    return axes


def pts_on_sphere(rings=10, n_per_ring=10, r=1):
    results = np.zeros((rings*n_per_ring, 3))
    n = 0
    for phi in np.linspace(0, 2*np.pi, rings):
        for theta in np.linspace(0, np.pi, n_per_ring):
            x = r*np.sin(theta)*np.cos(phi)
            y = r*np.sin(theta)*np.sin(phi)
            z = r*np.cos(theta)
            results[n, :] = [x,y,z]
            n += 1
    return results


def pts_on_ellipsoid(a,b,c,d,e,f,g,h,i, rings=50, n_per_ring=50):
    """
    a*x**2 + b*y**2 + c*z**2 + 2*x*y + 2*x*z + 2*y*z + 2*x +2*y + 2*z = 1
    ell = (a*x**2 + b*y**2 + c*z**2 + 2*d*x*y + 2*e*x*z + 2*f*y*z + 2*g*x +2*h*y + 2*i*z - 1).subs(x, r*sin(theta)*cos(phi)) \
                                                                       .subs(y, r*sin(theta)*sin(phi)) \
                                                                       .subs(z, r*cos(theta))
    solve(ell, r)
    """
    results = np.zeros((rings*n_per_ring*2, 3))
    n = 0
    for phi in np.linspace(0, 2*np.pi, rings):
        for theta in np.linspace(0, np.pi, n_per_ring):
            r1 = (-g*sin(theta)*cos(phi) - h*sin(phi)*sin(theta) - i*cos(theta) + sqrt(a*sin(theta)**2*cos(phi)**2 + b*sin(phi)**2*sin(theta)**2 + c*cos(theta)**2 + 2*d*sin(phi)*sin(theta)**2*cos(phi) + e*(-sin(phi - 2*theta) + sin(phi + 2*theta))/2 + f*(cos(phi - 2*theta) - cos(phi + 2*theta))/2 + g**2*sin(theta)**2*cos(phi)**2 + 2*g*h*sin(phi)*sin(theta)**2*cos(phi) + g*i*(-sin(phi - 2*theta) + sin(phi + 2*theta))/2 + h**2*sin(phi)**2*sin(theta)**2 + h*i*(cos(phi - 2*theta) - cos(phi + 2*theta))/2 + i**2*cos(theta)**2))/(a*sin(theta)**2*cos(phi)**2 + b*sin(phi)**2*sin(theta)**2 + c*cos(theta)**2 + 2*d*sin(phi)*sin(theta)**2*cos(phi) + e*(-sin(phi - 2*theta) + sin(phi + 2*theta))/2 + f*(cos(phi - 2*theta) - cos(phi + 2*theta))/2)
            r2 = -(g*sin(theta)*cos(phi) + h*sin(phi)*sin(theta) + i*cos(theta) + sqrt(a*sin(theta)**2*cos(phi)**2 + b*sin(phi)**2*sin(theta)**2 + c*cos(theta)**2 + 2*d*sin(phi)*sin(theta)**2*cos(phi) - e*(sin(phi - 2*theta) - sin(phi + 2*theta))/2 + f*(cos(phi - 2*theta) - cos(phi + 2*theta))/2 + g**2*sin(theta)**2*cos(phi)**2 + 2*g*h*sin(phi)*sin(theta)**2*cos(phi) - g*i*(sin(phi - 2*theta) - sin(phi + 2*theta))/2 + h**2*sin(phi)**2*sin(theta)**2 + h*i*(cos(phi - 2*theta) - cos(phi + 2*theta))/2 + i**2*cos(theta)**2))/(a*sin(theta)**2*cos(phi)**2 + b*sin(phi)**2*sin(theta)**2 + c*cos(theta)**2 + 2*d*sin(phi)*sin(theta)**2*cos(phi) - e*(sin(phi - 2*theta) - sin(phi + 2*theta))/2 + f*(cos(phi - 2*theta) - cos(phi + 2*theta))/2)
            for r in [r1,r2]:
                if ~np.isnan(r):
                    x = r*np.sin(theta)*np.cos(phi)
                    y = r*np.sin(theta)*np.sin(phi)
                    z = r*np.cos(theta)
                    results[n, :] = [x,y,z]
                    n += 1
    return results[:n,:]


def pts_on_ellipsoid2(x0,y0,z0,a,b,c,d,e,f, rings=50, n_per_ring=50):
    """
    (x-x0)**2/a**2 + (y-y0)**2/b**2 + (z-z0)**2/c**2 + d*x*y + e*x*z + f*z*y = 1
    ell = (x-x0)**2/a**2 + (y-y0)**2/b**2 + (z-z0)**2/c**2 + d*x*y + e*x*z + f*z*y - 1
    ell = expand(ell)
    soln = solve(ell.subs(x, r*sin(theta)*cos(phi)).subs(y, r*sin(theta)*sin(phi)).subs(z, r*cos(theta)), r)
    """
    results = np.zeros((rings*n_per_ring*2, 3))
    n = 0
    for phi in np.linspace(0, 2*np.pi, rings):
        for theta in np.linspace(0, np.pi, n_per_ring):
            r1 = (a**2*b**2*z0*cos(theta) + a**2*c**2*y0*sin(phi)*sin(theta) - a*b*c*sqrt(a**2*b**2*c**2*d*sin(phi)*sin(theta)**2*cos(phi) + a**2*b**2*c**2*e*(-sin(phi - 2*theta) + sin(phi + 2*theta))/4 + a**2*b**2*c**2*f*(cos(phi - 2*theta) - cos(phi + 2*theta))/4 - a**2*b**2*d*z0**2*sin(phi)*sin(theta)**2*cos(phi) - a**2*b**2*e*z0**2*(-sin(phi - 2*theta) + sin(phi + 2*theta))/4 - a**2*b**2*f*z0**2*(cos(phi - 2*theta) - cos(phi + 2*theta))/4 + a**2*b**2*cos(theta)**2 - a**2*c**2*d*y0**2*sin(phi)*sin(theta)**2*cos(phi) - a**2*c**2*e*y0**2*(-sin(phi - 2*theta) + sin(phi + 2*theta))/4 - a**2*c**2*f*y0**2*(cos(phi - 2*theta) - cos(phi + 2*theta))/4 + a**2*c**2*sin(phi)**2*sin(theta)**2 - a**2*y0**2*cos(theta)**2 + a**2*y0*z0*(cos(phi - 2*theta) - cos(phi + 2*theta))/2 - a**2*z0**2*sin(phi)**2*sin(theta)**2 - b**2*c**2*d*x0**2*sin(phi)*sin(theta)**2*cos(phi) - b**2*c**2*e*x0**2*(-sin(phi - 2*theta) + sin(phi + 2*theta))/4 - b**2*c**2*f*x0**2*(cos(phi - 2*theta) - cos(phi + 2*theta))/4 + b**2*c**2*sin(theta)**2*cos(phi)**2 - b**2*x0**2*cos(theta)**2 + b**2*x0*z0*(-sin(phi - 2*theta) + sin(phi + 2*theta))/2 - b**2*z0**2*sin(theta)**2*cos(phi)**2 - c**2*x0**2*sin(phi)**2*sin(theta)**2 + 2*c**2*x0*y0*sin(phi)*sin(theta)**2*cos(phi) - c**2*y0**2*sin(theta)**2*cos(phi)**2) + b**2*c**2*x0*sin(theta)*cos(phi))/(a**2*b**2*c**2*d*sin(phi)*sin(theta)**2*cos(phi) + a**2*b**2*c**2*e*(-sin(phi - 2*theta) + sin(phi + 2*theta))/4 + a**2*b**2*c**2*f*(cos(phi - 2*theta) - cos(phi + 2*theta))/4 + a**2*b**2*cos(theta)**2 + a**2*c**2*sin(phi)**2*sin(theta)**2 + b**2*c**2*sin(theta)**2*cos(phi)**2)
            r2 = (a**2*b**2*z0*cos(theta) + a**2*c**2*y0*sin(phi)*sin(theta) + a*b*c*sqrt(a**2*b**2*c**2*d*sin(phi)*sin(theta)**2*cos(phi) + a**2*b**2*c**2*e*(-sin(phi - 2*theta) + sin(phi + 2*theta))/4 + a**2*b**2*c**2*f*(cos(phi - 2*theta) - cos(phi + 2*theta))/4 - a**2*b**2*d*z0**2*sin(phi)*sin(theta)**2*cos(phi) - a**2*b**2*e*z0**2*(-sin(phi - 2*theta) + sin(phi + 2*theta))/4 - a**2*b**2*f*z0**2*(cos(phi - 2*theta) - cos(phi + 2*theta))/4 + a**2*b**2*cos(theta)**2 - a**2*c**2*d*y0**2*sin(phi)*sin(theta)**2*cos(phi) - a**2*c**2*e*y0**2*(-sin(phi - 2*theta) + sin(phi + 2*theta))/4 - a**2*c**2*f*y0**2*(cos(phi - 2*theta) - cos(phi + 2*theta))/4 + a**2*c**2*sin(phi)**2*sin(theta)**2 - a**2*y0**2*cos(theta)**2 + a**2*y0*z0*(cos(phi - 2*theta) - cos(phi + 2*theta))/2 - a**2*z0**2*sin(phi)**2*sin(theta)**2 - b**2*c**2*d*x0**2*sin(phi)*sin(theta)**2*cos(phi) - b**2*c**2*e*x0**2*(-sin(phi - 2*theta) + sin(phi + 2*theta))/4 - b**2*c**2*f*x0**2*(cos(phi - 2*theta) - cos(phi + 2*theta))/4 + b**2*c**2*sin(theta)**2*cos(phi)**2 - b**2*x0**2*cos(theta)**2 + b**2*x0*z0*(-sin(phi - 2*theta) + sin(phi + 2*theta))/2 - b**2*z0**2*sin(theta)**2*cos(phi)**2 - c**2*x0**2*sin(phi)**2*sin(theta)**2 + 2*c**2*x0*y0*sin(phi)*sin(theta)**2*cos(phi) - c**2*y0**2*sin(theta)**2*cos(phi)**2) + b**2*c**2*x0*sin(theta)*cos(phi))/(a**2*b**2*c**2*d*sin(phi)*sin(theta)**2*cos(phi) + a**2*b**2*c**2*e*(-sin(phi - 2*theta) + sin(phi + 2*theta))/4 + a**2*b**2*c**2*f*(cos(phi - 2*theta) - cos(phi + 2*theta))/4 + a**2*b**2*cos(theta)**2 + a**2*c**2*sin(phi)**2*sin(theta)**2 + b**2*c**2*sin(theta)**2*cos(phi)**2)
            for r in [r1,r2]:
                if ~np.isnan(r):
                    x = r*np.sin(theta)*np.cos(phi)
                    y = r*np.sin(theta)*np.sin(phi)
                    z = r*np.cos(theta)
                    results[n, :] = [x,y,z]
                    n += 1
    return results[:n, :]


def ellipse(a,b,f,p,q,d):
    results = []
    for theta in np.linspace(0, 2*np.pi, 100):
        r1 = (-p*cos(theta) - q*sin(theta) + sqrt(-4*a*d*cos(theta)**2 - 4*b*d*sin(theta)**2 - 4*d*f*sin(2*theta) + p**2*cos(theta)**2 + p*q*sin(2*theta) + q**2*sin(theta)**2))/(2*(a*cos(theta)**2 + b*sin(theta)**2 + f*sin(2*theta)))
        r2 = -(p*cos(theta) + q*sin(theta) + sqrt(-4*a*d*cos(theta)**2 - 4*b*d*sin(theta)**2 - 4*d*f*sin(2*theta) + p**2*cos(theta)**2 + p*q*sin(2*theta) + q**2*sin(theta)**2))/(2*a*cos(theta)**2 + 2*b*sin(theta)**2 + 2*f*sin(2*theta))
        if ~np.isnan(r1):
            results.append((r1*np.cos(theta), r1*sin(theta)))
        if ~np.isnan(r2):
            results.append((r2*np.cos(theta), r2*sin(theta)))

    return np.array(results)


def fit_ellipse(XY):
    x = XY[:,[0]]
    y = XY[:,[1]]
    D = np.hstack([x**2, y**2, 2*x*y, 2*x, 2*y])
    v = np.linalg.inv(D.T.dot(D)).dot(D.T.dot(np.ones((XY.shape[0], 1))))
    #print(v)
    a,b,f,p,q = v[:,0]
    A3 = np.array([[a,f,p],
                   [f,b,q],
                   [p,q,-1]])
    A2 = A3[:2, :2]
    vpq = np.vstack([p,q])
    offset = -np.linalg.inv(A2).dot(vpq)
    T = np.eye(3)
    T[2, :2] = offset[:,0]
    B3 = T.dot(A3).dot(T.T)
    B2 = B3[:2, :2]/-B3[2,2]
    ev, rotM = np.linalg.eig(B2)
    gain = np.sqrt(1/ev)

    return offset, gain, rotM
