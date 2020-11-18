"""
Reference for optimizing:
http://www.anuncommonlab.com/articles/how-kalman-filters-work/part2.html
"""

import numpy as np
from numpy import sqrt
from .quaternion import Quaternion
import time
from .filter_log import DataLog


G = 9.81
KTS2MS = 0.514444
FT2M = 0.3048


class FixedWingEKF:

    def __init__(self):

        self.q = np.array([1.0, 0, 0, 0], dtype=float)
        self.a = np.array([0.0, 0, -G], dtype=float)
        self.w = np.array([0.0, 0, 0], dtype=float)
        self.alt = 0  # not a state, just passing through for display

        self.P = np.zeros((10, 10))

        # self.P[0, 0] = .3**2  # varies from ~.7 to 1.0
        # # ~ 10 deg error in pitch/roll ~ sin(10/2)
        # self.P[1, 1] = (.1**2)
        # self.P[2, 2] = (.1 ** 2)
        # self.P[3, 3] = 1  # 180 deg error for heading
        #self.P[:4, :4] = 10
        # self.P[0, 0] = 1
        # self.P[2, 2] = 1

        # Accel init error
        # use 2 deg error => sin(2deg) = .03
        #self.P[4:7, 4:7] = np.diag(np.ones((3,)))*(.03*G)**2

        # Accel model is coordinated turn assumption. How far from this could we be?
        aerr_x = (.01*G)**2 / 1
        aerr_y = (.1*G)**2 / 1
        aerr_z = (.01*G)**2 / 1
        werr_x = (10*np.pi/180)**2 / 1  # 10 deg / sec
        werr_y = (10*np.pi/180)**2 / 1  # 10 deg / sec
        werr_z = (2*np.pi/180)**2 / 1   # 10 deg / sec
        self.Q = np.diag(np.hstack([np.zeros((4,)),
                                    aerr_x, aerr_y, aerr_z,
                                    werr_x, werr_y, werr_z]))

        accel_err = 1.5
        self.Raccel = np.diag(accel_err*np.ones((3,)))
        gyro_err = .0001
        self.Rgyro = np.diag(gyro_err * np.ones((3,)))

        mag_err = .01
        self.Rmag = np.diag(mag_err * np.ones((2,)))

        self.last_update = time.time()
        self.t0 = time.time()
        self.log = DataLog()

    def setQ_small(self):
        aerr_x = (.01*G)**2 / 1
        aerr_y = (.01*G)**2 / 1
        aerr_z = (.01*G)**2 / 1
        werr_x = (10*np.pi/180)**2 / 1  # 10 deg / sec
        werr_y = (10*np.pi/180)**2 / 1  # 10 deg / sec
        werr_z = (2*np.pi/180)**2 / 1   # 10 deg / sec
        self.Q = np.diag(np.hstack([np.zeros((4,)),
                                    aerr_x, aerr_y, aerr_z,
                                    werr_x, werr_y, werr_z]))

    def state_vec(self):
        return np.vstack(np.concatenate([self.q, self.a, self.w]))

    def set_state_vec(self, x):
        q = Quaternion(*x[:4].flatten())
        q.normalize()
        self.q = q.as_ndarray()
        self.a = x[4:7].flatten()
        self.w = x[7:10].flatten()

        #self.force_P_pos_def_sym()

    def force_P_pos_def_sym(self):
        # Make sure the P is positive definite symmetric -- think that's the right term! :)
        # Maybe only periodically do this?
        i, j = np.tril_indices(10)
        idx = self.P[i, j] < 0
        self.P[i[idx], j[idx]] = 0
        self.P[j, i] = self.P[i, j]
        self.log.set_state(self.state_vec(), self.P)

    def predict(self, dt):

        # Quaternion prediction
        q = Quaternion(*self.q)        
        
        # We are using this updated q. I don't know the pro/con of doing this, but probably
        # isn't the spirit of the EKF. Should try without.

        phi = q.euler_angles()[0]
        v_ac = (q*Quaternion.from_vec([0,1.0,0])*q.inv()).as_ndarray()[1:]
        v_ac[2] = 0
        v_ac /= sqrt(v_ac[0]**2 + v_ac[1]**2)

        ac = v_ac*G*np.tan(phi)
        ag = np.array([0,0,-G])
        a = ag + ac

        ab = (q.inv()*Quaternion.from_vec(a)*q).as_ndarray()[1:]
        self.a = ab

        self.q = q * Quaternion(1, *(.5*self.w*dt))
        self.q.normalize()
        self.q = self.q.as_ndarray()

        # Gyro prediction
        #self.w = (q.inv()*Quaternion.from_vec([0,0,-G/self.tas*np.tan(phi)])*q).as_ndarray()[1:]
        # If we do nothing here, it is the constant w assumption

        F = self.calc_F_air(dt)
        self.P = F.dot(self.P).dot(F.T) + self.Q*dt
        
        #self.force_P_pos_def_sym()
        
        self.log.predict(dt, self.state_vec(), self.P)

    def update_accel(self, accels):
        self.log.accel(accels)

        y = accels - self.a
        y = np.vstack(y)

        H = np.zeros((3, 10))

        H[0, 4] = 1
        H[1, 5] = 1
        H[2, 6] = 1
        S = H.dot(self.P).dot(H.T) + self.Raccel
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        self.Ka = K
        # K = np.zeros((10, 3))
        # K[4:7, 0:3] = np.diag([.009, .05, .001])
        x = self.state_vec() + K.dot(y)

        self.P = (np.eye(10) - K.dot(H)).dot(self.P)
        #A = np.eye(10) - K.dot(H)
        #self.P = A.dot(self.P).dot(A.T) + K.dot(self.Raccel).dot(K.T)  # Joseph form
        self.set_state_vec(x)

    def update_gyro(self, gyros):
        self.log.gyro(gyros)
        y = gyros - self.w
        y = np.vstack(y)

        H = np.zeros((3, 10))
        H[0, 7] = 1
        H[1, 8] = 1
        H[2, 9] = 1

        S = H.dot(self.P).dot(H.T) + self.Rgyro
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        self.Kw = K
        # K = np.zeros((10, 3))
        # K[7:10, :3] = np.diag([.97, .97, .68])
        x = self.state_vec() + K.dot(y)

        self.P = (np.eye(10) - K.dot(H)).dot(self.P)
        #A = np.eye(10) - K.dot(H)
        #self.P = A.dot(self.P).dot(A.T) + K.dot(self.Rgyro).dot(K.T)  # Joseph form
        self.set_state_vec(x)

    def update_mag(self, mags):
        """
        See derive.py for discussion of how this works.
        """
        self.log.mag(mags)
        q = Quaternion(*self.q)
        roll, pitch, heading = q.euler_angles()
        
        # Undo roll and pitch from mag sensor, then calculate heading vector
        qtmp = Quaternion.axis_angle(np.array([1.0, 0, 0]), roll)*Quaternion.axis_angle(np.array([0, 1.0, 0]), pitch)

        #sensor_heading = mags[:2]
        # Note: we're only taking the first two components
        sensor_heading = (qtmp * Quaternion.from_vec(mags) * qtmp.inv()).as_ndarray()[1:3]
        sensor_heading /= np.linalg.norm(sensor_heading)
        sensor_heading[1] *= -1

        h = np.vstack([np.cos(heading), np.sin(heading)])
        y = np.vstack(sensor_heading) - h
        # print("sensor_head, state_heading", sensor_heading, h.flatten() )
        
        H = self.calc_mag_H()
        S = H.dot(self.P).dot(H.T) + self.Rmag
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        self.Km = K
        x = self.state_vec() + K.dot(y)

        self.P = (np.eye(10) - K.dot(H)).dot(self.P)
        #A = np.eye(10) - K.dot(H)
        #self.P = A.dot(self.P).dot(A.T) + K.dot(self.Rmag).dot(K.T)  # Joseph form
        self.set_state_vec(x)        

    def update_bearing(self, bearing):
        """
        bearing uses same definition of psi; inertial x axis is North
        """
        q0, q1, q2, q3 = self.q
        v_head = np.array([-2*q2**2 - 2*q3**2 + 1, 2*q0*q3 + 2*q1*q2])
        v_bearing = np.array([np.cos(bearing), np.sin(bearing)])
        y = np.vstack(v_bearing) - np.vstack(v_head)

        H = np.zeros((2, 11), dtype=float)
        H[0, 2] = -4*q2
        H[0, 3] = -4*q3
        H[1, 0] = 2*q3
        H[1, 1] = 2*q2
        H[1, 2] = 2*q1
        H[1, 3] = 2*q0

        S = H.dot(self.P).dot(H.T) + (1*np.pi/180)**2
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        x = self.state_vec() + K.dot(y)

        self.P = (np.eye(11) - K.dot(H)).dot(self.P)
        self.set_state_vec(x)

    def quaternion(self):
        return Quaternion(*self.q)

    def eulers(self):
        """
        returns: roll, pitch, heading
        """
        a, b, c, d = self.q
        phi = np.arctan2(2*(a*b + c*d),
                            1-2*(b**2 + c**2))
        theta = np.arcsin(2*(a*c - d*b))
        psi = np.arctan2(2*(a*d+b*c),
                            1-2*(c**2+d**2))
        return np.array([phi, theta, psi])

    def turn_rate(self):
        q = self.quaternion()
        return (q*Quaternion.from_vec(self.w)*q.inv()).as_ndarray()[3]

    def __repr__(self):
        euler_angles = self.quaternion().euler_angles()*180/np.pi
        return f"{time.time()-self.t0:5.0f}{euler_angles[0]:7.1f}{euler_angles[1]:7.1f}{euler_angles[2]:7.1f} {self.w[0]*180/np.pi:6.2f}{self.w[1]*180/np.pi:6.2f}{self.w[2]*180/np.pi:6.2f} {self.tas/KTS2MS:.1f} {self.a[0]/G:6.2f}{self.a[1]/G:6.2f}{self.a[2]/G:6.2f}"

    def calc_mag_H(self):
        q0, q1, q2, q3 = self.q
        H = np.zeros((2,10))        
        x0 = 2*q3
        x1 = 2*q0
        x2 = 2*q1
        x3 = q2*x2 + q3*x1
        x4 = -2*q2**2 - 2*q3**2 + 1
        x5 = x3**2
        x6 = x4**2 + x5
        x7 = x6**(-3/2)
        x8 = x4*x7
        x9 = x3*x8
        x10 = 2*q2
        x11 = 1/sqrt(x6)
        x12 = 4*q2
        x13 = x12*x4 - x2*x3
        x14 = 4*q3
        x15 = -x1*x3 + x14*x4
        x16 = 2*x11
        x17 = x5*x7
        x18 = x3*x7
        H[0, 0] = -x0*x9
        H[0, 1] = -x10*x9
        H[0, 2] = -x11*x12 + x13*x8
        H[0, 3] = -x11*x14 + x15*x8
        H[1, 0] = q3*x16 - x0*x17
        H[1, 1] = q2*x16 - x10*x17
        H[1, 2] = x11*x2 + x13*x18
        H[1, 3] = x1*x11 + x15*x18
        return H

    def calc_mag_H_old(self):
        q0, q1, q2, q3 = self.q
        H = np.zeros((3,10))
        H[0, 0] = -2.0*q0**3 - 2.0*q0*q1**2 + 2.0*q0*q2**2 + 2.0*q0*q3**2 + 2.0*q0
        H[0, 1] = -2.0*q0**2*q1 - 2.0*q1**3 + 2.0*q1*q2**2 + 2.0*q1*q3**2 + 2.0*q1
        H[0, 2] = -2.0*q0**2*q2 - 2.0*q1**2*q2 + 2.0*q2**3 + 2.0*q2*q3**2 - 2.0*q2
        H[0, 3] = -2.0*q0**2*q3 - 2.0*q1**2*q3 + 2.0*q2**2*q3 + 2.0*q3**3 - 2.0*q3
        H[1, 0] = 4.0*q0**2*q3 - 4.0*q0*q1*q2 - 2.0*q3
        H[1, 1] = 4.0*q0*q1*q3 - 4.0*q1**2*q2 + 2.0*q2
        H[1, 2] = 4.0*q0*q2*q3 - 4.0*q1*q2**2 + 2.0*q1
        H[1, 3] = 4.0*q0*q3**2 - 2.0*q0 - 4.0*q1*q2*q3
        H[2, 0] = -4.0*q0**2*q2 - 4.0*q0*q1*q3 + 2.0*q2
        H[2, 1] = -4.0*q0*q1*q2 - 4.0*q1**2*q3 + 2.0*q3
        H[2, 2] = -4.0*q0*q2**2 + 2.0*q0 - 4.0*q1*q2*q3
        H[2, 3] = -4.0*q0*q2*q3 - 4.0*q1*q3**2 + 2.0*q1
        return H

    def calc_F_air(self, dt):
        return self.calc_F_air_const_w(dt)

    def calc_F_air_const_w(self, dt):
        """
        Jacobian with wpredict: [wx,wy,wz]
        """
        q0, q1, q2, q3 = self.q
        wx, wy, wz = self.w
        F = np.zeros((10, 10))
        x0 = dt/2
        x1 = wx*x0
        x2 = -x1
        x3 = wy*x0
        x4 = -x3
        x5 = wz*x0
        x6 = -x5
        x7 = q1*x0
        x8 = -x7
        x9 = q2*x0
        x10 = -x9
        x11 = q3*x0
        x12 = -x11
        x13 = q0*x0
        x14 = 1.0*q0**2
        x15 = q1**2
        x16 = 1.0*x15
        x17 = q2**2
        x18 = 1.0*x17
        x19 = 1.0*q3**2
        x20 = x14 - x16 + x18 - x19
        x21 = q1*q2
        x22 = q0*q3
        x23 = x21 - x22
        x24 = 0.25*x20**2 + x23**2
        x25 = 1/sqrt(x24)
        x26 = -2*x15 - 2*x17 + 1
        x27 = 1/x26
        x28 = x25*x27
        x29 = 1.0*x28
        x30 = x20*x29
        x31 = G*q3
        x32 = q1*x31
        x33 = x30*x32
        x34 = G*q0
        x35 = 2*q0*q1 + 2*q2*q3
        x36 = 2.0*x21 - 2.0*x22
        x37 = x35*x36
        x38 = 0.5*x37
        x39 = 0.5*x20
        x40 = x27/x24**(3/2)
        x41 = x40*(-q0*x39 + q3*x23)
        x42 = x38*x41
        x43 = x35*x39
        x44 = x41*x43
        x45 = G*x28
        x46 = x38*x45
        x47 = q1*x34
        x48 = x29*x36
        x49 = x46 + x47*x48
        x50 = x31*x44 + x33 + x34*x42 + x49
        x51 = x16*x45
        x52 = G*q2
        x53 = q1*x52
        x54 = x30*x53
        x55 = G*q1
        x56 = x29*x35
        x57 = q0*x52
        x58 = x56*x57
        x59 = x32*x56
        x60 = x58 - x59
        x61 = x36*x51 + x42*x55 + x44*x52 + x54 + x60
        x62 = x32*x48
        x63 = x35*x45
        x64 = x14*x63
        x65 = x19*x63
        x66 = x30*x47
        x67 = x39*x63
        x68 = -x31*x42 + x34*x44 - x62 + x64 + x65 + x66 + x67
        x69 = x47*x56
        x70 = q3*x52
        x71 = x56*x70
        x72 = -G - x69 - x71
        x73 = x48*x53 + x72
        x74 = -x20*x51 + x42*x52 - x44*x55 + x73
        x75 = x28*x38
        x76 = x34*x75
        x77 = x28*x43
        x78 = x31*x77
        x79 = x52 + x76 + x78
        x80 = x14*x45
        x81 = x40*(q1*x39 - q2*x23)
        x82 = x38*x81
        x83 = x43*x81
        x84 = 2.0*x25/x26**2
        x85 = x47*x84
        x86 = x20*x35
        x87 = x84*x86
        x88 = 1.0*x22*x45
        x89 = x20*x88 + x60
        x90 = x31*x83 + x32*x87 + x34*x82 + x36*x80 + x37*x85 + x89
        x91 = x30*x57
        x92 = G*x84
        x93 = x15*x92
        x94 = x53*x87
        x95 = x37*x93 + x49 + x52*x83 + x55*x82 + x91 + x94
        x96 = x36*x88
        x97 = x37*x84
        x98 = x20*x80 - x31*x82 - x32*x97 + x34*x83 + x72 + x85*x86 - x96
        x99 = x48*x57
        x100 = x18*x63 + x35*x51 + x53*x97
        x101 = x100 + x52*x82 - x55*x83 - x66 - x67 - x86*x93 + x99
        x102 = -x31 + x52*x77 + x55*x75
        x103 = x52*x75
        x104 = x55*x77
        x105 = x48*x70
        x106 = x17*x92
        x107 = x40*(-q1*x23 - q2*x39)
        x108 = x107*x38
        x109 = x107*x43
        x110 = x105 + x106*x37 + x108*x52 - x109*x55 - x33 + x46 - x94
        x111 = x19*x45
        x112 = -x108*x31 + x109*x34 - x111*x36 + x57*x87 - x70*x97 + x89
        x113 = G + x108*x34 + x109*x31 + x111*x20 + x57*x97 + x69 + x70*x87 + x71 + x96
        x114 = x30*x70 + x67
        x115 = x100 + x106*x86 + x108*x55 + x109*x52 + x114 + x62
        x116 = x40*(q0*x23 + q3*x39)
        x117 = x116*x38
        x118 = x116*x43
        x119 = x114 + x117*x34 + x118*x31 - x64 - x65 + x99
        x120 = x18*x45
        x121 = x117*x55 + x118*x52 + x120*x20 + x73
        x122 = -x105 - x117*x31 + x118*x34 - x46 + x91
        x123 = x117*x52 - x118*x55 + x120*x36 - x54 - x58 + x59
        x124 = x34*x77
        x125 = x31*x75
        x126 = x124 - x125 - x55
        x127 = x103 - x104 - x34
        F[0, 0] = 1
        F[0, 1] = x2
        F[0, 2] = x4
        F[0, 3] = x6
        F[0, 7] = x8
        F[0, 8] = x10
        F[0, 9] = x12
        F[1, 0] = x1
        F[1, 1] = 1
        F[1, 2] = x5
        F[1, 3] = x4
        F[1, 7] = x13
        F[1, 8] = x12
        F[1, 9] = x9
        F[2, 0] = x3
        F[2, 1] = x6
        F[2, 2] = 1
        F[2, 3] = x1
        F[2, 7] = x11
        F[2, 8] = x13
        F[2, 9] = x8
        F[3, 0] = x5
        F[3, 1] = x3
        F[3, 2] = x2
        F[3, 3] = 1
        F[3, 7] = x10
        F[3, 8] = x7
        F[3, 9] = x13
        F[4, 0] = q0*x50 + q1*x61 - q2*x74 + q3*x68 + x79
        F[4, 1] = q0*x90 + q1*x95 - q2*x101 + q3*x98 + x102
        F[4, 2] = q0*x113 + q1*x115 - q2*x110 + q3*x112 - x103 + x104 + x34
        F[4, 3] = q0*x119 + q1*x121 - q2*x123 + q3*x122 + x126
        F[5, 0] = q0*x68 + q1*x74 + q2*x61 - q3*x50 + x126
        F[5, 1] = q0*x98 + q1*x101 + q2*x95 - q3*x90 + x127
        F[5, 2] = q0*x112 + q1*x110 + q2*x115 - q3*x113 + x102
        F[5, 3] = q0*x122 + q1*x123 + q2*x121 - q3*x119 - x52 - x76 - x78
        F[6, 0] = q0*x74 - q1*x68 + q2*x50 + q3*x61 + x127
        F[6, 1] = q0*x101 - q1*x98 + q2*x90 + q3*x95 - x124 + x125 + x55
        F[6, 2] = q0*x110 - q1*x112 + q2*x113 + q3*x115 + x79
        F[6, 3] = q0*x123 - q1*x122 + q2*x119 + q3*x121 + x102
        F[7, 7] = 1
        F[8, 8] = 1
        F[9, 9] = 1
        return F
