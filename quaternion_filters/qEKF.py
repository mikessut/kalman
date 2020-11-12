
import numpy as np
from numpy import sqrt
from .quaternion import Quaternion
import time
from .filter_log import DataLog


class qEKF:

    def __init__(self):

        self.q = np.array([1.0, 0, 0, 0], dtype=float)
        self.w = np.array([0.0, 0, 0], dtype=float)
        self.wb = np.zeros((3,))
        self.ab = np.zeros((3,))

        self.P = np.zeros((13, 13))
        # position relative to north
        self.P[0, 0] = 1*0
        #self.P[3, 3] = 1*0
        #self.P[1:4, 1:4] = np.diag(.01*np.ones((3,)))
        self.P[0, 0] = .3**2  # varies from ~.7 to 1.0
        # ~ 10 deg error in pitch/roll ~ sin(10/2)
        self.P[1, 1] = (.1**2)
        self.P[2, 2] = (.1 ** 2)
        self.P[3, 3] = 1  # 180 deg error for heading

        # Bias errors
        self.P[7:10, 7:10] = np.diag(np.ones((3,)))*(5*np.pi/180)**2
        self.P[10:13, 10:13] = np.diag(np.ones((3,))) * (2) ** 2

        qerr = .001*0
        # assumes constant ang rate; could be wrong by as much as 10deg/sec in .5 sec?
        werr = (10*np.pi/180)**2 / .5
        wb_err = (.1*np.pi/180)**2 / 60
        ab_err = (.1)**2 / 60
        self.Q = np.diag(np.hstack([qerr*np.ones((4,)), werr*np.ones((3,)),
                                    wb_err*np.ones((3,)), ab_err*np.ones((3,))]))

        accel_err = .01
        self.Raccel = np.diag(accel_err*np.ones((3,)))
        gyro_err = .001
        self.Rgyro = np.diag(gyro_err * np.ones((3,)))

        mag_err = .004
        self.Rmag = np.diag(mag_err * np.ones((3,)))

        self.last_update = time.time()
        self.log = DataLog()

    def state_vec(self):
        return np.vstack(np.concatenate([self.q, self.w, self.wb, self.ab]))

    def set_state_vec(self, x):
        q = Quaternion(*x[:4].flatten())
        q.normalize()
        self.q = q.as_ndarray()
        self.w = x[4:7].flatten()
        self.wb = x[7:10].flatten()
        self.ab = x[10:13].flatten()
        self.log.set_state(self.state_vec(), self.P)

    def predict(self, dt=None):

        if dt is None:
            dt = time.time() - self.last_update
            self.last_update = time.time()

        F = self.calc_F(dt)
        self.P = F.dot(self.P).dot(F.T) + self.Q*dt

        q = Quaternion(*self.q) * Quaternion(1, *(.5*self.w*dt))
        q.normalize()
        self.q = q.as_ndarray()
        self.log.predict(dt, self.state_vec(), self.P)

    def update_accel(self, accels):
        self.log.accel(accels)
        q = Quaternion(*self.q)
        g = np.array([0,0,9.8], dtype=float)
        y = accels - ((q.inv() * Quaternion.from_vec(g) * q).as_ndarray()[1:] + self.ab)
        y = np.vstack(y)

        H = self.calc_accel_H()
        S = H.dot(self.P).dot(H.T) + self.Raccel
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        x = self.state_vec() + K.dot(y)

        self.P = (np.eye(13) - K.dot(H)).dot(self.P)
        self.set_state_vec(x)

    def update_gyro(self, gyros):
        self.log.gyro(gyros)
        y = gyros - (self.w + self.wb)
        y = np.vstack(y)

        H = np.zeros((3, 13))
        H[0, 4] = 1
        H[0, 7] = 1
        H[1, 5] = 1
        H[1, 8] = 1
        H[2, 6] = 1
        H[2, 9] = 1

        S = H.dot(self.P).dot(H.T) + self.Rgyro
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        x = self.state_vec() + K.dot(y)

        self.P = (np.eye(13) - K.dot(H)).dot(self.P)
        self.set_state_vec(x)

    def update_mag(self, mags):
        """
        See derive.py for discussion of how this works.
        """
        self.log.mag(mags)
        q = Quaternion(*self.q)
        mag_inertial = (q * Quaternion.from_vec(mags) * q.inv()).as_ndarray()[1:]
        mag_inertial[2] = 0.0
        mag_inertial /= np.linalg.norm(mag_inertial)
        mag_body = (q.inv() * Quaternion.from_vec(mag_inertial) * q).as_ndarray()[1:]

        y = mag_body - (q.inv() * Quaternion.from_vec(np.array([1.0, 0, 0])) * q).as_ndarray()[1:]
        y = np.vstack(y)
        H = self.calc_mag_H()
        S = H.dot(self.P).dot(H.T) + self.Rmag
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        x = self.state_vec() + K.dot(y)
        #import pdb; pdb.set_trace()

        self.P = (np.eye(13) - K.dot(H)).dot(self.P)
        self.set_state_vec(x)

    def quaternion(self):
        return Quaternion(*self.q)

    def __repr__(self):
        return f"{euler_angles[0]:7.1f}{euler_angles[1]:7.1f}{euler_angles[2]:7.1f}  {k.wb[0]*180/np.pi:6.2f}{k.wb[1]*180/np.pi:6.2f}{k.wb[2]*180/np.pi:6.2f} {k.ab[0]:6.2f}{k.ab[1]:6.2f}{k.ab[2]:6.2f} {k.w[0]*180/np.pi:6.2f}{k.w[1]*180/np.pi:6.2f}{k.w[2]*180/np.pi:6.2f}"

    def calc_F(self, dt):
        q0, q1, q2, q3 = self.q
        wx, wy, wz = self.w
        F = np.zeros((13, 13))
        F[0, 0] = 1.00000000000000
        F[0, 1] = -0.5*dt*wx
        F[0, 2] = -0.5*dt*wy
        F[0, 3] = -0.5*dt*wz
        F[0, 4] = -0.5*dt*q1
        F[0, 5] = -0.5*dt*q2
        F[0, 6] = -0.5*dt*q3
        F[1, 0] = 0.5*dt*wx
        F[1, 1] = 1.00000000000000
        F[1, 2] = 0.5*dt*wz
        F[1, 3] = -0.5*dt*wy
        F[1, 4] = 0.5*dt*q0
        F[1, 5] = -0.5*dt*q3
        F[1, 6] = 0.5*dt*q2
        F[2, 0] = 0.5*dt*wy
        F[2, 1] = -0.5*dt*wz
        F[2, 2] = 1.00000000000000
        F[2, 3] = 0.5*dt*wx
        F[2, 4] = 0.5*dt*q3
        F[2, 5] = 0.5*dt*q0
        F[2, 6] = -0.5*dt*q1
        F[3, 0] = 0.5*dt*wz
        F[3, 1] = 0.5*dt*wy
        F[3, 2] = -0.5*dt*wx
        F[3, 3] = 1.00000000000000
        F[3, 4] = -0.5*dt*q2
        F[3, 5] = 0.5*dt*q1
        F[3, 6] = 0.5*dt*q0
        F[4, 4] = 1
        F[5, 5] = 1
        F[6, 6] = 1
        F[7, 7] = 1
        F[8, 8] = 1
        F[9, 9] = 1
        F[10, 10] = 1
        F[11, 11] = 1
        F[12, 12] = 1
        return F

    def calc_accel_H(self):
        q0, q1, q2, q3 = self.q
        wx, wy, wz = self.w
        H = np.zeros((3, 13))
        H[0, 0] = 39.2*q0**2*q2 - 39.2*q0*q1*q3 - 19.6*q2
        H[0, 1] = 39.2*q0*q1*q2 - 39.2*q1**2*q3 + 19.6*q3
        H[0, 2] = 39.2*q0*q2**2 - 19.6*q0 - 39.2*q1*q2*q3
        H[0, 3] = 39.2*q0*q2*q3 - 39.2*q1*q3**2 + 19.6*q1
        H[0, 10] = 1
        H[1, 0] = -39.2*q0**2*q1 - 39.2*q0*q2*q3 + 19.6*q1
        H[1, 1] = -39.2*q0*q1**2 + 19.6*q0 - 39.2*q1*q2*q3
        H[1, 2] = -39.2*q0*q1*q2 - 39.2*q2**2*q3 + 19.6*q3
        H[1, 3] = -39.2*q0*q1*q3 - 39.2*q2*q3**2 + 19.6*q2
        H[1, 11] = 1
        H[2, 0] = -19.6*q0**3 + 19.6*q0*q1**2 + 19.6*q0*q2**2 - 19.6*q0*q3**2 + 19.6*q0
        H[2, 1] = -19.6*q0**2*q1 + 19.6*q1**3 + 19.6*q1*q2**2 - 19.6*q1*q3**2 - 19.6*q1
        H[2, 2] = -19.6*q0**2*q2 + 19.6*q1**2*q2 + 19.6*q2**3 - 19.6*q2*q3**2 - 19.6*q2
        H[2, 3] = -19.6*q0**2*q3 + 19.6*q1**2*q3 + 19.6*q2**2*q3 - 19.6*q3**3 + 19.6*q3

        return H

    def calc_mag_H(self):
        q0, q1, q2, q3 = self.q
        H = np.zeros((3, 13))
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
