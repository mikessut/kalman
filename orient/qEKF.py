
import numpy as np
from numpy import sqrt
from quaternion import Quaternion
import time


class qEKF:

    def __init__(self):

        self.q = np.array([1.0, 0, 0, 0], dtype=float)
        self.w = np.array([0.0, 0, 0], dtype=float)

        self.P = np.zeros((7, 7))
        qerr = .001
        werr = .1
        self.Q = np.diag(np.hstack([qerr*np.ones((4,)), werr*np.ones((3,))]))

        accel_err = 1
        self.Raccel = np.diag(accel_err*np.ones((3,)))
        gyro_err = 1
        self.Rgyro = np.diag(gyro_err * np.ones((3,)))

        self.last_update = time.clock()

    def predict(self, dt=None):

        if dt is None:
            dt = time.clock() - self.last_update
            self.last_update = time.clock()

        F = self.calc_F(dt)
        self.P = F.dot(self.P).dot(F.T) + self.Q

        q = Quaternion(1, *(.5*self.w*dt)) * Quaternion(*self.q)
        q.normalize()
        self.q = q.as_ndarray()

    def update_accel(self, accels):
        q = Quaternion(*self.q)
        g = np.array([0,0,9.8], dtype=float)
        y = accels - (q.inv() * Quaternion.from_vec(g) * q).as_ndarray()[1:]
        y = np.vstack(y)

        H = self.calc_accel_H()
        S = H.dot(self.P).dot(H.T) + self.Raccel
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        x = np.vstack(np.concatenate([self.q, self.w])) + K.dot(y)
        q0, q1, q2, q3, wx, wy, wz = x.flatten()
        q = Quaternion(q0,q1,q2,q3)
        q.normalize()
        self.q = q.as_ndarray()
        self.w = np.array([wx,wy,wz])

        self.P = (np.eye(7) - K.dot(H)).dot(self.P)

    def update_gyro(self, gyros):
        y = gyros - self.w
        y = np.vstack(y)

        H = np.zeros((3, 7))
        H[0, 4] = 1
        H[1, 5] = 1
        H[2, 6] = 1

        S = H.dot(self.P).dot(H.T) + self.Rgyro
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        x = np.vstack(np.concatenate([self.q, self.w])) + K.dot(y)
        q0, q1, q2, q3, wx, wy, wz = x.flatten()
        q = Quaternion(q0, q1, q2, q3)
        q.normalize()
        self.q = q.as_ndarray()
        self.w = np.array([wx, wy, wz])

        self.P = (np.eye(7) - K.dot(H)).dot(self.P)

    def quaternion(self):
        return Quaternion(*self.q)

    def calc_F(self, dt):
        q0, q1, q2, q3 = self.q
        wx, wy, wz = self.w
        F = np.zeros((7, 7))
        F[0, 0] = (-0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) * (
                    0.5 * dt * q1 * wx + 0.5 * dt * q2 * wy + 0.5 * dt * q3 * wz - 0.5 * dt * wx * (
                        0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) - 0.5 * dt * wy * (
                                0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) - 0.5 * dt * wz * (
                                0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) - q0) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2) + 1 / sqrt(
            (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                        0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                        0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                        -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2)
        F[0, 1] = -0.5 * dt * wx / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                    0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                    -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) * (
                              -0.5 * dt * q0 * wx + 0.5 * dt * q2 * wz - 0.5 * dt * q3 * wy + 0.5 * dt * wx * (
                                  -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) + 0.5 * dt * wy * (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) - 0.5 * dt * wz * (
                                          0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) - q1) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[0, 2] = -0.5 * dt * wy / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                    0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                    -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) * (
                              -0.5 * dt * q0 * wy - 0.5 * dt * q1 * wz + 0.5 * dt * q3 * wx - 0.5 * dt * wx * (
                                  0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) + 0.5 * dt * wy * (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) + 0.5 * dt * wz * (
                                          0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) - q2) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[0, 3] = -0.5 * dt * wz / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                    0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                    -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) * (
                              -0.5 * dt * q0 * wz + 0.5 * dt * q1 * wy - 0.5 * dt * q2 * wx + 0.5 * dt * wx * (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) - 0.5 * dt * wy * (
                                          0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) + 0.5 * dt * wz * (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) - q3) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[0, 4] = -0.5 * dt * q1 / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                    0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                    -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              -0.5 * dt * q0 * (
                                  0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) + 0.5 * dt * q1 * (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) - 0.5 * dt * q2 * (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) + 0.5 * dt * q3 * (
                                          0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2)) * (
                              -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[0, 5] = -0.5 * dt * q2 / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                    0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                    -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              -0.5 * dt * q0 * (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) + 0.5 * dt * q1 * (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) + 0.5 * dt * q2 * (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) - 0.5 * dt * q3 * (
                                          0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1)) * (
                              -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[0, 6] = -0.5 * dt * q3 / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                    0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                    -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              -0.5 * dt * q0 * (
                                  0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) - 0.5 * dt * q1 * (
                                          0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) + 0.5 * dt * q2 * (
                                          0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) + 0.5 * dt * q3 * (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0)) * (
                              -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[1, 0] = 0.5 * dt * wx / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                   0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                   -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) * (
                              0.5 * dt * q1 * wx + 0.5 * dt * q2 * wy + 0.5 * dt * q3 * wz - 0.5 * dt * wx * (
                                  0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) - 0.5 * dt * wy * (
                                          0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) - 0.5 * dt * wz * (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) - q0) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[1, 1] = (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) * (
                    -0.5 * dt * q0 * wx + 0.5 * dt * q2 * wz - 0.5 * dt * q3 * wy + 0.5 * dt * wx * (
                        -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) + 0.5 * dt * wy * (
                                0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) - 0.5 * dt * wz * (
                                0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) - q1) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2) + 1 / sqrt(
            (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                        0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                        0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                        -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2)
        F[1, 2] = -0.5 * dt * wz / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                    0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                    -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) * (
                              -0.5 * dt * q0 * wy - 0.5 * dt * q1 * wz + 0.5 * dt * q3 * wx - 0.5 * dt * wx * (
                                  0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) + 0.5 * dt * wy * (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) + 0.5 * dt * wz * (
                                          0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) - q2) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[1, 3] = 0.5 * dt * wy / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                   0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                   -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) * (
                              -0.5 * dt * q0 * wz + 0.5 * dt * q1 * wy - 0.5 * dt * q2 * wx + 0.5 * dt * wx * (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) - 0.5 * dt * wy * (
                                          0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) + 0.5 * dt * wz * (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) - q3) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[1, 4] = 0.5 * dt * q0 / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                   0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                   -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) * (-0.5 * dt * q0 * (
                    0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) + 0.5 * dt * q1 * (
                                                                                                                -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) - 0.5 * dt * q2 * (
                                                                                                                0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) + 0.5 * dt * q3 * (
                                                                                                                0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2)) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[1, 5] = 0.5 * dt * q3 / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                   0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                   -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) * (-0.5 * dt * q0 * (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) + 0.5 * dt * q1 * (
                                                                                                                0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) + 0.5 * dt * q2 * (
                                                                                                                -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) - 0.5 * dt * q3 * (
                                                                                                                0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1)) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[1, 6] = -0.5 * dt * q2 / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                    0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                    -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) * (-0.5 * dt * q0 * (
                    0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) - 0.5 * dt * q1 * (
                                                                                                                0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) + 0.5 * dt * q2 * (
                                                                                                                0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) + 0.5 * dt * q3 * (
                                                                                                                -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0)) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[2, 0] = 0.5 * dt * wy / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                   0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                   -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) * (
                              0.5 * dt * q1 * wx + 0.5 * dt * q2 * wy + 0.5 * dt * q3 * wz - 0.5 * dt * wx * (
                                  0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) - 0.5 * dt * wy * (
                                          0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) - 0.5 * dt * wz * (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) - q0) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[2, 1] = 0.5 * dt * wz / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                   0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                   -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) * (
                              -0.5 * dt * q0 * wx + 0.5 * dt * q2 * wz - 0.5 * dt * q3 * wy + 0.5 * dt * wx * (
                                  -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) + 0.5 * dt * wy * (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) - 0.5 * dt * wz * (
                                          0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) - q1) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[2, 2] = (0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) * (
                    -0.5 * dt * q0 * wy - 0.5 * dt * q1 * wz + 0.5 * dt * q3 * wx - 0.5 * dt * wx * (
                        0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) + 0.5 * dt * wy * (
                                -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) + 0.5 * dt * wz * (
                                0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) - q2) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2) + 1 / sqrt(
            (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                        0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                        0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                        -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2)
        F[2, 3] = -0.5 * dt * wx / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                    0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                    -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) * (
                              -0.5 * dt * q0 * wz + 0.5 * dt * q1 * wy - 0.5 * dt * q2 * wx + 0.5 * dt * wx * (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) - 0.5 * dt * wy * (
                                          0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) + 0.5 * dt * wz * (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) - q3) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[2, 4] = -0.5 * dt * q3 / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                    0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                    -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) * (-0.5 * dt * q0 * (
                    0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) + 0.5 * dt * q1 * (
                                                                                                                -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) - 0.5 * dt * q2 * (
                                                                                                                0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) + 0.5 * dt * q3 * (
                                                                                                                0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2)) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[2, 5] = 0.5 * dt * q0 / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                   0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                   -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) * (-0.5 * dt * q0 * (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) + 0.5 * dt * q1 * (
                                                                                                                0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) + 0.5 * dt * q2 * (
                                                                                                                -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) - 0.5 * dt * q3 * (
                                                                                                                0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1)) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[2, 6] = 0.5 * dt * q1 / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                   0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                   -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) * (-0.5 * dt * q0 * (
                    0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) - 0.5 * dt * q1 * (
                                                                                                                0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) + 0.5 * dt * q2 * (
                                                                                                                0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) + 0.5 * dt * q3 * (
                                                                                                                -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0)) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[3, 0] = 0.5 * dt * wz / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                   0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                   -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) * (
                              0.5 * dt * q1 * wx + 0.5 * dt * q2 * wy + 0.5 * dt * q3 * wz - 0.5 * dt * wx * (
                                  0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) - 0.5 * dt * wy * (
                                          0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) - 0.5 * dt * wz * (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) - q0) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[3, 1] = -0.5 * dt * wy / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                    0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                    -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) * (
                              -0.5 * dt * q0 * wx + 0.5 * dt * q2 * wz - 0.5 * dt * q3 * wy + 0.5 * dt * wx * (
                                  -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) + 0.5 * dt * wy * (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) - 0.5 * dt * wz * (
                                          0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) - q1) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[3, 2] = 0.5 * dt * wx / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                   0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                   -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) * (
                              -0.5 * dt * q0 * wy - 0.5 * dt * q1 * wz + 0.5 * dt * q3 * wx - 0.5 * dt * wx * (
                                  0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) + 0.5 * dt * wy * (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) + 0.5 * dt * wz * (
                                          0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) - q2) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[3, 3] = (0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) * (
                    -0.5 * dt * q0 * wz + 0.5 * dt * q1 * wy - 0.5 * dt * q2 * wx + 0.5 * dt * wx * (
                        0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) - 0.5 * dt * wy * (
                                0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) + 0.5 * dt * wz * (
                                -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) - q3) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2) + 1 / sqrt(
            (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                        0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                        0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                        -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2)
        F[3, 4] = 0.5 * dt * q2 / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                   0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                   -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) * (-0.5 * dt * q0 * (
                    0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) + 0.5 * dt * q1 * (
                                                                                                                -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) - 0.5 * dt * q2 * (
                                                                                                                0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) + 0.5 * dt * q3 * (
                                                                                                                0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2)) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[3, 5] = -0.5 * dt * q1 / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                    0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                    -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) * (-0.5 * dt * q0 * (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) + 0.5 * dt * q1 * (
                                                                                                                0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) + 0.5 * dt * q2 * (
                                                                                                                -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) - 0.5 * dt * q3 * (
                                                                                                                0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1)) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[3, 6] = 0.5 * dt * q0 / sqrt((0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                    0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                                   0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                                   -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) + (
                              0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) * (-0.5 * dt * q0 * (
                    0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) - 0.5 * dt * q1 * (
                                                                                                                0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) + 0.5 * dt * q2 * (
                                                                                                                0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) + 0.5 * dt * q3 * (
                                                                                                                -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0)) / (
                              (0.5 * dt * q0 * wx - 0.5 * dt * q2 * wz + 0.5 * dt * q3 * wy + q1) ** 2 + (
                                  0.5 * dt * q0 * wy + 0.5 * dt * q1 * wz - 0.5 * dt * q3 * wx + q2) ** 2 + (
                                          0.5 * dt * q0 * wz - 0.5 * dt * q1 * wy + 0.5 * dt * q2 * wx + q3) ** 2 + (
                                          -0.5 * dt * q1 * wx - 0.5 * dt * q2 * wy - 0.5 * dt * q3 * wz + q0) ** 2) ** (
                              3 / 2)
        F[4, 4] = 1
        F[5, 5] = 1
        F[6, 6] = 1
        return F

    def calc_accel_H(self):
        q0, q1, q2, q3 = self.q
        wx, wy, wz = self.w
        H = np.zeros((3, 7))
        H[0, 0] = 4.0 * q0 ** 2 * q2 / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 - 4.0 * q0 * q1 * q3 / (
                    q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 - 2.0 * q2 / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
        H[0, 1] = 4.0 * q0 * q1 * q2 / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 - 4.0 * q1 ** 2 * q3 / (
                    q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 + 2.0 * q3 / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
        H[0, 2] = 4.0 * q0 * q2 ** 2 / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 - 2.0 * q0 / (
                    q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) - 4.0 * q1 * q2 * q3 / (
                              q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2
        H[0, 3] = 4.0 * q0 * q2 * q3 / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 - 4.0 * q1 * q3 ** 2 / (
                    q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 + 2.0 * q1 / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
        H[1, 0] = -4.0 * q0 ** 2 * q1 / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 - 4.0 * q0 * q2 * q3 / (
                    q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 + 2.0 * q1 / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
        H[1, 1] = -4.0 * q0 * q1 ** 2 / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 + 2.0 * q0 / (
                    q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) - 4.0 * q1 * q2 * q3 / (
                              q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2
        H[1, 2] = -4.0 * q0 * q1 * q2 / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 - 4.0 * q2 ** 2 * q3 / (
                    q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 + 2.0 * q3 / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
        H[1, 3] = -4.0 * q0 * q1 * q3 / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 - 4.0 * q2 * q3 ** 2 / (
                    q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 + 2.0 * q2 / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
        H[2, 0] = -2.0 * q0 ** 3 / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 + 2.0 * q0 * q1 ** 2 / (
                    q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 + 2.0 * q0 * q2 ** 2 / (
                              q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 - 2.0 * q0 * q3 ** 2 / (
                              q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 + 2.0 * q0 / (
                              q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
        H[2, 1] = -2.0 * q0 ** 2 * q1 / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 + 2.0 * q1 ** 3 / (
                    q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 + 2.0 * q1 * q2 ** 2 / (
                              q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 - 2.0 * q1 * q3 ** 2 / (
                              q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 - 2.0 * q1 / (
                              q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
        H[2, 2] = -2.0 * q0 ** 2 * q2 / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 + 2.0 * q1 ** 2 * q2 / (
                    q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 + 2.0 * q2 ** 3 / (
                              q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 - 2.0 * q2 * q3 ** 2 / (
                              q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 - 2.0 * q2 / (
                              q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
        H[2, 3] = -2.0 * q0 ** 2 * q3 / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 + 2.0 * q1 ** 2 * q3 / (
                    q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 + 2.0 * q2 ** 2 * q3 / (
                              q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 - 2.0 * q3 ** 3 / (
                              q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 2 + 2.0 * q3 / (
                              q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
        return H