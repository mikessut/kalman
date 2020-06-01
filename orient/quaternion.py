
import numpy as np
from scipy.optimize import minimize

"""
https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Using_quaternion_as_rotations
https://en.wikipedia.org/wiki/Quaternion#Hamilton_product

Can q be solved/fit to two vectors?
https://math.stackexchange.com/questions/2910115/solve-for-rotation-quaternion-that-rotates-x-into-y

Example:
q = Quaternion.axis_angle(np.array([0, 0, 1.0]), np.pi/2)
p = Quaternion.from_point(np.array([1.0, 0, 0]))

p2 = q * p * q.inv()

porig = q.inv() * p2 *q
"""

class Quaternion:

    def __init__(self, a, b, c, d):
        """
        Quaternion as a + i*b + j*c + k*d
        :param a:
        :param b:
        :param c:
        :param d:
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    @staticmethod
    def axis_angle(axis: np.ndarray, angle: float):
        """

        :param axis: 3 long numpy array of axis of rotation. It will be normalized if it isn't already.
        :param angle: angle of ration in radians
        :return:
        """
        axis /= np.linalg.norm(axis)
        axis_sin = axis * np.sin(angle/2)
        return Quaternion(np.cos(angle/2), axis_sin[0], axis_sin[1], axis_sin[2])

    @staticmethod
    def from_vec(pt: np.ndarray):
        return Quaternion(0.0, pt[0], pt[1], pt[2])

    def __mul__(self, other):
        """
        https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
        :param other:
        :return:
        """
        return Quaternion(self.a*other.a - self.b*other.b - self.c*other.c - self.d*other.d,
                          self.a*other.b + self.b*other.a + self.c*other.d - self.d*other.c,
                          self.a*other.c - self.b*other.d + self.c*other.a + self.d*other.b,
                          self.a*other.d + self.b*other.c - self.c*other.b + self.d*other.a)

    def inv(self):
        mag = self.a**2 + self.b**2 + self.c**2 + self.d**2
        return Quaternion(self.a / mag,
                          -self.b/mag,
                          -self.c/mag,
                          -self.d/mag)

    def norm(self):
        return np.sqrt(self.a**2 + self.b**2 + self.c**2 + self.d**2)

    def conj(self):
        return Quaternion(self.a, -self.b, -self.c, -self.d)

    def normalize(self):
        n = self.norm()
        self.a /= n
        self.b /= n
        self.c /= n
        self.d /= n

    def get_axis_angle(self):
        """
        Took from: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Using_quaternion_as_rotations
        ... recovering axis angle
        :return:
        """
        #angle = np.arccos(self.a)*2
        angle = 2*np.arctan2(np.linalg.norm([self.b, self.c, self.d]), self.a)
        axis = np.array([self.b, self.c, self.d])
        axis /= np.linalg.norm(axis)
        return axis, angle

    def euler_angles(self):
        """
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        phi: bank about x axis
        theta: rotation about y
        psi: heading
        :return:
        """
        phi = np.arctan2(2*(self.a*self.b + self.c*self.d),
                         1-2*(self.b**2 + self.c**2))
        theta = np.arcsin(2*(self.a*self.c - self.d*self.b))
        psi = np.arctan2(2*(self.a*self.d+self.b*self.c),
                         1-2*(self.c**2+self.d**2))
        return np.array([phi, theta, psi])

    def as_ndarray(self):
        return np.array([self.a, self.b, self.c, self.d])

    def __getitem__(self, item):
        return [self.a, self.b, self.c, self.d][item]

    def __repr__(self):
        return f"{self.a} + {self.b} * i + {self.c} * j + {self.d} * k"


def fit_quaternion(inertial, body, q0=Quaternion(1.0, 0.0, 0.0, 0.0)):

    def func(X):
        q = Quaternion(X[0], X[1], X[2], X[3])
        q.normalize()
        p = Quaternion.from_vec(inertial)
        p2 = q.inv() * p * q

        p2 = np.array([p2[1], p2[2], p2[3]])
        return sum((p2 - body) ** 2)

    def constraint(X):
        q = Quaternion(X[0], X[1], X[2], X[3])
        return 1 - q.norm()

    x = minimize(func, [q0[0], q0[1], q0[2], q0[3]], constraints={'type': 'eq', 'fun': constraint}, tol=.01)
    q = Quaternion(*x.x)
    q.normalize()
    #import pdb; pdb.set_trace()
    return q, x.fun


def fit_small_q(inertial, body, q0: Quaternion):

    def func(X):
        dq = Quaternion(1, X[0], X[1], X[2])
        q = dq * q0

        v2 = (q.inv() * Quaternion.from_vec(inertial) * q).as_ndarray()[1:]
        return sum((v2 - body)**2)

    x = minimize(func, [0,0,0])
    q = Quaternion(1, *x.x) * q0
    q.normalize()
    return q, x.fun


def example():
    input = np.array([1.0, 0, 0])
    #output = np.array([np.cos(10*np.pi/180), np.sin(10*np.pi/180), 0])
    output = np.array([.9, .1, 0])

    x = fit_quaternion(input, output)
    return x


def rot_between_q():
    q1 = Quaternion(1, 0, 0, 0)
    ang = 10*np.pi/180
    q2 = Quaternion(np.cos(ang/2), 0, 0, 1)

    del_q = q2 * q1.conj()
    return del_q