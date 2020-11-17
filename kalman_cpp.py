import ctypes
import numpy as np
from quaternion_filters.quaternion import Quaternion

c_float_p = ctypes.POINTER(ctypes.c_float)

class KalmanCpp:

    NSTATES = 10

    def __init__(self):
        self.clib = ctypes.CDLL('./cpp_kalman/libkalman.so')
        self.clib.CreateKalmanClass.restype = ctypes.c_voidp
        self.clib.state_ptr.restype = c_float_p
        self.ptr = self.clib.CreateKalmanClass()

    def print_state(self):
        self.clib.PrintState(ctypes.c_voidp(self.ptr))

    def predict(self, dt):
        self.clib.predict(ctypes.c_voidp(self.ptr), 
                          ctypes.c_float(dt))

    def update_accel(self, a):
        self.clib.update_accel(ctypes.c_voidp(self.ptr), 
                               ctypes.c_float(a[0]),
                               ctypes.c_float(a[1]),
                               ctypes.c_float(a[2]))

    def __getattr__(self, val):
        if val == 'a':
            return self.state_vec()[4:7]
        elif val == 'w':
            return self.state_vec()[7:10]
        elif val == 'q':
            return self.state_vec()[:4]
    
    def state_vec(self):
        p = self.clib.state_ptr(ctypes.c_voidp(self.ptr)) 
        return np.array([p[n] for n in range(self.NSTATES)])

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