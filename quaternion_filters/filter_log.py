
import time
import numpy as np
from quaternion import Quaternion

G = 9.81
KTS2MS = 0.514444


class DataLog:

    def __init__(self):
        self.log = []

    def predict(self, dt, x, P):
        self.log.append((time.time(), 'predict', dt, x, P))

    def set_state(self, x, P):
        self.log.append((time.time(), 'set_state', x, P))

    def gyro(self, gyros):
        self.log.append((time.time(), 'gyro', gyros))

    def accel(self, accel):
        self.log.append((time.time(), 'accel', accel))

    def tas(self, tas):
        self.log.append((time.time(), 'tas', tas))

    def mag(self, mag):
        if len(mag) == 3:
            self.log.append((time.time(), 'mag', mag))

    def predict_times(self):
        return np.array([x[0] for x in self.log if x[1] == 'predict'])

    def predict_dt(self):
        return np.array([x[2] for x in self.log if x[1] == 'predict'])

    def get_mag(self):
        t = np.array([x[0] for x in self.log if x[1] == 'mag'])
        return t, np.array([x[2] for x in self.log if x[1] == 'mag'])

    def get_eulers(self):
        t = np.array([x[0] for x in self.log if x[1] == 'set_state'])
        return t, np.array([Quaternion(*x[2][:4].flatten()).euler_angles()*180/np.pi
                            for x in self.log if x[1] == 'set_state'])

    def get_gyro(self):
        """raw measurement"""
        t = np.array([x[0] for x in self.log if x[1] == 'gyro'])
        return t, np.array([x[2] for x in self.log if x[1] == 'gyro'])

    def get_accel(self):
        """raw measurement"""
        t = np.array([x[0] for x in self.log if x[1] == 'accel'])
        return t, np.array([x[2] for x in self.log if x[1] == 'accel'])

    def get_tas(self):
        """raw measurement"""
        t = np.array([x[0] for x in self.log if x[1] == 'tas'])
        return t, np.array([x[2]/KTS2MS for x in self.log if x[1] == 'tas'])

    def get_a_state(self):
        """ from state"""
        t = np.array([x[0] for x in self.log if x[1] == 'set_state'])
        return t, np.array([x[2][4:7].flatten()/G
                            for x in self.log if x[1] == 'set_state'])

    def get_w_state(self):
        """ from state"""
        t = np.array([x[0] for x in self.log if x[1] == 'set_state'])
        return t, np.array([x[2][7:10].flatten()*180/np.pi
                            for x in self.log if x[1] == 'set_state'])

    def get_tas_state(self):
        """ from state"""
        t = np.array([x[0] for x in self.log if x[1] == 'set_state'])
        return t, np.array([x[2][10]/KTS2MS
                            for x in self.log if x[1] == 'set_state'])

    def get_P_diag(self):
        t = np.array([x[0] for x in self.log if x[1] == 'set_state'])
        return t, np.array([np.diag(x[3])
                            for x in self.log if x[1] == 'set_state'])
