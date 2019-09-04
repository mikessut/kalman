
import kalman
from kalman import K2ms, g
import unittest
import matplotlib.pylab as plt
import numpy as np


class TestImplementation(unittest.TestCase):

    def test_self_level(self):

        tf = 5*60
        dt = .038
        t = 0

        kf = kalman.EKF()

        kf.setstate('TAS', 100*K2ms)
        kf.setstate('phi', 20*np.pi/180)

        xh = []

        while t < tf:
            kf.predict(dt)
            kf.update_gyro([0,0,0])
            kf.update_accel([0,0, g])

            xh.append(kf.x)

            t += dt
            if np.abs(kf.getstate('phi')*180/np.pi) < 1:
                break

        xh = np.hstack(xh)
        t = np.arange(xh.shape[1])*dt
        print(xh.shape)
        plt.plot(t, xh[kf.statei('phi'), :]*180/np.pi)
        plt.xlabel('Time (sec)')
        plt.ylabel('Roll (Phi) (deg)')
        plt.grid(True)
        plt.show()

        self.assertTrue(np.abs(kf.getstate('phi')) < 1)
