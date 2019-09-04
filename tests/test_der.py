
from kalman_derive import *
from kalman import K2ms
import unittest
import numpy as np


class TestDerive(unittest.TestCase):

    def test_accel_centripetal(self):
        """
        Check centripetal Accelerations in all quadrants.  Make sure that
        body accel is 2g for a 60 deg turn
        """
        for deg in np.linspace(0, 360, 9):
            self.assertEqual(ab[2,0].subs(psi, deg*pi/180).\
                                     subs(phi, 60*pi/180).\
                                     subs(theta,0).\
                                     subs(g, 1).evalf(), -2.0)
            self.assertEqual(ab[2,0].subs(psi, deg*pi/180).\
                                     subs(phi, -60*pi/180).\
                                     subs(theta,0).\
                                     subs(g, 1).evalf(), -2.0)
            # Coordinated turn should only have accel in body z direction.
            # Check that x and y are zero.
            for ax in range(2):
                val = ab[ax,0].subs(psi, deg*pi/180).\
                             subs(phi, 60*pi/180).\
                             subs(theta,0).\
                             subs(g, 1).evalf()
                self.assertTrue(np.isclose(float(val), 0.0))
                val = ab[ax,0].subs(psi, deg*pi/180).\
                             subs(phi, -60*pi/180).\
                             subs(theta,0).\
                             subs(g, 1).evalf()
                self.assertTrue(np.isclose(float(val), 0.0))

    def test_rotation_sign(self):
        thetaval = 10*pi/180
        # self.assertTrue(rot.subs(theta, thetaval).subs(phi, 18*pi/180).\
        #                     subs(TAS, 100*K2ms).subs(g, 9.81).evalf() > 0)
# 
        # self.assertTrue(rot.subs(theta, thetaval).subs(phi, -18*pi/180).\
        #                     subs(TAS, 100*K2ms).subs(g, 9.81).evalf() < 0)

        self.assertTrue(rotb[2,0].subs(theta, thetaval).subs(phi, 18*pi/180).\
                            subs(TAS, 100*K2ms).subs(g, 9.81).evalf() > 0)

        self.assertTrue(rotb[2,0].subs(theta, thetaval).subs(phi, -18*pi/180).\
                            subs(TAS, 100*K2ms).subs(g, 9.81).evalf() < 0)

    def test_std_rate_turn(self):
        # 26.3 deg bank at 180 knots should be standard rate
        val = rot.subs(phi, 26.3*pi/180).subs(g, 9.81).subs(TAS, 180*K2ms).evalf()
        self.assertTrue(np.isclose(float(val)*180/np.pi, 3.0, .01))

        val = rot.subs(phi, -26.3*pi/180).subs(g, 9.81).subs(TAS, 180*K2ms).evalf()
        self.assertTrue(np.isclose(float(val)*180/np.pi, -3.0, .01))
