import unittest
import numpy as np
import fixgw.netfix as netfix
import time


class TestPyEfisConnection(unittest.TestCase):

	def test_connect(self):

            c = netfix.Client('127.0.0.1', 3490)
            c.connect()

            t = np.arange(0,5, .01)
            p = 10*np.sin(2*np.pi/5*t)
            t0 = time.time()
            n = 0
            while True:
                while t[n] > (time.time()-t0):
                    time.sleep(.003)
                c.writeValue("PITCH", p[n])
                n += 1
                if n == len(p):
                    break
