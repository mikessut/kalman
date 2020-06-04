
from sympy import symbols, sqrt, diff, simplify
import quaternion
from quaternion import Quaternion
import numpy as np

quaternion.np.sqrt = sqrt

q0, q1, q2, q3 = symbols(['q0', 'q1', 'q2', 'q3'])
wx, wy, wz = symbols(['wx', 'wy', 'wz'])
wbx, wby, wbz = symbols(['wbx', 'wby', 'wbz'])
abx, aby, abz = symbols(['abx', 'aby', 'abz'])

# Magnetometer sensor output
mx, my, mz = symbols(['mx', 'my', 'mz'])
dt = symbols('dt')
states = [q0, q1, q2, q3, wx, wy, wz, wbx, wby, wbz, abx, aby, abz]
# Predict

q = Quaternion(1, .5*wx*dt, .5*wy*dt, .5*wz*dt) * Quaternion(q0, q1, q2, q3)
q.normalize()

nextstate = [q[0], q[1], q[2], q[3], wx, wy, wz, wbx, wby, wbz, abx, aby, abz]
for n in range(7):
    print(f"newq[{n}] = {nextstate[n]}")

F = np.zeros((13, 13), dtype=object)
for n in range(13):
    for m in range(13):
        tmp = (diff(nextstate[n], states[m]))
        F[n, m] = tmp
        if tmp != 0:
            print(f"F[{n}, {m}] = {tmp}")

# Accel update
#

# Map state into measurement space
q = Quaternion(q0, q1, q2, q3)
g = np.array([0,0,9.8], dtype=float)
accel = (q.inv() * Quaternion.from_vec(g) * q).as_ndarray()[1:] + np.array([abx, aby, abz])
H = np.zeros((3, 13), dtype=object)
for n in range(3):
    for m in range(13):
        tmp = diff(accel[n], states[m])
        H[n, m] = tmp
        if tmp != 0:
            print(f"H[{n}, {m}] = {tmp}")


# gyro update
gyros = np.array([wx,wy,wz]) + np.array([wbx, wby, wbz])
H = np.zeros((3, 13), dtype=object)
for n in range(3):
    for m in range(13):
        tmp = diff(gyros[n], states[m])
        H[n, m] = tmp
        if tmp != 0:
            print(f"H[{n}, {m}] = {tmp}")


# Magnetometer update
# Only consider the normalized vector projected onto the global xy plane

# 1.  Rotate the magnetometer measurement to global coordinates
# 2.  Only look at component in xy plane and normalize.
# 3.  Rotate this back to body coordinates -- this is the "measurement

# Kalman eqn:
# y = z - H*x
# innovation = measurement - (func translate state to measurement)
q = Quaternion(q0, q1, q2, q3)
mag_inertial = (q * Quaternion.from_vec(np.array([mx, my, mz])) * q.inv()).as_ndarray()[1:]
mag_inertial[2] = 0
mag_inertial /= sqrt(mag_inertial[0]**2 + mag_inertial[1]**2)

mag_body = (q.inv() * Quaternion.from_vec(mag_inertial) * q).as_ndarray()[1:]

# Only portion that we need to determin Jacobian is transvering "north vector"
# to body coordinates

north_body = (q.inv() * Quaternion.from_vec(np.array([1.0, 0, 0])) * q).as_ndarray()[1:]
H = np.zeros((3, 13), dtype=object)
for n in range(3):
    for m in range(13):
        tmp = diff(north_body[n], states[m])
        H[n, m] = tmp
        if tmp != 0:
            print(f"H[{n}, {m}] = {tmp}")
