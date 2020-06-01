
from sympy import symbols, sqrt, diff, simplify
import quaternion
from quaternion import Quaternion
import numpy as np

quaternion.np.sqrt = sqrt

q0, q1, q2, q3 = symbols(['q0', 'q1', 'q2', 'q3'])
wx, wy, wz = symbols(['wx', 'wy', 'wz'])
dt = symbols('dt')
states = [q0, q1, q2, q3, wx, wy, wz]
# Predict

q = Quaternion(1, .5*wx*dt, .5*wy*dt, .5*wz*dt) * Quaternion(q0, q1, q2, q3)
q.normalize()

nextstate = [q[0], q[1], q[2], q[3], wx, wy, wz]
for n in range(7):
    print(f"newq[{n}] = {nextstate[n]}")

F = np.zeros((7, 7), dtype=object)
for n in range(7):
    for m in range(7):
        tmp = (diff(nextstate[n], states[m]))
        F[n, m] = tmp
        if tmp != 0:
            print(f"F[{n}, {m}] = {tmp}")

# Accel update
#

# Map state into measurement space
q = Quaternion(q0, q1, q2, q3)
g = np.array([0,0,1], dtype=float)
accel = (q.inv() * Quaternion.from_vec(g) * q).as_ndarray()[1:]
H = np.zeros((3, 7), dtype=object)
for n in range(3):
    for m in range(7):
        tmp = diff(accel[n], states[m])
        H[n, m] = tmp
        if tmp != 0:
            print(f"H[{n}, {m}] = {tmp}")


# gyro update


