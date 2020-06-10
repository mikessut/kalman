
import socket
from orient import qEKF
import numpy as np
import msvcrt
import matplotlib.pyplot as plt

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

s.bind(('', 5555))

k = qEKF.qEKF()

done = False
while not done:

    msg, adr = s.recvfrom(8192)
    cols = msg.split(b',')

    accels = np.array([float(x) for x in cols[2:5]])
    gyros = np.array([float(x) for x in cols[6:9]])
    mags = np.array([float(x) for x in cols[10:13]])

    k.predict()

    #import pdb; pdb.set_trace()
    if len(accels) == 3:
        #accels[0] += 1
        k.update_accel(accels)

    if len(gyros) == 3:
        gyros[0] += 2*np.pi/180
        k.update_gyro(gyros)

    if len(mags) == 3:
        k.update_mag(mags)
    #k.log.mag(mags)

    #print(k.quaternion().euler_angles()*180/np.pi)
    euler_angles = k.quaternion().euler_angles()*180/np.pi
    print(f"{euler_angles[0]:7.1f}{euler_angles[1]:7.1f}{euler_angles[2]:7.1f}  {k.wb[0]*180/np.pi:6.2f}{k.wb[1]*180/np.pi:6.2f}{k.wb[2]*180/np.pi:6.2f} {k.ab[0]:6.2f}{k.ab[1]:6.2f}{k.ab[2]:6.2f} {k.w[0]*180/np.pi:6.2f}{k.w[1]*180/np.pi:6.2f}{k.w[2]*180/np.pi:6.2f}")
    d = np.diag(k.P)
    #print(d[7:10])

    if msvcrt.kbhit():
        done = True


f, ax = plt.subplots(4,1, sharex=True)
ax[0].plot(*k.log.get_eulers())
ax[0].set_ylabel('Eulers (deg)')
ax[1].plot(*k.log.get_w())
ax[1].set_ylabel('w (dps)')
ax[2].plot(*k.log.get_wb())
ax[2].set_ylabel('wb (dps)')
ax[3].plot(*k.log.get_ab())
ax[3].set_ylabel('ab ms2')

f, ax = plt.subplots(4,1, sharex=True)
t, P = k.log.get_P_diag()
ax[0].plot(t, P[:, :4])
ax[1].plot(t, P[:, 4:7])
ax[2].plot(t, P[:, 7:10])
ax[3].plot(t, P[:, 10:13])

plt.show()

import pdb; pdb.set_trace()