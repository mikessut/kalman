
import socket
from quaternion_filters import qEKF, fixed_wing_EKF
import numpy as np
try:
    import msvcrt
except ModuleNotFoundError:
    pass
import matplotlib.pyplot as plt
from util import head360
try:
    import fixgw.netfix as netfix
except ModuleNotFoundError:
    pass
import threading
import time
from latlong import latlong2dist


SEND2FIXGW = True

codes = {1: 'gps',
         3: 'accel',
         5: 'mag',
         4: 'gyro',
         6: 'pos',
         7: 'unkn',
         8: 'unkn2',
         }


class FIXGWInterface:

    def __init__(self):
        self.client = netfix.Client('127.0.0.1', 3490)
        self.client.connect()

    def update(self, k: fixed_wing_EKF.FixedWingEKF):
        roll, pitch, head = k.quaternion().euler_angles()*180/np.pi
        self.client.writeValue("PITCH", -pitch)
        self.client.writeValue("ROLL", roll)
        self.client.writeValue("HEAD", head360(-head*np.pi/180))

        self.client.writeValue("ALAT", -k.accel()[0]/fixed_wing_EKF.G)

        # TODO: this isn't really IAS
        self.client.writeValue("IAS", k.tas()/fixed_wing_EKF.KTS2MS)


def run_func(done, k, fixgw, raw_log):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    s.bind(('', 5555))

    gps = []

    while not done.is_set():

        msg, adr = s.recvfrom(8192)
        raw_log.append((time.time(), msg))
        cols = msg.split(b',')

        data = {}
        for c, name in codes.items():
            data[name] = []

        for n in range(1, len(cols), 4):
            c = int(cols[n])
            data[codes[c]] = np.array([float(x) for x in cols[n+1:n+4]])

        k.predict()

        #import pdb; pdb.set_trace()
        if len(data['accel']) == 3:
            #accels[0] += 1
            k.update_accel(data['accel'])

        if len(data['gyro']) == 3:
            #data['gyro'][0] += 2*np.pi/180
            k.update_gyro(data['gyro'])

        if len(data['mag']) == 3:
            k.update_mag(data['mag'])

        if len(data['gps']) > 0:
            if len(gps) == 2:
                gps[0] = gps[1]
                gps[1] = (time.time(), data['gps'])  # time, (lat, lng, alt)
                dt = gps[1][0] - gps[0][0]
                d = latlong2dist(gps[0][1][0], gps[0][1][1],
                                 gps[1][1][0], gps[1][1][1])
                print(f"dt: {dt}, d: {d}, speed (m/s): {d/dt}")
                k.update_tas(d/dt)
            else:
                gps.append((time.time(), data['gps']))



        #print(k.quaternion().euler_angles()*180/np.pi)
        euler_angles = k.quaternion().euler_angles()*180/np.pi
        print(k)
        if SEND2FIXGW:
            fixgw.update(k.quaternion())



#k = qEKF.qEKF()
k = fixed_wing_EKF.FixedWingEKF()
if SEND2FIXGW:
    fixgw = FIXGWInterface()
else:
    fixgw = None
done = threading.Event()
raw_log = []
done.clear()

thread = threading.Thread(target=run_func, args=(done, k, fixgw, raw_log))
thread.start()

while True:
    tmp = input()
    if tmp == 'q':
        done.set()
        break

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
