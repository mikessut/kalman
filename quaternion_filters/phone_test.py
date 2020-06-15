
import socket
import quaternion
from quaternion_filters import fixed_wing_EKF
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
import argparse


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

        self.client.writeValue("ALAT", -k.a[1]/fixed_wing_EKF.G)

        # TODO: this isn't really IAS
        self.client.writeValue("IAS", k.tas/fixed_wing_EKF.KTS2MS)

        # Dummy in GPS altitude
        self.client.writeValue("ALT", k.alt/fixed_wing_EKF.FT2M)



class StreamMocker:

    def __init__(self, fn, plot=False):
        if plot:
            self.plot_path(fn)
        self.fid = open(fn, 'rb')
        self.t0 = time.time()
        line = self.fid.readline()
        self.t0_file = float(line.split(b',')[0])
        self.next_line = self.fid.readline()
        self.next_time = float(self.next_line.split(b',')[0])

    def recvfrom(self, l):
        while (time.time() - self.t0) < (self.next_time - self.t0_file):
            time.sleep(.002)

        line = self.next_line
        self.next_line = self.fid.readline()
        self.next_time = float(self.next_line.split(b',')[0])
        return line, ''

    def plot_path(self, fn):
        gps = []
        with open(fn) as fid:
            line = fid.readline()
            while len(line) > 0:
                cols = [float(x) for x in line.split(',')]
                datatypes = cols[1::4]
                if 1.0 in datatypes:
                    idx = 2 + datatypes.index(1.0)
                    gps.append([cols[0]] + [x for x in cols[idx:idx+3]])
                line = fid.readline()

        gps = np.array(gps)
        plt.ion()
        self.f, self.ax = plt.subplots()
        self.ax.plot(gps[:, 2], gps[:, 1])
        self.ax.axis('equal')
        self.f.canvas.draw()
        self.f.canvas.flush_events()
        plt.show()


def run_func(args, done, k, fixgw, raw_log):

    if args.stream_file is None:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(('', 5555))
    else:
        s = StreamMocker(args.stream_file)

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
            data[codes[c]] = np.array([float(cols[0])] + [float(x) for x in cols[n+1:n+4]])

        k.predict()

        #import pdb; pdb.set_trace()
        if len(data['accel']) == 4:
            #accels[0] += 1
            k.update_accel(data['accel'][1:])

        if len(data['gyro']) == 4:
            #data['gyro'][0] += 2*np.pi/180
            k.update_gyro(data['gyro'][1:])

        if len(data['mag']) == 4:
            k.update_mag(data['mag'][1:] + np.array(args.mag_offset))

        if len(data['gps']) > 0:
            if len(gps) == 2:
                gps[0] = gps[1]
                gps[1] = data['gps']  # time, lat, lng, alt
                dt = gps[1][0] - gps[0][0]
                d = latlong2dist(gps[0][1], gps[0][2],
                                 gps[1][1], gps[1][2])
                print(f"dt: {dt}, d: {d}, speed (m/s): {d/dt}")
                if args.dummy_tas is None:
                    k.update_tas(d/dt)
                k.alt = data['gps'][3]
            else:
                gps.append(data['gps'])

        if args.dummy_tas is not None:
            k.update_tas(args.dummy_tas*fixed_wing_EKF.KTS2MS)
        # k.update_tas(120*fixed_wing_EKF.KTS2MS)



        #print(k.quaternion().euler_angles()*180/np.pi)
        euler_angles = k.quaternion().euler_angles()*180/np.pi
        print(k)
        if SEND2FIXGW:
            fixgw.update(k)


parser = argparse.ArgumentParser()
parser.add_argument('--stream-file', default=None)
parser.add_argument('--dummy-tas', default=None, type=float)
parser.add_argument('--mag-offset', help='in native mag units (e.g. -7,11.6,0)',
                    default=np.array([0,0,0.0]), nargs=3,
                    type=float)
args = parser.parse_args()
print(args)


#k = qEKF.qEKF()
k = fixed_wing_EKF.FixedWingEKF()
# q = k.quaternion()
# q = q*quaternion.Quaternion.axis_angle([0,0,1.0], 150*np.pi/180)
# k.q = q.as_ndarray()
# k.P[:4, :4] = np.zeros((4,4))
if SEND2FIXGW:
    fixgw = FIXGWInterface()
else:
    fixgw = None
done = threading.Event()
raw_log = []
done.clear()

thread = threading.Thread(target=run_func, args=(args, done, k, fixgw, raw_log))
thread.start()

while True:
    tmp = input()
    if tmp == 'q':
        done.set()
        break

f, ax = plt.subplots(4,1, sharex=True)
ax[0].plot(*k.log.get_eulers())
ax[0].set_ylabel('Eulers (deg)')
ax[1].plot(*k.log.get_w_state())
ax[1].plot(*k.log.get_gyro(), 'k')
ax[1].set_ylabel('w_state (dps)')
ax[2].plot(*k.log.get_a_state())
ax[2].set_ylabel('a_state (g)')
ax[2].plot(*k.log.get_accel(), 'k')
ax[3].plot(*k.log.get_tas_state())
ax[3].plot(*k.log.get_tas())
ax[3].set_ylabel('tas (kts)')

f, ax = plt.subplots(4,1, sharex=True)
t, P = k.log.get_P_diag()
ax[0].plot(t, P[:, :4])
ax[0].set_ylabel('Q')
ax[1].plot(t, P[:, 4:7])
ax[1].set_ylabel('a')
ax[2].plot(t, P[:, 7:10])
ax[2].set_ylabel('w')
ax[3].plot(t, P[:, 10])
ax[3].set_ylabel('tas')

plt.show()

import pdb; pdb.set_trace()
