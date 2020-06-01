
import socket
import kalman
from numpy import pi
import numpy as np

KALMAN_MODE = True
RAW_MODE = False
SENSOR_MODE = False
NUM_INIT = 100
n = 0

k = kalman.EKF()
k.mode = 'air'
k.x[k.state_names.index('TAS'), 0] = 120 / kalman.K2ms

init = {'mag': np.zeros(3)}
#k.x[k.state_names.index('magze'), 0] = 1400

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

s.bind(('', 5555))
last = 0


class SensorTriad:

    NUM_INIT = NUM_INIT

    def __init__(self):
        self._init = True
        self._ctr = 0
        self._offset = np.zeros((3, ))
        self._value = np.zeros((3, ))

    def update(self, *args):
        if len(args) < 3:
            return
        x, y, z = args
        if self._init:
            self._offset += np.array([x, y, z])
            self._ctr += 1
            if self._ctr >= self.NUM_INIT:
                self._init = False
                self._offset /= self.NUM_INIT
        self._value = np.array([x, y, z]) - self._offset

    def get(self):
        if self._init:
            return np.zeros((3,))
        return self._value

    def __getattr__(self, item):
        if item not in ['x', 'y', 'z']:
            raise ValueError("Only supports x,y,z . methods")
        if item == 'x':
            return self.get()[0]
        elif item == 'y':
            return self.get()[1]
        elif item == 'z':
            return self.get()[2]

    def initialized(self):
        return not self._init


class AccelTriad(SensorTriad):

    def update(self, *args):
        if len(args) < 3:
            return
        x, y, z = args
        if self._init:
            self._offset += np.array([x, y, z])
            self._ctr += 1
            if self._ctr >= self.NUM_INIT:
                self._init = False
                self._offset /= self.NUM_INIT
                self._offset[2] -= kalman.g
        self._value = np.array([x, y, z]) - self._offset


gyros = SensorTriad()
accels = AccelTriad()
mags = SensorTriad()

while True:
    try:
        msg, adr = s.recvfrom(8192)
        if RAW_MODE:
            print(msg)
        cols = msg.split(b',')
        dt = float(cols[0]) - last
        last = float(cols[0])
        accels.update(*[float(x) for x in cols[2:5]])
        gyros.update(*[float(x) for x in cols[6:9]])
        mags.update(*[float(x) for x in cols[10:13]])
        if SENSOR_MODE:
            print(f"{accels.x:7.2f}{accels.y:7.2f}{accels.z:7.2f}{gyros.x:7.3f}{gyros.y:7.3f}{gyros.z:7.3f}")
        if False:
            init['mag'] /= NUM_INIT
            init['mag'] = k.Rot_sns.dot(np.vstack(init['mag']))[:,0]
            print(init['mag'])
            k.x[k.state_names.index('psi'), 0] = -np.arctan2(init['mag'][1], init['mag'][0])
            in_plane_mag = np.linalg.norm(init['mag'][:2])
            out_of_plane_ang = np.arctan2(init['mag'][2], in_plane_mag)
            k.x[k.state_names.index('magxe'), 0] = in_plane_mag
            k.x[k.state_names.index('magze'), 0] = np.linalg.norm(init['mag'])*np.sin(out_of_plane_ang)
            print("mag init to:", k.x[k.state_names.index('psi'), 0]*180/pi,
               k.x[k.state_names.index('magxe'), 0], k.x[k.state_names.index('magze'), 0])
            n += 1

        if KALMAN_MODE:
            if (not accels.initialized()) or (not gyros.initialized()):
                print("init")
                continue
            k.predict()
            #k.update(gyros, accels, mag, .1)
            #import pdb; pdb.set_trace()
            k.update_gyro(gyros.get())
            k.update_accel(accels.get())
            #k.update_mag(mag)
            #k.update_gyros_accels(gyros, accels)
            roll = k.x[k.state_names.index('phi'), 0]*180/pi
            pitch = k.x[k.state_names.index('theta'), 0]*180/pi
            yaw = k.x[k.state_names.index('psi'), 0]*180/pi
            droll = k.x[k.state_names.index('p'), 0]*180/pi
            dpitch = k.x[k.state_names.index('q'), 0]*180/pi
            dyaw = k.x[k.state_names.index('r'), 0]*180/pi
            magx = k.x[k.state_names.index('magxe'), 0]
            magz = k.x[k.state_names.index('magze'), 0]
            TAS = k.x[k.state_names.index('TAS'), 0]/kalman.K2ms
            ax = k.x[k.state_names.index('ax'), 0]
            ay = k.x[k.state_names.index('ay'), 0]
            az = k.x[k.state_names.index('az'), 0]
            # {magx:7.1f} {magy:7.1f} {magz:7.1f} {TAS:7.1f}  {ax:7.3f} {ay:7.3f} {az:7.3f} {droll:7.3f} {dpitch:7.3f}
            print(f"{roll:7.1f} {pitch:7.1f} {yaw:7.1f} {magx:7.1f} {magz:7.1f} {dyaw:7.1f}")
    except (KeyboardInterrupt, SystemExit):
        import pdb; pdb.set_trace()
