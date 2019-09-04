
import socket
import kalman
from numpy import pi
import numpy as np

KALMAN_MODE = False
RAW_MODE = True
NUM_INIT = 10
n = 0

k = kalman.EKF()

init = {'mag': np.zeros(3)}
#k.x[k.state_names.index('magze'), 0] = 1400

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

s.bind(('', 5555))
last = 0

while True:
    try:
        msg, adr = s.recvfrom(8192)
        if RAW_MODE:
            print(msg)
        cols = msg.split(b',')
        dt = float(cols[0]) - last
        last = float(cols[0])
        accels = [float(x) for x in cols[2:5]]
        gyros = [float(x) for x in cols[6:9]]
        mag = [float(x) for x in cols[10:13]]
        if (len(accels) < 3) or (len(gyros) < 3) or (len(mag) < 3):
            continue
        if n < NUM_INIT:
            init['mag'] += mag
            n += 1
            continue
        elif n == NUM_INIT:
            # init kalman
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
            k.predict()
            #k.update(gyros, accels, mag, .1)
            #import pdb; pdb.set_trace()
            k.update_gyro(gyros)
            k.update_accel(accels)
            k.update_mag(mag)
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
