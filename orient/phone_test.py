
import socket
import qEKF
import numpy as np

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

s.bind(('', 5555))

k = qEKF.qEKF()


while True:
    try:
        msg, adr = s.recvfrom(8192)
        cols = msg.split(b',')

        accels = np.array([float(x) for x in cols[2:5]])
        gyros = np.array([float(x) for x in cols[6:9]])
        mags = np.array([float(x) for x in cols[10:13]])

        k.predict()

        #import pdb; pdb.set_trace()
        if len(accels) == 3:
            k.update_accel(accels)

        if len(gyros) == 3:
            k.update_gyro(gyros)

        #print(k.quaternion().euler_angles()*180/np.pi)
        euler_angles = k.quaternion().euler_angles()*180/np.pi
        print(f"{euler_angles[0]:7.1f}{euler_angles[1]:7.1f}{euler_angles[2]:7.1f}")

    except (KeyboardInterrupt, SystemExit):
        import pdb; pdb.set_trace()
