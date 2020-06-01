
from qEKF import qEKF
import numpy as np

k = qEKF()

k.predict(.002)
ang = 10*np.pi/180
#k.update_accel(np.array([np.sin(ang), 0, np.cos(ang)]))

k.update_gyro(np.array([0,0, 10*np.pi/180]))

print("w:", k.w)
print("euler:", k.quaternion().euler_angles()*180/np.pi)


k.predict(.1)
print("w:", k.w)
print("euler:", k.quaternion().euler_angles()*180/np.pi)
