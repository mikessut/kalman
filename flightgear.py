

"""

fgcommand("add-io-channel", {"config": "generic,socket,out,10,localhost,6789,udp,fixgw", "name": "test"});
"""

import socket
import struct
#import rospy
#import fixgw.plugin as plugin
import fixgw.netfix as netfix
import time
from quaternion_filters.fixed_wing_EKF import FixedWingEKF, KTS2MS
import numpy as np
import quaternion

netfix_client = netfix.Client('127.0.0.1', 3490)
netfix_client.connect()

client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
client.bind(('', 6789))

G_FTS2 = 32.17405  # ft / s**2
G_MS2 = 9.81


def eulers(q):
    a, b, c, d = q.components
    phi = np.arctan2(2*(a*b + c*d),
                        1-2*(b**2 + c**2))
    theta = np.arcsin(2*(a*c - d*b))
    psi = np.arctan2(2*(a*d+b*c),
                        1-2*(c**2+d**2))
    return np.array([phi, theta, psi])


def normalize_heading(val_deg):
    if val_deg > 360:
        return normalize_heading(val_deg - 360)
    elif val_deg < 0:
        return normalize_heading(val_deg + 360)
    else:
        return val_deg


if __name__ == '__main__':
    t = time.time()
    kf = FixedWingEKF()


    while True:
        data, addr = client.recvfrom(1024)
        #print(data, addr)
        
        #data = struct.unpack('>' + 'L' + 'f'*9, data)
        data = struct.unpack('>' + 'f'*11, data)

        airspeed, altitude, head, roll, pitch, rollrate, pitchrate, yawrate, ax, ay, az = data
        ax /= G_FTS2
        ay /= G_FTS2
        az /= G_FTS2

        #print(airspeed, head)
        #print(f"pitch, roll, yaw: {pitchrate:.1f} {rollrate:.1f} {yawrate:.1f} ")
        #print(f"accels:           {ax:.2f} {ay:.2f} {az:.2f}")
        #print(ros.is_connected)
        #ros_topic.publish(roslibpy.Message({'data': 'test'}))
        dt = time.time() - t
        t = time.time()
        kf.predict(dt)

        a = np.array([ax, ay, az]) * G_MS2

        # Rotate a into z down coordinate frame
        q = quaternion.from_rotation_vector(np.array([1, 0, 0]) * np.pi)
        a = quaternion.as_float_array(q * np.quaternion(0, *a) * q.inverse())[1:]
        #print((a / G_MS2).round(2))
        
        kf.update_accel(a)

        w = np.array([rollrate, -pitchrate, -yawrate])
        print((w * 180/np.pi).round(1))
        kf.update_gyro(w)

        kf.update_tas(airspeed * KTS2MS)

        q = quaternion.from_float_array(kf.state_vec().flatten()[:4])
        
        es = eulers(q) * 180 / np.pi
        #print(es)
        #print(kf.tas / KTS2MS)  
        roll, pitch, head = es
        head = normalize_heading(-head)
        #print(head)

        #print(np.round(kf.tas / KTS2MS, 1), ((a - kf.a) / G_MS2).round(2))
        #print(np.round(kf.tas / KTS2MS - airspeed, 1), ((w - kf.w) * 180/np.pi).round(2))

        netfix_client.writeValue("PITCH", -pitch)
        netfix_client.writeValue("ROLL", roll)
        netfix_client.writeValue("HEAD", head)
        #netfix_client.writeValue("IAS", airspeed)
        netfix_client.writeValue("IAS", kf.tas / KTS2MS)
        netfix_client.writeValue("ROT", -quaternion.as_float_array(q*np.quaternion(0, *kf.w)*q.inverse())[3]*180/np.pi)
        netfix_client.writeValue("ALAT", -kf.a[1] / G_MS2)
        netfix_client.writeValue("ALT", altitude)


