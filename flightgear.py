

"""

fgcommand("add-io-channel", {"config": "generic,socket,out,10,localhost,6789,udp,fgfs_fixgw", "name": "test"});
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
import plotting
import queue
import threading
from pyqtgraph.Qt import QtGui 

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


def euler_to_quaternion(roll, pitch, head):
    q = np.quaternion(1, 0, 0, 0)
    q = q * quaternion.from_rotation_vector(np.array([0, 0, 1.0])*-head)
    q = q * quaternion.from_rotation_vector(np.array([0, 1.0, 0])*-pitch)
    q = q * quaternion.from_rotation_vector(np.array([1.0, 0, 0])*roll)
    return q


def synthetic_mag(q, inclination, magnitude):
    m = np.array([np.cos(inclination), 0, -np.sin(inclination)])*magnitude
    return quaternion.as_float_array(q.inverse() * np.quaternion(0, *m) * q)[1:]


def flightgear_loop(plot_q):
    t = time.time()
    kf = FixedWingEKF()
    #q = quaternion.from_rotation_vector(np.array([0, 0, 1])*-170*np.pi/180)
    #kf.q = quaternion.as_float_array(q)

    while True:
        data, addr = client.recvfrom(1024)
        #print(data, addr)
        
        #data = struct.unpack('>' + 'L' + 'f'*9, data)
        data = struct.unpack('>' + 'f'*11, data)

        airspeed, altitude, head, roll, pitch, rollrate, pitchrate, yawrate, ax, ay, az = data
        ax /= G_FTS2
        ay /= G_FTS2
        az /= G_FTS2
        a = np.array([ax, ay, az]) * G_MS2
        w = np.array([rollrate, -pitchrate, -yawrate])
        qtrue = euler_to_quaternion(roll*np.pi/180, pitch*np.pi/180, head*np.pi/180)
        m = synthetic_mag(qtrue, 60*np.pi/180, 60)

        dt = time.time() - t
        t = time.time()
        kf.predict(dt)

        # Accel update
        # Rotate a into z down coordinate frame
        q = quaternion.from_rotation_vector(np.array([1, 0, 0]) * np.pi)
        a = quaternion.as_float_array(q * np.quaternion(0, *a) * q.inverse())[1:]
        #print((a / G_MS2).round(2))
        kf.update_accel(a)

        # Gyro update        
        #print((w * 180/np.pi).round(1))
        kf.update_gyro(w)

        # Magnetometer update
        #print("mag", m.round(1))
        #print("true head", head)
        kf.update_mag(m)

        #q = kf.quaternion()
        
        es = kf.eulers() * 180 / np.pi
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
        netfix_client.writeValue("IAS", airspeed)

        turn_rate0 = -quaternion.as_float_array(q*np.quaternion(0, *w)*q.inverse())[3]*180/np.pi 
        turn_rate = -kf.turn_rate()*180/np.pi

        plot_q.put(np.array([turn_rate0, turn_rate, 
                             kf.P[0, 0],
                             kf.P[1, 1],
                             kf.P[2, 2],
                             kf.P[3, 3]]))
        netfix_client.writeValue("ROT", turn_rate)
        netfix_client.writeValue("ALAT", -kf.a[1] / G_MS2)
        netfix_client.writeValue("ALT", altitude)
        #print(kf.P)


if __name__ == '__main__':
    q = queue.Queue()
    thread = threading.Thread(target=flightgear_loop, args=(q,))
    thread.start()
    p = plotting.Plotter(1000, q)  
    #p.app.exec() # This plots from qt start
    QtGui.QApplication.instance().exec_()

    