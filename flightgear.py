

"""

fgcommand("add-io-channel", {"config": "generic,socket,out,10,192.168.2.103,6789,udp,fgfs_fixgw", "name": "test"});
print(getprop("/orientation/heading-deg"));
"""

import socket
import struct
#import rospy
#import fixgw.plugin as plugin
import fixgw.netfix as netfix
import time
from quaternion_filters.fixed_wing_EKF import FixedWingEKF, KTS2MS
from quaternion_filters.fixed_wing_EKF_eulers import FixedWingEKFEulers, KTS2MS
from kalman_cpp import KalmanCpp
import numpy as np
import quaternion
import plotting
import queue
import threading
from pyqtgraph.Qt import QtGui 
import sys

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
    q = q * quaternion.from_rotation_vector(np.array([0, 0, 1.0])*head)
    q = q * quaternion.from_rotation_vector(np.array([0, 1.0, 0])*pitch)
    q = q * quaternion.from_rotation_vector(np.array([1.0, 0, 0])*roll)
    return q


def synthetic_mag(q, inclination, magnitude):
    m = np.array([np.cos(inclination), 0, np.sin(inclination)])*magnitude
    return quaternion.as_float_array(q.inverse() * np.quaternion(0, *m) * q)[1:]


class Timer:

    def __init__(self):
        self.n = 0
        self.t = 0

    def start(self):
        self.n += 1
        self.tstart = time.time()

    def stop(self):
        self.t += time.time() - self.tstart
        if self.n % 100 == 0:
            print(f"avg exec time: {self.t / self.n}")


def flightgear_loop(plot_q, run_bool):
    t = time.time()
    #kf = FixedWingEKF()
    kf = KalmanCpp()
    #kf.set_heading(180)
    #kf.es[2] = 180*np.pi/180
    #q = quaternion.from_rotation_vector(np.array([0, 0, 1])*180*np.pi/180)
    #kf.q = quaternion.as_float_array(q)

    timer = Timer()

    ctr = 0
    initializing = True

    logfid = open("log.ssv", "w")

    while run_bool.is_set():
        data, addr = client.recvfrom(1024)
        data = struct.unpack('>' + 'f'*11, data)

        # rotation rates are in rad/sec
        # accels are in ft/s2
        airspeed, altitude, head, roll, pitch, rollrate, pitchrate, yawrate, ax, ay, az = data
        #airspeed, altitude, head, roll, pitch, rollrate, pitchrate, yawrate, ax, ay, az = (0 for _ in range(11))
        ax /= G_FTS2
        ay /= G_FTS2
        az /= G_FTS2
        a = np.array([ax, ay, az]) * G_MS2
        w = np.array([rollrate, pitchrate, yawrate])
        qtrue = euler_to_quaternion(roll*np.pi/180, pitch*np.pi/180, head*np.pi/180)
                
        m = synthetic_mag(qtrue, 60*np.pi/180, 60)

        dt = time.time() - t
        t = time.time()
        kf.predict(dt, airspeed * KTS2MS)
        

        # Accel update
        # Rotate a into z down coordinate frame
        #q = quaternion.from_rotation_vector(np.array([1, 0, 0]) * np.pi)
        #a = quaternion.as_float_array(q * np.quaternion(0, *a) * q.inverse())[1:]
        #print((a / G_MS2).round(2))        
        kf.update_accel(a)
        
        # Gyro update        
        #print((w * 180/np.pi).round(1))
        kf.update_gyro(w)
        
        
        # Magnetometer update
        #print("mag", m.round(1))
        #print("true head", head)
        kf.update_mag(m)

        es = kf.eulers() * 180 / np.pi
        #print(es)
        #print(kf.tas / KTS2MS)  
        roll, pitch, head = es
        #print(roll, pitch, head)
        head = normalize_heading(head)
        #print(head)

        #print(kf.P[4:7, 4:7])
        #print(kf.Km.round(3))
        #print(np.round(kf.tas / KTS2MS, 1), ((a - kf.a) / G_MS2).round(2))
        #print(np.round(kf.tas / KTS2MS - airspeed, 1), ((w - kf.w) * 180/np.pi).round(2))

        netfix_client.writeValue("PITCH", pitch)
        netfix_client.writeValue("ROLL", roll)
        netfix_client.writeValue("HEAD", head)
        netfix_client.writeValue("IAS", airspeed)

        #turn_rate0 = -quaternion.as_float_array(q*np.quaternion(0, *w)*q.inverse())[3]*180/np.pi 
        turn_rate = kf.turn_rate()*180/np.pi

        #plot_q.put(np.array([0, turn_rate, 
        #                     kf.P[0, 0],
        #                     kf.P[1, 1],
        #                     kf.P[2, 2]]))
        #                     #kf.P[3, 3]]))
        #plot_q.put(np.array([kf.P[0, 0], kf.P[1, 1], kf.P[2,2], kf.P[3,3]]))
        #plot_q.put(np.array([kf.a[0], a[0], kf.a[1], a[1]]))
        #plot_q.put(np.array([kf.w[0], w[0], kf.w[1], w[1]]))
        netfix_client.writeValue("ROT", turn_rate)
        netfix_client.writeValue("ALAT", kf.a[1] / G_MS2)
        netfix_client.writeValue("ALT", altitude)
        #print(kf.P)
        logfid.write(f"{time.time()} {a[0]} {a[1]} {a[2]} {w[0]} {w[1]} {w[2]}\n")

        ctr += 1

    logfid.close()


if __name__ == '__main__':
    q = queue.Queue()
    run_bool = threading.Event()
    run_bool.set()
    thread = threading.Thread(target=flightgear_loop, args=(q, run_bool))
    thread.start()
    #p = plotting.Plotter(1000, q, ['Proll', 'Ppitch', 'Phead'])
    #p = plotting.Plotter(1000, q, ['Pq0', 'Pq1', 'Pq2', 'Pq3'])
    #p = plotting.Plotter(1000, q, ['kf.ax', 'ax', 'kf.ay', 'ay'])
    #p = plotting.Plotter(1000, q, ['kf.wx', 'wx', 'kf.wy', 'wy'])
    #p.app.exec() # This plots from qt start
    #QtGui.QApplication.instance().exec_()

    while True:
        try:
            time.sleep(.5)
        except KeyboardInterrupt:
            run_bool.clear()
            print("Interrupte received. Exiting...")
            break


    