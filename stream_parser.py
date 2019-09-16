
import numpy as np
from numpy import pi
from kalman import K2ms, ft2m
import matplotlib.pylab as plt
import kalman
import argparse
try:
    import fixgw.netfix as netfix
except ModuleNotFoundError:
    pass
import time
import sys

codes = {1: 'gps',
         3: 'accel',
         5: 'mag',
         4: 'gyro',
         6: 'pos',
         7: 'unkn',
         8: 'unkn2',
         }


def latlong2dist(lat1, long1, lat2, long2):
    R = 6357000
    dLat = (lat2 - lat1)*np.pi/180
    dLong = (long2 - long1)*np.pi/180
    a = np.sin(dLat/2) * np.sin(dLat/2) \
        + np.cos(lat1*np.pi/180) * np.cos(lat2*np.pi/180) * \
          np.sin(dLong/2) * np.sin(dLong/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c  # Distance in m
    return d


def latlong2bearing(lat1, long1, lat2, long2):
    """
    # https://www.igismap.com/formula-to-find-bearing-or-heading-angle-between-two-points-latitude-longitude/
    # θ = atan2(sin(Δlong)*cos(lat2), cos(lat1)*sin(lat2) − sin(lat1)*cos(lat2)*cos(Δlong))

    Definition of direction is the traditional North 0 deg, east 90 deg, etc.
    """

    dLong = (long2 - long1)*np.pi/180
    return np.arctan2(np.sin(dLong)*np.cos(lat2*np.pi/180),
                      np.cos(lat1*np.pi/180)*np.sin(lat2*np.pi/180) - np.sin(lat1*np.pi/180)*np.cos(lat2*np.pi/180)*np.cos(dLong))


def psi2heading(psi):
    """
    Yaw angle between 0 and 360
    psi is in radians
    """
    if isinstance(psi, np.ndarray):
        return np.vectorize(psi2heading)(psi)
    if psi < 0:
        return psi2heading(psi + 2*pi)
    if psi > 2*pi:
        return psi2heading(psi - 2*pi)
    return psi*180/pi


def parse_stream_file(fn, filter=None, zero_time=False):
    data = {}
    for c, name in codes.items():
        data[name] = []

    with open(fn, 'r') as fid:
        line = fid.readline()
        while len(line) > 0:
            cols = line.split(',')
            t = cols[0]
            for n in range(1, len(cols), 4):
                c = int(cols[n])
                data[codes[c]].append([float(t)] +  [float(x) for x in cols[n+1:n+4]])

            line = fid.readline()

    for sensor in data.keys():
        data[sensor] = np.array(data[sensor])
        if zero_time:
            data[sensor][:, 0] = data[sensor][:, 0] - data[sensor][0, 0]
        if filter is not None:
            idx = filter(data[sensor][:,0])
            data[sensor] = data[sensor][idx, :]


    if len(data['pos']) > 0:
        data['speed'] = np.sqrt((data['pos'][1:,1] - data['pos'][:-1,1])**2 + (data['pos'][1:,2] - data['pos'][:-1,2])**2)/np.diff(data['pos'][:,0])
    if len(data['gps']) > 0:
        data['speed2'] = latlong2dist(data['gps'][1:, 1], data['gps'][1:, 2],
                                      data['gps'][:-1, 1], data['gps'][:-1, 2])/np.diff(data['gps'][:,0])
        data['bearing'] = latlong2bearing(data['gps'][1:, 1], data['gps'][1:, 2],
                                          data['gps'][:-1, 1], data['gps'][:-1, 2])

    for sns in ['gyro', 'mag', 'accel']:
        data[f'{sns}_rot'] = np.zeros(data[sns].shape)
        for n in range(len(data[sns])):
            data[f'{sns}_rot'][n, 0] = data[sns][n, 0]
            data[f'{sns}_rot'][n, 1:] = kalman.EKF.Rot_sns.dot(np.vstack(data[sns][n, 1:]))[:, 0]

    return data


class pyEfisInterface:

    def __init__(self, playback_speed=1.0):
        self.client = netfix.Client('127.0.0.1', 3490)
        self.client.connect()
        self.t0 = time.time()
        self.playback_speed = playback_speed
        self.t0stream = None

    def update(self, t, k, sensors):
        # wait for sim time to meet stream time
        if self.t0stream is None:
            self.t0stream = t
        while (t - self.t0stream) > (time.time() - self.t0)*self.playback_speed:
            time.sleep(.003)

        self.client.writeValue("PITCH", k.getstate("theta")*180/np.pi)
        self.client.writeValue("ROLL", k.getstate("phi")*180/np.pi)
        self.client.writeValue("HEAD", psi2heading(k.getstate("psi")))
        self.client.writeValue("IAS", k.getstate("TAS")/K2ms)
        if "gps" in sensors.keys():
            self.client.writeValue("ALT", sensors['gps'][2]/ft2m)
        # TODO: rate of turn
        # slip/skid
        # Altitude
        t = int(np.round(t))
        sys.stdout.write(f"{t//60:02d}:{t%60:02d} IAS: {k.getstate('TAS')/K2ms:5.0f}\r")


def kalman_filter(fn, filter=None, mag_offset=np.array([0,0,0]),
                  mag_gain=np.array([1,1,1]),
                  zero_time=False,
                  send_to_pyEfis=False,
                  playback_speed=1.0):

    k = kalman.EKF()
    time = []
    x = []
    ctr = 0
    NINIT = 50
    mag = np.zeros(3)
    prev = {}
    first_t = True
    if send_to_pyEfis:
        pyEfis = pyEfisInterface(playback_speed=playback_speed)

    with open(fn, 'r') as fid, open("output.txt", "w") as fout:
        line = fid.readline()
        while len(line) > 0:
            cols = line.split(',')
            t = float(cols[0])
            if first_t:
                t0 = t
                first_t = False
            if zero_time:
                t -= t0
            if filter is not None:
                if not filter(t):
                    line = fid.readline()
                    continue

            sensors = {}
            for n in range(1, len(cols), 4):
                c = int(cols[n])
                sensors[codes[c]] = np.array([float(x) for x in cols[n+1:n+4]])
                if codes[c] == 'mag':
                    sensors[codes[c]] -= mag_offset
                    sensors[codes[c]] /= mag_gain

            if ctr < NINIT:
                if 'mag' in sensors.keys():
                    mag += sensors['mag']
                    ctr += 1
            elif ctr == NINIT:
                mag /= NINIT
                mag = k.Rot_sns.dot(np.vstack(mag))[:,0]
                print(mag)
                k.x[k.state_names.index('psi'), 0] = -np.arctan2(mag[1], mag[0])
                in_plane_mag = np.linalg.norm(mag[:2])
                out_of_plane_ang = np.arctan2(mag[2], in_plane_mag)
                k.x[k.state_names.index('magxe'), 0] = in_plane_mag
                k.x[k.state_names.index('magze'), 0] = np.linalg.norm(mag)*np.sin(out_of_plane_ang)
                print("mag init to:", k.x[k.state_names.index('psi'), 0]*180/pi,
                   k.x[k.state_names.index('magxe'), 0], k.x[k.state_names.index('magze'), 0])
                ctr += 1
                print("State:", k)
                #import pdb; pdb.set_trace()
            else:
                dt = t - prev['t']
                # print("sensors:", sensors)
                # print(k)
                # if t > 163934:
                #     import pdb; pdb.set_trace()
                k.predict(dt)
                fout.write(f"p {dt}\n")
                if 'mag' in sensors.keys():
                    k.update_mag(sensors['mag'])
                    fout.write(f"m {sensors['mag'][0]} {sensors['mag'][1]} {sensors['mag'][2]}\n")
                if 'gyro' in sensors.keys():
                    k.update_gyro(sensors['gyro'])
                    fout.write(f"g {sensors['gyro'][0]} {sensors['gyro'][1]} {sensors['gyro'][2]}\n")
                if 'accel' in sensors.keys():
                    k.update_accel(sensors['accel'])
                    fout.write(f"a {sensors['accel'][0]} {sensors['accel'][1]} {sensors['accel'][2]}\n")

                if ('gps' in sensors.keys()) and ('gps' in prev.keys()):
                    TAS = latlong2dist(sensors['gps'][0], sensors['gps'][1],
                                       prev['gps'][1][0], prev['gps'][1][1])/(t-prev['gps'][0])
                    k.update_TAS(TAS)
                    fout.write(f"t {TAS}\n")
                ctr += 1

                if send_to_pyEfis:
                    pyEfis.update(t, k, sensors)

                x.append(k.x)
                time.append(t)



            prev['t'] = t
            if 'gps' in sensors.keys():
                prev['gps'] = [t, sensors['gps']]
            line = fid.readline()
    return np.array(time), np.hstack(x), k


def plot_stream_data(data):

    plt.figure(1)
    plt.clf()
    for n, sns in enumerate(['gyro','accel', 'mag']):
        plt.subplot(4,1,1+n)
        plt.plot(data[sns][:,0], data[sns][:,1:])
        plt.ylabel(sns)

    plt.figure(2)
    plt.subplot(311)
    plt.plot(data['pos'][1:,0], data['speed']/K2ms)
    plt.plot(data['pos'][1:,0], data['speed2']/K2ms)
    plt.ylabel('Speed (knots)')

    plt.subplot(312)
    plt.plot(data['gps'][:,0], data['gps'][:,3]/ft2m)
    plt.ylabel('Alt (ft)')

    plt.subplot(313)
    plt.plot(data['gps'][1:,0], np.unwrap(data['bearing'])*180/np.pi)

def plot_gyro_pqr(t, x, data):
    plt.figure(5)
    plt.clf()
    for n, statevar in enumerate(['p', 'q', 'r']):
        ax = plt.subplot(3,1,1+n)
        plt.plot(t, x[kalman.EKF.statei(statevar), :]*180/np.pi)
        plt.plot(data['gyro_rot'][:,0], data['gyro_rot'][:,1+n]*180/np.pi)
        plt.ylabel(f'{statevar}')
        plt.grid(True)
        if n == 0:
            plt.title('pqr (blue) vs. raw gyro (orange)')


def plot_accel_raw(t, x, data):
    plt.figure(6)
    plt.clf()
    for n, statevar in enumerate(['ax', 'ay', 'az']):
        ax = plt.subplot(3,1,1+n)
        plt.plot(data['accel_rot'][:,0], data['accel_rot'][:,1+n]/9.81)
        plt.plot(t, x[kalman.EKF.statei(statevar), :]/9.81)
        plt.ylabel(f'{statevar}')
        plt.grid(True)
        if n == 0:
            plt.title('a states (orange) vs. raw accel (blue)')


def plot_kf(t, x, data):
    plt.figure(3)
    plt.clf()
    ax = plt.subplot(311)
    plt.plot(t, x[kalman.EKF.statei('TAS'), :]/K2ms)
    plt.grid(True)
    plt.ylabel('TAS (kts)')
    plt.title('KF outputs')

    plt.subplot(312, sharex=ax)
    plt.plot(t, x[kalman.EKF.statei('phi'), :]*180/np.pi)
    plt.ylabel('roll')
    plt.grid(True)

    plt.subplot(313, sharex=ax)
    plt.plot(t, x[kalman.EKF.statei('theta'), :]*180/np.pi)
    plt.ylabel('pitch')
    plt.grid(True)


def plot_yaw(t, x):
    plt.figure(4)
    plt.clf()

    ax = plt.subplot(411)
    plt.plot(t, np.unwrap(x[kalman.EKF.statei('psi')])*180/np.pi)
    plt.ylabel('Heading (deg)')
    plt.grid(True)
    plt.title('Yaw info')

    plt.subplot(412, sharex=ax)
    plt.plot(t, np.unwrap(x[kalman.EKF.statei('r')])*180/np.pi, label='KF state')
    plt.plot(data['gyro_rot'][:,0], data['gyro_rot'][:,3]*180/np.pi, label='raw gyro')
    plt.ylabel('Yaw Rate (deg/sec)')
    plt.legend(loc=0)
    plt.grid(True)

    plt.subplot(413, sharex=ax)
    plt.plot(data['mag_rot'][:,0], data['mag_rot'][:,1:])
    plt.ylabel('mag')
    plt.grid(True)

    plt.subplot(414, sharex=ax)
    plt.plot(data['mag'][:,0], np.arctan2(-data['mag'][:,2], data['mag'][:,1])*180/np.pi)
    plt.ylabel('mag head')
    plt.grid(True)

def plot_pitch(t, x, data):
    plt.figure(7)
    plt.clf()

    ax = plt.subplot(211)
    plt.plot(data['gyro_rot'][:,0], data['gyro_rot'][:,2]*180/np.pi)
    plt.plot(t, x[kalman.EKF.statei('q'), :]*180/np.pi)
    plt.ylabel('q, raw gyro y')
    plt.grid(True)
    plt.title('Pitch (theta, q)')

    plt.subplot(212, sharex=ax)
    plt.plot(t, x[kalman.EKF.statei('theta'), :]*180/np.pi)
    plt.ylabel('Theta')
    plt.grid(True)


def plot_roll(t, x, data):
    plt.figure(8)
    plt.clf()

    ax = plt.subplot(311)
    plt.plot(data['gyro_rot'][:,0], data['gyro_rot'][:,1]*180/np.pi)
    plt.plot(t, x[kalman.EKF.statei('p'), :]*180/np.pi)
    plt.ylabel('p, raw gyro x')
    plt.grid(True)
    plt.title('Roll (q, phi)')

    plt.subplot(312, sharex=ax)
    plt.plot(t, x[kalman.EKF.statei('phi'), :]*180/np.pi)
    plt.ylabel('phi')
    plt.grid(True)

    plt.subplot(313, sharex=ax)
    plt.plot(t, x[kalman.EKF.statei('psi'), :]*180/np.pi)
    plt.ylabel('psi')
    plt.grid(True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pyEfis', help="Send data to pyEfis", default=False, action="store_true")
    parser.add_argument('-s','--speed', default=2.0, type=float)

    args = parser.parse_args()
    print(args)
    #sys.exit()

    # Data is taking off runway 31 at KBIS

    idx_func = lambda t: (t > 952092) # & (t < 952160)
    fn = 'mystream_8_7_13_24_30.csv'  # KBIS
    mag_offset = np.array([0,0,0])

    fn = 'mystream_8_13_8_26_19.csv'  # KFSD
    mag_offset = np.array([5, 22, 0])
    idx_func = lambda t: t > 0 #t < 164100

    fn = 'mystream_8_16_11_55_48_rv_flight.csv'  # KFSD
    mag_offset = np.array([5, 22, 0])
    idx_func = lambda t: (t > 360) & (t < 1800) #t < 360  # takeoff occurs at about 360 seconds


    data = parse_stream_file(fn, idx_func, zero_time=True)

    data['mag_cal'] = data['mag'].copy()
    mag_offset = np.array([6.563601938902331, -23.364970231751286, 0.6140873985288852])
    mag_gain = np.array([50.43649099163166, 55.90023271512351, 50])
    data['mag_cal'][:,1:] -= mag_offset
    data['mag_cal'][:,1:] /= mag_gain
    # plt.ion()
    # plt.close('all')
    # plot_stream_data(data)

    t, x, kf = kalman_filter(fn, idx_func, mag_offset, mag_gain, zero_time=True,
                             send_to_pyEfis=args.pyEfis,
                             playback_speed=args.speed)

    plt.ion()
    plot_kf(t, x, data)
    plot_stream_data(data)
    plot_yaw(t, x)
    plot_gyro_pqr(t, x, data)
    plot_accel_raw(t, x, data)
    plot_pitch(t, x, data)
    plot_roll(t, x, data)
