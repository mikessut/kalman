
import numpy as np
from numpy import pi, tan, sin, cos
from numpy import arctan as atan

g = 9.81
K2ms = 1852/3600;   # Convert knots to m/s; 1852m = 1nm
ft2m = 1/3.28084    # Convert ft to m

def DCM(phi,theta,psi):
    """
    phi: roll
    theta: pitch
    psi: yaw
    """
    sinR = sin(phi);
    cosR = cos(phi);
    sinP = sin(theta);
    cosP = cos(theta);
    sinY = sin(psi);
    cosY = cos(psi);

    # H = np.zeros((3,3))
    # H[0,0] = cosP * cosY;
    # H[0,1] = cosP * sinY;
    # H[0,2] = -sinP;
    # H[1,0] = sinR * sinP * cosY - cosR * sinY;
    # H[1,1] = sinR * sinP * sinY + cosR * cosY;
    # H[1,2] = sinR * cosP;
    # H[2,0] = cosR * sinP * cosY + sinR * sinY;
    # H[2,1] = cosR * sinP * sinY - sinR * cosY;
    # H[2,2] = cosR * cosP;
    HEB = np.array([[cosP * cosY, cosP * sinY, -sinP],
                    [sinR * sinP * cosY - cosR * sinY, sinR * sinP * sinY + cosR * cosY, sinR * cosP],
                    [cosR * sinP * cosY + sinR * sinY, cosR * sinP * sinY - sinR * cosY, cosR * cosP]])
    return HEB


class KalmanMatricies:

    def __init__(self, nstates, nsensors):
        self.nstates = nstates
        self.nsensors = nsensors
        self.P = np.zeros((nstates, nstates))
        self.Q = np.zeros((nstates, nstates))
        self.R = np.zeros((nstates, nsensors))

    def predict(self, F):
        assert(F.shape == (self.nstates, self.nstates))
        self.P = F.dot(self.P).dot(F.T) + self.Q
        assert(self.P.shape == (self.nstates, self.nstates))

    def update_sensors(self, H, sns_idx):
        assert(H.shape == (len(sns_idx), self.nstates))
        S = H.dot(self.P).dot(H.T) + self.R[np.ix_(sns_idx, sns_idx)]
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        self.P = (np.eye(self.nstates) - K.dot(H)).dot(self.P)
        assert(self.P.shape == (self.nstates, self.nstates))
        assert(K.shape == (self.nstates, len(sns_idx)))
        return K

    def set_Q_cross_term(self, i, j, val):
        self.Q[i, j] = val
        self.Q[j, i] = val


class EKF:

    state_names = ['p', 'q', 'r', 'ax', 'ay', 'az', 'phi',
                   'theta', 'psi', 'TAS', 'magxe', 'magze']
    sensor_names = ['wx', 'wy', 'wz', 'ax', 'ay', 'az', 'TAS', 'magx', 'magy', 'magz']
    dt = .038

    # Rotate 180 deg about x axis
    Rot_sns = np.array([[1, 0, 0],
                        [0, cos(pi), -sin(pi)],
                        [0, sin(pi), cos(pi)]])

    def __init__(self):
        nstates = len(self.state_names)
        self.nstates = nstates
        nsensors = 10
        self.nsensors = nsensors
        dt = self.dt

        self.x = np.zeros((nstates, 1))

        self.mode = 'gnd'
        self.gnd = KalmanMatricies(nstates, nsensors)
        self.air = KalmanMatricies(nstates, nsensors)

        # x deg/sec => convert to radians / dt secs
        # this means 2000 deg/sec / sec error in model   seems HUGE!!
        p_err = (2*pi/180*dt)**2
        q_err = (2*pi/180*dt)**2
        r_err = (2*pi/180*dt)**2
        ax_err = (2*g*dt)**2  # confidence that ax, ay, az is from phi/TAS
        ay_err = (.2*g*dt)**2
        az_err = (.2*g*dt)**2
        pitch_err = (.01*pi/180*dt)**2  # confidence that pitch is theta + dt*q
        roll_err =  (.01*pi/180*dt)**2    # confident that roll is atan(TAS/g*<p,q,r projected to earth_rot_z>)
        yaw_err =   (.01*pi/180*dt)**2             # confidence that yaw is psi + dt*<

        gnd_orient_err = (17*pi/180*dt)**2
        earth_mag_err = (1/30*dt)
        psi_err = .1/10*pi/180*dt  # gnd only
        const_TAS_err = (6.9*K2ms*dt)**2
        #np.fill_diagonal(self.Q,
        #                np.hstack([np.ones(3)*rot_err, np.ones(3)*aerr,
        #                           np.ones(2)*orient_err, psi_err**2,
        #                           np.ones(3)*earth_mag_err**2]))
        i = np.arange(nstates)
        self.air.Q[i, i] = np.hstack([p_err, q_err, r_err,
                                  ax_err, ay_err, az_err,
                                  roll_err, pitch_err, yaw_err,
                                  const_TAS_err,
                                  np.ones(2)*earth_mag_err**2])

        self.gnd.Q[i, i] = np.hstack([np.ones(2)*r_err, (10*pi/180*dt)**2,
                           ax_err, ay_err, az_err,
                                  np.ones(2)*gnd_orient_err, psi_err**2,
                                  const_TAS_err,
                                  np.ones(2)*(earth_mag_err/1000)**2])

        #self.air.set_Q_cross_term(self.statei('ax'), self.statei('theta'), (.1*g * 4*pi/180  *dt**2))
        #self.air.set_Q_cross_term(self.statei('q'), self.statei('phi'), (.1*pi/180 * .1*pi/180  /dt**2)*20)

        BW = 50
        # Gyro ~ 0.01 deg/rt-Hz
        gyro_err = (.01*np.sqrt(BW))**2
        # Accel 300 ug/rt-Hz
        #accel_err = (300e-6*g*np.sqrt(BW))**2
        accel_err = (.5*g)**2
        mag_err = (.1)**2
        TAS_err = (20*K2ms)**2
        #np.fill_diagonal(self.R,
        #                 np.hstack([np.ones(3)*gyro_err, np.ones(3)*accel_err,
        #                            np.ones(3)*mag_err]))
        R = np.zeros((self.nsensors, self.nsensors))
        i = np.arange(self.sensori('wx'), self.sensori('wx')+3)
        R[i, i] = np.ones(3)*gyro_err
        i = np.arange(self.sensori('ax'), self.sensori('ax')+3)
        R[i, i] = np.ones(3)*accel_err
        i = np.arange(self.sensori('magx'), self.sensori('magx')+3)
        R[i, i] = np.ones(3)*mag_err
        R[self.sensori('TAS'), self.sensori('TAS')] = TAS_err

        self.air.R = R
        self.gnd.R = R.copy()

        #np.fill_diagonal(self.H[:7, :7], np.ones(7))
        #np.fill_diagonal(self.H[3:6, 3:6], np.ones(3))

        #self.x[self.state_names.index('TAS'), 0] = .1
        self.x[self.state_names.index('az'), 0] = -g

    @classmethod
    def statei(cls, name):
        """
        Return the index in the state vector of the named state.
        """
        return cls.state_names.index(name)

    @classmethod
    def sensori(cls, name):
        """
        Return the index in the sensor vector of the named sensor.
        """
        return cls.sensor_names.index(name)

    def setstate(self, state_name, val):
        self.x[self.statei(state_name), 0] = val

    def getstate(self, state_name):
        return self.x[self.statei(state_name), 0]

    def predict(self, dt=None):
        if self.x[self.statei('TAS'), 0] > 50*K2ms:
            self.mode = 'air'
            self.predict_air(dt)
        else:
            self.mode = 'gnd'
            self.predict_gnd(dt)

    def predict_gnd(self, dt=None):
        x = {}
        for i, name in enumerate(self.state_names):
            x[name] = self.x[i, 0]

        if dt is None:
            dt = self.dt
        assert(dt > 0)

        self.x[0, 0] = 0
        self.x[1, 0] = 0
        self.x[2, 0] = x['r']
        self.x[3, 0] = x['ax']
        self.x[4, 0] = x['ay']
        self.x[5, 0] = x['az']
        self.x[6, 0] = 0
        self.x[7, 0] = 0
        self.x[8, 0] = dt*x['r'] + x['psi']
        self.x[9, 0] = x['TAS'] + x['ax']*dt
        self.x[10, 0] = x['magxe']
        self.x[11, 0] = x['magze']
        F = np.zeros((self.nstates, self.nstates))


        F[2, 2] = 1

        F[3, 3] = 1

        F[4, 4] = 1

        F[5, 5] = 1



        F[8, 2] = dt
        F[8, 8] = 1

        F[9, 3] = dt
        F[9, 9] = 1

        F[10, 10] = 1

        F[11, 11] = 1
        self.gnd.predict(F)

    def predict_air(self, dt=None):
        x = {}
        for i, name in enumerate(self.state_names):
            x[name] = self.x[i, 0]

        if dt is None:
            dt = self.dt
        assert(dt > 0)
        self.x[0, 0] = x['p']
        self.x[1, 0] = x['q']
        self.x[2, 0] = x['r']
        self.x[3, 0] = g*sin(x['theta'])
        self.x[4, 0] = g*(sin(x['phi'])*sin(x['psi'])*sin(x['theta']) + cos(x['phi'])*cos(x['psi']))*cos(x['psi'])*tan(x['phi']) - g*(sin(x['phi'])*sin(x['theta'])*cos(x['psi']) - sin(x['psi'])*cos(x['phi']))*sin(x['psi'])*tan(x['phi']) - g*sin(x['phi'])*cos(x['theta'])
        self.x[5, 0] = -g*(sin(x['phi'])*sin(x['psi']) + sin(x['theta'])*cos(x['phi'])*cos(x['psi']))*sin(x['psi'])*tan(x['phi']) + g*(-sin(x['phi'])*cos(x['psi']) + sin(x['psi'])*sin(x['theta'])*cos(x['phi']))*cos(x['psi'])*tan(x['phi']) - g*cos(x['phi'])*cos(x['theta'])
        self.x[6, 0] = dt*x['p'] + x['phi']
        self.x[7, 0] = dt*x['q'] + x['theta']
        self.x[8, 0] = x['psi'] + dt*g*tan(x['phi'])/x['TAS']
        self.x[9, 0] = x['TAS'] + x['ax']*dt
        self.x[10, 0] = x['magxe']
        self.x[11, 0] = x['magze']
        F = np.zeros((self.nstates, self.nstates))
        F[0, 0] = 1

        F[1, 1] = 1

        F[2, 2] = 1

        F[3, 7] = g*cos(x['theta'])

        F[4, 6] = g*(-cos(x['theta']) + 1)*cos(x['phi'])
        F[4, 7] = g*sin(x['phi'])*sin(x['theta'])

        F[5, 6] = g*(cos(x['theta']) - 1 - 1/cos(x['phi'])**2)*sin(x['phi'])
        F[5, 7] = g*sin(x['theta'])*cos(x['phi'])

        F[6, 0] = dt
        F[6, 6] = 1

        F[7, 1] = dt
        F[7, 7] = 1

        F[8, 6] = dt*g/(x['TAS']*cos(x['phi'])**2)
        F[8, 8] = 1
        F[8, 9] = -dt*g*tan(x['phi'])/x['TAS']**2

        F[9, 3] = dt
        F[9, 9] = 1

        F[10, 10] = 1

        F[11, 11] = 1
        self.air.predict(F)

    def update_gyro(self, gyros):
        assert(len(gyros) == 3)
        gyros = self.Rot_sns.dot(np.vstack(gyros))

        y = np.vstack([gyros]) - self.x[:3, [0]]
        assert(y.shape == (3,1))

        H = np.zeros((3, self.nstates))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        i = self.sensori('wx')
        if self.mode == 'gnd':
            K = self.gnd.update_sensors(H, list(range(i, i+3)))
        elif self.mode == 'air':
            K = self.air.update_sensors(H, list(range(i, i+3)))
        else:
            raise Exception("Invalid mode")

        self.x = self.x + K.dot(y)

    def update_accel(self, accels):
        assert(len(accels) == 3)
        accels = self.Rot_sns.dot(np.vstack(accels))

        y = accels - self.x[slice(3,6), [0]]
        assert(y.shape == (3,1))

        H = np.zeros((3, self.nstates))
        H[0, 3] = 1
        H[1, 4] = 1
        H[2, 5] = 1
        i = self.sensori('ax')
        if self.mode == 'gnd':
            K = self.gnd.update_sensors(H, list(range(i, i+3)))
        elif self.mode == 'air':
            K = self.air.update_sensors(H, list(range(i, i+3)))
        else:
            raise Exception("Invalid mode")

        self.x = self.x + K.dot(y)
        assert(self.x.shape == (self.nstates, 1))

    def update_mag(self, mag):
        assert(len(mag) == 3)
        mag = self.Rot_sns.dot(np.vstack(mag))

        x = {}
        for i, name in enumerate(self.state_names):
            x[name] = self.x[i, 0]

        i = self.state_names.index('magxe')
        magb = DCM(x['phi'], x['theta'], x['psi']).dot(np.vstack([self.x[i,0], 0, self.x[i+1,0]]))
        y = mag - magb
        assert(y.shape == (3,1))

        H = np.zeros((3, self.nstates))
        H[0, 7] = -x['magxe']*sin(x['theta'])*cos(x['psi']) - x['magze']*cos(x['theta'])
        H[0, 8] = -x['magxe']*sin(x['psi'])*cos(x['theta'])
        H[0, 10] = cos(x['psi'])*cos(x['theta'])
        H[0, 11] = -sin(x['theta'])
        H[1, 6] = x['magxe']*(sin(x['phi'])*sin(x['psi']) + sin(x['theta'])*cos(x['phi'])*cos(x['psi'])) + x['magze']*cos(x['phi'])*cos(x['theta'])
        H[1, 7] = (x['magxe']*cos(x['psi'])*cos(x['theta']) - x['magze']*sin(x['theta']))*sin(x['phi'])
        H[1, 8] = -x['magxe']*(sin(x['phi'])*sin(x['psi'])*sin(x['theta']) + cos(x['phi'])*cos(x['psi']))
        H[1, 10] = sin(x['phi'])*sin(x['theta'])*cos(x['psi']) - sin(x['psi'])*cos(x['phi'])
        H[1, 11] = sin(x['phi'])*cos(x['theta'])
        H[2, 6] = -x['magxe']*sin(x['phi'])*sin(x['theta'])*cos(x['psi']) + x['magxe']*sin(x['psi'])*cos(x['phi']) - x['magze']*sin(x['phi'])*cos(x['theta'])
        H[2, 7] = (x['magxe']*cos(x['psi'])*cos(x['theta']) - x['magze']*sin(x['theta']))*cos(x['phi'])
        H[2, 8] = x['magxe']*(sin(x['phi'])*cos(x['psi']) - sin(x['psi'])*sin(x['theta'])*cos(x['phi']))
        H[2, 10] = sin(x['phi'])*sin(x['psi']) + sin(x['theta'])*cos(x['phi'])*cos(x['psi'])
        H[2, 11] = cos(x['phi'])*cos(x['theta'])
        i = self.sensori('magx')
        if self.mode == 'gnd':
            K = self.gnd.update_sensors(H, list(range(i, i+3)))
        elif self.mode == 'air':
            K = self.air.update_sensors(H, list(range(i, i+3)))
        else:
            raise Exception("Invalid mode")

        self.x = self.x + K.dot(y)
        assert(self.x.shape == (self.nstates, 1))

    def update_TAS(self, TAS):

        H = np.zeros((1, self.nstates))
        H[0, self.statei('TAS')] = 1

        if self.mode == 'gnd':
            K = self.gnd.update_sensors(H, [self.sensori('TAS')])
        elif self.mode == 'air':
            K = self.air.update_sensors(H, [self.sensori('TAS')])

        y = TAS - self.x[self.statei('TAS'), 0]

        self.x = self.x + K.dot(np.array([[y]]))
        assert(self.x.shape == (self.nstates, 1))

    def __repr__(self):
        s = ''
        for n, name in enumerate(self.state_names):
            s += f"{name:<7s}{self.x[n,0]:10.2e}\n"
        return s
