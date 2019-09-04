
from sympy import *
import numpy as np
from DCM_sym import HEB
import re

def jacobian(vector, vars, output_prefix=''):
    M = np.zeros((len(vector), len(vars)), dtype=object)
    for m in range(len(vector)):
        for n in range(len(vars)):
            M[m,n] = simplify(diff(vector[m], vars[n]))
            tmp = M[m,n]
            if tmp != 0:
                if MATLAB:
                    tmp = str(tmp).replace('**', "^")
                    print(f'F({m+1},{n+1}) = {tmp};')
                else:
                    for state_name in vars:
                        #tmp = str(tmp).replace(str(state_name), f"x['{str(state_name)}']")
                        tmp = re.sub(f"([^\w]|^)({state_name})([^\w]|$)", f"\\1x['{str(state_name)}']\\3", str(tmp))
                    print(f'{output_prefix}[{m}, {n}] = {tmp}')
        print()
    return M

def sub_py_dict(vector, vars, output_prefix=''):
    for m in range(len(vector)):
        if MATLAB:
            print(f'x({m+1}) = {x[m, 0]};')
        else:
            tmp = str(vector[m])
            for state_name in statevars:
                #tmp = tmp.replace(str(state_name), f"x['{str(state_name)}']")
                tmp = re.sub(f"([^\w]|^)({state_name})([^\w]|$)", f"\\1x['{str(state_name)}']\\3", tmp)
            print(f'{output_prefix}[{m}, 0] = {tmp}')

# x(0)     Body-axis roll rate, pr, rad/s
# x(1)     Body-axis pitch rate, qr, rad/s
# x(2)     Body-axis yaw rate, rr,rad/s
# x(3)     Body axis accel-x (positve out the nose)
# x(4)     Body axis accel-y (positive out the right wing)
# x(5)     Body axis accel-z (positive towards the ground)
# x(6)     Roll angle of body WRT Earth, phir, rad
# x(7)     Pitch angle of body WRT Earth, thetar, rad
# x(8)     Yaw angle of body WRT Earth, psir, rad
# x(9)     Speed m/s
# Mag x component in earth coords is by definition 0
# x(10)    Mag x earth
# x(11)    Mag z earth
p, q, r = symbols(['p', 'q', 'r'])
phi, theta, psi = symbols(['phi', 'theta', 'psi'])
ax, ay, az = symbols(['ax', 'ay', 'az'])
# gyro sensors
wx, wy, wz = symbols(['wx', 'wy', 'wz'])
magx, magy, magz = symbols(['magx', 'magy', 'magz'])

magxe, magze = symbols(['magxe', 'magze'])
TAS = symbols('TAS')
g = symbols('g')
dt = symbols('dt')
nstates = 12
statevars = np.array([p, q, r, ax, ay, az, phi, theta, psi, TAS, magxe, magze])
MATLAB = False

# x = np.vstack([p, q, r, phi, theta, psi, TAS])

rot = tan(phi)*g/TAS
rotb = HEB.dot(np.vstack([0,0,rot]))
gb = HEB.dot(np.vstack([0,0,-g]))

# centripital accel in earth coords
#ace = np.vstack([-g*tan(phi)*cos(psi+pi/2), -g*tan(phi)*sin(psi+pi/2), 0])
ace = np.vstack([-g*tan(phi)*sin(psi), g*tan(phi)*cos(psi), 0])
acb = HEB.dot(ace)

ab = acb + gb

x = np.zeros((nstates, 1), dtype=object)
#x[:3, 0] = [p, rotb[1,0], rotb[2,0]]
x[:3, 0] = [p, q, r]

# Body coords accel
x[3:6, [0]] = ab

# rotations in earth axes
rote = HEB.T.dot(np.vstack([p, q, r]))
# phi, theta, psi
x[6, 0] = phi + p*dt
x[7, 0] = theta + q*dt
x[8, 0] = psi + rot*dt  # or rote[2,0]*dt?

x[9, 0] = TAS + ax*dt

x[10, 0] = magxe
x[11, 0] = magze

sub_py_dict(x[:, 0], statevars, 'self.x')
print("F = np.zeros((self.nstates, self.nstates))")
F = jacobian(x[:, 0], statevars, 'F')

# H matrix
# H rows: p, q, r, ax, ay, az, magx, magy, magz
sensors = [wx, wy, wz, ax, ay, az, TAS, magx, magy, magz]

# H cols: states
magb = HEB.dot(np.vstack([magxe, 0, magze]))
Hx = np.vstack([p,
                q,
                r,
                ax,
                ay,
                az,
                TAS,
                magb])

print("mag in body:")
for n in range(3):
    print(f'mag{"xyz"[n]}b = {magb[n,0]}')
#mag2psi = atan2(mage[1,0], mage[0,0])
jacobian(Hx[:,0], statevars, 'H')

# Use a different prediction model on the ground.
xgnd = np.zeros((nstates, 1), dtype=object)
xgnd[2, 0] = r
xgnd[3:6, 0] = [ax, ay, az]
xgnd[8, 0] = psi + r*dt
xgnd[9, 0] = TAS
xgnd[10:12, 0] = [magxe, magze]

print()
print("Ground model:")
sub_py_dict(xgnd[:,0], statevars, 'self.xgnd')
print()
jacobian(xgnd[:,0], statevars, 'self.F')
