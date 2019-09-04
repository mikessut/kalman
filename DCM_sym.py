
# sinR = sin(Phi);
# cosR = cos(Phi);
# sinP = sin(Theta);
# cosP = cos(Theta);
# sinY = sin(Psi);
# cosY = cos(Psi);
#
# H(1,1) = cosP * cosY;
# H(1,2) = cosP * sinY;
# H(1,3) = -sinP;
# H(2,1) = sinR * sinP * cosY - cosR * sinY;
# H(2,2) = sinR * sinP * sinY + cosR * cosY;
# H(2,3) = sinR * cosP;
# H(3,1) = cosR * sinP * cosY + sinR * sinY;
# H(3,2) = cosR * sinP * sinY - sinR * cosY;
# H(3,3) = cosR * cosP;

import numpy as np
from sympy import *

phi, theta, psi = symbols(['phi', 'theta', 'psi'])
sinR = sin(phi)
cosR = cos(phi)
sinP = sin(theta)
cosP = cos(theta)
sinY = sin(psi)
cosY = cos(psi)
HEB2 = np.zeros((3,3), dtype=object)
HEB2[1-1,1-1] = cosP * cosY;
HEB2[1-1,2-1] = cosP * sinY;
HEB2[1-1,3-1] = -sinP;
HEB2[2-1,1-1] = sinR * sinP * cosY - cosR * sinY;
HEB2[2-1,2-1] = sinR * sinP * sinY + cosR * cosY;
HEB2[2-1,3-1] = sinR * cosP;
HEB2[3-1,1-1] = cosR * sinP * cosY + sinR * sinY;
HEB2[3-1,2-1] = cosR * sinP * sinY - sinR * cosY;
HEB2[3-1,3-1] = cosR * cosP;
HEB = np.array([[cosP * cosY, cosP * sinY, -sinP],
                [sinR * sinP * cosY - cosR * sinY, sinR * sinP * sinY + cosR * cosY, sinR * cosP],
                [cosR * sinP * cosY + sinR * sinY, cosR * sinP * sinY - sinR * cosY, cosR * cosP]])

for m in range(3):
    for n in range(3):
        assert(simplify(HEB[m,n] - HEB2[m,n]) == 0)


def derivation():
    # opposite signs from what is shown in: https://en.wikipedia.org/wiki/Davenport_chained_rotations
    # Not sure I understand why...
    R = np.array([[1, 0, 0],
                  [0, cosR, sinR],
                  [0, -sinR, cosR]])
    P =  np.array([[cosP, 0, -sinP],
                   [0, 1, 0],
                   [sinP, 0, cosP]])
    Y = np.array([[cosY, sinY, 0],
                  [-sinY, cosY, 0],
                  [0, 0, 1]])

    result = R.dot(P).dot(Y)
    print(result)
    for m in range(3):
        for n in range(3):
            assert(simplify(result[m,n] - HEB[m,n]) == 0)
