
import numpy as np

def head360(theta):
    """
    theta is in radians

    returns angle in degrees between 0 and 360
    """
    if isinstance(theta, np.ndarray):
        return np.vectorize(head360)(psi)
    if theta < 0:
        return head360(theta + 2*np.pi)
    elif theta > 2*np.pi:
        return head360(theta - 2*np.pi)
    else:
        return theta*180/np.pi
