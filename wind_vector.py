

import numpy as np


head_track = 360
head_mag = 10

gndspd = 250
tas = 235

# wind = total - aircraft
vtot = np.array([np.cos(head_track*np.pi/180), np.sin(head_track*np.pi/180)])*gndspd
vaircraft = np.array([np.cos(head_mag*np.pi/180), np.sin(head_mag*np.pi/180)])*tas

vwind = vtot - vaircraft

print(vwind)
print(np.linalg.norm(vwind), np.arctan2(vwind[1], vwind[0])*180/np.pi)
