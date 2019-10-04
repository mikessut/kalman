
from kalman import ft2m
import numpy as np
from util import head360



RHO0 = 1.225 # kg/m^3
P0 = 101325 # Pa (sea level std pressure)
L = .0065  # K/m  (std lapse rate)
T0 = 288.15  # C (sea level std temperature)
g = 9.81
R = 8.31447  # J/mol/K
M = 0.0289654 # kg/mol molar mass
inHg2PA = 3386.39

def tas(ias, TdegC, altitude, altimeter=29.92126):
    """
    temp: degC
    alt:  indicated altitude in feet
    https://en.wikipedia.org/wiki/True_airspeed
    https://en.wikipedia.org/wiki/Density_of_air

    Actual pressure from alt setting.
    https://wahiduddin.net/calc/density_altitude.htm (eqn 16)
    This reference was most useful getting results to match the online calculator.

    Output is very close to calculator at: https://www.dauntless-soft.com/products/Freebies/TrueAirspeedCalculator/
    """
    T = 273.15 + TdegC
    h = altitude*ft2m
    p_sealevel = altimeter*inHg2PA
    p1 = p_sealevel*(1-L*h/T0)**(g*M/R/L)
    p = (p_sealevel**(L*R/g/M) - L/T0*P0**(L*R/g/M)*h)**(g*M/L/R)
    print(p1, p)
    rho = p*M/R/T

    PA = (1-(p/P0)**(R*L/g/M))*T0/L/ft2m
    rho_DA = p1*M/R/T
    p_DA = rho_DA/M*R*(T0 - L*h)
    #DA = (1-(p_DA/P0)**(R*L/g/M))*T0/L/ft2m
    DA = T0/L*(1-(R*T0*rho/M/P0)**(L*R/(g*M-L*R)))/ft2m

    return ias*np.sqrt(RHO0/rho), PA, DA


def wind_vector(v_total, v_aircraft, mag_angle=True):
    """
    inputs are two component vectors

    Wind as a vector is returned
    """
    # wind = total - aircraft


    vwind = v_total - v_aircraft
    if mag_angle:
        return np.linalg.norm(vwind), head360(np.arctan2(vwind[1], vwind[0]))
    else:
        return vwind



head_track = 360
head_mag = 10

gndspd = 250
tas1 = 235

vtot = np.array([np.cos(head_track*np.pi/180), np.sin(head_track*np.pi/180)])*gndspd
vaircraft = np.array([np.cos(head_mag*np.pi/180), np.sin(head_mag*np.pi/180)])*tas1

print(wind_vector(vtot,vaircraft))
