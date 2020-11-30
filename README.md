Overview
========

This is an effort to derive and implement an Extended Kalman Filter to use for
an arcraft AHRS system.  

1. Derivations of the Kalman filter equations are presented in Jupyter Notebooks.
2. The Kalman filter is implemented in both python/numpy as well as c++.
   a. The KF can be tested with output from the [Flightgear simulator](https://www.flightgear.org/) using 
      `flightgear.py`.  This script can run the KF with either the numpy or c++ implementations
   b. 

References
==========

https://core.ac.uk/download/pdf/41774601.pdf

Requirements
============

1. `pyEfis <https://github.com/makerplane/pyEfis>`_
2. `FIX-Gateway <https://github.com/makerplane/FIX-Gateway>`_

Test data set
=============

A dataset collected from a phone is used for development/testing.  This data
consists of accels, gyros, magnetometer and GPS data.  Speed and altitude is derived from the
GPS data.

Know issues/TODOs
=================

1. Slip Skid
2. Rate of turn
3. Airspeed (as derived from GPS) is very jumpy in the test data set.
4. Implementation with baro and pitot sensors
  a. Derive TAS from IAS using OAT/pressure
5. Should the magnetic vector be included as states? Or perhaps use the IGRF model to determine
   magnetic inclination.
6. Is wind vector desired? Probably -- that requires a wind frame and then body frame?  This
   definitely requires a magnetic declination model.
