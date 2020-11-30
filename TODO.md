

1. Filter initialization
    a. mag_update
    b. const w or predicted w from turn?
    c. CAN debug
    d. With real sensors, do we need to add bias states?
4. Port to STM32
    a. Airspeed calcs
    b. Altitude calcs
    c. VS calcs
    d. Filter altitude simple IIR?
    e. Debug output
        over CAN? 
    f. Be able to flash STM32 from linux
    g. Predict timer for KF
    h. Send CAN to STM32
        - Mag hard/soft iron compensation
    i. Filter tuning
        - Accel Q needs to be higher
        - Check Q*dt
5. Flight test
    a. Measure actual sensor noise
    b. Mag calibration