



#define CONST_RHO0      1.225 // kg/m^3  (sea level standard temperature air density)
#define CONST_P0        101325 // Pa (sea level std pressure)
#define CONST_L         0.0065  // K/m  (std lapse rate)
#define CONST_T0        288.15  // C (sea level std temperature)
#define CONST_g         9.81
#define CONST_R         8.31447  // J/mol/K
#define CONST_M         0.0289654 // kg/mol molar mass
#define CONV_INHG2PA    3386.39
#define CONV_MS2KNOTS   1.94384
#define CONV_FT2M       (1.0/3.28084)
#define CONV_M2FT       3.28084

void airspeed_altitude(float abs_press, float diff_press, float alt_setting, float oat,
                       float *altitude, float *ias, float *tas);
