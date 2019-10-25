
#include "pressure.h"
#include <math.h>


void airspeed_altitude(float abs_press, float diff_press, float alt_setting, float oat,
                       float *altitude, float *ias, float *tas)
{
  float p_sealevel = alt_setting*CONV_INHG2PA;
  float h = -(pow(abs_press, CONST_L*CONST_R/CONST_g/CONST_M)
              - pow(p_sealevel, CONST_L*CONST_R/CONST_g/CONST_M))*CONST_T0/CONST_L/pow(CONST_P0, CONST_L*CONST_R/CONST_g/CONST_M);
  float p = pow(pow(p_sealevel, CONST_L*CONST_R/CONST_g/CONST_M)
                - CONST_L/CONST_T0*pow(CONST_P0, CONST_L*CONST_R/CONST_g/CONST_M)*h, CONST_g*CONST_M/CONST_L/CONST_R);

  float rho = p*CONST_M/CONST_R/(273.15 + oat);
  *altitude = h * CONV_M2FT;
  *ias = sqrt(diff_press*2/CONST_RHO0)*CONV_MS2KNOTS;
  *tas = (*ias) * sqrt(CONST_RHO0/rho);
}
