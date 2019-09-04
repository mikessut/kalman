
#include "dcm.h"
#include <cmath>


Eigen::Matrix3f dcm(float phi, float theta, float psi) {
  Eigen::Matrix3f R;
  R(0,0) = cos(theta)*cos(psi);
  R(0,1) = cos(theta) * sin(psi);
  R(0,2) = -sin(theta);
  R(1,0) = sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi);
  R(1,1) = sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi);
  R(1,2) = sin(phi) * cos(theta);
  R(2,0) = cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi);
  R(2,1) = cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi);
  R(2,2) = cos(phi) * cos(theta);
  return R;
}
