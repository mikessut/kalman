
#include "kalman.h"


Kalman::Kalman() {
  
  // Setup P matrix
  P = Matrix<float, NSTATES, NSTATES>::Zero();
  P(I_Q0, I_Q0) = pow(.3, 2);
  P(I_Q1, I_Q1) = pow(.1, 2);
  P(I_Q2, I_Q2) = pow(.1, 2);
  P(I_Q3, I_Q3) = 100;  // Allows for faster init of mag to heading

  for (int i=0; i < 3; i++) {
    // 2 degree error
    // sin(2deg) = .03
    P(I_AX+i, I_AX+i) = pow(.03*g, 2);
  }

  float aerr_x = pow(.4*g*DT, 2.0);  // confidence that ax, ay, az is from phi/TAS
  float aerr_y = pow(.4*g*DT, 2.0);
  float aerr_z = pow(.2*g*DT, 2.0);
  float werr_x = pow(10*PI/180, 2) / 1;
  float werr_y = pow(10*PI/180, 2) / 1;
  float werr_z = pow(.1*PI/180, 2) / 1;

  Matrix<float, NSTATES, 1> tmpv;
  // Build Q matrix
  tmpv << 0, 0, 0, 0, 
          aerr_x, aerr_y, aerr_z,
          werr_x, werr_y, werr_z;
  Q = tmpv.asDiagonal();

  // Initialize state vector
  x = Matrix<float, NSTATES, 1>::Zero();
  x(I_Q0, 0) = 1.0;
  x(I_AZ, 0) = -g;
}


void Kalman::predict(float dt) {
  Matrix<float, NSTATES, 1> lastx = x;

  Eigen::Quaternion<float> q(x(0), x(1), x(2), x(3));
  Eigen::Quaternion<float> qd(1, x(I_WX)*.5, x(I_WY)*.5, x(I_WZ)*.5);

  q = q * qd;
  q.normalize();
    
  float phi = roll(q);
  // Centripetal acceleration direction (right wing into inertial coord sys)
  Eigen::Matrix<float, 3, 1> ac = (q*Eigen::Quaternion<float>(0, 0, -1.0, 0)*q.inverse()).vec();
  ac(2) = 0;
  ac.normalize();
  ac *= g*tan(phi);

  Eigen::Matrix<float, 3, 1> ag(0, 0, g);
  Eigen::Matrix<float, 3, 1> a = ag + ac;

  // Rotate into body coord sys
  Eigen::Matrix<float, 3, 1> ab = (q.inverse()*Eigen::Quaternion<float>(0, a(0), a(1), a(2))*q).vec();

  // update state vector
  for (int i=0; i < 3; i++)
    x(I_AX+i) = ab(i);

  x(I_Q0) = q.w();
  x(I_Q1) = q.x();
  x(I_Q2) = q.y();
  x(I_Q3) = q.z();

  Matrix<float, NSTATES, NSTATES> F = calcF(dt);
  P = F * P * F.transpose() + Q;
}


void Kalman::update_accel(Matrix<float, 3, 1> a) {
  Matrix<float, 3, 1> y = a - x.block(I_AX, 0, 3, 1);
  Matrix<float, 3, NSTATES> H = Matrix<float, 3, NSTATES>::Zero();
  H(0,4) = 1;
  H(1,5) = 1;
  H(2,6) = 1;
  // (3x10)*(10x10)*(10x3)
  // 3x3
  Matrix<float, 3, 3> S = H * P * H.transpose() + Eigen::Matrix<float, 3, 3>::Identity() * Raccel;
  // (10x10)*(10x3)*(3x3)
  Matrix<float, NSTATES, 3> K = P * H.transpose() * S.inverse();
  x = x + K*y;
  P = (Eigen::Matrix<float, NSTATES, NSTATES>::Identity() - K*H)*P;
}


void Kalman::update_gyro(Matrix<float, 3, 1> w) {
  Matrix<float, 3, 1> y = w - x.block(I_WX, 0, 3, 1);
  Matrix<float, 3, NSTATES> H = Matrix<float, 3, NSTATES>::Zero();
  H(0,7) = 1;
  H(1,8) = 1;
  H(2,9) = 1;
  Matrix<float, 3, 3> S = H * P * H.transpose() + Eigen::Matrix<float, 3, 3>::Identity() * Raccel;
  Matrix<float, NSTATES, 3> K = P * H.transpose() * S.inverse();
  x = x + K*y;
  P = (Eigen::Matrix<float, NSTATES, NSTATES>::Identity() - K*H)*P;
}


void Kalman::update_mag(Matrix<float, 3, 1> m)
{
  
}

Eigen::Matrix<float, NSTATES, NSTATES> Kalman::calcF(float dt) {
  Eigen::Matrix<float, NSTATES, NSTATES> F = Matrix<float, NSTATES, NSTATES>::Zero();
  float q0, q1, q2, q3;
  float wx, wy, wz;
  q0 = x(I_Q0);
  q1 = x(I_Q1);
  q2 = x(I_Q2);
  q3 = x(I_Q3);
  wx = x(I_WX);
  wy = x(I_WY);
  wz = x(I_WZ);

  return F;
}


ostream& operator<<(ostream &ofs, const Kalman &k) {
  for (int i=0; i < NSTATES; i++)
    ofs << k.x(i,0) << ",";
  return ofs;
};


float roll(Eigen::Quaternion<float> q) {
  return atan2(2*(q.w()*q.x() + q.y()*q.z()),
                 1-2*(q.x()*q.x() + q.y()*q.y()));
}