
#include "kalman.h"


Kalman::Kalman() {
  
  // Setup P matrix
  P = Matrix<float, NSTATES, NSTATES>::Zero();
  // P(I_Q0, I_Q0) = pow(.3, 2);
  // P(I_Q1, I_Q1) = pow(.1, 2);
  // P(I_Q2, I_Q2) = pow(.1, 2);
  // P(I_Q3, I_Q3) = 100;  // Allows for faster init of mag to heading
// 
  // for (int i=0; i < 3; i++) {
  //   // 2 degree error
  //   // sin(2deg) = .03
  //   P(I_AX+i, I_AX+i) = pow(.03*g, 2);
  // }

  float aerr_x = pow(.01*g, 2.0);  // confidence that ax, ay, az is from phi/TAS
  float aerr_y = pow(.1*g, 2.0);
  float aerr_z = pow(.01*g, 2.0);
  float werr_x = pow(10*PI/180, 2) / 1;
  float werr_y = pow(10*PI/180, 2) / 1;
  float werr_z = pow(2*PI/180, 2) / 1;

  Matrix<float, NSTATES, 1> tmpv;
  // Build Q matrix
  tmpv << 0, 0, 0, 0, 
          aerr_x, aerr_y, aerr_z,
          werr_x, werr_y, werr_z;
  Q = tmpv.asDiagonal();

  // Initialize state vector
  x = Matrix<float, NSTATES, 1>::Zero();
  x(I_Q0) = 1.0;
  x(I_AZ) = -g;
}


void Kalman::predict(float dt) {
  Eigen::Quaternion<float> q(x(0), x(1), x(2), x(3));
  
  float roll = q2roll(q);
  Eigen::Matrix<float, 3, 1> ac = (q*Eigen::Quaternion<float>(0, 0, 1.0, 0)*q.inverse()).vec();
  ac(2) = 0;
  ac.normalize();
  ac *= g*tan(roll);

  Eigen::Matrix<float, 3, 1> ag(0, 0, -g);
  Eigen::Matrix<float, 3, 1> a = ag + ac;

  // Rotate into body coord sys
  Eigen::Matrix<float, 3, 1> ab = (q.inverse()*Eigen::Quaternion<float>(0, a(0), a(1), a(2))*q).vec();

  Eigen::Quaternion<float> qd(1, x(I_WX)*.5*dt, x(I_WY)*.5*dt, x(I_WZ)*.5*dt);

  q = q * qd;
  q.normalize();

  // update state vector
  for (int i=0; i < 3; i++)
    x(I_AX+i) = ab(i);

  x(I_Q0) = q.w();
  x(I_Q1) = q.x();
  x(I_Q2) = q.y();
  x(I_Q3) = q.z();

  Matrix<float, NSTATES, NSTATES> F = calcF(dt);
  P = F * P * F.transpose() + Q*dt;
}


void Kalman::update_accel(Matrix<float, 3, 1> a) {
  Matrix<float, 3, 1> y = a - x.block(I_AX, 0, 3, 1);
  Matrix<float, 3, NSTATES> H = Matrix<float, 3, NSTATES>::Zero();
  H(0,4) = 1;
  H(1,5) = 1;
  H(2,6) = 1;
  Matrix<float, 3, 3> S = H * P * H.transpose() + Eigen::Matrix<float, 3, 3>::Identity() * Raccel;
  Matrix<float, NSTATES, 3> K = P * H.transpose() * S.inverse();
  x = x + K*y;
  P = (Eigen::Matrix<float, NSTATES, NSTATES>::Identity() - K*H)*P;
  q_normalize();
}


void Kalman::update_gyro(Matrix<float, 3, 1> w) {
  Matrix<float, 3, 1> y = w - x.block(I_WX, 0, 3, 1);
  Matrix<float, 3, NSTATES> H = Matrix<float, 3, NSTATES>::Zero();
  H(0,7) = 1;
  H(1,8) = 1;
  H(2,9) = 1;
  Matrix<float, 3, 3> S = H * P * H.transpose() + Eigen::Matrix<float, 3, 3>::Identity() * Rgyro;
  Matrix<float, NSTATES, 3> K = P * H.transpose() * S.inverse();
  x = x + K*y;
  P = (Eigen::Matrix<float, NSTATES, NSTATES>::Identity() - K*H)*P;
  q_normalize();
}


void Kalman::update_mag(Matrix<float, 3, 1> m)
{
  Quaternion<float> q(x(I_Q0), x(I_Q1), x(I_Q2), x(I_Q3));
  float roll = q2roll(q);
  float pitch = q2pitch(q);
  float heading = q2heading(q);

  // Remove roll and pitch from sensor reading to compute sensor heading vector
  Quaternion<float> qtmp = Quaternion<float>(AngleAxis<float>(roll, Matrix<float, 3, 1>(1.0, 0, 0))) *
                           Quaternion<float>(AngleAxis<float>(pitch, Matrix<float, 3, 1>(0, 1.0, 0)));
  Matrix<float, 3, 1> sensor_heading = (qtmp * Quaternion<float>(0, m(0), m(1), m(2)) * qtmp.inverse()).vec();
  sensor_heading(2) = 0;
  sensor_heading.normalize();
  sensor_heading(1) *= -1;

  Matrix<float, 2, 1> y = sensor_heading.block(0, 0, 2, 1) - Matrix<float, 2, 1>(cos(heading), sin(heading));

  Matrix<float, 2, NSTATES> H = calc_mag_H();
  Matrix<float, 2, 2> S = H * P * H.transpose() + Eigen::Matrix<float, 2, 2>::Identity() * Rmag;
  Matrix<float, NSTATES, 2> K = P * H.transpose() * S.inverse();
  x = x + K*y;
  P = (Eigen::Matrix<float, NSTATES, NSTATES>::Identity() - K*H)*P;
  q_normalize();
}


void Kalman::q_normalize() {
  Quaternion<float> q(x(I_Q0), x(I_Q1), x(I_Q2), x(I_Q3));
  q.normalize();
  x(I_Q0) = q.w();
  x(I_Q1) = q.x();
  x(I_Q2) = q.y();
  x(I_Q3) = q.z();
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

  float x0 = (1.0/2.0)*dt;
  float x1 = wx*x0;
  float x2 = -x1;
  float x3 = wy*x0;
  float x4 = -x3;
  float x5 = wz*x0;
  float x6 = -x5;
  float x7 = q1*x0;
  float x8 = -x7;
  float x9 = q2*x0;
  float x10 = -x9;
  float x11 = q3*x0;
  float x12 = -x11;
  float x13 = q0*x0;
  float x14 = 1.0*pow(q0, 2);
  float x15 = pow(q1, 2);
  float x16 = 1.0*x15;
  float x17 = pow(q2, 2);
  float x18 = 1.0*x17;
  float x19 = 1.0*pow(q3, 2);
  float x20 = x14 - x16 + x18 - x19;
  float x21 = q1*q2;
  float x22 = q0*q3;
  float x23 = x21 - x22;
  float x24 = 0.25*pow(x20, 2) + pow(x23, 2);
  float x25 = pow(x24, -1.0/2.0);
  float x26 = -2*x15 - 2*x17 + 1;
  float x27 = 1.0/x26;
  float x28 = x25*x27;
  float x29 = 1.0*x28;
  float x30 = x20*x29;
  float x31 = g*q3;
  float x32 = q1*x31;
  float x33 = x30*x32;
  float x34 = g*q0;
  float x35 = 2*q0*q1 + 2*q2*q3;
  float x36 = 2.0*x21 - 2.0*x22;
  float x37 = x35*x36;
  float x38 = 0.5*x37;
  float x39 = 0.5*x20;
  float x40 = x27/pow(x24, 3.0/2.0);
  float x41 = x40*(-q0*x39 + q3*x23);
  float x42 = x38*x41;
  float x43 = x35*x39;
  float x44 = x41*x43;
  float x45 = g*x28;
  float x46 = x38*x45;
  float x47 = q1*x34;
  float x48 = x29*x36;
  float x49 = x46 + x47*x48;
  float x50 = x31*x44 + x33 + x34*x42 + x49;
  float x51 = x16*x45;
  float x52 = g*q2;
  float x53 = q1*x52;
  float x54 = x30*x53;
  float x55 = g*q1;
  float x56 = x29*x35;
  float x57 = q0*x52;
  float x58 = x56*x57;
  float x59 = x32*x56;
  float x60 = x58 - x59;
  float x61 = x36*x51 + x42*x55 + x44*x52 + x54 + x60;
  float x62 = x32*x48;
  float x63 = x35*x45;
  float x64 = x14*x63;
  float x65 = x19*x63;
  float x66 = x30*x47;
  float x67 = x39*x63;
  float x68 = -x31*x42 + x34*x44 - x62 + x64 + x65 + x66 + x67;
  float x69 = x47*x56;
  float x70 = q3*x52;
  float x71 = x56*x70;
  float x72 = -g - x69 - x71;
  float x73 = x48*x53 + x72;
  float x74 = -x20*x51 + x42*x52 - x44*x55 + x73;
  float x75 = x28*x38;
  float x76 = x34*x75;
  float x77 = x28*x43;
  float x78 = x31*x77;
  float x79 = x52 + x76 + x78;
  float x80 = x14*x45;
  float x81 = x40*(q1*x39 - q2*x23);
  float x82 = x38*x81;
  float x83 = x43*x81;
  float x84 = 2.0*x25/pow(x26, 2);
  float x85 = x47*x84;
  float x86 = x20*x35;
  float x87 = x84*x86;
  float x88 = 1.0*x22*x45;
  float x89 = x20*x88 + x60;
  float x90 = x31*x83 + x32*x87 + x34*x82 + x36*x80 + x37*x85 + x89;
  float x91 = x30*x57;
  float x92 = g*x84;
  float x93 = x15*x92;
  float x94 = x53*x87;
  float x95 = x37*x93 + x49 + x52*x83 + x55*x82 + x91 + x94;
  float x96 = x36*x88;
  float x97 = x37*x84;
  float x98 = x20*x80 - x31*x82 - x32*x97 + x34*x83 + x72 + x85*x86 - x96;
  float x99 = x48*x57;
  float x100 = x18*x63 + x35*x51 + x53*x97;
  float x101 = x100 + x52*x82 - x55*x83 - x66 - x67 - x86*x93 + x99;
  float x102 = -x31 + x52*x77 + x55*x75;
  float x103 = x52*x75;
  float x104 = x55*x77;
  float x105 = x48*x70;
  float x106 = x17*x92;
  float x107 = x40*(-q1*x23 - q2*x39);
  float x108 = x107*x38;
  float x109 = x107*x43;
  float x110 = x105 + x106*x37 + x108*x52 - x109*x55 - x33 + x46 - x94;
  float x111 = x19*x45;
  float x112 = -x108*x31 + x109*x34 - x111*x36 + x57*x87 - x70*x97 + x89;
  float x113 = g + x108*x34 + x109*x31 + x111*x20 + x57*x97 + x69 + x70*x87 + x71 + x96;
  float x114 = x30*x70 + x67;
  float x115 = x100 + x106*x86 + x108*x55 + x109*x52 + x114 + x62;
  float x116 = x40*(q0*x23 + q3*x39);
  float x117 = x116*x38;
  float x118 = x116*x43;
  float x119 = x114 + x117*x34 + x118*x31 - x64 - x65 + x99;
  float x120 = x18*x45;
  float x121 = x117*x55 + x118*x52 + x120*x20 + x73;
  float x122 = -x105 - x117*x31 + x118*x34 - x46 + x91;
  float x123 = x117*x52 - x118*x55 + x120*x36 - x54 - x58 + x59;
  float x124 = x34*x77;
  float x125 = x31*x75;
  float x126 = x124 - x125 - x55;
  float x127 = x103 - x104 - x34;
  F(0, 0) = 1;
  F(0, 1) = x2;
  F(0, 2) = x4;
  F(0, 3) = x6;
  F(0, 7) = x8;
  F(0, 8) = x10;
  F(0, 9) = x12;
  F(1, 0) = x1;
  F(1, 1) = 1;
  F(1, 2) = x5;
  F(1, 3) = x4;
  F(1, 7) = x13;
  F(1, 8) = x12;
  F(1, 9) = x9;
  F(2, 0) = x3;
  F(2, 1) = x6;
  F(2, 2) = 1;
  F(2, 3) = x1;
  F(2, 7) = x11;
  F(2, 8) = x13;
  F(2, 9) = x8;
  F(3, 0) = x5;
  F(3, 1) = x3;
  F(3, 2) = x2;
  F(3, 3) = 1;
  F(3, 7) = x10;
  F(3, 8) = x7;
  F(3, 9) = x13;
  F(4, 0) = q0*x50 + q1*x61 - q2*x74 + q3*x68 + x79;
  F(4, 1) = q0*x90 + q1*x95 - q2*x101 + q3*x98 + x102;
  F(4, 2) = q0*x113 + q1*x115 - q2*x110 + q3*x112 - x103 + x104 + x34;
  F(4, 3) = q0*x119 + q1*x121 - q2*x123 + q3*x122 + x126;
  F(5, 0) = q0*x68 + q1*x74 + q2*x61 - q3*x50 + x126;
  F(5, 1) = q0*x98 + q1*x101 + q2*x95 - q3*x90 + x127;
  F(5, 2) = q0*x112 + q1*x110 + q2*x115 - q3*x113 + x102;
  F(5, 3) = q0*x122 + q1*x123 + q2*x121 - q3*x119 - x52 - x76 - x78;
  F(6, 0) = q0*x74 - q1*x68 + q2*x50 + q3*x61 + x127;
  F(6, 1) = q0*x101 - q1*x98 + q2*x90 + q3*x95 - x124 + x125 + x55;
  F(6, 2) = q0*x110 - q1*x112 + q2*x113 + q3*x115 + x79;
  F(6, 3) = q0*x123 - q1*x122 + q2*x119 + q3*x121 + x102;
  F(7, 7) = 1;
  F(8, 8) = 1;
  F(9, 9) = 1;

  return F;
}


Matrix<float, 2, NSTATES> Kalman::calc_mag_H() {
  Eigen::Matrix<float, 2, NSTATES> H = Matrix<float, 2, NSTATES>::Zero();
  float q0, q1, q2, q3;
  q0 = x(I_Q0);
  q1 = x(I_Q1);
  q2 = x(I_Q2);
  q3 = x(I_Q3);

  float x0 = 2*q3;
  float x1 = 2*q0;
  float x2 = 2*q1;
  float x3 = q2*x2 + q3*x1;
  float x4 = -2*pow(q2, 2) - 2*pow(q3, 2) + 1;
  float x5 = pow(x3, 2);
  float x6 = pow(x4, 2) + x5;
  float x7 = pow(x6, -3.0/2.0);
  float x8 = x4*x7;
  float x9 = x3*x8;
  float x10 = 2*q2;
  float x11 = pow(x6, -1.0/2.0);
  float x12 = 4*q2;
  float x13 = x12*x4 - x2*x3;
  float x14 = 4*q3;
  float x15 = -x1*x3 + x14*x4;
  float x16 = 2*x11;
  float x17 = x5*x7;
  float x18 = x3*x7;
  H(0, 0) = -x0*x9;
  H(0, 1) = -x10*x9;
  H(0, 2) = -x11*x12 + x13*x8;
  H(0, 3) = -x11*x14 + x15*x8;
  H(1, 0) = q3*x16 - x0*x17;
  H(1, 1) = q2*x16 - x10*x17;
  H(1, 2) = x11*x2 + x13*x18;
  H(1, 3) = x1*x11 + x15*x18;
  return H;
}


ostream& operator<<(ostream &ofs, const Kalman &k) {
  for (int i=0; i < NSTATES; i++)
    ofs << k.x(i,0) << ",";
  return ofs;
};


float q2roll(Eigen::Quaternion<float> q) {
  return atan2(2*(q.w()*q.x() + q.y()*q.z()),
                 1-2*(q.x()*q.x() + q.y()*q.y()));
}


float q2pitch(Eigen::Quaternion<float> q) {
  return asin(2*(q.w()*q.y() - q.z()*q.x()));
}


float q2heading(Eigen::Quaternion<float> q) {
  return atan2(2*(q.w()*q.z() + q.x()*q.y()),
               1-2*(q.y()*q.y() + q.z()*q.z()));
}