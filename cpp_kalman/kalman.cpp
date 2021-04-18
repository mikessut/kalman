#include "kalman.h"


Kalman::Kalman() {
  initialize();
}


void Kalman::initialize() {
  // Setup P matrix
  P = Matrix<float, NSTATES, NSTATES>::Zero();
  
  float aerr_x = pow(2*g, 2.0);  // confidence that ax, ay, az is from phi/TAS
  float aerr_y = pow(2*g, 2.0);
  float aerr_z = pow(4*g, 2.0);
  float werr_x = pow(500*PI/180, 2) / 1;
  float werr_y = pow(500*PI/180, 2) / 1;
  float werr_z = pow(50*PI/180, 2) / 1;

  Matrix<float, NSTATES, 1> tmpv;
  #ifdef BIAS_STATES
  float wberr = pow(.5*PI/180, 2) / 240.0;
  float aberr = pow(.1*G, 2) / 240.0;
  tmpv << 0, 0, 0, 0, 
          aerr_x, aerr_y, aerr_z,
          werr_x, werr_y, werr_z,
          wberr, wberr, wberr,
          aberr, aberr, aberr;
  #else
  tmpv << 0, 0, 0, 0, 
          aerr_x, aerr_y, aerr_z,
          werr_x, werr_y, werr_z;
  #endif
  // Build Q matrix
  Q = tmpv.asDiagonal();

  // Initialize state vector
  x = Matrix<float, NSTATES, 1>::Zero();
  x(I_Q0) = 1.0;
  x(I_AZ) = -g;
}


void Kalman::predict(float dt, float tas) {
  Eigen::Quaternion<float> q(x(0), x(1), x(2), x(3));

  tas = max(float(KF_MIN_SPEED_MS), tas);
  
  float roll = q2roll(q);
  Eigen::Matrix<float, 3, 1> ac = (q*Eigen::Quaternion<float>(0, 0, 1.0, 0)*q.inverse()).vec();
  ac(2) = 0;
  ac.normalize();
  float tan_roll = tan(roll);
  ac *= g*tan_roll;

  Eigen::Matrix<float, 3, 1> ag(0, 0, -g);
  Eigen::Matrix<float, 3, 1> a = ag + ac;

  // Rotate into body coord sys
  Eigen::Matrix<float, 3, 1> ab = (q.inverse()*Eigen::Quaternion<float>(0, a(0), a(1), a(2))*q).vec();

  // w prediction
  Matrix<float, 3, 1> wpredict = (q.inverse() * Quaternion<float>(0, 0, 0, g/tas*tan_roll) * q).vec();

  // q prediction
  Eigen::Quaternion<float> qd(1, x(I_WX)*.5*dt, x(I_WY)*.5*dt, x(I_WZ)*.5*dt);
  q = q * qd;
  q.normalize();

  // update state vector
  for (int i=0; i < 3; i++) {
    x(I_AX+i) = ab(i);
    x(I_WX+i) = wpredict(i);
  }

  x(I_Q0) = q.w();
  x(I_Q1) = q.x();
  x(I_Q2) = q.y();
  x(I_Q3) = q.z();

  Matrix<float, NSTATES, NSTATES> F = calcF(dt, tas);
  P = F * P * F.transpose() + Q*dt;
}


void Kalman::update_accel(Matrix<float, 3, 1> a) {
  #ifdef BIAS_STATES
  Matrix<float, 3, 1> y = a - x.block(I_AX, 0, 3, 1) - x.block(I_ABX, 0, 3, 1);
  #else
  Matrix<float, 3, 1> y = a - x.block(I_AX, 0, 3, 1);
  #endif
  Matrix<float, 3, NSTATES> H = Matrix<float, 3, NSTATES>::Zero();
  H(0,4) = 1;
  H(1,5) = 1;
  H(2,6) = 1;
  #ifdef BIAS_STATES
  H(0,13) = 1;
  H(1,14) = 1;
  H(2,15) = 1;
  #endif

  Matrix<float, 3, 3> S = H * P * H.transpose() + Eigen::Matrix<float, 3, 3>::Identity() * Raccel;
  Matrix<float, NSTATES, 3> K = P * H.transpose() * S.inverse();
  x = x + K*y;
  P = (Eigen::Matrix<float, NSTATES, NSTATES>::Identity() - K*H)*P;
  q_normalize();
}


void Kalman::update_gyro(Matrix<float, 3, 1> w) {
  #ifdef BIAS_STATES
  Matrix<float, 3, 1> y = w - x.block(I_WX, 0, 3, 1) - x.block(I_WBX, 0, 3, 1);
  #else
  Matrix<float, 3, 1> y = w - x.block(I_WX, 0, 3, 1); 
  #endif
  Matrix<float, 3, NSTATES> H = Matrix<float, 3, NSTATES>::Zero();
  H(0,7) = 1;
  H(1,8) = 1;
  H(2,9) = 1;
  #ifdef BIAS_STATES
  H(0,10) = 1;
  H(1,11) = 1;
  H(2,12) = 1;
  #endif
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
  Quaternion<float> qtmp = Quaternion<float>(AngleAxis<float>(pitch, Matrix<float, 3, 1>(0, 1.0, 0))) *
                           Quaternion<float>(AngleAxis<float>(roll, Matrix<float, 3, 1>(1.0, 0, 0)));
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


void Kalman::update_gps_bearing(float gps_heading)
{
  // Use vector representation of heading to avoid issues of wrapping around 0/360 deg
  // Represent heading as a vector
  Eigen::Matrix<float, 2, 1> meas(cos(gps_heading), sin(gps_heading));

  // Get heading from quaternion
  float state_heading = q2heading(Quaternion<float>(x(I_Q0), x(I_Q1), x(I_Q2), x(I_Q3)));
  Matrix<float, 2, 1> y = meas - Matrix<float, 2, 1>(cos(state_heading), sin(state_heading));

  Matrix<float, 2, NSTATES> H = Matrix<float, 2, NSTATES>::Zero();
  float q0 = x(I_Q0);
  float q1 = x(I_Q1);
  float q2 = x(I_Q2);
  float q3 = x(I_Q3);

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

  Matrix<float, 2, 2> S = H * P * H.transpose() + Eigen::Matrix<float, 2, 2>::Identity() * Rgps_heading;
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


Eigen::Matrix<float, NSTATES, NSTATES> Kalman::calcF(float dt, float tas) {
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
  float x14 = pow(q1, 2);
  float x15 = 2*x14;
  float x16 = pow(q2, 2);
  float x17 = 2*x16;
  float x18 = -x15 - x17 + 1;
  float x19 = 1.0/x18;
  float x20 = G*q3;
  float x21 = x19*x20;
  float x22 = 2*q0;
  float x23 = 2*q2;
  float x24 = q1*x23 - q3*x22;
  float x25 = pow(q0, 2);
  float x26 = pow(q3, 2);
  float x27 = -x14 + x16 + x25 - x26;
  float x28 = pow(x24, 2) + pow(x27, 2);
  float x29 = pow(x28, -1.0/2.0);
  float x30 = 2*x27;
  float x31 = q1*x30;
  float x32 = x29*x31;
  float x33 = x21*x32;
  float x34 = G*q0;
  float x35 = x19*x34;
  float x36 = 2*x24;
  float x37 = q3*x36;
  float x38 = x22*x27;
  float x39 = q1*x22 + q3*x23;
  float x40 = x39/pow(x28, 3.0/2.0);
  float x41 = x40*(x37 - x38);
  float x42 = x24*x41;
  float x43 = x35*x42;
  float x44 = x27*x41;
  float x45 = x21*x44;
  float x46 = G*x19;
  float x47 = x29*x39;
  float x48 = x24*x47;
  float x49 = x46*x48;
  float x50 = q1*x36;
  float x51 = x29*x50;
  float x52 = x35*x51;
  float x53 = x49 + x52;
  float x54 = x33 + x43 + x45 + x53;
  float x55 = x29*x46;
  float x56 = x15*x55;
  float x57 = G*q2;
  float x58 = x19*x57;
  float x59 = x32*x58;
  float x60 = G*q1;
  float x61 = x19*x60;
  float x62 = 2*x47;
  float x63 = q1*x21;
  float x64 = x62*x63;
  float x65 = x22*x47*x58;
  float x66 = -x64 + x65;
  float x67 = x24*x56 + x42*x61 + x44*x58 + x59 + x66;
  float x68 = x46*x47;
  float x69 = x27*x68;
  float x70 = x21*x51;
  float x71 = x32*x35;
  float x72 = 2*x68;
  float x73 = x25*x72;
  float x74 = x26*x72;
  float x75 = x73 + x74;
  float x76 = -x21*x42 + x35*x44 + x69 - x70 + x71 + x75;
  float x77 = q1*x35*x62;
  float x78 = q3*x58;
  float x79 = x62*x78;
  float x80 = -G - x77 - x79;
  float x81 = x51*x58 + x80;
  float x82 = -x27*x56 + x42*x58 - x44*x61 + x81;
  float x83 = x19*x48;
  float x84 = x34*x83;
  float x85 = x27*x47;
  float x86 = x21*x85;
  float x87 = x57 + x84 + x86;
  float x88 = x25*x55;
  float x89 = x36*x88;
  float x90 = x40*(-x23*x24 + x31);
  float x91 = x24*x90;
  float x92 = x35*x91;
  float x93 = x27*x90;
  float x94 = x21*x93;
  float x95 = pow(x18, -2);
  float x96 = 4*x95;
  float x97 = q1*x96;
  float x98 = x48*x97;
  float x99 = x34*x98;
  float x100 = x85*x97;
  float x101 = x100*x20;
  float x102 = x29*x38;
  float x103 = x102*x21;
  float x104 = x103 + x66;
  float x105 = x101 + x104 + x89 + x92 + x94 + x99;
  float x106 = x102*x58;
  float x107 = G*x95;
  float x108 = 4*x107;
  float x109 = x108*x14;
  float x110 = x100*x57;
  float x111 = x106 + x109*x48 + x110 + x53 + x58*x93 + x61*x91;
  float x112 = x22*x24;
  float x113 = x112*x29;
  float x114 = x113*x21;
  float x115 = -x114 + x80;
  float x116 = x100*x34 + x115 - x20*x98 - x21*x91 + x30*x88 + x35*x93;
  float x117 = x113*x58;
  float x118 = -x69;
  float x119 = x15*x68 + x17*x68 + x57*x98;
  float x120 = -x109*x85 + x117 + x118 + x119 + x58*x91 - x61*x93 - x71;
  float x121 = x19*x85;
  float x122 = x121*x57 - x20 + x60*x83;
  float x123 = x57*x83;
  float x124 = x121*x60;
  float x125 = x29*x58;
  float x126 = x125*x37;
  float x127 = -x33;
  float x128 = x108*x16;
  float x129 = x40*(-x23*x27 - x50);
  float x130 = x129*x24;
  float x131 = x129*x27;
  float x132 = -x110 + x126 + x127 + x128*x48 + x130*x58 - x131*x61 + x49;
  float x133 = x26*x55;
  float x134 = x133*x30;
  float x135 = x57*x96;
  float x136 = x135*x48;
  float x137 = q0*x136;
  float x138 = x135*x85;
  float x139 = q3*x138;
  float x140 = x130*x35;
  float x141 = x131*x21;
  float x142 = G + x114 + x134 + x137 + x139 + x140 + x141 + x77 + x79;
  float x143 = q0*x138 - q3*x136 + x104 - x130*x21 + x131*x35 - x133*x36;
  float x144 = q3*x30;
  float x145 = x125*x144;
  float x146 = x145 + x69;
  float x147 = x119 + x128*x85 + x130*x61 + x131*x58 + x146 + x70;
  float x148 = x40*(x112 + x144);
  float x149 = x148*x24;
  float x150 = x149*x35;
  float x151 = x148*x27;
  float x152 = x151*x21;
  float x153 = x117 + x146 + x150 + x152 - x73 - x74;
  float x154 = x17*x55;
  float x155 = x149*x61 + x151*x58 + x154*x27 + x81;
  float x156 = -x49;
  float x157 = x106 - x126 - x149*x21 + x151*x35 + x156;
  float x158 = x64 - x65;
  float x159 = x149*x58 - x151*x61 + x154*x24 + x158 - x59;
  float x160 = x121*x34;
  float x161 = x20*x83;
  float x162 = x160 - x161 - x60;
  float x163 = x123 - x124 - x34;
  float x164 = 1.0/tas;
  float x165 = 4*x164;
  float x166 = q0*x165;
  float x167 = q1*x166*x58;
  float x168 = x14*x165;
  float x169 = 2*x164;
  float x170 = x169*x39;
  float x171 = x170*x58;
  float x172 = -x171;
  float x173 = x165*x25;
  float x174 = x39*x95;
  float x175 = 8*x164*x174;
  float x176 = x14*x175;
  float x177 = q1*x175*x57;
  float x178 = q0*x177;
  float x179 = x170*x21;
  float x180 = x166*x63 + x179;
  float x181 = x166*x78;
  float x182 = x165*x26;
  float x183 = x170*x35;
  float x184 = q3*x177;
  float x185 = x16*x175;
  float x186 = x16*x165;
  float x187 = x170*x61;
  float x188 = q1*x165*x78 + x187;
  float x189 = pow(q1, 3);
  float x190 = x169*x46;
  float x191 = x169*x61;
  float x192 = x164*x17;
  float x193 = x15*x164;
  float x194 = x169*x26;
  float x195 = x107*x165*x39;
  float x196 = x174*x60;
  float x197 = x169*x25;
  float x198 = pow(q2, 3);
  float x199 = x174*x57;
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
  F(4, 0) = q0*x54 + q1*x67 - q2*x82 + q3*x76 + x87;
  F(4, 1) = q0*x105 + q1*x111 - q2*x120 + q3*x116 + x122;
  F(4, 2) = q0*x142 + q1*x147 - q2*x132 + q3*x143 - x123 + x124 + x34;
  F(4, 3) = q0*x153 + q1*x155 - q2*x159 + q3*x157 + x162;
  F(5, 0) = q0*x76 + q1*x82 + q2*x67 + q3*(x127 + x156 - x43 - x45 - x52) + x162;
  F(5, 1) = q0*x116 + q1*x120 + q2*x111 + q3*(-x101 - x103 + x158 - x89 - x92 - x94 - x99) + x163;
  F(5, 2) = q0*x143 + q1*x132 + q2*x147 + q3*(x115 - x134 - x137 - x139 - x140 - x141) + x122;
  F(5, 3) = q0*x157 + q1*x159 + q2*x155 + q3*(-x117 + x118 - x145 - x150 - x152 + x75) - x57 - x84 - x86;
  F(6, 0) = q0*x82 - q1*x76 + q2*x54 + q3*x67 + x163;
  F(6, 1) = q0*x120 - q1*x116 + q2*x105 + q3*x111 - x160 + x161 + x60;
  F(6, 2) = q0*x132 - q1*x143 + q2*x142 + q3*x147 + x87;
  F(6, 3) = q0*x159 - q1*x157 + q2*x153 + q3*x155 + x122;
  F(7, 0) = -x167 + x168*x21 + x172;
  F(7, 1) = -x173*x58 + x176*x20 - x178 + x180;
  F(7, 2) = -x181 + x182*x61 - x183 + x184 - x185*x34;
  F(7, 3) = -x186*x35 + x188;
  F(8, 0) = x168*x35 + x188;
  F(8, 1) = x173*x61 + x176*x34 + x181 + x183 + x184;
  F(8, 2) = x178 + x180 + x182*x58 + x185*x20;
  F(8, 3) = x167 + x171 + x186*x21;
  F(9, 0) = x183 - x189*x190 + x191*x25 + x191*x26 - x192*x61;
  F(9, 1) = pow(q0, 3)*x190 + x173*x196 + x182*x196 - x186*x196 - x187 - x189*x195 - x192*x35 - x193*x35 + x194*x35;
  F(9, 2) = pow(q3, 3)*x190 - x168*x199 + x172 + x173*x199 + x182*x199 - x192*x21 - x193*x21 - x195*x198 + x197*x21;
  F(9, 3) = x179 - x190*x198 - x193*x58 + x194*x58 + x197*x58;
  F(10, 10) = 1;
  F(11, 11) = 1;
  F(12, 12) = 1;
  F(13, 13) = 1;
  F(14, 14) = 1;
  F(15, 15) = 1;

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

float Kalman::roll() {
  return q2roll(Quaternion<float>(x(0), x(1), x(2), x(3)));
}

float Kalman::pitch() {
  return q2pitch(Quaternion<float>(x(0), x(1), x(2), x(3)));
}

float Kalman::heading() {
  return q2heading(Quaternion<float>(x(0), x(1), x(2), x(3)));
}

float positive_heading(float head_rad) {
  if (head_rad < 0) {
    return head_rad + 2*PI;
  } else {
    return head_rad;
  }
}

float Kalman::turn_rate() {
  Quaternion<float> q(x(0), x(1), x(2), x(3));
  return (q * Quaternion<float>(0, x(I_WX), x(I_WY), x(I_WZ)) * q.inverse()).vec()(2);
}

void Kalman::set_heading(float heading_to_set) {
  Quaternion<float> q(x(0), x(1), x(2), x(3));
  float current_heading = heading();
  // Remove heading from quaternion
  q = Quaternion<float>(AngleAxis<float>(current_heading, Matrix<float, 3, 1>(0, 0, -1.0))) * q;
  // rotate back with new heading
  q = Quaternion<float>(AngleAxis<float>(heading_to_set, Matrix<float, 3, 1>(0, 0, 1.0))) * q;
  x(I_Q0) = q.w();
  x(I_Q1) = q.x();
  x(I_Q2) = q.y();
  x(I_Q3) = q.z();
}