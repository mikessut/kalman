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
  float x22 = pow(q0, 2);
  float x23 = 1.0*x22;
  float x24 = 1.0*x14;
  float x25 = 1.0*x16;
  float x26 = pow(q3, 2);
  float x27 = 1.0*x26;
  float x28 = x23 - x24 + x25 - x27;
  float x29 = q1*q2;
  float x30 = q0*q3;
  float x31 = x29 - x30;
  float x32 = 0.25*pow(x28, 2) + pow(x31, 2);
  float x33 = pow(x32, -1.0/2.0);
  float x34 = 1.0*x33;
  float x35 = q1*x34;
  float x36 = x28*x35;
  float x37 = x21*x36;
  float x38 = 2.0*x29 - 2.0*x30;
  float x39 = G*q0;
  float x40 = x19*x39;
  float x41 = x38*x40;
  float x42 = q0*q1;
  float x43 = 2*q2*q3 + 2*x42;
  float x44 = 0.5*x43;
  float x45 = pow(x32, -3.0/2.0);
  float x46 = 0.5*x28;
  float x47 = x45*(-q0*x46 + q3*x31);
  float x48 = x44*x47;
  float x49 = x46*x47;
  float x50 = x43*x49;
  float x51 = G*x19;
  float x52 = x33*x51;
  float x53 = x38*x52;
  float x54 = x44*x53;
  float x55 = x35*x41 + x54;
  float x56 = x21*x50 + x37 + x41*x48 + x55;
  float x57 = G*q2;
  float x58 = x19*x57;
  float x59 = x36*x58;
  float x60 = G*q1;
  float x61 = x19*x60;
  float x62 = x38*x48;
  float x63 = x43*x58;
  float x64 = x34*x63;
  float x65 = q0*x64;
  float x66 = x21*x35;
  float x67 = x43*x66;
  float x68 = x65 - x67;
  float x69 = x24*x53 + x49*x63 + x59 + x61*x62 + x68;
  float x70 = x38*x66;
  float x71 = x43*x52;
  float x72 = x23*x71;
  float x73 = x27*x71;
  float x74 = x36*x40;
  float x75 = x46*x71;
  float x76 = x40*x43;
  float x77 = -x21*x62 + x49*x76 - x70 + x72 + x73 + x74 + x75;
  float x78 = x28*x52;
  float x79 = x38*x58;
  float x80 = x35*x76;
  float x81 = q3*x64;
  float x82 = -G - x80 - x81;
  float x83 = x35*x79 + x82;
  float x84 = -x24*x78 + x48*x79 - x50*x61 + x83;
  float x85 = x33*x44;
  float x86 = x41*x85;
  float x87 = x33*x46;
  float x88 = x43*x87;
  float x89 = x21*x88;
  float x90 = x57 + x86 + x89;
  float x91 = x45*(q1*x46 - q2*x31);
  float x92 = x44*x91;
  float x93 = x46*x91;
  float x94 = x43*x93;
  float x95 = x43/pow(x18, 2);
  float x96 = x39*x95;
  float x97 = 2.0*x33;
  float x98 = q1*x97;
  float x99 = x96*x98;
  float x100 = x20*x95;
  float x101 = x28*x98;
  float x102 = 1.0*x30;
  float x103 = x102*x78 + x68;
  float x104 = x100*x101 + x103 + x21*x94 + x23*x53 + x38*x99 + x41*x92;
  float x105 = q0*x34;
  float x106 = x28*x58;
  float x107 = x105*x106;
  float x108 = G*x95;
  float x109 = x108*x97;
  float x110 = x109*x14;
  float x111 = x38*x92;
  float x112 = x57*x95;
  float x113 = x101*x112;
  float x114 = x107 + x110*x38 + x111*x61 + x113 + x55 + x63*x93;
  float x115 = x102*x53;
  float x116 = x38*x98;
  float x117 = -x100*x116 - x111*x21 - x115 + x23*x78 + x28*x99 + x76*x93 + x82;
  float x118 = x105*x79;
  float x119 = x112*x116 + x24*x71 + x25*x71;
  float x120 = -x110*x28 + x118 + x119 - x61*x94 - x74 - x75 + x79*x92;
  float x121 = x38*x85;
  float x122 = x121*x61 - x20 + x63*x87;
  float x123 = x79*x85;
  float x124 = x61*x88;
  float x125 = q3*x34;
  float x126 = x125*x79;
  float x127 = x109*x16;
  float x128 = x45*(-q1*x31 - q2*x46);
  float x129 = x128*x44;
  float x130 = x128*x46;
  float x131 = x130*x43;
  float x132 = -x113 + x126 + x127*x38 + x129*x79 - x131*x61 - x37 + x54;
  float x133 = x112*x97;
  float x134 = q0*x133;
  float x135 = x129*x38;
  float x136 = q3*x133;
  float x137 = x103 + x130*x76 + x134*x28 - x135*x21 - x136*x38 - x27*x53;
  float x138 = G + x115 + x129*x41 + x131*x21 + x134*x38 + x136*x28 + x27*x78 + x80 + x81;
  float x139 = x106*x125 + x75;
  float x140 = x119 + x127*x28 + x130*x63 + x135*x61 + x139 + x70;
  float x141 = x45*(q0*x31 + q3*x46);
  float x142 = x141*x44;
  float x143 = x141*x46;
  float x144 = x143*x43;
  float x145 = x118 + x139 + x142*x41 + x144*x21 - x72 - x73;
  float x146 = x142*x38;
  float x147 = x143*x63 + x146*x61 + x25*x78 + x83;
  float x148 = x107 - x126 + x143*x76 - x146*x21 - x54;
  float x149 = x142*x79 - x144*x61 + x25*x53 - x59 - x65 + x67;
  float x150 = x76*x87;
  float x151 = x121*x21;
  float x152 = x150 - x151 - x60;
  float x153 = x123 - x124 - x39;
  float x154 = 1.0/tas;
  float x155 = 4*x154;
  float x156 = x155*x58;
  float x157 = x156*x42;
  float x158 = x14*x155;
  float x159 = 2*x154;
  float x160 = x159*x43;
  float x161 = x160*x58;
  float x162 = -x161;
  float x163 = x155*x22;
  float x164 = 8*x154;
  float x165 = x14*x164;
  float x166 = x112*x164;
  float x167 = x166*x42;
  float x168 = x160*x21;
  float x169 = x155*x30*x61 + x168;
  float x170 = x156*x30;
  float x171 = x155*x26;
  float x172 = x160*x40;
  float x173 = q1*q3;
  float x174 = x166*x173;
  float x175 = x16*x164;
  float x176 = x155*x16;
  float x177 = x160*x61;
  float x178 = x156*x173 + x177;
  float x179 = pow(q1, 3);
  float x180 = x159*x51;
  float x181 = x159*x61;
  float x182 = x154*x17;
  float x183 = x15*x154;
  float x184 = x159*x26;
  float x185 = x108*x155;
  float x186 = x60*x95;
  float x187 = x159*x22;
  float x188 = pow(q2, 3);
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
  F(4, 0) = q0*x56 + q1*x69 - q2*x84 + q3*x77 + x90;
  F(4, 1) = q0*x104 + q1*x114 - q2*x120 + q3*x117 + x122;
  F(4, 2) = q0*x138 + q1*x140 - q2*x132 + q3*x137 - x123 + x124 + x39;
  F(4, 3) = q0*x145 + q1*x147 - q2*x149 + q3*x148 + x152;
  F(5, 0) = q0*x77 + q1*x84 + q2*x69 - q3*x56 + x152;
  F(5, 1) = q0*x117 + q1*x120 + q2*x114 - q3*x104 + x153;
  F(5, 2) = q0*x137 + q1*x132 + q2*x140 - q3*x138 + x122;
  F(5, 3) = q0*x148 + q1*x149 + q2*x147 - q3*x145 - x57 - x86 - x89;
  F(6, 0) = q0*x84 - q1*x77 + q2*x56 + q3*x69 + x153;
  F(6, 1) = q0*x120 - q1*x117 + q2*x104 + q3*x114 - x150 + x151 + x60;
  F(6, 2) = q0*x132 - q1*x137 + q2*x138 + q3*x140 + x90;
  F(6, 3) = q0*x149 - q1*x148 + q2*x145 + q3*x147 + x122;
  F(7, 0) = -x157 + x158*x21 + x162;
  F(7, 1) = x100*x165 - x163*x58 - x167 + x169;
  F(7, 2) = -x170 + x171*x61 - x172 + x174 - x175*x96;
  F(7, 3) = -x176*x40 + x178;
  F(8, 0) = x158*x40 + x178;
  F(8, 1) = x163*x61 + x165*x96 + x170 + x172 + x174;
  F(8, 2) = x100*x175 + x167 + x169 + x171*x58;
  F(8, 3) = x157 + x161 + x176*x21;
  F(9, 0) = x172 - x179*x180 + x181*x22 + x181*x26 - x182*x61;
  F(9, 1) = pow(q0, 3)*x180 + x163*x186 + x171*x186 - x176*x186 - x177 - x179*x185 - x182*x40 - x183*x40 + x184*x40;
  F(9, 2) = pow(q3, 3)*x180 - x112*x158 + x112*x163 + x112*x171 + x162 - x182*x21 - x183*x21 - x185*x188 + x187*x21;
  F(9, 3) = x168 - x180*x188 - x183*x58 + x184*x58 + x187*x58;

  #ifdef BIAS_STATES
  for (int i=0; i < 6; i++) {
    F(10+i, 10+i) = 1.0;
  }
  #endif

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