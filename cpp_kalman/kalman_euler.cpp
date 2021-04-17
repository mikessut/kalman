#include "kalman_euler.h"


KalmanEuler::KalmanEuler() {
  initialize();
}


void KalmanEuler::initialize() {
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
  tmpv << 0, 0, 0,
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
  x(I_AZ) = -g;
}


void KalmanEuler::predict(float dt, float tas) {
  float heading = x(I_HEADING);
  float pitch = x(I_PITCH);
  float roll = x(I_ROLL);

  Quaternion<float> q = Quaternion<float>(AngleAxis<float>(heading, Matrix<float, 3, 1>(0, 0, 1.0)));
  q *= Quaternion<float>(AngleAxis<float>(pitch, Matrix<float, 3, 1>(0, 1.0, 0)));
  q *= Quaternion<float>(AngleAxis<float>(roll, Matrix<float, 3, 1>(1.0, 0, 0)));

  tas = max(float(KF_MIN_SPEED_MS), tas);

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

  // Euler angle prediction (via quaternion calculation)
  Eigen::Quaternion<float> qd(1, x(I_WX)*.5*dt, x(I_WY)*.5*dt, x(I_WZ)*.5*dt);
  q = q * qd;
  q.normalize();

  // update state vector
  x(I_HEADING) = q2heading(q);
  x(I_PITCH) = q2pitch(q);
  x(I_ROLL) = q2pitch(q);
  
  for (int i=0; i < 3; i++) {
    x(I_AX+i) = ab(i);
    x(I_WX+i) = wpredict(i);
  }

  Matrix<float, NSTATES, NSTATES> F = calcF(dt, tas);
  P = F * P * F.transpose() + Q*dt;
}


void KalmanEuler::update_accel(Matrix<float, 3, 1> a) {
  #ifdef BIAS_STATES
  Matrix<float, 3, 1> y = a - x.block(I_AX, 0, 3, 1) - x.block(I_ABX, 0, 3, 1);
  #else
  Matrix<float, 3, 1> y = a - x.block(I_AX, 0, 3, 1);
  #endif
  Matrix<float, 3, NSTATES> H = Matrix<float, 3, NSTATES>::Zero();
  H(0,3) = 1;
  H(1,4) = 1;
  H(2,5) = 1;
  #ifdef BIAS_STATES
  H(0,12) = 1;
  H(1,13) = 1;
  H(2,14) = 1;
  #endif

  Matrix<float, 3, 3> S = H * P * H.transpose() + Eigen::Matrix<float, 3, 3>::Identity() * Raccel;
  Matrix<float, NSTATES, 3> K = P * H.transpose() * S.inverse();
  x = x + K*y;
  P = (Eigen::Matrix<float, NSTATES, NSTATES>::Identity() - K*H)*P;
}


void KalmanEuler::update_gyro(Matrix<float, 3, 1> w) {
  #ifdef BIAS_STATES
  Matrix<float, 3, 1> y = w - x.block(I_WX, 0, 3, 1) - x.block(I_WBX, 0, 3, 1);
  #else
  Matrix<float, 3, 1> y = w - x.block(I_WX, 0, 3, 1); 
  #endif
  Matrix<float, 3, NSTATES> H = Matrix<float, 3, NSTATES>::Zero();
  H(0,6) = 1;
  H(1,7) = 1;
  H(2,8) = 1;
  #ifdef BIAS_STATES
  H(0,9) = 1;
  H(1,10) = 1;
  H(2,11) = 1;
  #endif
  Matrix<float, 3, 3> S = H * P * H.transpose() + Eigen::Matrix<float, 3, 3>::Identity() * Rgyro;
  Matrix<float, NSTATES, 3> K = P * H.transpose() * S.inverse();
  x = x + K*y;
  P = (Eigen::Matrix<float, NSTATES, NSTATES>::Identity() - K*H)*P;
}


void KalmanEuler::update_gps_bearing(float gps_heading)
{
  Matrix<float, 2, 1> meas(cos(gps_heading), sin(gps_heading));
  Matrix<float, 2, 1> h(cos(x(I_HEADING)), sin(x(I_HEADING)));

  Matrix<float, 2, 1> y = meas - h;

  Matrix<float, 2, NSTATES> H = Matrix<float, 2, NSTATES>::Zero();
  H(0, 0) = -sin(x(I_HEADING));
  H(1, 0) = cos(x(I_HEADING));

  Matrix<float, 2, 2> S = H * P * H.transpose() + Eigen::Matrix<float, 2, 2>::Identity() * 0.001;
  Matrix<float, NSTATES, 2> K = P * H.transpose() * S.inverse();
  x = x + K*y;
  P = (Eigen::Matrix<float, NSTATES, NSTATES>::Identity() - K*H)*P;
}

/*
void KalmanEuler::update_mag(Matrix<float, 3, 1> m)
{
}
*/


Eigen::Matrix<float, NSTATES, NSTATES> KalmanEuler::calcF(float dt, float tas) {
  Eigen::Matrix<float, NSTATES, NSTATES> F = Matrix<float, NSTATES, NSTATES>::Zero();
  float heading = x(I_HEADING);
  float pitch = x(I_PITCH);
  float roll = x(I_ROLL);
  float wx = x(I_WX);
  float wy = x(I_WY);
  float wz = x(I_WZ);
  
  float x0 = (1.0/2.0)*roll;
  float x1 = sin(x0);
  float x2 = (1.0/2.0)*heading;
  float x3 = sin(x2);
  float x4 = (1.0/2.0)*pitch;
  float x5 = sin(x4);
  float x6 = x3*x5;
  float x7 = x1*x6;
  float x8 = (1.0/2.0)*x7;
  float x9 = cos(x0);
  float x10 = cos(x2);
  float x11 = cos(x4);
  float x12 = x10*x11;
  float x13 = x12*x9;
  float x14 = (1.0/2.0)*x13;
  float x15 = x14 + x8;
  float x16 = dt*wx;
  float x17 = x15*x16;
  float x18 = x11*x3;
  float x19 = x18*x9;
  float x20 = (1.0/2.0)*x19;
  float x21 = -x20;
  float x22 = x10*x5;
  float x23 = x1*x22;
  float x24 = (1.0/2.0)*x23;
  float x25 = x21 + x24;
  float x26 = dt*wy;
  float x27 = x25*x26;
  float x28 = x1*x18;
  float x29 = (1.0/2.0)*x28;
  float x30 = x22*x9;
  float x31 = (1.0/2.0)*x30;
  float x32 = x29 + x31;
  float x33 = dt*wz;
  float x34 = x32*x33;
  float x35 = x1*x12;
  float x36 = x6*x9;
  float x37 = -x36;
  float x38 = x35 + x37;
  float x39 = -x23;
  float x40 = x19 + x39;
  float x41 = (1.0/2.0)*x16;
  float x42 = x13 + x7;
  float x43 = (1.0/2.0)*x26;
  float x44 = -x35;
  float x45 = x36 + x44;
  float x46 = (1.0/2.0)*x33;
  float x47 = x28 + x30;
  float x48 = x40*x41 + x42*x43 + x45*x46 + x47;
  float x49 = pow(dt, 2);
  float x50 = (1.0/4.0)*x49;
  float x51 = pow(wx, 2)*x50 + pow(wy, 2)*x50 + pow(wz, 2)*x50 + 1;
  float x52 = 1.0/x51;
  float x53 = 2*x52;
  float x54 = x48*x53;
  float x55 = -x54*(x17 + x27 + x34 + x38);
  float x56 = -x31;
  float x57 = -x29 + x56;
  float x58 = x26*x57;
  float x59 = x25*x33;
  float x60 = (1.0/2.0)*x36;
  float x61 = -x60;
  float x62 = (1.0/2.0)*x35;
  float x63 = x61 + x62;
  float x64 = x16*x63;
  float x65 = x38*x43 + x40 - x41*x47 + x42*x46;
  float x66 = x53*x65;
  float x67 = pow(x65, 2);
  float x68 = pow(x48, 2);
  float x69 = -x53*x68 + 1;
  float x70 = -x53*x67 + x69;
  float x71 = x38 - x40*x43 + x41*x42 + x46*x47;
  float x72 = x54*x71;
  float x73 = -x40*x46 + x41*x45 + x42 - x43*x47;
  float x74 = x66*x73;
  float x75 = 1.0/(pow(x70, 2) + pow(x72 + x74, 2));
  float x76 = x75*(-x72 - x74);
  float x77 = (1.0/2.0)*x17 + (1.0/2.0)*x27 + (1.0/2.0)*x34 + x63;
  float x78 = x53*x71;
  float x79 = (1.0/2.0)*x58;
  float x80 = x15 + (1.0/2.0)*x59 - 1.0/2.0*x64 + x79;
  float x81 = x53*x73;
  float x82 = -x15*x46 + x25 + x32*x41 - x43*x63;
  float x83 = x16*x25;
  float x84 = x33*x63;
  float x85 = x15*x26;
  float x86 = (1.0/2.0)*x85;
  float x87 = x57 + (1.0/2.0)*x83 + (1.0/2.0)*x84 - x86;
  float x88 = x70*x75;
  float x89 = -x62;
  float x90 = x61 + x89;
  float x91 = x16*x90;
  float x92 = x29 + x56;
  float x93 = x26*x92;
  float x94 = x20 + x24;
  float x95 = x33*x94;
  float x96 = -x54*(x13 - x7 + x91 + x93 + x95);
  float x97 = -x24;
  float x98 = x21 + x97;
  float x99 = x26*x98;
  float x100 = x33*x92;
  float x101 = -x8;
  float x102 = x101 + x14;
  float x103 = x102*x16;
  float x104 = (1.0/2.0)*x100 - 1.0/2.0*x103 + x90 + (1.0/2.0)*x99;
  float x105 = x102 + (1.0/2.0)*x91 + (1.0/2.0)*x93 + (1.0/2.0)*x95;
  float x106 = x16*x92;
  float x107 = x102*x33;
  float x108 = x26*x90;
  float x109 = (1.0/2.0)*x106 + (1.0/2.0)*x107 - 1.0/2.0*x108 + x98;
  float x110 = -x102*x43 + x41*x94 - x46*x90 + x92;
  float x111 = x16*x57;
  float x112 = x60 + x89;
  float x113 = x112*x26;
  float x114 = -x14;
  float x115 = x101 + x114;
  float x116 = x115*x33;
  float x117 = -x54*(x111 + x113 + x116 + x40);
  float x118 = x112*x33;
  float x119 = x20 + x97;
  float x120 = x119*x16;
  float x121 = -x30;
  float x122 = x121 - x28;
  float x123 = x112 + x115*x41 - x119*x43 - x46*x57;
  float x124 = (1.0/2.0)*x111 + (1.0/2.0)*x113 + (1.0/2.0)*x116 + x119;
  float x125 = x112*x16;
  float x126 = x119*x33;
  float x127 = (1.0/2.0)*x125 + (1.0/2.0)*x126 + x15 - x79;
  float x128 = (1.0/2.0)*x118 - 1.0/2.0*x120 + x57 + x86;
  float x129 = x49/pow(x51, 2);
  float x130 = wx*x129;
  float x131 = dt*x66;
  float x132 = dt*x54;
  float x133 = x130*x68 - x132*x40;
  float x134 = dt*x52;
  float x135 = x134*x40;
  float x136 = x135*x71;
  float x137 = x134*x73;
  float x138 = x137*x47;
  float x139 = x65*x73;
  float x140 = x48*x71;
  float x141 = x134*x42;
  float x142 = x134*x65;
  float x143 = x141*x48 + x142*x45;
  float x144 = wy*x129;
  float x145 = -x132*x42 + x144*x68;
  float x146 = x141*x71;
  float x147 = x142*x47;
  float x148 = x135*x48;
  float x149 = wz*x129;
  float x150 = -x132*x45 + x149*x68;
  float x151 = x134*x47;
  float x152 = x151*x48;
  float x153 = x135*x65;
  float x154 = x152 - x153;
  float x155 = x141*x73;
  float x156 = x134*x45;
  float x157 = x155 + x156*x71;
  float x158 = pow(1 - pow(-x54*x73 + x66*x71, 2), -1.0/2.0);
  float x159 = x48*x73;
  float x160 = x65*x71;
  float x161 = -x135*x73 + x141*x65 - x151*x71;
  float x162 = x134*x38;
  float x163 = pow(x71, 2);
  float x164 = -x163*x53 + x69;
  float x165 = x71*x81;
  float x166 = x48*x66;
  float x167 = 1.0/(pow(x164, 2) + pow(x165 + x166, 2));
  float x168 = x167*(-x165 - x166);
  float x169 = x164*x167;
  float x170 = -x19;
  float x171 = x170 + x39;
  float x172 = dt*x78;
  float x173 = x71*x73;
  float x174 = x48*x65;
  float x175 = G*x112;
  float x176 = -x175;
  float x177 = G*x25;
  float x178 = x170 + x23;
  float x179 = x178*x42;
  float x180 = 2*x179;
  float x181 = x122*x45;
  float x182 = x122*x38;
  float x183 = x180 + x181 - x182;
  float x184 = pow(x42, 2);
  float x185 = x178*x40;
  float x186 = pow(x122, 2) + x184 + x185 + x38*x45;
  float x187 = pow(x183, 2) + pow(x186, 2);
  float x188 = pow(x187, -1.0/2.0);
  float x189 = tan(roll);
  float x190 = x188*x189;
  float x191 = x183*x190;
  float x192 = x177*x191;
  float x193 = G*x115;
  float x194 = x186*x190;
  float x195 = x193*x194;
  float x196 = G*x42;
  float x197 = pow(x178, 2);
  float x198 = 2*x13 + 2*x7;
  float x199 = x115*x198;
  float x200 = x112*x45;
  float x201 = x122*x32;
  float x202 = x190*(x197 + x199 + 2*x200 + 2*x201);
  float x203 = x196*x202;
  float x204 = G*x178;
  float x205 = x115*x40;
  float x206 = x15*x178;
  float x207 = x32*x38;
  float x208 = x45*x57;
  float x209 = x190*(x179 + x181 + x205 + x206 + x207 + x208);
  float x210 = x204*x209;
  float x211 = x183*x196;
  float x212 = (1.0/2.0)*x183;
  float x213 = (1.0/2.0)*x186;
  float x214 = x189/pow(x187, 3.0/2.0);
  float x215 = x214*(-x212*(2*x197 + 2*x199 + 4*x200 + 4*x201) - x213*(x180 + 2*x181 + 2*x205 + 2*x206 + 2*x207 + 2*x208));
  float x216 = x211*x215;
  float x217 = x186*x215;
  float x218 = x204*x217;
  float x219 = x176 + x192 - x195 + x203 - x210 + x216 - x218;
  float x220 = G*x57;
  float x221 = x175*x194;
  float x222 = G*x38;
  float x223 = G*x122;
  float x224 = x183*x215;
  float x225 = x191*x220 + x193 + x202*x222 - x209*x223 - x217*x223 - x221 + x222*x224;
  float x226 = x177*x194;
  float x227 = x191*x193 + x196*x209 + x196*x217 + x202*x204 + x204*x224 - x220 + x226;
  float x228 = x191*x204 + x194*x196 - x222;
  float x229 = x191*x196;
  float x230 = x194*x204;
  float x231 = -x223 + x229 - x230;
  float x232 = -x177;
  float x233 = G*x45;
  float x234 = G*x32;
  float x235 = x194*x234;
  float x236 = x175*x191;
  float x237 = x235 - x236;
  float x238 = -x202*x223 + x209*x233 + x217*x233 - x223*x224 + x232 + x237;
  float x239 = -x191*x223 + x194*x233 - x196;
  float x240 = x191*x222 - x194*x223 + x204;
  float x241 = x112*x239 + x240*x57;
  float x242 = x114 + x8;
  float x243 = G*x98;
  float x244 = x60 + x62;
  float x245 = G*x244;
  float x246 = G*x92;
  float x247 = x121 + x28;
  float x248 = x178*x247;
  float x249 = x242*x45;
  float x250 = x198*x244;
  float x251 = x122*x94;
  float x252 = x190*(x248 + 2*x249 + x250 + 2*x251);
  float x253 = x247*x42;
  float x254 = x122*(-x13 + x7);
  float x255 = x38*x94;
  float x256 = x178*x90;
  float x257 = x244*x40;
  float x258 = x45*x98;
  float x259 = x190*(x253 + x254 + x255 + x256 + x257 + x258);
  float x260 = x214*(-x212*(2*x248 + 4*x249 + 2*x250 + 4*x251) - x213*(2*x253 + 2*x254 + 2*x255 + 2*x256 + 2*x257 + 2*x258));
  float x261 = x186*x260;
  float x262 = x183*x260;
  float x263 = x191*x245 + x194*x246 + x196*x259 + x196*x261 + x204*x252 + x204*x262 - x243;
  float x264 = G*x242;
  float x265 = x191*x246;
  float x266 = x194*x245;
  float x267 = x196*x252;
  float x268 = x204*x259;
  float x269 = x211*x260;
  float x270 = x204*x261;
  float x271 = -x264 + x265 - x266 + x267 - x268 + x269 - x270;
  float x272 = x191*x243 - x194*x264 + x222*x252 + x222*x262 - x223*x259 - x223*x261 + x245;
  float x273 = G*x194*x94 - x191*x264 - x223*x252 - x223*x262 + x233*x259 + x233*x261 - x246;
  float x274 = pow(x189, 2) + 1;
  float x275 = x188*x274;
  float x276 = x211*x275;
  float x277 = x186*x275;
  float x278 = x204*x277;
  float x279 = x178*x45;
  float x280 = x198*x32;
  float x281 = x25*x45;
  float x282 = x115*x122;
  float x283 = x190*(x279 + x280 + 2*x281 + 2*x282);
  float x284 = x196*x283;
  float x285 = x42*x45;
  float x286 = x115*x38;
  float x287 = x15*x45;
  float x288 = x122*x178;
  float x289 = x178*x57;
  float x290 = x32*x40;
  float x291 = x190*(x285 + x286 + x287 + x288 + x289 + x290);
  float x292 = x204*x291;
  float x293 = x214*(-x212*(2*x279 + 2*x280 + 4*x281 + 4*x282) - x213*(2*x285 + 2*x286 + 2*x287 + 2*x288 + 2*x289 + 2*x290));
  float x294 = x211*x293;
  float x295 = x186*x293;
  float x296 = x204*x295;
  float x297 = x232 - x235 + x236 + x276 - x278 + x284 - x292 + x294 - x296;
  float x298 = G*x15;
  float x299 = x183*x275;
  float x300 = x183*x293;
  float x301 = x191*x298 + x222*x283 + x222*x299 + x222*x300 - x223*x277 - x223*x291 - x223*x295 - x226 + x234;
  float x302 = x191*x234 + x196*x277 + x196*x291 + x196*x295 + x204*x283 + x204*x299 + x204*x300 + x221 - x298;
  float x303 = G*x274;
  float x304 = -x192 + x195;
  float x305 = x176 + x186*x188*x303*x45 - x223*x283 - x223*x299 - x223*x300 + x233*x291 + x233*x295 + x304;
  float x306 = x15*x240 + x239*x25;
  float x307 = x223 - x229 + x230;
  float x308 = 1.0/tas;
  float x309 = x189*x308;
  float x310 = x175*x309;
  float x311 = x222*x309;
  float x312 = x196*x309;
  float x313 = x220*x309;
  float x314 = x309*x47;
  float x315 = x177*x314;
  float x316 = x193*x309;
  float x317 = x223*x309;
  float x318 = -x178*x313;
  float x319 = x309*x40;
  float x320 = x204*x309;
  float x321 = x310*x47;
  float x322 = G*x309;
  float x323 = -x206*x322;
  float x324 = x274*x308;
  float x325 = x324*x47;
  float x326 = x223*x324;
  float x327 = x222*x324;
  float x328 = 2*x42;
  float x329 = 2*x311;
  float x330 = 2*x312;
  float x331 = x303*x308;
  F(0, 0) = x76*(x55 - x66*(x42 + x58 + x59 - x64)) + x88*(x54*x87 + x66*x82 + x77*x78 + x80*x81);
  F(0, 1) = x76*(-x66*(x100 - x103 + x37 + x44 + x99) + x96) + x88*(x104*x81 + x105*x78 + x109*x54 + x110*x66);
  F(0, 2) = x76*(x117 - x66*(x118 - x120 + x122 + x85)) + x88*(x123*x66 + x124*x78 + x127*x54 + x128*x81);
  F(0, 6) = x76*(x130*x67 + x131*x47 + x133) + x88*(-x130*x139 - x130*x140 + x136 - x138 + x143);
  F(0, 7) = x76*(-x131*x38 + x144*x67 + x145) + x88*(x137*x38 - x139*x144 - x140*x144 + x146 - x147 - x148);
  F(0, 8) = x76*(-x131*x42 + x149*x67 + x150) + x88*(-x139*x149 - x140*x149 + x154 + x157);
  F(1, 0) = -x158*(-x54*x82 + x66*x87 - x77*x81 + x78*x80);
  F(1, 1) = -x158*(x104*x78 - x105*x81 + x109*x66 - x110*x54);
  F(1, 2) = -x158*(-x123*x54 - x124*x81 + x127*x66 + x128*x78);
  F(1, 6) = -x158*(x130*x159 - x130*x160 - x156*x48 + x161);
  F(1, 7) = -x158*(x144*x159 - x144*x160 + x154 - x155 + x162*x71);
  F(1, 8) = -x158*(-x137*x45 + x146 + x147 + x148 + x149*x159 - x149*x160);
  F(2, 0) = x168*(x55 - x78*(x122 + x83 + x84 - x85)) + x169*(x54*x80 + x66*x77 + x78*x82 + x81*x87);
  F(2, 1) = x168*(-x78*(x106 + x107 - x108 + x171) + x96) + x169*(x104*x54 + x105*x66 + x109*x81 + x110*x78);
  F(2, 2) = x168*(x117 - x78*(x125 + x126 + x42 - x58)) + x169*(x123*x78 + x124*x66 + x127*x81 + x128*x54);
  F(2, 6) = x168*(x130*x163 + x133 - x172*x42) + x169*(-x130*x173 - x130*x174 - x152 + x153 + x157);
  F(2, 7) = x168*(x144*x163 + x145 + x172*x40) + x169*(-x144*x173 - x144*x174 + x161 + x162*x48);
  F(2, 8) = x168*(x149*x163 + x150 - x172*x47) + x169*(-x136 + x138 + x143 - x149*x173 - x149*x174);
  F(3, 0) = x122*x238 + x15*x228 + x219*x42 + x225*x38 + x227*x40 + x231*x25 + x241;
  F(3, 1) = x122*x273 + x228*x90 + x231*x92 + x239*x242 + x240*x98 + x263*x40 + x271*x42 + x272*x38;
  F(3, 2) = x112*x231 + x122*x305 + x228*x57 + x297*x42 + x301*x38 + x302*x40 + x306;
  F(4, 0) = x15*x307 + x225*x47 + x227*x42 + x228*x25 + x238*x38 + x239*x57 + x240*x63 + x40*(x175 - x203 + x210 - x216 + x218 + x304);
  F(4, 1) = x102*x240 + x228*x92 + x239*x98 + x263*x42 + x272*x47 + x273*x38 + x307*x90 + x40*(x264 - x265 + x266 - x267 + x268 - x269 + x270);
  F(4, 2) = x112*x228 + x119*x240 + x15*x239 + x301*x47 + x302*x42 + x305*x38 + x307*x57 + x40*(x177 + x237 - x276 + x278 - x284 + x292 - x294 + x296);
  F(5, 0) = x219*x47 + x225*x40 + x227*x45 + x228*x32 + x231*x63 + x238*x42 + x306;
  F(5, 1) = x102*x231 + x228*x94 + x239*x92 + x240*x90 + x263*x45 + x271*x47 + x272*x40 + x273*x42;
  F(5, 2) = x115*x228 + x119*x231 + x241 + x297*x47 + x301*x40 + x302*x45 + x305*x42;
  F(6, 0) = x15*x311 + x25*x317 + x310*x42 - x312*x63 + x313*x40 - x315 - x316*x38 + x318;
  F(6, 1) = -x102*x312 + x242*x312 + x243*x319 - x244*x311 - x246*x314 + x311*x90 + x317*x92 - x320*x98;
  F(6, 2) = -x119*x312 + x122*x310 - x178*x327 - x196*x325 - x207*x322 + x25*x312 + x298*x319 + x313*x38 - x321 + x323 + x326*x42 + x327*x40;
  F(7, 0) = -x15*x317 + x25*x329 - x310*x40 + x313*x328 - x316*x47 - x320*x63;
  F(7, 1) = -x102*x320 - x245*x314 - x264*x319 - x317*x90 + x329*x92 + x330*x98;
  F(7, 2) = -x119*x320 - x122*x313 + x15*x330 - x177*x319 - x204*x325 - x234*x314 + 2*x310*x38 - x326*x40 + x327*x328;
  F(8, 0) = x179*x322 - x182*x322 - x205*x322 + x317*x63 + x321 + x323;
  F(8, 1) = x102*x317 - x171*x311 + x247*x312 - x257*x322 + x264*x314 - x320*x90;
  F(8, 2) = x119*x317 + x184*x331 - x185*x331 - x290*x322 - x311*x42 + x312*x45 + x315 + x318 + x326*x47 - x331*pow(x38, 2);
  F(9, 9) = 1;
  F(10, 10) = 1;
  F(11, 11) = 1;
  F(12, 12) = 1;
  F(13, 13) = 1;
  F(14, 14) = 1;

  return F;
}


/*
Matrix<float, 2, NSTATES> KalmanEuler::calc_mag_H() {
  // TODO
}
*/


ostream& operator<<(ostream &ofs, const KalmanEuler &k) {
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

float KalmanEuler::roll() {
  return q2roll(Quaternion<float>(x(0), x(1), x(2), x(3)));
}

float KalmanEuler::pitch() {
  return q2pitch(Quaternion<float>(x(0), x(1), x(2), x(3)));
}

float KalmanEuler::heading() {
  return q2heading(Quaternion<float>(x(0), x(1), x(2), x(3)));
}

float positive_heading(float head_rad) {
  if (head_rad < 0) {
    return head_rad + 2*PI;
  } else {
    return head_rad;
  }
}


float KalmanEuler::turn_rate() {
  Quaternion<float> q(x(0), x(1), x(2), x(3));
  return (q * Quaternion<float>(0, x(I_WX), x(I_WY), x(I_WZ)) * q.inverse()).vec()(2);
}

void KalmanEuler::set_heading(float heading_to_set) {
  x(I_HEADING) = heading_to_set;
}