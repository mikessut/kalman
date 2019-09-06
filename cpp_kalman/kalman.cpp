
#include "kalman.h"


KalmanMatricies::KalmanMatricies() {
    P = Matrix<float, NSTATES, NSTATES>::Zero();
    Q = Matrix<float, NSTATES, NSTATES>::Zero();
    R = Matrix<float, NSTATES, NSENSORS>::Zero();
}


void KalmanMatricies::predict(Eigen::Matrix<float, NSTATES, NSTATES> F) {
  P = F*P*F.transpose() + Q;
}


Matrix<float, NSTATES, Dynamic> KalmanMatricies::update_sensors(Eigen::Matrix<float, Dynamic, NSTATES> H,
                    int sns_idx, int nsensors) {
  Matrix<float, Dynamic, Dynamic> S = H*P*H.transpose() + R.block(sns_idx, sns_idx, nsensors, nsensors);
  Matrix<float, NSTATES, Dynamic> K = P*H.transpose()*S.inverse();
  P = (Matrix<float, NSTATES, NSTATES>::Identity() - K*H*P);
  return K;
}


Kalman::Kalman() : air(), gnd() {
  float p_err = pow(2*PI/180*DT, 2.0);
  float q_err = pow(2*PI/180*DT, 2.0);
  float r_err = pow(2*PI/180*DT, 2.0);
  float ax_err = pow(2*g*DT, 2.0);  // confidence that ax, ay, az is from phi/TAS
  float ay_err = pow(.2*g*DT, 2.0);
  float az_err = pow(.2*g*DT, 2.0);
  float pitch_err = pow(.01*PI/180*DT, 2.0);  // confidence that pitch is theta + dt*q
  float roll_err =  pow(.01*PI/180*DT, 2.0);    // confident that roll is atan(TAS/g*<p,q,r projected to earth_rot_z>)
  float yaw_err =   pow(.01*PI/180*DT, 2.0);             // confidence that yaw is psi + dt*<

  float gnd_orient_err = pow(17.0*PI/180*DT, 2.0);
  float earth_mag_err = pow(1.0/30.0*DT, 2.0);
  float psi_err = .1/10*PI/180*DT;  // gnd only
  float const_TAS_err = pow(6.9*K2ms*DT, 2.0);

  Matrix<float, NSTATES, 1> tmpv;
  tmpv << p_err, q_err, r_err,
          ax_err, ay_err, az_err,
          roll_err, pitch_err, yaw_err,
          const_TAS_err,
          earth_mag_err, earth_mag_err;
  air.Q = tmpv.asDiagonal();

  // R init
  float BW = 50;
  // Gyro ~ 0.01 deg/rt-Hz
  float gyro_err = pow(.01*sqrt(BW), 2.0);
  // Accel 300 ug/rt-Hz
  float accel_err = pow(.5*g, 2.0);
  float mag_err = pow(.1, 2.0);
  float TAS_err = pow(1*K2ms, 2.0);

  Matrix <float, 3, 1> tmp3v;
  tmp3v << gyro_err, gyro_err, gyro_err;
  air.R.block(0,0, 3,3) = tmp3v.asDiagonal();
  tmp3v << accel_err,accel_err,accel_err;
  air.R.block(3,3, 3,3) = tmp3v.asDiagonal();
  air.R(6,6) = TAS_err;
  tmp3v << mag_err,mag_err,mag_err;
  air.R.block(7,7, 3,3) = tmp3v.asDiagonal();

  x = Matrix<float, NSTATES, 1>::Zero();
  x(I_AZ, 0) = -g;

  Rot_sns << 1, 0, 0,
             0, cos(PI), sin(PI),
             0, -sin(PI), cos(PI);
}


void Kalman::predict(float dt) {
  if (x[I_TAS, 0] > AIR_GND_SWITCH_SPEED*K2ms) {
    predict_air(dt);
  } else {
    predict_gnd(dt);
  }
}

void Kalman::predict_air(float dt) {
  Matrix<float, NSTATES, 1> lastx(x);

  x(0, 0) = lastx(I_P,0);
  x(1, 0) = lastx(I_Q,0);
  x(2, 0) = lastx(I_R,0);
  x(3, 0) = g*sin(lastx(I_PITCH,0));
  x(4, 0) = g*(sin(lastx(I_ROLL,0))*sin(lastx(I_YAW,0))*sin(lastx(I_PITCH,0)) + cos(lastx(I_ROLL,0))*cos(lastx(I_YAW,0)))*cos(lastx(I_YAW,0))*tan(lastx(I_ROLL,0)) - g*(sin(lastx(I_ROLL,0))*sin(lastx(I_PITCH,0))*cos(lastx(I_YAW,0)) - sin(lastx(I_YAW,0))*cos(lastx(I_ROLL,0)))*sin(lastx(I_YAW,0))*tan(lastx(I_ROLL,0)) - g*sin(lastx(I_ROLL,0))*cos(lastx(I_PITCH,0));
  x(5, 0) = -g*(sin(lastx(I_ROLL,0))*sin(lastx(I_YAW,0)) + sin(lastx(I_PITCH,0))*cos(lastx(I_ROLL,0))*cos(lastx(I_YAW,0)))*sin(lastx(I_YAW,0))*tan(lastx(I_ROLL,0)) + g*(-sin(lastx(I_ROLL,0))*cos(lastx(I_YAW,0)) + sin(lastx(I_YAW,0))*sin(lastx(I_PITCH,0))*cos(lastx(I_ROLL,0)))*cos(lastx(I_YAW,0))*tan(lastx(I_ROLL,0)) - g*cos(lastx(I_ROLL,0))*cos(lastx(I_PITCH,0));
  x(6, 0) = dt*lastx(I_P,0) + lastx(I_ROLL,0);
  x(7, 0) = dt*lastx(I_Q,0) + lastx(I_PITCH,0);
  x(8, 0) = lastx(I_YAW,0) + dt*g*tan(lastx(I_ROLL,0))/lastx(I_TAS,0);
  x(9, 0) = lastx(I_TAS,0) + lastx(I_AX,0)*dt;
  x(10, 0) = lastx(I_MX,0);
  x(11, 0) = lastx(I_MZ,0);

  Matrix<float, NSTATES, NSTATES> F = Matrix<float, NSTATES, NSTATES>::Zero();
  F(0, 0) = 1;

  F(1, 1) = 1;

  F(2, 2) = 1;

  F(3, 7) = g*cos(lastx(I_PITCH,0));

  F(4, 6) = g*(-cos(lastx(I_PITCH,0)) + 1)*cos(lastx(I_ROLL,0));
  F(4, 7) = g*sin(lastx(I_ROLL,0))*sin(lastx(I_PITCH,0));

  F(5, 6) = g*(cos(lastx(I_PITCH,0)) - 1 - 1/cos(lastx(I_ROLL,0))/cos(lastx(I_ROLL,0)))*sin(lastx(I_ROLL,0));
  F(5, 7) = g*sin(lastx(I_PITCH,0))*cos(lastx(I_ROLL,0));

  F(6, 0) = dt;
  F(6, 6) = 1;

  F(7, 1) = dt;
  F(7, 7) = 1;

  F(8, 6) = dt*g/(lastx(I_TAS,0)*cos(lastx(I_ROLL,0))*cos(lastx(I_ROLL,0)));
  F(8, 8) = 1;
  F(8, 9) = -dt*g*tan(lastx(I_ROLL,0))/lastx(I_TAS,0)/lastx(I_TAS,0);

  F(9, 3) = dt;
  F(9, 9) = 1;

  F(10, 10) = 1;

  F(11, 11) = 1;
  air.predict(F);
}

void Kalman::predict_gnd(float dt) {

}

void Kalman::update_accel(Matrix<float, 3, 1> a) {
  a = Rot_sns*a;
  Matrix<float, 3, 1> y = a - x.block(I_AX, 0, 3, 1);
  Matrix<float, 3, NSTATES> H = Matrix<float, 3,NSTATES>::Zero();
  H(0,3) = 1;
  H(1,4) = 1;
  H(2,5) = 1;
  // TODO: air/gnd mode switch
  Matrix<float, NSTATES, 3> K = air.update_sensors(H, I_AX, 3);
  x = x + K*y;
}
