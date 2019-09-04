
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
  Matrix<float, NSTATES, Dynamic> S = H*P*H.transpose() + R.block(sns_idx, sns_idx, nsensors, nsensors);
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
}
