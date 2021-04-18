/**
 * Standard Rate Turn
 * 
 * 
 * tas = 92.6  # m/s^2  => close to 180 knots
 * roll = 25*np.pi/180
 * g = 9.81
 * 
 * turn_rate = g / tas * tan(roll)  # 1/s
 * print(2*np.pi / turn_rate)
 * print(127 / .02)
 * 
 * q = Quaternion.from_axis_angle([1, 0, 0], roll)
 * w = q2np(q.inverse() * Quaternion(0, 0, 0, turn_rate) * q)[1:]
 * print(w)
 * 
 * v_ac = q2np(q * Quaternion(0, 0, 1, 0) * q.inverse())[1:]
 * v_ac[2] = 0
 * v_ac /= sqrt(v_ac[0]**2 + v_ac[1]**2)
 * 
 * ac = v_ac*g*tan(roll)
 * ag = np.array([0,0,-g])  
 * a = ag + ac
 * ab = q2np(q.inverse() * Quaternion(0, *a) * q)[1:]
 * print(ab)
 * 
 * 127.188925896105
 * 6350.0
 * [0 0.0208775161359882 0.0447719778366767]
 * [0 -4.44089209850063e-16 -10.8241373850220]
 */

//#include "kalman_euler.h"
#include "kalman.h"
#include <Eigen/Dense>

using namespace std;

int test_turn() {
  //KalmanEuler kf = KalmanEuler();
  Kalman kf = Kalman();

  //kf.x(2) = 25 * M_PI / 180.0;
  Eigen::Quaternion<float> q = Eigen::Quaternion<float>(Eigen::AngleAxis<float>(25 * M_PI / 180.0, Matrix<float, 3, 1>(1.0, 0, 0)));
  kf.x(0) = q.w();
  kf.x(1) = q.x();
  kf.x(2) = q.y();
  kf.x(3) = q.z();


  cout << kf.x(0) << endl;
  cout << kf.x(1) << endl;
  cout << kf.x(2) << endl;

  for (int i = 0; i < 6350; i++) {
    kf.predict(0.02, 92.6);
    kf.update_gyro(Matrix<float, 3, 1>(0, 0.020877, .04477198));
    kf.update_accel(Matrix<float, 3, 1>(0, 0, -10.824));
    // kf.update_gps_bearing(-5 * M_PI / 180.0);

    
    cout << positive_heading(kf.heading()) * 180.0 / M_PI << endl;
    cout << kf.pitch() * 180.0 / M_PI << endl;
    cout << kf.roll() * 180.0 / M_PI << endl;

    cout << kf.get_P(0, 0) << endl;
    cout << kf.get_P(1, 1) << endl;
    cout << kf.get_P(2, 2) << endl;

    // for (int j = 3; j < NSTATES; j++) 
    //   cout << kf.x(j) << endl;
    cout << endl;
  }

  return 0;
}

int test_heading_update() {
  //KalmanEuler kf = KalmanEuler();
  Kalman kf = Kalman();

  //kf.x(2) = 25 * M_PI / 180.0;
  //Eigen::Quaternion<float> q = Eigen::Quaternion<float>(Eigen::AngleAxis<float>(25 * M_PI / 180.0, Matrix<float, 3, 1>(1.0, 0, 0)));
  //kf.x(0) = q.w();
  //kf.x(1) = q.x();
  //kf.x(2) = q.y();
  //kf.x(3) = q.z();


  for (int i = 0; i < 500; i++) {
    kf.predict(0.02, 92.6);
    kf.update_gyro(Matrix<float, 3, 1>(0, 0.0, 0));
    kf.update_accel(Matrix<float, 3, 1>(0, 0, -9.81));
    kf.update_gps_bearing(-5 * M_PI / 180.0);

    
    cout << positive_heading(kf.heading()) * 180.0 / M_PI << endl;
    cout << kf.pitch() * 180.0 / M_PI << endl;
    cout << kf.roll() * 180.0 / M_PI << endl;

    // cout << kf.get_P(0, 0) << endl;
    // cout << kf.get_P(1, 1) << endl;
    // cout << kf.get_P(2, 2) << endl;

    // for (int j = 3; j < NSTATES; j++) 
    //   cout << kf.x(j) << endl;
    cout << endl;
  }

  return 0;
}

int main() {
  test_turn();
  //test_heading_update();
}