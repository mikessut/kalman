
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

#ifndef KALMAN_H
#define KALMAN_H

// Approximate DT
#define DT .038
#define PI 3.141592653589793
#define g  9.81
#define G  g
#define K2ms  1852.0/3600.0   // Convert knots to m/s; 1852m = 1nm
#define ft2m  1.0/3.28084

#define ACCEL_MEAS_ERR  pow(1.5 , 2)   // m/s^2
#define GYRO_MEAS_ERR   pow(0.01, 2)   // rad/sec
#define MAG_MEAS_ERR    pow(0.3 , 2)  // in normalized heading vector units sin(20deg) ??

// TAS shows up in the denominator of some of the calculations and thus can't go to
// zero.  This parameter limits the minimum value used in the calcualtions.
#define KF_MIN_SPEED_MS    10*K2ms

#define NSTATES 10
// States
// Quaternion of orientation q0, q1, q2, q3
// body acceleration
// body rotational rates

#define I_Q0    0
#define I_Q1    1
#define I_Q2    2
#define I_Q3    3
#define I_AX    4
#define I_AY    5
#define I_AZ    6
#define I_WX    7
#define I_WY    8
#define I_WZ    9
#define I_WBX   10
#define I_WBY   11
#define I_WBZ   12

using namespace Eigen;
using namespace std;


class Kalman
{
private:
  int nstates;
  Matrix<float, NSTATES, NSTATES> P, Q;
  
  Matrix<float, NSTATES, Dynamic> update_sensors(Eigen::Matrix<float, Dynamic, NSTATES> H,
                      int sns_idx, int nsensors);

  float Raccel = ACCEL_MEAS_ERR;
  float Rgyro = GYRO_MEAS_ERR;
  float Rmag = MAG_MEAS_ERR;
  Matrix<float, NSTATES, NSTATES> calcF(float dt, float tas);
  Matrix<float, 2, NSTATES> calc_mag_H();
  void q_normalize();


public:
  Matrix <float, NSTATES, 1> x;  // should be private, but public for testing
  Kalman();
  void initialize();
  void predict(float dt, float tas);
  void update_accel(Matrix<float, 3, 1> a);
  void update_gyro(Matrix<float, 3, 1> w);
  void update_mag(Matrix<float, 3, 1> m);
  
  void printDiag(Matrix<float, Dynamic, Dynamic> M) {
    for (int i=0; i < M.rows(); i++)
        cout << M(i,i) << endl;
  };
  void printQDiag() { printDiag(Q); };
  //void printRDiag() { printDiag(R); };
  void printStates() { cout << x << endl << endl; }
  //void statesCSV(ofstream &ofs) {
  //  for (int i=0; i < NSTATES; i++)
  //    ofs << x(i,0) << ",";
  //}
  friend ostream& operator<<(ostream &ofs, const Kalman &k);
  void printState() {
    for (int i=0; i < NSTATES; i++) 
      cout << x(i) << endl;
  }
  float * Qdiaganol() {
    return Q.diagonal().data();
  }

  float roll();
  float pitch();
  float heading();

  float turn_rate();

  float get_P(int i, int j) {
    return P(i, j);
  }
};

ostream& operator<<(ostream &ofs, const Kalman &k);

float q2roll(Eigen::Quaternion<float> q);
float q2pitch(Eigen::Quaternion<float> q);
float q2heading(Eigen::Quaternion<float> q);

float positive_heading(float);
#endif
