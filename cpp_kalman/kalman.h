
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

#ifndef KALMAN_H
#define KALMAN_H

// Approximate DT
#define DT .038
#define PI 3.141592653589793
#define g  9.81
#define K2ms  1852.0/3600.0   // Convert knots to m/s; 1852m = 1nm
#define ft2m  1.0/3.28084

#define ACCEL_MEAS_ERR  1.5
#define GYRO_MEAS_ERR  .0001
#define MAG_MEAS_ERR  .01


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
  Matrix<float, NSTATES, NSTATES> calcF(float dt);

public:
  Matrix <float, NSTATES, 1> x;  // should be private, but public for testing
  Kalman();
  void predict(float dt);
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
};

ostream& operator<<(ostream &ofs, const Kalman &k);

float roll(Eigen::Quaternion<float> q);
#endif
