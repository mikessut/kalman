
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

#ifndef KALMAN_H
#define KALMAN_H

#define NSTATES 12
#define NSENSORS 10

// Approximate DT
#define DT .038
#define PI 3.141592653589793
#define g  9.81
#define K2ms  1852.0/3600.0   // Convert knots to m/s; 1852m = 1nm
#define ft2m  1.0/3.28084

#define AIR_GND_SWITCH_SPEED 50  // switch between gnd/air filter at this speed in knots

// Indecies of all of the EKF states
#define I_P      0
#define I_Q      1
#define I_R      2
#define I_AX     3
#define I_AY     4
#define I_AZ     5
#define I_ROLL   6
#define I_PITCH  7
#define I_YAW    8
#define I_TAS    9
#define I_MX     10
#define I_MZ     11

// Indicies of sensors (R might be only place this is used?)
#define IS_WX 0
#define IS_WY 1
#define IS_WZ 2
#define IS_AX 3
#define IS_AY 4
#define IS_AZ 5
#define IS_TAS 6
#define IS_MX  7
#define IS_MY  8
#define IS_MZ  9


using namespace Eigen;
using namespace std;

typedef enum {
  KF_GND = 0,
  KF_AIR = 1
} KF_SEL_FILTER;

class KalmanMatricies
{
private:

public:
  KalmanMatricies();
  Matrix<float, NSTATES, NSTATES> P, Q;
  Matrix<float, NSENSORS, NSENSORS> R;
  void predict(Eigen::Matrix<float, NSTATES, NSTATES> F);

  Matrix<float, NSTATES, Dynamic> update_sensors(Eigen::Matrix<float, Dynamic, NSTATES> H,
                      int sns_idx, int nsensors);
};


class Kalman
{
private:
  int nstates;
  KalmanMatricies air;
  KalmanMatricies gnd;
  Matrix<float, 3, 3> Rot_sns;
  KF_SEL_FILTER active_filter;
  Matrix<float, NSTATES, Dynamic> update_sensors(Eigen::Matrix<float, Dynamic, NSTATES> H,
                      int sns_idx, int nsensors);

public:
  Matrix <float, NSTATES, 1> x;  // should be private, but public for testing
  Kalman();
  void predict(float dt);
  void predict_air(float dt);
  void predict_gnd(float dt);
  void update_accel(Matrix<float, 3, 1> a);
  void update_gyro(Matrix<float, 3, 1> w);
  void update_mag(Matrix<float, 3, 1> m);
  void update_TAS(float tas);
  void printAirP() { std::cout << air.P << std::endl; }
  void printAirQ() { std::cout << air.Q << std::endl; }
  void printDiag(Matrix<float, Dynamic, Dynamic> M) {
    for (int i=0; i < M.rows(); i++)
        cout << M(i,i) << endl;
  };
  void printAirQDiag() { printDiag(air.Q); };
  void printAirRDiag() { printDiag(air.R); };
  void printStates() { cout << x << endl << endl; }
  //void statesCSV(ofstream &ofs) {
  //  for (int i=0; i < NSTATES; i++)
  //    ofs << x(i,0) << ",";
  //}
  friend ostream& operator<<(ostream &ofs, const Kalman &k);
};

ostream& operator<<(ostream &ofs, const Kalman &k);

#endif
