
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

#define NSTATES 12
#define NSENSORS 10

// Approximate DT
#define DT .038
#define PI 3.141592653589793
#define g  9.81
#define K2ms  1852.0/3600.0   // Convert knots to m/s; 1852m = 1nm
#define ft2m  1.0/3.28084

#define AIR_GND_SWITCH_SPEED 40  // switch between gnd/air filter at this speed in knots

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


using Eigen::MatrixXd;
using Eigen::Matrix3f;
using Eigen::Matrix;
using Eigen::Dynamic;

using namespace std;


class KalmanMatricies
{
private:

public:
  KalmanMatricies();
  Matrix<float, NSTATES, NSTATES> P, Q;
  Matrix<float, NSTATES, NSENSORS> R;
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


public:
  Matrix <float, NSTATES, 1> x;  // should be private, but public for testing
  Kalman();
  void predict(float dt);
  void predict_air(float dt);
  void predict_gnd(float dt);
  void update_accel(Matrix<float, 3, 1> a);
  void printAirP() { std::cout << air.P << std::endl; }
  void printAirQ() { std::cout << air.Q << std::endl; }
  void printDiag(Matrix<float, Dynamic, Dynamic> M) {
    for (int i=0; i < M.rows(); i++)
        cout << M(i,i) << endl;
  };
  void printAirQDiag() { printDiag(air.Q); };
  void printAirRDiag() { printDiag(air.R); };
  void printStates() { cout << x << endl; }
};
