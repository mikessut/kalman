
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
  Matrix <float, NSTATES, 1> x;

public:
  Kalman();
  void printAirP() { std::cout << air.P << std::endl; }
  void printAirQ() { std::cout << air.Q << std::endl; }
  void printDiag(Matrix<float, Dynamic, Dynamic> M) {
    for (int i=0; i < M.rows(); i++) {
        for (int j=0; j < M.cols(); j++) {
            if (i == j)
              cout << M(i,j) << endl;
        }
      }
  };
  void printAirQDiag() { printDiag(air.Q); };
  void printAirRDiag() { printDiag(air.R); };
};
