
#include <iostream>
#include "kalman.h"
#include "dcm.h"
#include <Eigen/Core>

using namespace std;
using namespace Eigen;
/*

g++ -I .\eigen test_eigen.cpp -o test_eigen

*/

int main() {

  Kalman k;
  cout << "hello world" <<endl;


  // Eigen::Matrix<float, 5, 5> M = Eigen::Matrix<float,5,5>::Zero();
  // cout << M << endl;
  // Eigen::Matrix<float, 2, 1> tmpv;
  // tmpv << 1, 2;
  // M.block(1,1, 2,2) = tmpv.asDiagonal();
  // cout << M << endl;
  //k.printAirRDiag();
  k.x(I_TAS,0) = 50;
  k.printStates();
  cout << endl;
  //cout << "x(0) " << k.x(0,0) << endl;
  k.predict_air(.038);
  Matrix<float,3,1> a;
  a << 2, -.2, -9;
  k.update_accel(a);
  //k.predict_air(.038);
  k.printStates();
}
