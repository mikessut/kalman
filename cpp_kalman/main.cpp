
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
  k.printAirRDiag();
}
