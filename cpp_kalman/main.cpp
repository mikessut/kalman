
#include <iostream>
#include <fstream>
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
  //cout << "x(0) " << k.x(0,0) << endl;
  k.predict(.038);
  k.printStates();
  Matrix<float,3,1> a;
  a << 2, -.2, -9;
  k.update_accel(a);
  k.printStates();
  a << .01 , 0 , -.02;
  k.update_gyro(a);
  k.printStates();
  a << .1 , .1 , .8;
  k.update_mag(a);
  k.printStates();
  k.update_TAS(55);
  //k.predict_air(.038);
  k.printStates();

  ifstream ifs("output.txt");
  ofstream ofs("test.out");
  float dt, x, y, z;
  string c;
  int ctr = 0;
  while (ctr < 10) {
    ifs >> c;
    if (c == "p") {
      ifs >> dt;
      cout << "p " << dt << endl;
    } else if (c == "a") {
      ifs >> x;
      ifs >> y;
      ifs >> z;
      cout << "a" << x << ", " << y << ", " << z << endl;
    }
    ctr++;
  }
}
