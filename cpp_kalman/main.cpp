
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
  k.x(NSTATES-2,0) = 3.89e-01;
  k.x(NSTATES-1,0) = 9.06e-01;
  k.x(8,0) = -1.0821426;

  ifstream ifs("output.txt");
  ofstream ofs("test.out");
  float dt, x, y, z;
  string c;
  int ctr = 0;
  float t = 0;
  while (ctr < 10) {
  //while (ifs) {
    ifs >> c;
    if (c == "p") {
      ofs << t << ",";
      ofs << k;
      ofs << endl;

      ifs >> dt;
      k.predict(dt);
      t += dt;
    } else if (c == "a") {
      ifs >> x;
      ifs >> y;
      ifs >> z;
      k.update_accel(Vector3f(x,y,z));
    } else if (c == "g") {
      ifs >> x;
      ifs >> y;
      ifs >> z;
      k.update_gyro(Vector3f(x,y,z));
    } else if (c == "m") {
      ifs >> x;
      ifs >> y;
      ifs >> z;
      k.update_mag(Vector3f(x,y,z));
    } else if (c == "t") {
      ifs >> x;
      k.update_TAS(x);
    }
    ctr++;
  }
}
