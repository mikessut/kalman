
#include <iostream>
#include <fstream>
#include "kalman.h"
#include "pressure.h"
#include <Eigen/Core>

using namespace std;
using namespace Eigen;
/*

g++ -I .\eigen test_eigen.cpp -o test_eigen

*/


int main() {

  float altitude, ias, tas;
  //airspeed_altitude(float abs_press, float diff_press, float alt_setting, float oat,
  airspeed_altitude(80000.0, 5000.0, 30.12, 19.2,
                    &altitude, &ias, &tas);

  printf("Alt: %.0f; IAS: %.1f; TAS: %.1f", altitude, ias, tas);
  
  Kalman k;

  cout << k << endl;

  k.predict(.01);
  k.update_accel(Vector3f(.1,0,-9));
  cout << k << endl;
  return 0;
 
  ifstream ifs("../output.txt");
  ofstream ofs("test.out");
  
  float dt, x, y, z;
  string c;
  int ctr = 0;
  float t = 0;
  //while (ctr < 10) {

  while (ifs) {
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
    }
    ctr++;
  }

  return 0;
}
