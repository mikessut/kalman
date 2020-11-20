
#include "kalman.h"

#define LIBEXPORT extern "C" __attribute__((__visibility__("default")))


LIBEXPORT void* CreateKalmanClass() {
    return new Kalman;
}


LIBEXPORT void* PrintState(void* ptr) {
    Kalman* ref = reinterpret_cast<Kalman*>(ptr);
    ref->printState();
}


LIBEXPORT void predict(void *ptr, float dt, float tas) {
    Kalman* ref = reinterpret_cast<Kalman*>(ptr);
    ref->predict(dt, tas);
}


LIBEXPORT void update_accel(void *ptr, float x, float y, float z) {
    Kalman* ref = reinterpret_cast<Kalman*>(ptr);
    ref->update_accel(Vector3f(x, y, z));
}


LIBEXPORT void update_gyro(void *ptr, float x, float y, float z) {
    Kalman* ref = reinterpret_cast<Kalman*>(ptr);
    ref->update_gyro(Vector3f(x, y, z));
}


LIBEXPORT void update_mag(void *ptr, float x, float y, float z) {
    Kalman* ref = reinterpret_cast<Kalman*>(ptr);
    ref->update_mag(Vector3f(x, y, z));
}


LIBEXPORT void* state_ptr(void *ptr) {
    Kalman* ref = reinterpret_cast<Kalman*>(ptr);
    ref->x.data();
}

LIBEXPORT float get_P(void *ptr, int i, int j) {
    Kalman* ref = reinterpret_cast<Kalman*>(ptr);
    return ref->get_P(i, j);
}