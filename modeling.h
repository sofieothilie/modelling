#ifndef MODELING_H
#define MODELING_H

#include <stdint.h>
#include <Python.h>

typedef int32_t int_t;
typedef double real_t;

int simulate(real_t* model_data, int_t n_x, int_t n_y, int_t n_z, double r_dt, int r_max_iter, int r_snapshot_freq, double r_sensor_height, int_t r_model_nx, int_t r_model_ny, double* sign, int sign_len, int fs);

PyObject *modeling_py_wrapper(PyObject *self, PyObject *args, PyObject* kwargs);



#endif // MODELING_H