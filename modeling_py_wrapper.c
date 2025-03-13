#define _XOPEN_SOURCE 600

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "modeling.h"

PyObject *modeling_py_wrapper(PyObject *self, PyObject *args, PyObject* kwargs)
{
    import_array();

    PyArrayObject* model = NULL; // 3D array of media
    int Nx = 0, Ny = 0, Nz = 0;
    double dt =0.001;
    int max_iter=4000;
    int snapshot_freq=20;

    double sensor_height=0.5;
    int fs=8e6;

    PyArrayObject* signature_wave = NULL;
    

    static char *kwlist[] = {"model", "signature_wave", "res", "dt", "max_iter","snapshot_freq", "sensor_height", "sampling", NULL};


    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!(iii)diidi|$", kwlist,
                                     &PyArray_Type, &model,
                                     &PyArray_Type, &signature_wave,
                                     &Nx, &Ny, &Nz,
                                     &dt, &max_iter,&snapshot_freq, &sensor_height, &fs))
    {
        printf("[ERROR] Illegal parameters passed to method\n");
        Py_RETURN_NONE;
    }



    if (!model) {
        printf("[ERROR] Model is NULL.\n");
        Py_RETURN_NONE;
    }

    // Ensure correct NumPy array type
    if (PyArray_TYPE(signature_wave) != NPY_DOUBLE)
    {
        PyErr_Format(PyExc_TypeError, "Array must be a NumPy array of type double.");
        printf("[ERROR] Model array is not of type double.\n");
        return NULL;
    }

    if (!PyArray_ISBEHAVED(model))
    {
        PyErr_Format(PyExc_ValueError, "Array must be a well-behaved NumPy array. Slicing it beforehand can be a problem.");
        printf("[ERROR] Model array is not well-behaved.\n");
        return NULL;
    }

    

    // Retrieve data pointer
    real_t* model_data = (real_t*)PyArray_DATA(model);
    size_t model_dims[2] = {
        (size_t)PyArray_DIM(model, 0),
        (size_t)PyArray_DIM(model, 1),
    };

    if (!model_data) {
        printf("[ERROR] Model data pointer is NULL.\n");
        Py_RETURN_NONE;
    }

    simulate(model_data, Nx, Ny, Nz, dt, max_iter, snapshot_freq, sensor_height, model_dims[0], model_dims[1], (double*)PyArray_DATA(signature_wave), PyArray_DIM(signature_wave, 0), fs);

    Py_RETURN_NONE;
}
