#define _XOPEN_SOURCE 600

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "modeling.h"

PyObject *modeling_py_wrapper(PyObject *self, PyObject *args, PyObject* kwargs)
{
    printf("[DEBUG] Entering modeling_py_wrapper\n");

    import_array();

    PyArrayObject* model = NULL; // 3D array of media
    int Nx = 0, Ny = 0, Nz = 0;

    static char *kwlist[] = {"model", "res", NULL};

    printf("[DEBUG] Parsing arguments...\n");

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!(iii)|$", kwlist,
                                     &PyArray_Type, &model,
                                     &Nx, &Ny, &Nz))
    {
        printf("[ERROR] Illegal parameters passed to method\n");
        Py_RETURN_NONE;
    }

    printf("[DEBUG] Parsed arguments successfully.\n");
    printf("[DEBUG] Model: %p, Nx: %d, Ny: %d, Nz: %d\n", model, Nx, Ny, Nz);

    if (!model) {
        printf("[ERROR] Model is NULL.\n");
        Py_RETURN_NONE;
    }

    // Ensure correct NumPy array type
    if (PyArray_TYPE(model) != NPY_DOUBLE)
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

    printf("[DEBUG] Model array is valid.\n");

    // Retrieve data pointer
    real_t* model_data = (real_t*)PyArray_DATA(model);
    size_t model_dims[2] = {
        (size_t)PyArray_DIM(model, 0),
        (size_t)PyArray_DIM(model, 1),
    };

    printf("[DEBUG] Model dimensions: %zu x %zu\n", model_dims[0], model_dims[1]);

    if (!model_data) {
        printf("[ERROR] Model data pointer is NULL.\n");
        Py_RETURN_NONE;
    }

    printf("[DEBUG] Calling simulate function...\n");
    simulate(model, Nx, Ny, Nz);
    printf("[DEBUG] Simulation completed.\n");

    Py_RETURN_NONE;
}
