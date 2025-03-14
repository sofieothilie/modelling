#include "modeling_wrapper.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/*
Python module creation
*/

static PyMethodDef methods[] = {
   {"launch_model", modeling_py_wrapper, METH_VARARGS | METH_KEYWORDS,  "simulate wave on model."},
    {NULL, NULL, 0, NULL}  // Sentinel value to indicate the end
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "modeling",  // Module name
    "module for simulating the waves send on the sleipner model",
    -1,
    methods  // Module methods
};

PyMODINIT_FUNC PyInit_modeling(void)
{
    import_array();
    return PyModule_Create(&module);
}
