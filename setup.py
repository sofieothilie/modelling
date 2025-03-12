from setuptools import setup, Extension
import numpy

# Define the extension module
module = Extension(
    'modeling',  # Module name
    sources=['simulate.c', 'modeling_py_wrapper.c', 'module_setup.c'],  # Source file(s)
    include_dirs=[numpy.get_include()],  # Include NumPy headers
    extra_compile_args=['-std=c99', '-fopenmp'],  # Use C99 standard for compatibility
    extra_link_args = ["-fopenmp"]
)

# Setup function
setup(
    name='modeling',
    version='1.0',
    description='module for simulating the waves send on the sleipner model',
    ext_modules=[module],
)