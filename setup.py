from setuptools import setup, Extension
import sys,os

#append your numpy package, might not be necessary, but was for me


import numpy

os.environ["CC"] = "gcc"
os.environ["CXX"] = "gcc"

if 'CUDA_PATH' in os.environ:
   CUDA_PATH = os.environ['CUDA_PATH']
else:
   print("Could not find CUDA_PATH in environment variables. Defaulting to /usr/local/cuda!")
   CUDA_PATH = "/usr/local/cuda"

if not os.path.isdir(CUDA_PATH):
   print("CUDA_PATH {} not found. Please update the CUDA_PATH variable and rerun".format(CUDA_PATH))
   exit(0)


# Define the extension module
module = Extension(
    'modeling',  # Module name
    sources=['src/modeling_py_wrapper.c', 'src/module_setup.c'],  # Source file(s)
    include_dirs=[numpy.get_include(), os.path.join(CUDA_PATH, "include")],  # Include NumPy headers
    libraries=["modeling", "cudart"],  # Libraries to link with (no need for 'lib' prefix in Linux)
    library_dirs=["./cuda_build", os.path.join(CUDA_PATH, "lib64")],  # Directories where the libraries are located
    extra_compile_args=['-fPIC'],  # Compiler flags, -fPIC for position-independent code (required for shared libraries)
   #  extra_link_args=["-lcudart_static"]  # Linking with the static CUDA runtime library (if needed)
)


# Setup function
setup(
    name='modeling',
    version='2.0',
    description='module for simulating the waves send on the sleipner model',
    ext_modules=[module],
)