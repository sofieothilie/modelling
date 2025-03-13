from setuptools import setup, Extension
import sys,os

#append your numpy package, might not be necessary, but was for me
sys.path.append('C:\\Users\\guill\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages')


import numpy

os.environ["CC"] = "cl"
os.environ["CXX"] = "cl"

if 'CUDA_PATH' in os.environ:
   CUDA_PATH = os.environ['CUDA_PATH']
else:
   print("Could not find CUDA_PATH in environment variables. Defaulting to /usr/local/cuda!")
   CUDA_PATH = "/usr/local/cuda"

if not os.path.isdir(CUDA_PATH):
   print("CUDA_PATH {} not found. Please update the CUDA_PATH variable and rerun".format(CUDA_PATH))
   exit(0)

python_include_path = "C:/Users/guill/AppData/Local/Programs/Python/Python312"  # Example path, modify it according to your setup


# Define the extension module
module = Extension(
    'modeling',  # Module name
    sources=['modeling_py_wrapper.c', 'module_setup.c'],  # Source file(s)
    include_dirs=[numpy.get_include(), os.path.join(CUDA_PATH, "include"), python_include_path],  # Include NumPy headers
    libraries=["libmodeling", "cudart_static"],
    library_dirs=[".", os.path.join(CUDA_PATH, "lib/x64")],
    extra_compile_args=['/MD',],
    extra_link_args = ["/NODEFAULTLIB:LIBCMT"]
)

# Setup function
setup(
    name='modeling',
    version='2.0',
    description='module for simulating the waves send on the sleipner model',
    ext_modules=[module],
)