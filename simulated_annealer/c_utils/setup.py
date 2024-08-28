"""-----------------------------------------------------------------------------

Setup of the MCMC C++ extension

-----------------------------------------------------------------------------"""
from distutils.core import setup, Extension

import os

import numpy as np
numpy_include = np.get_include()

extra_compile_args = ["-w", "-std=c++17", "-O3"]

c_utils_module = Extension('_c_interface',
                        language='c++',
                        include_dirs=[numpy_include],
                        library_dirs=[],
                        sources=['c_interface_wrap.cxx', 'mcmc.cpp'],
                        extra_compile_args=extra_compile_args)


setup(name='c_utils',
      py_modules=['c_utils'],
      ext_modules=[c_utils_module])

