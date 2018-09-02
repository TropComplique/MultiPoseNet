from distutils.core import setup
from Cython.Build import cythonize
import numpy


"""
Just run this:
python setup_heatmap_creation.py build_ext --inplace; rm build/ -rf
"""

path_to_source_code = 'detector/input_pipeline/heatmap_creation/main.pyx'
setup(ext_modules=cythonize(path_to_source_code), include_dirs=[numpy.get_include()])
