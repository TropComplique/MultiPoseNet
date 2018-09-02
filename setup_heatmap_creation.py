from distutils.core import setup
from Cython.Build import cythonize
import numpy


"""
python setup_heatmap_creation.py build_ext --inplace
rm build/ -rf
"""


setup(ext_modules=cythonize('detector/input_pipeline/heatmap_creation/main.pyx'), include_dirs=[numpy.get_include()])
