
import setuptools
import numpy
import platform
import os
import importlib

from Cython.Build import cythonize
from distutils.extension import Extension

env_flag = os.environ.get("USE_AVX2")
if env_flag is not None:
    USE_AVX2 = env_flag == "1"
else:
    try:
        cpuinfo = importlib.import_module("cpuinfo")
        flags = cpuinfo.get_cpu_info().get("flags", [])
        USE_AVX2 = "avx2" in flags
    except Exception:
        USE_AVX2 = False

if platform.system() == 'Windows':
    extra_compile_args = ["/Ox", "/openmp"]
    if USE_AVX2:
        extra_compile_args.append("/arch:AVX2")
    extra_link_args = []
else:
    extra_compile_args = ["-O3", "-fopenmp", "-std=c++11"]
    if USE_AVX2:
        extra_compile_args.extend(["-mavx2", "-msse4.2"])
    extra_link_args = ['-fopenmp']

sourcefiles = ['deeds/registration.pyx', 'deeds/libs/deedsBCV0.cpp']
extensions = [Extension(name='deeds.registration', sources=sourcefiles, language="c++",
                        include_dirs=[numpy.get_include()], extra_compile_args=extra_compile_args, extra_link_args=extra_link_args)]

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="deeds",
    version="1.0.3",
    author="Marcin Wiktorowski",
    author_email="wiktorowski211@gmail.com",
    description="Python wrapper around efficient 3D discrete deformable registration for medical images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wiktorowski211/deeds-registration",
    packages=setuptools.find_packages(exclude=("tests",)),
    ext_modules=cythonize(extensions),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
)
