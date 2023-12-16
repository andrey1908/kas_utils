from setuptools import setup
import glob
import shutil
import os
import os.path as osp


def update_libraries(libraries,
        dst=osp.join(osp.dirname(__file__), "src/kas_utils/")):
    for old_library in glob.glob(f"{dst}/py_*.so"):
        os.remove(old_library)
    for library in libraries:
        shutil.copy(library, dst)

libraries = os.getenv("kas_utils_cpp_PYTHON_LIBRARIES")
if libraries:
    libraries = libraries.split(';')
else:
    src = osp.join(osp.dirname(__file__), "../cpp/build/")
    libraries = glob.glob(f"{src}/py_*.so")
update_libraries(libraries=libraries)

setup(
    name='kas_utils',
    version='0.0.1',
    packages=['kas_utils'],
    package_dir={'': 'src'},
    package_data={'': ['py_*.so']}
)
