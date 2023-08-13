from setuptools import setup
import glob
import shutil
import os
import os.path as osp


def update_libraries(source_path,
        dst=osp.join(osp.dirname(__file__), "src/kas_utils/")):
    for file in glob.glob(f"{dst}/py_*.so"):
        os.remove(file)

    for file in glob.glob(f"{source_path}/py_*.so"):
        shutil.copy(file, dst)


update_libraries(source_path=osp.join(osp.dirname(__file__), "../cpp/build/"))

setup(
    name='kas_utils',
    version='0.0.1',
    packages=['kas_utils'],
    package_dir={'': 'src'},
    package_data={'': ['py_*.so']}
)
