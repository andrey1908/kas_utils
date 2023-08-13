from setuptools import setup

setup(
    name='kas_utils',
    version='0.0.1',
    packages=['kas_utils'],
    package_dir={'': 'src'},
    package_data={'': ['py_depth_to_point_cloud.*.so']}
)
