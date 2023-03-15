Install opengv with python:
```
cd /usr/local/src
git clone --recurse-submodules https://github.com/laurentkneip/opengv
cd opengv
mkdir build
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON=ON
make -j4
make install
```