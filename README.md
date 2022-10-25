# upgrade PIP
```
curl https://bootstrap.pypa.io/get-pip.py | python -
```
# install pycuda

```
export CUDA_HOME=/usr/local/cuda
export LIBRARY_PATH=${CUDA_HOME}/lib64:$LIBRARY_PATH
export CPATH=$CPATH:${CUDA_HOME}/include
```
optionally try setting also this env variable
```
export C_INCLUDE_PATH=${CUDA_HOME}/include:${C_INCLUDE_PATH}
```

```
pip3 install 'pycuda<2021.1'
```
# install matplotlib
```commandline
sudo apt-get install python3-matplotlib
```