# Jetson installation
1. upgrade PIP
```
curl https://bootstrap.pypa.io/get-pip.py | python -
```
2. install pycuda

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
3. install matplotlib
```commandline
sudo apt-get install python3-matplotlib
```

# Desktop installation
```
pip install .
pip install pycuda==2022.2.1
pip install nvidia-pyindex
pip install nvidia-tensorrt
```