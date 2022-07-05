# 搭建pymc3环境

## Ubuntu20.04LTS

```
$ conda env create --file py38-pymc3.yaml
# 配置py38-pymc3环境变量
$ conda activate py38-pymc3
$ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
$ nano $CONDA_PREFIX/etc/conda/activate.d/env_vars.h
文件内容如下：
CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
$ mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
$ nano $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.h
文件内容如下：
export LD_LIBRARY_PATH=`echo $LD_LIBRARY_PATH | cut -d : -f 2-`
$ conda deactivate
$ conda activate py38-pymc3
```

**.theanorc配置**

```
[global]
floatX = float32
device = cuda
optimizer_including = cudnn

[gcc]
cxxflags = -I/media/samba/anaconda3/envs/py38-pymc3/include -L/media/samba/anaconda3/envs/py38-pymc3/lib -L/usr/lib/x86_64-linux-gnu -lrt -pthread -lresolv

[gpuarray]
preallocate = 0

[dnn]
enabled = True
library_path = /media/samba/anaconda3/envs/py38-pymc3/lib
include_path = /media/samba/anaconda3/envs/py38-pymc3/include

[cuda]
cuda = /media/samba/anaconda3/envs/py38-pymc3/bin

[lib]
cnmem = 0.5
```



## 参考

- [theano README](https://github.com/SNSerHello/MyNotes/tree/main/theano)
- [(Generalized) Linear and Hierarchical Linear Models in PyMC3](https://docs.pymc.io/en/v3/pymc-examples/examples/generalized_linear_models/GLM.html)