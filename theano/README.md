# Theano环境搭建

Theano曾经风靡一时，可惜后来因为大佬Yoshua Bengio进入谷歌后，谷歌着力开发Tensorflow，它也慢慢退出了历史舞台。现在因为一些老的AI算法使用到了Theano，我们也不想再将他们迁移到新的AI Framework，所以有时候还需要搭建一下theano环境。坦率的说，如果不够熟悉theano的话，那么在现在的开发环境下搭建theano环境还是有些麻烦的，这里主要介绍了使用Anaconda来搭建环境，一方面是相对比较容易，另外一方面是这个方面能够在Windows与Linux中使用，平台迁移起来也比较方便。

## 在Ubuntu20.04LTS中搭建theano环境

```
$ conda env create --file py38-theano.yaml

$ conda activate py38-theano
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
$ conda activate py38-theano
```

**注**：下载 [py38-theano.yaml](https://github.com/SNSerHello/MyNotes/blob/main/anaconda3/py38-theano.yaml)

**.theanrc配置**

```
[global]
floatX = float32
device = cuda
optimizer_including = cudnn

[gcc]
cxxflags = -I/media/samba/anaconda3/envs/py38-theano/include -L/media/samba/anaconda3/envs/py38-theano/lib -L/usr/lib/x86_64-linux-gnu -lrt -pthread -lresolv

[gpuarray]
preallocate = 0

[dnn]
enabled = True
library_path = /media/samba/anaconda3/envs/py38-theano/lib
include_path = /media/samba/anaconda3/envs/py38-theano/include

[cuda]
cuda = /media/samba/anaconda3/envs/py38-theano/bin

[lib]
cnmem = 0.5
```



## 在Ubuntu18.04LTS中搭建theano环境

```
$ conda env create --file py36-theano.yaml

$ conda activate py36-theano
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
$ conda activate py36-theano
```

**注**：下载 [py36-theano.yaml](https://github.com/SNSerHello/MyNotes/blob/main/anaconda3/py36-theano.yaml)

**.theanrc配置**

```
[global]
floatX = float32
device = cuda
optimizer_including = cudnn

[gcc]
cxxflags = -I/media/samba/anaconda3/envs/py36-theano/include -L/media/samba/anaconda3/envs/py36-theano/lib -L/media/samba/anaconda3/envs/py36-theano/lib64 -L/usr/lib/x86_64-linux-gnu -lrt -pthread -lresolv

[gpuarray]
preallocate = 0

[dnn]
enabled = True
library_path = /media/samba/anaconda3/envs/py36-theano/lib
include_path = /media/samba/anaconda3/envs/py36-theano/include

[cuda]
cuda = /media/samba/anaconda3/envs/py36-theano/bin

[lib]
cnmem = 0.5
```

在Ubuntu20.04LTS推荐使用python3.8+cuda11.3+cudnn8.2搭建环境，在Ubuntu18.04LTS推荐使用python3.6+cuda9.2+cudnn7.1搭建环境，这个不是必须的，我们可以在Ubuntu20.04LTS上使用python3.6+其他的搭配方法。如果Ubuntu上的GCC的版本比较高的话，我们可以使用Anaconda的toolchain来搭建编译环境，详见 [Anaconda3 README](https://github.com/SNSerHello/MyNotes/tree/main/anaconda3)。



## 参考

- [Anaconda3 README](https://github.com/SNSerHello/MyNotes/tree/main/anaconda3)