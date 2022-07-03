# Anaconda3环境搭建

## 安装Anaconda3

```
$ wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
$ ./Anaconda3-2020.11-Linux-x86_64.sh
```

`Anaconda3-2020.11-Linux-x86_64.sh`的python版本是3.8.5，其他Anaconda3安装可参考：https://repo.anaconda.com/archive/。此外，如果要使用anaconda的docker images，请参考：https://hub.docker.com/u/continuumio。

## 搭建python3.8+cuda11.3环境

```
$ conda env create --file py38-cu113.yaml
```

默认的cudnn版本是8.2，可以在[py38-cu113.yaml](https://github.com/SNSerHello/MyNotes/blob/main/anaconda3/py38-cu113.yaml)中制定cudnn版本，比如说`cudnn=7.5`等

## 搭建python3.8+cuda10.2环境

```
$ conda env create --file py38-cu102.yaml
```

一般它与`cudnn=7.6`搭配，所以制定这个环境中使用7.6版本

## 在Anaconda3中配置不同的CUDA环境

```
$ conda activate 你的环境名
$ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
$ nano $CONDA_PREFIX/etc/conda/activate.d/env_vars.h
文件内容如下：
CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
$ mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
$ nano $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.h
文件内容如下：
export LD_LIBRARY_PATH=`echo $LD_LIBRARY_PATH | cut -d : -f 2-`
```

不同的CUDA环境有不同的装载路径，通过上面的方式可以结合`conda activate 你的环境名`与`conda deactivate`命令行自动的关联`LD_LIBRARY_PATH`，并正确装载`so`文件

## 通过Anaconda3搭建theano环境

在Ubuntu20.04LTS建议搭建python3.8+cuda11.3+cuda8.2+theano1.0.5，在Ubuntu18.04LTS建议搭建python3.6+cuda9.2+cuda7.1+theano1.0.4，详见下方配置文件。

### a) Ubuntu20.04LTS

```
$ conda env create --file py38-theano.yaml
```

#### .theanorc

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

### a) Ubuntu18.04LTS

```
$ conda env create --file py36-theano.yaml
```

#### .theanorc

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

## 搭建Anaconda3的Docker编译环境

针对python3.6，python3.7，python3.8的不同编译需求，笔者专门制作了一个docker的Anaconda3编译环境，可以编译TVM, PaddlePaddle等。cmake, make等常用的编译工具已经内在在这些选择中，进入后按照正常的方式即可对source codes进行编译。

```
$ sudo docker login
$ sudo docker pull snser/anaconda3
$ sudo docker run --rm -itv your_path:/workspace -w /workspace snser/anaconda3 /bin/bash
$ conda env list
base                  *  /root/anaconda3
python3.6                /root/anaconda3/envs/python3.6
python3.7                /root/anaconda3/envs/python3.7
python3.8                /root/anaconda3/envs/python3.8
$ conda activate python3.8/7/6
```

在Anaconda中增加编译环境，一般的方式是安装 `conda-build`，它的好处在于能够自动的完成环境相关配置，比如说：库文件会被放在`$CONDA_PREFIX/lib下`，`LD_LIBRARY_PATH`能够自动的找到相关的路径等，但是更加自己的需求，安装对应的toolchain安装包，现在Anaconda支持如下：

| macOS             | Linux               | Windows             |
| ----------------- | ------------------- | ------------------- |
| ● clang_osx-64    | ●  gcc_linux-64     | ●  m2w64-gcc_win-64 |
| ● clangxx_osx-64  | ● gxx_linux-64      |                     |
| ● gfortran_osx-64 | ● gfortran_linux-64 |                     |

**如何安装？**

```
$ conda install gcc_linux-64 gxx_linux-64
```



## 参考

- [Anaconda compiler tools](https://docs.conda.io/projects/conda-build/en/latest/resources/compiler-tools.html)
