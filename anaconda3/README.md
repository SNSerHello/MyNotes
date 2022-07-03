# Anaconda3环境搭建

## 安装Anaconda3

```
$ wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
$ ./Anaconda3-2020.11-Linux-x86_64.sh
```

`Anaconda3-2020.11-Linux-x86_64.sh`的python版本是3.8.5，其他Anaconda3安装可参考：https://repo.anaconda.com/archive/

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
