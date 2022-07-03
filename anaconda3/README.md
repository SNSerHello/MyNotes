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
