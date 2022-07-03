# Anaconda3环境搭建

## 安装Anaconda3

```
$ wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
$ ./Anaconda3-2020.11-Linux-x86_64.sh
```

`Anaconda3-2020.11-Linux-x86_64.sh`的python版本是3.8.5，其他Anaconda3安装可参考：https://repo.anaconda.com/archive/

## 搭建python3.8+cu113环境

```
$ conda env create --file py38-cu113.yaml
```

