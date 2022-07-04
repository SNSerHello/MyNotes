# Docker环境搭建

## 如何安装nvidia-docker？

### 搭建nvidia-docker环境

```
$ curl https://get.docker.com | sh && sudo systemctl --now enable docker
$ sudo apt-get remove docker docker-engine docker.io containerd runc
$ sudo apt-get update
$ sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
$ sudo mkdir -p /etc/apt/keyrings
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
$ echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
$ sudo apt-get update
$ sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
$ sudo apt-get update
$ sudo apt-get install -y nvidia-docker2
```

### 验证nvidia-docker环境

```
$ sudo docker pull nvcr.io/nvidia/pytorch:22.06-py3
$ sudo nvidia-docker run --gpus all --rm -it nvcr.io/nvidia/pytorch:22.06-py3 /bin/bash
root@6c1e3b8ba5e4:/workspace# ipython
Python 3.8.13 | packaged by conda-forge | (default, Mar 25 2022, 06:04:10)
Type 'copyright', 'credits' or 'license' for more information
IPython 8.4.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import torch

In [2]: torch.cuda.is_available()
Out[2]: True
```

**Okay!**

## 如何改变docker images存放路径？

```
$ sudo service docker stop
$ sudo nano /etc/docke/daemon.json
{
  "data-root": "/path/to/your/docker",
  ...
}
$ sudo rsync -aP /var/lib/docker/ /path/to/your/docker
$ sudo mv /var/lib/docker /var/lib/docker.old
$ sudo service docker start
$ sudo rm -rf /var/lib/docker.old
```

## 下载Docker images

| Tensorflow v2.9.1                                          | Pytorch v1.13.0                                     | Mxnet v1.9.1                                 | PaddlePaddle v2.2.2                                      |
| ---------------------------------------------------------- | --------------------------------------------------- | -------------------------------------------- | -------------------------------------------------------- |
| `sudo docker pull nvcr.io/nvidia/tensorflow:22.06-tf2-py3` | `sudo docker pull nvcr.io/nvidia/pytorch:22.06-py3` | `docker pull nvcr.io/nvidia/mxnet:22.06-py3` | `sudo docker pull nvcr.io/nvidia/paddlepaddle:22.06-py3` |



## 参考

- [Docker Prerequisites](https://docs.docker.com/engine/install/ubuntu/)

- [Nvidia Docker Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

- [How to change docker root data directory](https://tienbm90.medium.com/how-to-change-docker-root-data-directory-89a39be1a70b)

- [HOW TO MOVE DOCKER DATA DIRECTORY TO ANOTHER LOCATION ON UBUNTU](https://www.guguweb.com/2019/02/07/how-to-move-docker-data-directory-to-another-location-on-ubuntu/)

- [NVIDIA Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)

- [Running PaddlePaddle](https://docs.nvidia.com/deeplearning/frameworks/paddle-paddle-release-notes/running.html#running)

- [Running PyTorch](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html#running)

- [Running TensorFlow](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running)

- [Running NVIDIA Optimized Deep Learning Framework, powered by Apache MXNet](https://docs.nvidia.com/deeplearning/frameworks/mxnet-release-notes/running.html#running)

  