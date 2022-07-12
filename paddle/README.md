# Paddle环境搭建

使用Anaconda3搭建PaddlePaddle与Paddlelite环境，它不依赖于操作系统，下述方法适合Ubuntu与Windows等环境。

## PaddlePaddle环境搭建

```
$ conda env create --file py37-paddle.yaml
$ conda activate py37-paddle
$ pip3 install --upgrade paddlepaddle-gpu paddlelite
```

**注意**

- paddlelite现在仅仅支持`python3.7`，paddlepaddle也支持`python3.8`，所以搭建paddle环境使用`python3.7`版本。
- `cuda10.2`和`cudnn7.6.5`搭配`python3.7`版本的paddlepaddle-gpu版本，使用高版本的时候会导致相关动态库找不到。

## 检查PaddlePaddle环境

```
import paddle
paddle.utils.run_check()
```

## PaddlePaddle Docker Images

### v2.3.1-gpu-cuda11.2-cudnn8

```
$ sudo docker pull paddlepaddle/paddle:2.3.1-gpu-cuda11.2-cudnn8
```

### v2.3.1-gpu-cuda10.2-cudnn7

```
$ sudo docker pull paddlepaddle/paddle:2.3.1-gpu-cuda10.2-cudnn7
```

### v2.3.0-gpu-cuda11.2-cudnn8

```
$ sudo docker pull paddlepaddle/paddle:2.3.0-gpu-cuda11.2-cudnn8
```

### v2.3.0-gpu-cuda10.2-cudnn7

```
$ sudo docker pull paddlepaddle/paddle:2.3.0-gpu-cuda10.2-cudnn7
```

### v2.2.2-gpu-cuda11.2-cudnn8

```
$ sudo docker pull paddlepaddle/paddle:2.2.2-gpu-cuda11.2-cudnn8
```

### v2.2.2-gpu-cuda10.2-cudnn7

```
$ sudo docker pull paddlepaddle/paddle:2.2.2-gpu-cuda10.2-cudnn7
```

## PaddlePaddle编译

### Anaconda3环境

使用[py37-paddle-dev](https://github.com/SNSerHello/MyNotes/blob/main/paddle/py37-paddle-dev.yaml)可以搭建PaddlePaddle-GPU编译环境（CUDA10.1+CUDNN7.6，请参考：[Paddle](https://github.com/SNSerHello/Paddle)）部分，[py37-paddle](https://github.com/SNSerHello/MyNotes/blob/main/paddle/py37-paddle.yaml)搭建的环境可以安装官方发布的PaddlePaddle-GPU（CUDA10.2+CUDNN7.6）。因为没有`cudatoolkit-dev=10.2`，采用`cudatoolkit=10.2`后缺乏编译环境，所以无法编译。

### Docker环境

```
$ sudo nvidia-docker run --rm -itv your_path/Paddle:/workspace -w /workspace paddlepaddle/paddle:2.3.1-gpu-cuda10.2-cudnn7 /bin/bash
```



## 参考

- [PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle)
- [PaddlePaddle/Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite)
- [PaddlePaddle Docker Images](https://hub.docker.com/r/paddlepaddle/paddle/tags)