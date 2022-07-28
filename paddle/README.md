# Paddle环境搭建

使用Anaconda3搭建PaddlePaddle与Paddlelite环境，它不依赖于操作系统，下述方法适合Ubuntu与Windows等环境。

## PaddlePaddle环境搭建

```bash
$ conda env create --file py37-paddle.yaml
$ conda activate py37-paddle
$ pip3 install --upgrade paddlepaddle-gpu paddlelite
```

**注意**

- paddlelite现在仅仅支持`python3.7`，paddlepaddle也支持`python3.8`，所以搭建paddle环境使用`python3.7`版本。
- `cuda10.2`和`cudnn7.6.5`搭配`python3.7`版本的paddlepaddle-gpu版本，使用高版本的时候会导致相关动态库找不到。
- PaddlePaddle的源码编译，请参考：[SNSerHello/Paddle](https://github.com/SNSerHello/Paddle)

## 检查PaddlePaddle环境

```python
import paddle
paddle.utils.run_check()
```

## PaddlePaddle Docker Images

### v2.3.1-gpu-cuda11.2-cudnn8

```bash
$ sudo docker pull paddlepaddle/paddle:2.3.1-gpu-cuda11.2-cudnn8
```

### v2.3.1-gpu-cuda10.2-cudnn7

```bash
$ sudo docker pull paddlepaddle/paddle:2.3.1-gpu-cuda10.2-cudnn7
```

### v2.3.0-gpu-cuda11.2-cudnn8

```bash
$ sudo docker pull paddlepaddle/paddle:2.3.0-gpu-cuda11.2-cudnn8
```

### v2.3.0-gpu-cuda10.2-cudnn7

```bash
$ sudo docker pull paddlepaddle/paddle:2.3.0-gpu-cuda10.2-cudnn7
```

### v2.2.2-gpu-cuda11.2-cudnn8

```bash
$ sudo docker pull paddlepaddle/paddle:2.2.2-gpu-cuda11.2-cudnn8
```

### v2.2.2-gpu-cuda10.2-cudnn7

```bash
$ sudo docker pull paddlepaddle/paddle:2.2.2-gpu-cuda10.2-cudnn7
```

## PaddlePaddle编译

### Anaconda3环境

使用[py37-paddle-dev](https://github.com/SNSerHello/MyNotes/blob/main/paddle/py37-paddle-dev.yaml)可以搭建PaddlePaddle-GPU编译环境（CUDA10.1+CUDNN7.6，请参考：[Paddle](https://github.com/SNSerHello/Paddle)）部分，[py37-paddle](https://github.com/SNSerHello/MyNotes/blob/main/paddle/py37-paddle.yaml)搭建的环境可以安装官方发布的PaddlePaddle-GPU（CUDA10.2+CUDNN7.6）。因为没有`cudatoolkit-dev=10.2`，采用`cudatoolkit=10.2`后缺乏编译环境，所以无法编译。CUDA11.3支持`compute_86`，`CUDNN≥8.0.2`时候支持`CUDNN_FMA_MATH`，可参考：[cudnnMathType_t](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnMathType_t)，搭建python3.8编译环境的话，请参考：[py38-paddle-dev.yaml](https://github.com/SNSerHello/MyNotes/blob/main/paddle/py38-paddle-dev.yaml)。

```bash
$ conda activate py38-paddle-dev
(py38-paddle-dev) $ git checkout v2.3.1
(py38-paddle-dev) $ mkdir build
(py38-paddle-dev) $ cd build
(py38-paddle-dev) $ export PADDLE_VERSION=2.3.1
(py38-paddle-dev) $ cmake .. \
	-DPY_VERSION=`python --version | cut -d ' ' -f 2 | cut -d '.' -f -2` \
	-DWITH_GPU=ON \
	-DWITH_TESTING=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX \
	-DCUDA_SDK_ROOT_DIR=$CONDA_PREFIX \
	-DCUDNN_ROOT=$CONDA_PREFIX \
	-DNCCL_ROOT=$CONDA_PREFIX \
	-DCUPTI_ROOT=$CONDA_PREFIX/pkgs/cuda-toolkit/extras/CUPTI \
	-DWITH_ONNXRUNTIME=ON \
	-DON_INFER=ON \
	-DCMAKE_CUDA_ARCHITECTURES=86
(py38-paddle-dev) $ ulimit -n 4096
(py38-paddle-dev) $ make -j
```


### Docker环境

```bash
$ sudo nvidia-docker run --rm \
	-itv your_path/Paddle:/workspace \
	-w /workspace \
	paddlepaddle/paddle:2.3.1-gpu-cuda10.2-cudnn7 \
	/bin/bash
```



## 参考

- [PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle)
- [PaddlePaddle/Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite)
- [PaddlePaddle Docker Images](https://hub.docker.com/r/paddlepaddle/paddle/tags)
