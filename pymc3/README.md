# 搭建pymc3环境

## Ubuntu20.04LTS

```
$ sudo apt install libopenblas-dev
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

[blas]
ldflags=-L/usr/lib/x86_64-linux-gnu -lopenblas -lpthread -lm

[cuda]
cuda = /media/samba/anaconda3/envs/py38-pymc3/bin

[lib]
cnmem = 0.5
```



## 检查pymc3环境

```
import os
import arviz as az
import bambi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import theano
import xarray as xr
from numpy.random import default_rng

print(f"Running on PyMC3 v{pm.__version__}")

# %config InlineBackend.figure_format = 'retina'
# Initialize random number generator
RANDOM_SEED = 8927
rng = default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

size = 50
true_intercept = 1
true_slope = 2
x = np.linspace(0, 1, size)
y = true_intercept + x * true_slope + rng.normal(scale=0.5, size=size)
data = pd.DataFrame({"x": x, "y": y})

model = bambi.Model("y ~ x", data)
fitted = model.fit(draws=1000)

x_axis = xr.DataArray(np.linspace(0, 1, num=100), dims=["x_plot"])
mu_pred = fitted.posterior["Intercept"] + fitted.posterior["x"] * x_axis
mu_mean = mu_pred.mean(dim=("chain", "draw"))
mu_plot = mu_pred.stack(sample=("chain", "draw"))
random_subset = rng.permutation(np.arange(len(mu_plot.sample)))[:200]
plt.scatter(x, y)
plt.plot(x_axis, mu_plot.isel(sample=random_subset), color="black", alpha=0.025)
plt.plot(x_axis, mu_mean, color="C1")
plt.show()
```





## 参考

- [theano README](https://github.com/SNSerHello/MyNotes/tree/main/theano)
- [(Generalized) Linear and Hierarchical Linear Models in PyMC3](https://docs.pymc.io/en/v3/pymc-examples/examples/generalized_linear_models/GLM.html)