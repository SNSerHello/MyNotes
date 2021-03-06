# Theano环境搭建

Theano曾经风靡一时，可惜后来因为大佬Yoshua Bengio进入谷歌后，谷歌着力开发Tensorflow，它也慢慢退出了历史舞台。现在因为一些老的AI算法使用到了Theano，我们也不想再将他们迁移到新的AI Framework，所以有时候还需要搭建一下theano环境。坦率的说，如果不够熟悉theano的话，那么在现在的开发环境下搭建theano环境还是有些麻烦的，这里主要介绍了使用Anaconda来搭建环境，一方面是相对比较容易，另外一方面是这个方面能够在Windows与Linux中使用，平台迁移起来也比较方便。

## 在Ubuntu20.04LTS中搭建theano环境

```bash
$ sudo apt install libopenblas-dev
$ conda env create --file py38-theano.yaml

$ conda activate py38-theano
$ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
$ nano $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
文件内容如下：
CUDA_ROOT=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CUDA_ROOT/lib:$LD_LIBRARY_PATH
rm ~/.theanorc
cp ~/theanorc.py38 ~/.theanorc
$ mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
$ nano $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
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

[blas]
ldflags=-L/usr/lib/x86_64-linux-gnu -lopenblas -lpthread -lm

[cuda]
root = /media/samba/anaconda3/envs/py38-theano
include_path = /media/samba/anaconda3/envs/py38-theano/include

[lib]
cnmem = 0.5
```



## 在Ubuntu18.04LTS中搭建theano环境

```bash
$ sudo apt install libopenblas-dev
$ conda env create --file py36-theano.yaml

$ conda activate py36-theano
$ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
$ nano $CONDA_PREFIX/etc/conda/activate.d/env_vars.h
文件内容如下：
CUDA_ROOT=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CUDA_ROOT/lib:$LD_LIBRARY_PATH
rm ~/.theanorc
cp ~/theanorc.py36 ~/.theanorc
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

[blas]
ldflags=-L/usr/lib/x86_64-linux-gnu -lopenblas -lpthread -lm

[cuda]
root = /media/samba/anaconda3/envs/py36-theano
include_path = /media/samba/anaconda3/envs/py36-theano/include

[lib]
cnmem = 0.5
```

在Ubuntu20.04LTS推荐使用python3.8+cuda11.3+cudnn8.2搭建环境，在Ubuntu18.04LTS推荐使用python3.6+cuda9.2+cudnn7.1搭建环境，这个不是必须的，我们可以在Ubuntu20.04LTS上使用python3.6+其他的搭配方法。如果Ubuntu上的GCC的版本比较高的话，我们可以使用Anaconda的toolchain来搭建编译环境，详见 [Anaconda3 README](https://github.com/SNSerHello/MyNotes/tree/main/anaconda3)。

## 在Windows中搭建theano环境

**前提条件**

- `cuda_10.2.89_441.22_windows.exe`被安装到`G:\CUDAv10.2`
- `Anaconda3-2020.11-Windows-x86_64.exe`被安装到`G:\Anaconda3`
- `OpenBLAS`被安装到`C:\OpenBLAS`

**安装py38-theano**

```bash
$ conda env create --file py38-theano-win.yaml
$ cd G:\Anaconda3\envs\py38-theano\Library\bin
$ ln -s nvrtc64_102_0.dll nvrtc64_70.dll
$ ln -s cublas64_10.dll cublas64_70.dll
```

**.theanrc配置**

```
[global]
floatX = float32
device = cuda
optimizer_including = cudnn

[cuda]
root = G:\CUDAv10.2\bin
include_path = G:\CUDAv10.2\include

[blas]
ldflags = -LC:\OpenBLAS\bin -lopenblas -lpthread -lm

[gpuarray]
preallocate = 0

[gcc]
cxxflags = -IG:\CUDAv10.2\include -IG:\Anaconda3\envs\py38-theano\Library\include -IC:\OpenBLAS\include -LG:\Anaconda3\envs\py38-theano\Library\lib\x64 -LC:\OpenBLAS\lib

[nvcc]
flags = -LG:\Anaconda3\envs\py38-theano\Library\lib\x64 --machine=64

[dnn]
enabled = True
library_path = G:\Anaconda3\envs\py38-theano\Library\lib\x64
include_path = G:\Anaconda3\envs\py38-theano\Library\include

[lib]
cnmem = 0.5
```

## 检查theano.config配置

theano解析.theanorc后会存放到`theano.config`中，我们可以查看`theano.config`来检查配置是否正确，如下：

```python
$ conda activate py38-theano
(py38-theano) $ ipython
Python 3.8.13 (default, Mar 28 2022, 11:38:47)
Type 'copyright', 'credits' or 'license' for more information
IPython 8.3.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import theano
~/anaconda3/envs/py38-theano/lib/python3.8/site-packages/theano/gpuarray/dnn.py:184: UserWarning: Your cuDNN version is more recent than Theano. If you encounter problems, try updating Theano or downgrading cuDNN to a version >= v5 and <= v7.
  warnings.warn("Your cuDNN version is more recent than "
Using cuDNN version 8201 on context None
Mapped name None to device cuda: NVIDIA GeForce RTX ...

In [2]: print(theano.config)
floatX (('float64', 'float32', 'float16'))
    Doc:  Default floating-point precision for python casts.

Note: float16 support is experimental, use at your own risk.
    Value:  float32

warn_float64 (('ignore', 'warn', 'raise', 'pdb'))
    Doc:  Do an action when a tensor variable with float64 dtype is created. They can't be run on the GPU with the current(old) gpu back-end and are slow with gamer GPUs.
    Value:  ignore

pickle_test_value (<function BoolParam.<locals>.booltype at 0x7f8a90520790>)
    Doc:  Dump test values while pickling model. If True, test values will be dumped with model.
    Value:  True

cast_policy (('custom', 'numpy+floatX'))
    Doc:  Rules for implicit type casting
    Value:  custom

int_division (('int', 'raise', 'floatX'))
    Doc:  What to do when one computes x / y, where both x and y are of integer types
    Value:  int

deterministic (('default', 'more'))
    Doc:  If `more`, sometimes we will select some implementation that are more deterministic, but slower. In particular, on the GPU, we will avoid using AtomicAdd. Sometimes we will still use non-deterministic implementaion, e.g. when we do not have a GPU implementation that is deterministic. Also see the dnn.conv.algo* flags to cover more cases.
    Value:  default

device (cpu, opencl*, cuda*)
    Doc:  Default device for computations. If cuda* or opencl*, change thedefault to try to move computation to the GPU. Do not use upper caseletters, only lower case even if NVIDIA uses capital letters.
    Value:  cuda

init_gpu_device (, opencl*, cuda*)
    Doc:  Initialize the gpu device to use, works only if device=cpu. Unlike 'device', setting this option will NOT move computations, nor shared variables, to the specified GPU. It can be used to run GPU-specific tests on a particular GPU.
    Value:

force_device (<function BoolParam.<locals>.booltype at 0x7f8a90520d30>)
    Doc:  Raise an error if we can't use the specified device
    Value:  False

conv.assert_shape (<function BoolParam.<locals>.booltype at 0x7f8a90520ee0>)
    Doc:  If True, AbstractConv* ops will verify that user-provided shapes match the runtime shapes (debugging option, may slow down compilation)
    Value:  False

print_global_stats (<function BoolParam.<locals>.booltype at 0x7f8a905260d0>)
    Doc:  Print some global statistics (time spent) at the end
    Value:  False

<theano.configdefaults.ContextsParam object at 0x7f8a90591ca0>
    Doc:
    Context map for multi-gpu operation. Format is a
    semicolon-separated list of names and device names in the
    'name->dev_name' format. An example that would map name 'test' to
    device 'cuda0' and name 'test2' to device 'opencl0:0' follows:
    "test->cuda0;test2->opencl0:0".

    Invalid context names are 'cpu', 'cuda*' and 'opencl*'

    Value:

print_active_device (<function BoolParam.<locals>.booltype at 0x7f8a905263a0>)
    Doc:  Print active device at when the GPU device is initialized.
    Value:  True

<theano.configparser.ConfigParam object at 0x7f8a90591d90>
    Doc:  This flag is deprecated and will be removed in next Theano release.
    Value:  False

gpuarray.preallocate (<class 'float'>)
    Doc:  If negative it disables the allocation cache. If
             between 0 and 1 it enables the allocation cache and
             preallocates that fraction of the total GPU memory.  If 1
             or greater it will preallocate that amount of memory (in
             megabytes).
    Value:  0.0

gpuarray.sched (('default', 'multi', 'single'))
    Doc:  The sched parameter passed for context creation to pygpu.
                With CUDA, using "multi" is equivalent to using the parameter
                cudaDeviceScheduleBlockingSync. This is useful to lower the
                CPU overhead when waiting for GPU. One user found that it
                speeds up his other processes that was doing data augmentation.

    Value:  default

gpuarray.single_stream (<function BoolParam.<locals>.booltype at 0x7f8a90526700>)
    Doc:
             If your computations are mostly lots of small elements,
             using single-stream will avoid the synchronization
             overhead and usually be faster.  For larger elements it
             does not make a difference yet.  In the future when true
             multi-stream is enabled in libgpuarray, this may change.
             If you want to make sure to have optimal performance,
             check both options.

    Value:  True

cuda.root (<class 'str'>)
    Doc:  Location of the cuda installation
    Value:  /media/samba/anaconda3/envs/py38-theano

cuda.include_path (<class 'str'>)
    Doc:  Location of the cuda includes
    Value:  /media/samba/anaconda3/envs/py38-theano/include

<theano.configparser.ConfigParam object at 0x7f8a905291f0>
    Doc:  This flag is deprecated; use dnn.conv.algo_fwd.
    Value:  True

<theano.configparser.ConfigParam object at 0x7f8a905293d0>
    Doc:  This flag is deprecated; use `dnn.conv.algo_bwd_filter` and `dnn.conv.algo_bwd_data` instead.
    Value:  True

<theano.configparser.ConfigParam object at 0x7f8a90529430>
    Doc:  This flag is deprecated; use dnn.conv.algo_bwd_data and dnn.conv.algo_bwd_filter.
    Value:  True

dnn.conv.algo_fwd (('small', 'none', 'large', 'fft', 'fft_tiling', 'winograd', 'winograd_non_fused', 'guess_once', 'guess_on_shape_change', 'time_once', 'time_on_shape_change'))
    Doc:  Default implementation to use for cuDNN forward convolution.
    Value:  small

dnn.conv.algo_bwd_data (('none', 'deterministic', 'fft', 'fft_tiling', 'winograd', 'winograd_non_fused', 'guess_once', 'guess_on_shape_change', 'time_once', 'time_on_shape_change'))
    Doc:  Default implementation to use for cuDNN backward convolution to get the gradients of the convolution with regard to the inputs.
    Value:  none

dnn.conv.algo_bwd_filter (('none', 'deterministic', 'fft', 'small', 'winograd_non_fused', 'fft_tiling', 'guess_once', 'guess_on_shape_change', 'time_once', 'time_on_shape_change'))
    Doc:  Default implementation to use for cuDNN backward convolution to get the gradients of the convolution with regard to the filters.
    Value:  none

dnn.conv.precision (('as_input_f32', 'as_input', 'float16', 'float32', 'float64'))
    Doc:  Default data precision to use for the computation in cuDNN convolutions (defaults to the same dtype as the inputs of the convolutions, or float32 if inputs are float16).
    Value:  as_input_f32

dnn.base_path (<class 'str'>)
    Doc:  Install location of cuDNN.
    Value:  /media/samba/anaconda3/envs/py38-theano

dnn.include_path (<class 'str'>)
    Doc:  Location of the cudnn header
    Value:  /media/samba/anaconda3/envs/py38-theano/include

dnn.library_path (<class 'str'>)
    Doc:  Location of the cudnn link library.
    Value:  /media/samba/anaconda3/envs/py38-theano/lib

dnn.bin_path (<class 'str'>)
    Doc:  Location of the cuDNN load library (on non-windows platforms, this is the same as dnn.library_path)
    Value:  /media/samba/anaconda3/envs/py38-theano/lib

dnn.enabled (('auto', 'True', 'False', 'no_check'))
    Doc:  'auto', use cuDNN if available, but silently fall back to not using it if not present. If True and cuDNN can not be used, raise an error. If False, disable cudnn even if present. If no_check, assume present and the version between header and library match (so less compilation at context init)
    Value:  True

magma.include_path (<class 'str'>)
    Doc:  Location of the magma header
    Value:

magma.library_path (<class 'str'>)
    Doc:  Location of the magma library
    Value:

magma.enabled (<function BoolParam.<locals>.booltype at 0x7f8a9052b550>)
    Doc:   If True, use magma for matrix computation. If False, disable magma
    Value:  False

assert_no_cpu_op (('ignore', 'warn', 'raise', 'pdb'))
    Doc:  Raise an error/warning if there is a CPU op in the computational graph.
    Value:  ignore

<theano.configparser.ConfigParam object at 0x7f8a90529d00>
    Doc:  Default compilation mode
    Value:  Mode

cxx (<class 'str'>)
    Doc:  The C++ compiler to use. Currently only g++ is supported, but supporting additional compilers should not be too difficult. If it is empty, no C++ code is compiled.
    Value:  /home/yanqing/anaconda3/envs/py38-theano/bin/x86_64-conda-linux-gnu-c++

linker (('cvm', 'c|py', 'py', 'c', 'c|py_nogc', 'vm', 'vm_nogc', 'cvm_nogc'))
    Doc:  Default linker used if the theano flags mode is Mode
    Value:  cvm

allow_gc (<function BoolParam.<locals>.booltype at 0x7f8af3a66f70>)
    Doc:  Do we default to delete intermediate results during Theano function calls? Doing so lowers the memory requirement, but asks that we reallocate memory at the next function call. This is implemented for the default linker, but may not work for all linkers.
    Value:  True

optimizer (('o4', 'o3', 'o2', 'o1', 'unsafe', 'fast_run', 'fast_compile', 'merge', 'None'))
    Doc:  Default optimizer. If not None, will use this optimizer with the Mode
    Value:  o4

optimizer_verbose (<function BoolParam.<locals>.booltype at 0x7f8af3a19820>)
    Doc:  If True, we print all optimization being applied
    Value:  False

on_opt_error (('warn', 'raise', 'pdb', 'ignore'))
    Doc:  What to do when an optimization crashes: warn and skip it, raise the exception, or fall into the pdb debugger.
    Value:  warn

nocleanup (<function BoolParam.<locals>.booltype at 0x7f8af3a19a60>)
    Doc:  Suppress the deletion of code files that did not compile cleanly
    Value:  False

on_unused_input (('raise', 'warn', 'ignore'))
    Doc:  What to do if a variable in the 'inputs' list of  theano.function() is not used in the graph.
    Value:  raise

tensor.cmp_sloppy (<class 'int'>)
    Doc:  Relax tensor._allclose (0) not at all, (1) a bit, (2) more
    Value:  0

tensor.local_elemwise_fusion (<function BoolParam.<locals>.booltype at 0x7f8af3a19dc0>)
    Doc:  Enable or not in fast_run mode(fast_run optimization) the elemwise fusion optimization
    Value:  True

gpu.local_elemwise_fusion (<function BoolParam.<locals>.booltype at 0x7f8af3a19f70>)
    Doc:  Enable or not in fast_run mode(fast_run optimization) the gpu elemwise fusion optimization
    Value:  True

lib.amdlibm (<function BoolParam.<locals>.booltype at 0x7f8af3a1a160>)
    Doc:  Use amd's amdlibm numerical library
    Value:  False

gpuelemwise.sync (<function BoolParam.<locals>.booltype at 0x7f8af3a1a310>)
    Doc:  when true, wait that the gpu fct finished and check it error code.
    Value:  True

traceback.limit (<class 'int'>)
    Doc:  The number of stack to trace. -1 mean all.
    Value:  8

traceback.compile_limit (<class 'int'>)
    Doc:  The number of stack to trace to keep during compilation. -1 mean all. If greater then 0, will also make us save Theano internal stack trace.
    Value:  0

experimental.unpickle_gpu_on_cpu (<function BoolParam.<locals>.booltype at 0x7f8af3a1a5e0>)
    Doc:  Allow unpickling of pickled GpuArrays as numpy.ndarrays.This is useful, if you want to open a GpuArray without having cuda installed.If you have cuda installed, this will force unpickling tobe done on the cpu to numpy.ndarray.Please be aware that this may get you access to the data,however, trying to unpicke gpu functions will not succeed.This flag is experimental and may be removed any time, whengpu<>cpu transparency is solved.
    Value:  False

numpy.seterr_all (('ignore', 'warn', 'raise', 'call', 'print', 'log', 'None'))
    Doc:  ("Sets numpy's behaviour for floating-point errors, ", "see numpy.seterr. 'None' means not to change numpy's default, which can be different for different numpy releases. This flag sets the default behaviour for all kinds of floating-point errors, its effect can be overriden for specific errors by the following flags: seterr_divide, seterr_over, seterr_under and seterr_invalid.")
    Value:  ignore

numpy.seterr_divide (('None', 'ignore', 'warn', 'raise', 'call', 'print', 'log'))
    Doc:  Sets numpy's behavior for division by zero, see numpy.seterr. 'None' means using the default, defined by numpy.seterr_all.
    Value:  None

numpy.seterr_over (('None', 'ignore', 'warn', 'raise', 'call', 'print', 'log'))
    Doc:  Sets numpy's behavior for floating-point overflow, see numpy.seterr. 'None' means using the default, defined by numpy.seterr_all.
    Value:  None

numpy.seterr_under (('None', 'ignore', 'warn', 'raise', 'call', 'print', 'log'))
    Doc:  Sets numpy's behavior for floating-point underflow, see numpy.seterr. 'None' means using the default, defined by numpy.seterr_all.
    Value:  None

numpy.seterr_invalid (('None', 'ignore', 'warn', 'raise', 'call', 'print', 'log'))
    Doc:  Sets numpy's behavior for invalid floating-point operation, see numpy.seterr. 'None' means using the default, defined by numpy.seterr_all.
    Value:  None

warn.ignore_bug_before (('0.9', 'None', 'all', '0.3', '0.4', '0.4.1', '0.5', '0.6', '0.7', '0.8', '0.8.1', '0.8.2', '0.9', '0.10', '1.0', '1.0.1', '1.0.2', '1.0.3', '1.0.4', '1.0.5'))
    Doc:  If 'None', we warn about all Theano bugs found by default. If 'all', we don't warn about Theano bugs found by default. If a version, we print only the warnings relative to Theano bugs found after that version. Warning for specific bugs can be configured with specific [warn] flags.
    Value:  0.9

warn.argmax_pushdown_bug (<function BoolParam.<locals>.booltype at 0x7f8af3a1ac10>)
    Doc:  Warn if in past version of Theano we generated a bug with the theano.tensor.nnet.nnet.local_argmax_pushdown optimization. Was fixed 27 may 2010
    Value:  False

warn.gpusum_01_011_0111_bug (<function BoolParam.<locals>.booltype at 0x7f8af3fd3a60>)
    Doc:  Warn if we are in a case where old version of Theano had a silent bug with GpuSum pattern 01,011 and 0111 when the first dimensions was bigger then 4096. Was fixed 31 may 2010
    Value:  False

warn.sum_sum_bug (<function BoolParam.<locals>.booltype at 0x7f8af3a1ae50>)
    Doc:  Warn if we are in a case where Theano version between version 9923a40c7b7a and the 2 august 2010 (fixed date), generated an error in that case. This happens when there are 2 consecutive sums in the graph, bad code was generated. Was fixed 2 August 2010
    Value:  False

warn.sum_div_dimshuffle_bug (<function BoolParam.<locals>.booltype at 0x7f8af3a1d040>)
    Doc:  Warn if previous versions of Theano (between rev. 3bd9b789f5e8, 2010-06-16, and cfc6322e5ad4, 2010-08-03) would have given incorrect result. This bug was triggered by sum of division of dimshuffled tensors.
    Value:  False

warn.subtensor_merge_bug (<function BoolParam.<locals>.booltype at 0x7f8af3a1d1f0>)
    Doc:  Warn if previous versions of Theano (before 0.5rc2) could have given incorrect results when indexing into a subtensor with negative stride (for instance, for instance, x[a:b:-1][c]).
    Value:  False

warn.gpu_set_subtensor1 (<function BoolParam.<locals>.booltype at 0x7f8af3a1d3a0>)
    Doc:  Warn if previous versions of Theano (before 0.6) could have given incorrect results when moving to the gpu set_subtensor(x[int vector], new_value)
    Value:  False

warn.vm_gc_bug (<function BoolParam.<locals>.booltype at 0x7f8af3a1d550>)
    Doc:  There was a bug that existed in the default Theano configuration, only in the development version between July 5th 2012 and July 30th 2012. This was not in a released version. If your code was affected by this bug, a warning will be printed during the code execution if you use the `linker=vm,vm.lazy=True,warn.vm_gc_bug=True` Theano flags. This warning is disabled by default as the bug was not released.
    Value:  False

warn.signal_conv2d_interface (<function BoolParam.<locals>.booltype at 0x7f8af3a1d700>)
    Doc:  Warn we use the new signal.conv2d() when its interface changed mid June 2014
    Value:  False

warn.reduce_join (<function BoolParam.<locals>.booltype at 0x7f8af3a1d8b0>)
    Doc:  Your current code is fine, but Theano versions prior to 0.7 (or this development version) might have given an incorrect result. To disable this warning, set the Theano flag warn.reduce_join to False. The problem was an optimization, that modified the pattern "Reduce{scalar.op}(Join(axis=0, a, b), axis=0)", did not check the reduction axis. So if the reduction axis was not 0, you got a wrong answer.
    Value:  False

warn.inc_set_subtensor1 (<function BoolParam.<locals>.booltype at 0x7f8af3a1da60>)
    Doc:  Warn if previous versions of Theano (before 0.7) could have given incorrect results for inc_subtensor and set_subtensor when using some patterns of advanced indexing (indexing with one vector or matrix of ints).
    Value:  False

warn.round (<function BoolParam.<locals>.booltype at 0x7f8af3a1dc10>)
    Doc:  Warn when using `tensor.round` with the default mode. Round changed its default from `half_away_from_zero` to `half_to_even` to have the same default as NumPy.
    Value:  False

warn.inc_subtensor1_opt (<function BoolParam.<locals>.booltype at 0x7f8af3a1ddc0>)
    Doc:  Warn if previous versions of Theano (before 0.10) could have given incorrect results when computing inc_subtensor(zeros[idx], x)[idx], when idx is an array of integers with duplicated values.
    Value:  True

compute_test_value (('off', 'ignore', 'warn', 'raise', 'pdb'))
    Doc:  If 'True', Theano will run each op at graph build time, using Constants, SharedVariables and the tag 'test_value' as inputs to the function. This helps the user track down problems in the graph before it gets optimized.
    Value:  off

print_test_value (<function BoolParam.<locals>.booltype at 0x7f8af3a1e040>)
    Doc:  If 'True', the __eval__ of a Theano variable will return its test_value when this is available. This has the practical conseguence that, e.g., in debugging `my_var` will print the same as `my_var.tag.test_value` when a test value is defined.
    Value:  False

compute_test_value_opt (('off', 'ignore', 'warn', 'raise', 'pdb'))
    Doc:  For debugging Theano optimization only. Same as compute_test_value, but is used during Theano optimization
    Value:  off

unpickle_function (<function BoolParam.<locals>.booltype at 0x7f8af3a1e280>)
    Doc:  Replace unpickled Theano functions with None. This is useful to unpickle old graphs that pickled them when it shouldn't
    Value:  True

reoptimize_unpickled_function (<function BoolParam.<locals>.booltype at 0x7f8af3a1e430>)
    Doc:  Re-optimize the graph when a theano function is unpickled from the disk.
    Value:  False

exception_verbosity (('low', 'high'))
    Doc:  If 'low', the text of exceptions will generally refer to apply nodes with short names such as Elemwise{add_no_inplace}. If 'high', some exceptions will also refer to apply nodes with long descriptions  like:
    A. Elemwise{add_no_inplace}
            B. log_likelihood_v_given_h
            C. log_likelihood_h
    Value:  low

openmp (<function BoolParam.<locals>.booltype at 0x7f8af3a1e670>)
    Doc:  Allow (or not) parallel computation on the CPU with OpenMP. This is the default value used when creating an Op that supports OpenMP parallelization. It is preferable to define it via the Theano configuration file ~/.theanorc or with the environment variable THEANO_FLAGS. Parallelization is only done for some operations that implement it, and even for operations that implement parallelism, each operation is free to respect this flag or not. You can control the number of threads used with the environment variable OMP_NUM_THREADS. If it is set to 1, we disable openmp in Theano by default.
    Value:  False

openmp_elemwise_minsize (<class 'int'>)
    Doc:  If OpenMP is enabled, this is the minimum size of vectors for which the openmp parallelization is enabled in element wise ops.
    Value:  200000

check_input (<function BoolParam.<locals>.booltype at 0x7f8af3a1e8b0>)
    Doc:  Specify if types should check their input in their C code. It can be used to speed up compilation, reduce overhead (particularly for scalars) and reduce the number of generated C files.
    Value:  True

cache_optimizations (<function BoolParam.<locals>.booltype at 0x7f8af3a1ea60>)
    Doc:  WARNING: work in progress, does not work yet. Specify if the optimization cache should be used. This cache will any optimized graph and its optimization. Actually slow downs a lot the first optimization, and could possibly still contains some bugs. Use at your own risks.
    Value:  False

unittests.rseed (<class 'str'>)
    Doc:  Seed to use for randomized unit tests. Special value 'random' means using a seed of None.
    Value:  666

NanGuardMode.nan_is_error (<function BoolParam.<locals>.booltype at 0x7f8af3a1ed30>)
    Doc:  Default value for nan_is_error
    Value:  True

NanGuardMode.inf_is_error (<function BoolParam.<locals>.booltype at 0x7f8af3a1eee0>)
    Doc:  Default value for inf_is_error
    Value:  True

NanGuardMode.big_is_error (<function BoolParam.<locals>.booltype at 0x7f8af3a210d0>)
    Doc:  Default value for big_is_error
    Value:  True

NanGuardMode.action (('raise', 'warn', 'pdb'))
    Doc:  What NanGuardMode does when it finds a problem
    Value:  raise

optimizer_excluding (<class 'str'>)
    Doc:  When using the default mode, we will remove optimizer with these tags. Separate tags with ':'.
    Value:

optimizer_including (<class 'str'>)
    Doc:  When using the default mode, we will add optimizer with these tags. Separate tags with ':'.
    Value:  cudnn

optimizer_requiring (<class 'str'>)
    Doc:  When using the default mode, we will require optimizer with these tags. Separate tags with ':'.
    Value:

DebugMode.patience (<class 'int'>)
    Doc:  Optimize graph this many times to detect inconsistency
    Value:  10

DebugMode.check_c (<function BoolParam.<locals>.booltype at 0x7f8af3a21670>)
    Doc:  Run C implementations where possible
    Value:  True

DebugMode.check_py (<function BoolParam.<locals>.booltype at 0x7f8af3a21820>)
    Doc:  Run Python implementations where possible
    Value:  True

DebugMode.check_finite (<function BoolParam.<locals>.booltype at 0x7f8af3a219d0>)
    Doc:  True -> complain about NaN/Inf results
    Value:  True

DebugMode.check_strides (<class 'int'>)
    Doc:  Check that Python- and C-produced ndarrays have same strides. On difference: (0) - ignore, (1) warn, or (2) raise error
    Value:  0

DebugMode.warn_input_not_reused (<function BoolParam.<locals>.booltype at 0x7f8af3a21ca0>)
    Doc:  Generate a warning when destroy_map or view_map says that an op works inplace, but the op did not reuse the input for its output.
    Value:  True

DebugMode.check_preallocated_output (<class 'str'>)
    Doc:  Test thunks with pre-allocated memory as output storage. This is a list of strings separated by ":". Valid values are: "initial" (initial storage in storage map, happens with Scan),"previous" (previously-returned memory), "c_contiguous", "f_contiguous", "strided" (positive and negative strides), "wrong_size" (larger and smaller dimensions), and "ALL" (all of the above).
    Value:

DebugMode.check_preallocated_output_ndim (<class 'int'>)
    Doc:  When testing with "strided" preallocated output memory, test all combinations of strides over that number of (inner-most) dimensions. You may want to reduce that number to reduce memory or time usage, but it is advised to keep a minimum of 2.
    Value:  4

profiling.time_thunks (<function BoolParam.<locals>.booltype at 0x7f8af3a230d0>)
    Doc:  Time individual thunks when profiling
    Value:  True

profiling.n_apply (<class 'int'>)
    Doc:  Number of Apply instances to print by default
    Value:  20

profiling.n_ops (<class 'int'>)
    Doc:  Number of Ops to print by default
    Value:  20

profiling.output_line_width (<class 'int'>)
    Doc:  Max line width for the profiling output
    Value:  512

profiling.min_memory_size (<class 'int'>)
    Doc:  For the memory profile, do not print Apply nodes if the size
             of their outputs (in bytes) is lower than this threshold
    Value:  1024

profiling.min_peak_memory (<function BoolParam.<locals>.booltype at 0x7f8af3a23700>)
    Doc:  The min peak memory usage of the order
    Value:  False

profiling.destination (<class 'str'>)
    Doc:
             File destination of the profiling output

    Value:  stderr

profiling.debugprint (<function BoolParam.<locals>.booltype at 0x7f8af3a23940>)
    Doc:
             Do a debugprint of the profiled functions

    Value:  False

profiling.ignore_first_call (<function BoolParam.<locals>.booltype at 0x7f8af3a23af0>)
    Doc:
             Do we ignore the first call of a Theano function.

    Value:  False

optdb.position_cutoff (<class 'float'>)
    Doc:  Where to stop eariler during optimization. It represent the position of the optimizer where to stop.
    Value:  inf

optdb.max_use_ratio (<class 'float'>)
    Doc:  A ratio that prevent infinite loop in EquilibriumOptimizer.
    Value:  8.0

gcc.cxxflags (<class 'str'>)
    Doc:  Extra compiler flags for gcc
    Value:  -I/media/samba/anaconda3/envs/py38-theano/include -L/media/samba/anaconda3/envs/py38-theano/lib -L/usr/lib/x86_64-linux-gnu -lrt -pthread -lresolv

cmodule.warn_no_version (<function BoolParam.<locals>.booltype at 0x7f8af3a23e50>)
    Doc:  If True, will print a warning when compiling one or more Op with C code that can't be cached because there is no c_code_cache_version() function associated to at least one of those Ops.
    Value:  False

cmodule.remove_gxx_opt (<function BoolParam.<locals>.booltype at 0x7f8af3a27040>)
    Doc:  If True, will remove the -O* parameter passed to g++.This is useful to debug in gdb modules compiled by Theano.The parameter -g is passed by default to g++
    Value:  False

cmodule.compilation_warning (<function BoolParam.<locals>.booltype at 0x7f8af3a271f0>)
    Doc:  If True, will print compilation warnings.
    Value:  False

cmodule.preload_cache (<function BoolParam.<locals>.booltype at 0x7f8af3a273a0>)
    Doc:  If set to True, will preload the C module cache at import time
    Value:  False

cmodule.age_thresh_use (<class 'int'>)
    Doc:  In seconds. The time after which Theano won't reuse a compile c module.
    Value:  2073600

cmodule.debug (<function BoolParam.<locals>.booltype at 0x7f8af3a275e0>)
    Doc:  If True, define a DEBUG macro (if not exists) for any compiled C code.
    Value:  False

blas.ldflags (<class 'str'>)
    Doc:  lib[s] to include for [Fortran] level-3 blas implementation
    Value:  -L/usr/lib/x86_64-linux-gnu -lopenblas64 -lpthread -lm

blas.check_openmp (<function BoolParam.<locals>.booltype at 0x7f8af3a279d0>)
    Doc:  Check for openmp library conflict.
WARNING: Setting this to False leaves you open to wrong results in blas-related operations.
    Value:  True

metaopt.verbose (<class 'int'>)
    Doc:  0 for silent, 1 for only warnings, 2 for full output withtimings and selected implementation
    Value:  0

metaopt.optimizer_excluding (<class 'str'>)
    Doc:  exclude optimizers with these tags. Separate tags with ':'.
    Value:

metaopt.optimizer_including (<class 'str'>)
    Doc:  include optimizers with these tags. Separate tags with ':'.
    Value:

profile (<function BoolParam.<locals>.booltype at 0x7f8af3a27d30>)
    Doc:  If VM should collect profile information
    Value:  False

profile_optimizer (<function BoolParam.<locals>.booltype at 0x7f8af3a27ee0>)
    Doc:  If VM should collect optimizer profile information
    Value:  False

profile_memory (<function BoolParam.<locals>.booltype at 0x7f8af3a2b0d0>)
    Doc:  If VM should collect memory profile information and print it
    Value:  False

<theano.configparser.ConfigParam object at 0x7f8af3a291f0>
    Doc:  Useful only for the vm linkers. When lazy is None, auto detect if lazy evaluation is needed and use the appropriate version. If lazy is True/False, force the version used between Loop/LoopGC and Stack.
    Value:  None

warn.identify_1pexp_bug (<function BoolParam.<locals>.booltype at 0x7f8af3a2b310>)
    Doc:  Warn if Theano versions prior to 7987b51 (2011-12-18) could have yielded a wrong result due to a bug in the is_1pexp function
    Value:  False

on_shape_error (('warn', 'raise'))
    Doc:  warn: print a warning and use the default value. raise: raise an error
    Value:  warn

tensor.insert_inplace_optimizer_validate_nb (<class 'int'>)
    Doc:  -1: auto, if graph have less then 500 nodes 1, else 10
    Value:  -1

experimental.local_alloc_elemwise (<function BoolParam.<locals>.booltype at 0x7f8af3a2b670>)
    Doc:  DEPRECATED: If True, enable the experimental optimization local_alloc_elemwise. Generates error if not True. Use optimizer_excluding=local_alloc_elemwise to dsiable.
    Value:  True

experimental.local_alloc_elemwise_assert (<function BoolParam.<locals>.booltype at 0x7f8af3a2b700>)
    Doc:  When the local_alloc_elemwise is applied, add an assert to highlight shape errors.
    Value:  True

scan.allow_gc (<function BoolParam.<locals>.booltype at 0x7f8af3a2b940>)
    Doc:  Allow/disallow gc inside of Scan (default: False)
    Value:  False

scan.allow_output_prealloc (<function BoolParam.<locals>.booltype at 0x7f8af3a2baf0>)
    Doc:  Allow/disallow memory preallocation for outputs inside of scan (default: True)
    Value:  True

scan.debug (<function BoolParam.<locals>.booltype at 0x7f8af3a2bca0>)
    Doc:  If True, enable extra verbose output related to scan
    Value:  False

compile.wait (<class 'int'>)
    Doc:  Time to wait before retrying to acquire the compile lock.
    Value:  5

cycle_detection (('regular', 'fast'))
    Doc:  If cycle_detection is set to regular, most inplaces are allowed,but it is slower. If cycle_detection is set to faster, less inplacesare allowed, but it makes the compilation faster.The interaction of which one give the lower peak memory usage iscomplicated and not predictable, so if you are close to the peakmemory usage, triyng both could give you a small gain.
    Value:  regular

check_stack_trace (('off', 'log', 'warn', 'raise'))
    Doc:  A flag for checking the stack trace during the optimization process. default (off): does not check the stack trace of any optimization log: inserts a dummy stack trace that identifies the optimizationthat inserted the variable that had an empty stack trace.warn: prints a warning if a stack trace is missing and also a dummystack trace is inserted that indicates which optimization insertedthe variable that had an empty stack trace.raise: raises an exception if a stack trace is missing
    Value:  off

compile.timeout (<class 'int'>)
    Doc:  In seconds, time that a process will wait before deciding to
override an existing lock. An override only happens when the existing
lock is held by the same owner *and* has not been 'refreshed' by this
owner for more than this period. Refreshes are done every half timeout
period for running processes.
    Value:  120

compiledir_format (<class 'str'>)
    Doc:  Format string for platform-dependent compiled module subdirectory
(relative to base_compiledir). Available keys: device, gxx_version,
hostname, numpy_version, platform, processor, python_bitwidth,
python_int_bitwidth, python_version, short_platform, theano_version.
Defaults to 'compiledir_%(short_platform)s-%(processor)s-%(python_vers
ion)s-%(python_bitwidth)s'.
    Value:  compiledir_%(short_platform)s-%(processor)s-%(python_version)s-%(python_bitwidth)s

<theano.configparser.ConfigParam object at 0x7f8af3a29c10>
    Doc:  platform-independent root directory for compiled modules
    Value:  ~/.theano

<theano.configparser.ConfigParam object at 0x7f8af3a29d00>
    Doc:  platform-dependent cache directory for compiled modules
    Value:  ~/.theano/compiledir_Linux-5.13--generic-x86_64-with-glibc2.17-x86_64-3.8.13-64

<theano.configparser.ConfigParam object at 0x7f8af3a29d30>
    Doc:  Directory to cache pre-compiled kernels for the gpuarray backend.
    Value:  ~/.theano/compiledir_Linux-5.13--generic-x86_64-with-glibc2.17-x86_64-3.8.13-64/gpuarray_kernels

ctc.root (<class 'str'>)
    Doc:  Directory which contains the root of Baidu CTC library. It is assumed     that the compiled library is either inside the build, lib or lib64     subdirectory, and the header inside the include directory.
    Value:

```

## 检查theano环境

```python
from theano import function, config, shared, tensor as tt
import numpy
import time

import theano
print("theano version: {}".format(theano.__version__))

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tt.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any(
    [
        isinstance(x.op, tt.elemwise.Elemwise) and ("Gpu" not in type(x.op).__name__)
        for x in f.maker.fgraph.toposort()
    ]
):
    print("Used the cpu")
else:
    print("Used the gpu")
```



## 参考

- [Anaconda3 README](https://github.com/SNSerHello/MyNotes/tree/main/anaconda3)
- [Using the GPU](https://theano-pymc.readthedocs.io/en/latest/tutorial/using_gpu.html)