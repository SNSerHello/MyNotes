# Onnxruntime环境搭建

## Windows

```bash
$ conda activate py38-cu102
$ pip3 install --upgrade onnxruntime
```

在`CUDA10.2+CUDNN7.6`环境下，在使用`CUDAExecutionProvider`的时候，有可能会出现类似如下错误信息：`[E:onnxruntime:Default, provider_bridge_ort.cc:1022 onnxruntime::ProviderLibrary::Get] LoadLibrary failed with error 126 "" when trying to load %CONDA_PREFIX%\lib\site-packages\onnxruntime\capi\onnxruntime_providers_cuda.dll"`

### 如何解决？

```bash
$ conda activate py38-cu102
# 假设环境中已经安装了VS2017
$ "C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Visual Studio 2017\Visual Studio Tools\VC\x64 Native Tools Command Prompt for VS 2017.lnk"
$ dumpbin /dependents %CONDA_PREFIX%\Lib\site-packages\onnxruntime\capi\onnxruntime_providers_cuda.dll
```

**运行结果**

```bash
Microsoft (R) COFF/PE Dumper Version 14.16.27048.0
Copyright (C) Microsoft Corporation.  All rights reserved.

Dump of file %CONDA_PREFIX%\Lib\site-packages\onnxruntime\capi\onnxruntime_providers_cuda.dll

File Type: DLL

  Image has the following dependencies:

    cublas64_11.dll
    cudnn64_8.dll
    cufft64_10.dll
    onnxruntime_providers_shared.dll
    cudart64_110.dll
    KERNEL32.dll
    MSVCP140.dll
    VCRUNTIME140.dll
    VCRUNTIME140_1.dll
    api-ms-win-crt-math-l1-1-0.dll
    api-ms-win-crt-heap-l1-1-0.dll
    api-ms-win-crt-runtime-l1-1-0.dll
    api-ms-win-crt-stdio-l1-1-0.dll
    api-ms-win-crt-environment-l1-1-0.dll
    api-ms-win-crt-string-l1-1-0.dll

  Summary

        1000 .00cfg
      125000 .data
        C000 .gfids
        5000 .idata
        1000 .nvFatBi
    1533E000 .nv_fatb
       44000 .pdata
      180000 .rdata
       15000 .reloc
        1000 .rsrc
      F2D000 .text
        1000 .tls
```

从上面的信息可以看出，`onnxruntime`的CUDA依赖环境

- `cublas64_11.dll` => cuda11.x
- `cudnn64_8.dll` => cudnn8.x

现在我们的环境是`CUDA10.2+CUDNN7.6`，所以需要进行一定的修改才能运行

```bash
$ cp %CONDA_PREFIX%\Library\bin\cublas64_10.dll %CONDA_PREFIX%\Lib\site-packages\onnxruntime\capi\cublas64_11.dll
$ cp %CONDA_PREFIX%\Library\bin\cudnn64_7.dll %CONDA_PREFIX%\Lib\site-packages\onnxruntime\capi\cudnn64_8.dll
$ cp %CONDA_PREFIX%\Library\bin\cudart64_102.dll %CONDA_PREFIX%\Lib\site-packages\onnxruntime\capi\cudart64_110.dll
```

上面方法是拷贝相关文件到onnxruntime的dll目录中去，这样不会造成dll文件的污染。另外一种方法是在`%CONDA_PREFIX%\Library\bin`目录下面拷贝并重命名相关文件，即使日后onnxruntime更新后，也无需做此工作。从笔者的理解来看，如果出现这些情况，那么最好的方法是使用Anaconda3搭建`CUDA11.X+CUDNN8.X`环境（比如[py38-cu113](https://github.com/SNSerHello/MyNotes/blob/main/anaconda3/py38-cu113.yaml)）来完成运行onnxruntime，而不是在`CUDA10.2+CUDNN7.6`的环境上修修补补，造成日后环境的认为复杂性，一旦出现问题，很难搞清楚是哪方面的问题。



## 参考

- [anaconda3](https://github.com/SNSerHello/MyNotes/tree/main/anaconda3)
- [py38-cu102](https://github.com/SNSerHello/MyNotes/blob/main/anaconda3/py38-cu102.yaml)
- [py38-cu113](https://github.com/SNSerHello/MyNotes/blob/main/anaconda3/py38-cu113.yaml)