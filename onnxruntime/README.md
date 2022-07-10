# Onnxruntime环境搭建

## Windows

```
$ conda activate py38-cu102
$ pip3 install --upgrade onnxruntime
```

在`CUDA10.2+CUDNN7.6`环境下，在使用`CUDAExecutionProvider`的时候，有可能会出现类似如下错误信息：`[E:onnxruntime:Default, provider_bridge_ort.cc:1022 onnxruntime::ProviderLibrary::Get] LoadLibrary failed with error 126 "" when trying to load %CONDA_PREFIX%\lib\site-packages\onnxruntime\capi\onnxruntime_providers_cuda.dll"`

### 如何解决？

```
$ conda activate py38-cu102
# 假设环境中已经安装了VS2017
$ "C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Visual Studio 2017\Visual Studio Tools\VC\x64 Native Tools Command Prompt for VS 2017.lnk"
$ dumpbin /dependents %CONDA_PREFIX%\Lib\site-packages\onnxruntime\capi\onnxruntime_providers_cuda.dll
```

**运行结果**

```
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

```
$ cp %CONDA_PREFIX%\Library\bin\cublas64_10.dll %CONDA_PREFIX%\Lib\site-packages\onnxruntime\capi\cublas64_11.dll
$ cp %CONDA_PREFIX%\Library\bin\cudnn64_7.dll %CONDA_PREFIX%\Lib\site-packages\onnxruntime\capi\cudnn64_8.dll
$ cp %CONDA_PREFIX%\Library\bin\cudart64_102.dll %CONDA_PREFIX%\Lib\site-packages\onnxruntime\capi\cudart64_110.dll
```



## 参考

- [py38-cu102](https://github.com/SNSerHello/MyNotes/blob/main/anaconda3/py38-cu102.yaml)