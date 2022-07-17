# BLAS笔记

**BLAS**是Basic Linear Algebra Subprograms的缩写，即基本线性代数子程序，简介可以参考：https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms。它的函数分成3个levels，即

- **Level1** (vector - vector)，计算y的线性代数表达式为：
  $$y \leftarrow \alpha x + y$$
- **Level2** (matrix - vector)，计算y的线性代数表达式为：
  $$y \leftarrow \alpha A x + \beta y$$
  或者求解x向量
  $$T x = y$$
- **Level3** (matrix - matrix)，线性代数表达式为：
  $$C \leftarrow \alpha AB + \beta C$$

BLAS有很多实现方案，常见的如下所示

- CPU
  - Intel MKL (Only Intel CPU)
  - Netlib BLAS (Fortran77语言实现)
  - Netlib CBLAS (C语言实现)
  - GSL (GNU Scientific Libaray, 实现CBLAS接口)
  - OpenBLAS
  - LAPACK (Fortran77语言实现)
  - uBLAS (boost库的一部分)
  - Eigen BLAS (Fortran77与C语言的实现)
- GPU
  - cuBLAS (Only NVIDIA GPU)
  - rocBLAS (Only AMD GPU)
  - clBLAS (OpenCL BLAS实现，AMD主导)
  - clBLAST (tuned OpenCL BLAS实现)

对于CPU的BLAS方案，最常使用的是Intel MKL与OpenBLAS；GPU的BLAS方案，最常用的是cuBLAS（NVIDIA GPU）与clBLAST。clBLAS已经基本上停止维护，性能不佳，一般考虑使用clBLAST方案。

BLAS APIs的定义可以参考[netlib官网](http://www.netlib.org/blas/)，对于BLAS函数的前缀，比如S/D/C/Z，与后缀U/C等进行了说明，详见：https://www.gnu.org/software/gsl/doc/html/blas.html

| 前缀 | 说明                   |
| ---- | ---------------------- |
| S    | 单精度                 |
| D    | 双精度                 |
| C    | 单精度复数             |
| Z    | 双精度复数             |
| DS   | 输入单精度，输出双精度 |

| 后缀 | 说明                                   |
| ---- | -------------------------------------- |
| C    | 复数计算，做向量共轭（conjugated）     |
| U    | 复数计算，不做向量共轭（unconjugated） |

例子

sdot：实数单精度dot计算

ddot：实数双精度dot计算

dsdot：输入单精度实数，输出双精度实数的dot计算

cdotc：单精度复数计算，并对第一个输入当精度向量进行共轭计算

cdotu：单精度复数计算，无向量共轭计算

zdotc：双精度复数计算，并对第一个输入双精度向量进行共轭计算

zdotu：双精度复数计算，无向量共轭计算

| 常见操作 | 说明                           |
| -------- | :----------------------------- |
| **DOT**  | scalar product, $x^T y$      |
| **AXPY** | vector sum, $ax + y$           |
| **MV**   | matrix-vector product,$Ax$     |
| **SV**   | matrix-vector solve, $A^{-1}x$ |
| **MM**   | matrix-matrix product, $AB$    |
| **SM**   | matrix-matrix solve, $A^{-1}B$ |

| 常见缩写 | 说明                       |
| -------- | :------------------------- |
| **GE**   | general                    |
| **GB**   | general band               |
| **SY**   | symmetric                  |
| **SB**   | symmetric band (对称带状)  |
| **SP**   | symmetric packed(对称包装) |
| **HE**   | hermitian(赫米特)          |
| **HB**   | hermitian band             |
| **HP**   | hermitian packed           |
| **TR**   | triangular                 |
| **TB**   | triangular band            |
| **TP**   | triangular packed          |

对于包装的内存存储格式主要是为了节省内存，比如说

| UPLO | Triangular matrix ***A\***                                   | Packed storage in array AP                                   |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `U'  | ![$ \left( \begin{array}{cccc} a_{11} & a_{12} & a_{13} & a_{14} \\ & a_{22} & a_{23} & a_{24} \\ & & a_{33} & a_{34} \\ & & & a_{44} \end{array} \right) $](https://www.netlib.org/lapack/lug/img918.gif) | ![$ a_{11} \; \underbrace{a_{12} \: a_{22}} \; \underbrace{a_{13} \: a_{23} \: a_{33}} \; \underbrace{a_{14} \: a_{24} \: a_{34} \: a_{44}} $](https://www.netlib.org/lapack/lug/img926.gif) |
| `L'  | ![$ \left( \begin{array}{cccc} a_{11} & & & \\ a_{21} & a_{22} & & \\ a_{31} & a_{32} & a_{33} & \\ a_{41} & a_{42} & a_{43} & a_{44} \end{array} \right) $](https://www.netlib.org/lapack/lug/img920.gif) | ![$ \underbrace{a_{11} \: a_{21} \: a_{31} \: a_{41}} \; \underbrace{a_{22} \: a_{32} \: a_{42}} \; \underbrace{a_{33} \: a_{43}} \; a_{44} $](https://www.netlib.org/lapack/lug/img927.gif) |

## BLAS APIs

| 函数名              | CLBlast |
| ------------------- | ------- |
| srotg               | ×       |
| drotg               | ×       |
| srotmg              | ×       |
| drotmg              | ×       |
| srot                | ×       |
| drot                | ×       |
| srotm               | ×       |
| drotm               | ×       |
| sswap               | √       |
| dswap               | √       |
| cswap               | √       |
| zswap               | √       |
| hswap               | √       |
| sscal               | √       |
| dscal               | √       |
| cscal               | √       |
| zscal               | √       |
| hscal               | √       |
| scopy               | √       |
| dcopy               | √       |
| ccopy               | √       |
| zcopy               | √       |
| hcopy               | √       |
| saxpy               | √       |
| daxpy               | √       |
| caxpy               | √       |
| zaxpy               | √       |
| haxpy               | √       |
| sdot                | √       |
| ddot                | √       |
| hdot                | √       |
| cdotu               | √       |
| zdotu               | √       |
| cdotc               | √       |
| zdotc               | √       |
| snrm2               | √       |
| dnrm2               | √       |
| scnrm2              | √       |
| dznrm2              | √       |
| hnrm2               | √       |
| sasum               | √       |
| dasum               | √       |
| casum               | √       |
| dzasum              | √       |
| hasum               | √       |
| ssum                | √       |
| dsum                | √       |
| scsum               | √       |
| dzsum               | √       |
| hsum                | √       |
| samax               | √       |
| damax               | √       |
| camax               | √       |
| zamax               | √       |
| hamax               | √       |
| samin               | √       |
| damin               | √       |
| camin               | √       |
| zamin               | √       |
| hamin               | √       |
| smax                | √       |
| dmax                | √       |
| cmax                | √       |
| zmax                | √       |
| hmax                | √       |
| smin                | √       |
| dmin                | √       |
| cmin                | √       |
| zmin                | √       |
| hmin                | √       |
| sgemv               | √       |
| dgemv               | √       |
| cgemv               | √       |
| zgemv               | √       |
| hgemv               | √       |
| sgbmv               | √       |
| dgbmv               | √       |
| cgbmv               | √       |
| zgbmv               | √       |
| hgbmv               | √       |
| chemv               | √       |
| zhemv               | √       |
| chbmv               | √       |
| zhbmv               | √       |
| chpmv               | √       |
| zhpmv               | √       |
| ssymv               | √       |
| dsymv               | √       |
| hsymv               | √       |
| ssbmv               | √       |
| dsbmv               | √       |
| hsbmv               | √       |
| sspmv               | √       |
| dspmv               | √       |
| hspmv               | √       |
| strmv               | √       |
| dtrmv               | √       |
| ctrmv               | √       |
| ztrmv               | √       |
| htrmv               | √       |
| stbmv               | √       |
| dtbmv               | √       |
| ctbmv               | √       |
| ztbmv               | √       |
| htbmv               | √       |
| stpmv               | √       |
| dtpmv               | √       |
| ctpmv               | √       |
| ztpmv               | √       |
| htpmv               | √       |
| strsv               | √       |
| dtrsv               | √       |
| ctrsv               | √       |
| ztrsv               | √       |
| stbsv               | √       |
| dtbsv               | √       |
| ctbsv               | √       |
| ztbsv               | √       |
| stpsv               | √       |
| dtpsv               | √       |
| ctpsv               | √       |
| ztpsv               | √       |
| sger                | √       |
| dger                | √       |
| hger                | √       |
| cgeru               | √       |
| zgeru               | √       |
| cgerc               | √       |
| zgerc               | √       |
| cher                | √       |
| zher                | √       |
| chpr                | √       |
| zhpr                | √       |
| cher2               | √       |
| zher2               | √       |
| chpr2               | √       |
| zhpr2               | √       |
| ssyr                | √       |
| dsyr                | √       |
| hsyr                | √       |
| sspr                | √       |
| dspr                | √       |
| hspr                | √       |
| ssyr2               | √       |
| dsyr2               | √       |
| hsyr2               | √       |
| sspr2               | √       |
| dspr2               | √       |
| hspr2               | √       |
| sgemm               | √       |
| dgemm               | √       |
| cgemm               | √       |
| zgemm               | √       |
| hgemm               | √       |
| ssymm               | √       |
| dsymm               | √       |
| csymm               | √       |
| zsymm               | √       |
| hsymm               | √       |
| chemm               | √       |
| zhemm               | √       |
| ssyrk               | √       |
| dsyrk               | √       |
| csyrk               | √       |
| zsyrk               | √       |
| hsyrk               | √       |
| cherk               | √       |
| zherk               | √       |
| ssyr2k              | √       |
| dsyr2k              | √       |
| csyr2k              | √       |
| zsyr2k              | √       |
| hsyr2k              | √       |
| cher2k              | √       |
| zher2k              | √       |
| strmm               | √       |
| dtrmm               | √       |
| ctrmm               | √       |
| ztrmm               | √       |
| htrmm               | √       |
| strsm               | √       |
| dtrsm               | √       |
| ctrsm               | √       |
| ztrsm               | √       |
| shad                | √       |
| dhad                | √       |
| chad                | √       |
| zhad                | √       |
| somatcopy           | √       |
| domatcopy           | √       |
| comatcopy           | √       |
| zomatcopy           | √       |
| homatcopy           | √       |
| sim2col             | √       |
| dim2col             | √       |
| cim2col             | √       |
| zim2col             | √       |
| him2col             | √       |
| scol2im             | √       |
| dcol2im             | √       |
| ccol2im             | √       |
| zcol2im             | √       |
| hcol2im             | √       |
| sconvgem            | √       |
| dconvgemm           | √       |
| hconvgemm           | √       |
| saxpybatched        | √       |
| daxpybatched        | √       |
| caxpybatched        | √       |
| zaxpybatched        | √       |
| haxpybatched        | √       |
| sgemmbatched        | √       |
| dgemmbatched        | √       |
| cgemmbatched        | √       |
| zgemmbatched        | √       |
| hgemmbatched        | √       |
| sgemmstridedbatched | √       |
| dgemmstridedbatched | √       |
| cgemmstridedbatched | √       |
| zgemmstridedbatched | √       |
| hgemmstridedbatched | √       |
| sgemmwithtempbuffer | √       |
| dgemmwithtempbuffer | √       |
| cgemmwithtempbuffer | √       |
| zgemmwithtempbuffer | √       |
| hgemmwithtempbuffer | √       |
| sgemmtempbuffersize | √       |
| dgemmtempbuffersize | √       |
| cgemmtempbuffersize | √       |
| zgemmtempbuffersize | √       |
| hgemmtempbuffersize | √       |
| clearcached         | √       |
| fillcache           | √       |
| overrideparameters  | √       |



## 参考

- [oneMKL](https://oneapi-src.github.io/oneMKL/domains/blas/blas.html)
- [Wolfram Basic Linear Algebra](https://reference.wolfram.com/language/LowLevelLinearAlgebra/guide/BLASGuide.html)
- [Norm (mathematics)](https://en.wikipedia.org/wiki/Norm_(mathematics))
- [Hermitian matrix](https://en.wikipedia.org/wiki/Hermitian_matrix)
- [Packed Storage](https://www.netlib.org/lapack/lug/node123.html#:~:text=For%20complex%20Hermitian%20matrices%2C%20packing,the%20upper%20triangle%20by%20rows.)
- [Netlib BLAS Manual](http://netlib.org/blas/)
- [Wiki BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms)
- [GSL BLAS](https://www.gnu.org/software/gsl/doc/html/blas.html)
- [GSL BLAS APIs](https://www.gnu.org/software/gsl/doc/html/cblas.html)
- [BLAS APIs](http://www.netlib.org/blas/)

