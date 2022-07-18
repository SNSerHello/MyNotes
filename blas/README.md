# BLAS笔记

**BLAS**是Basic Linear Algebra Subprograms的缩写，即基本线性代数子程序，简介可以参考：[Wiki BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms)。它的函数分成3个levels，即

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

| Level  | 函数名              | CLBlast |
| ------ | ------------------- | ------- |
| Level1 | srotg               | ×       |
| Level1 | drotg               | ×       |
| Level1 | srotmg              | ×       |
| Level1 | drotmg              | ×       |
| Level1 | srot                | ×       |
| Level1 | drot                | ×       |
| Level1 | srotm               | ×       |
| Level1 | drotm               | ×       |
| Level1 | sswap               | √       |
| Level1 | dswap               | √       |
| Level1 | cswap               | √       |
| Level1 | zswap               | √       |
| Level1 | hswap               | √       |
| Level1 | sscal               | √       |
| Level1 | dscal               | √       |
| Level1 | cscal               | √       |
| Level1 | zscal               | √       |
| Level1 | hscal               | √       |
| Level1 | scopy               | √       |
| Level1 | dcopy               | √       |
| Level1 | ccopy               | √       |
| Level1 | zcopy               | √       |
| Level1 | hcopy               | √       |
| Level1 | saxpy               | √       |
| Level1 | daxpy               | √       |
| Level1 | caxpy               | √       |
| Level1 | zaxpy               | √       |
| Level1 | haxpy               | √       |
| Level1 | sdot                | √       |
| Level1 | ddot                | √       |
| Level1 | hdot                | √       |
| Level1 | cdotu               | √       |
| Level1 | zdotu               | √       |
| Level1 | cdotc               | √       |
| Level1 | zdotc               | √       |
| Level1 | snrm2               | √       |
| Level1 | dnrm2               | √       |
| Level1 | scnrm2              | √       |
| Level1 | dznrm2              | √       |
| Level1 | hnrm2               | √       |
| Level1 | sasum               | √       |
| Level1 | dasum               | √       |
| Level1 | casum               | √       |
| Level1 | dzasum              | √       |
| Level1 | hasum               | √       |
| Level1 | ssum                | √       |
| Level1 | dsum                | √       |
| Level1 | scsum               | √       |
| Level1 | dzsum               | √       |
| Level1 | hsum                | √       |
| Level1 | samax               | √       |
| Level1 | damax               | √       |
| Level1 | camax               | √       |
| Level1 | zamax               | √       |
| Level1 | hamax               | √       |
| Level1 | samin               | √       |
| Level1 | damin               | √       |
| Level1 | camin               | √       |
| Level1 | zamin               | √       |
| Level1 | hamin               | √       |
| Level1 | smax                | √       |
| Level1 | dmax                | √       |
| Level1 | cmax                | √       |
| Level1 | zmax                | √       |
| Level1 | hmax                | √       |
| Level1 | smin                | √       |
| Level1 | dmin                | √       |
| Level1 | cmin                | √       |
| Level1 | zmin                | √       |
| Level1 | hmin                | √       |
| Level2 | sgemv               | √       |
| Level2 | dgemv               | √       |
| Level2 | cgemv               | √       |
| Level2 | zgemv               | √       |
| Level2 | hgemv               | √       |
| Level2 | sgbmv               | √       |
| Level2 | dgbmv               | √       |
| Level2 | cgbmv               | √       |
| Level2 | zgbmv               | √       |
| Level2 | hgbmv               | √       |
| Level2 | chemv               | √       |
| Level2 | zhemv               | √       |
| Level2 | chbmv               | √       |
| Level2 | zhbmv               | √       |
| Level2 | chpmv               | √       |
| Level2 | zhpmv               | √       |
| Level2 | ssymv               | √       |
| Level2 | dsymv               | √       |
| Level2 | hsymv               | √       |
| Level2 | ssbmv               | √       |
| Level2 | dsbmv               | √       |
| Level2 | hsbmv               | √       |
| Level2 | sspmv               | √       |
| Level2 | dspmv               | √       |
| Level2 | hspmv               | √       |
| Level2 | strmv               | √       |
| Level2 | dtrmv               | √       |
| Level2 | ctrmv               | √       |
| Level2 | ztrmv               | √       |
| Level2 | htrmv               | √       |
| Level2 | stbmv               | √       |
| Level2 | dtbmv               | √       |
| Level2 | ctbmv               | √       |
| Level2 | ztbmv               | √       |
| Level2 | htbmv               | √       |
| Level2 | stpmv               | √       |
| Level2 | dtpmv               | √       |
| Level2 | ctpmv               | √       |
| Level2 | ztpmv               | √       |
| Level2 | htpmv               | √       |
| Level2 | strsv               | √       |
| Level2 | dtrsv               | √       |
| Level2 | ctrsv               | √       |
| Level2 | ztrsv               | √       |
| Level2 | stbsv               | √       |
| Level2 | dtbsv               | √       |
| Level2 | ctbsv               | √       |
| Level2 | ztbsv               | √       |
| Level2 | stpsv               | √       |
| Level2 | dtpsv               | √       |
| Level2 | ctpsv               | √       |
| Level2 | ztpsv               | √       |
| Level2 | sger                | √       |
| Level2 | dger                | √       |
| Level2 | hger                | √       |
| Level2 | cgeru               | √       |
| Level2 | zgeru               | √       |
| Level2 | cgerc               | √       |
| Level2 | zgerc               | √       |
| Level2 | cher                | √       |
| Level2 | zher                | √       |
| Level2 | chpr                | √       |
| Level2 | zhpr                | √       |
| Level2 | cher2               | √       |
| Level2 | zher2               | √       |
| Level2 | chpr2               | √       |
| Level2 | zhpr2               | √       |
| Level2 | ssyr                | √       |
| Level2 | dsyr                | √       |
| Level2 | hsyr                | √       |
| Level2 | sspr                | √       |
| Level2 | dspr                | √       |
| Level2 | hspr                | √       |
| Level2 | ssyr2               | √       |
| Level2 | dsyr2               | √       |
| Level2 | hsyr2               | √       |
| Level2 | sspr2               | √       |
| Level2 | dspr2               | √       |
| Level2 | hspr2               | √       |
| Level3 | sgemm               | √       |
| Level3 | dgemm               | √       |
| Level3 | cgemm               | √       |
| Level3 | zgemm               | √       |
| Level3 | hgemm               | √       |
| Level3 | ssymm               | √       |
| Level3 | dsymm               | √       |
| Level3 | csymm               | √       |
| Level3 | zsymm               | √       |
| Level3 | hsymm               | √       |
| Level3 | chemm               | √       |
| Level3 | zhemm               | √       |
| Level3 | ssyrk               | √       |
| Level3 | dsyrk               | √       |
| Level3 | csyrk               | √       |
| Level3 | zsyrk               | √       |
| Level3 | hsyrk               | √       |
| Level3 | cherk               | √       |
| Level3 | zherk               | √       |
| Level3 | ssyr2k              | √       |
| Level3 | dsyr2k              | √       |
| Level3 | csyr2k              | √       |
| Level3 | zsyr2k              | √       |
| Level3 | hsyr2k              | √       |
| Level3 | cher2k              | √       |
| Level3 | zher2k              | √       |
| Level3 | strmm               | √       |
| Level3 | dtrmm               | √       |
| Level3 | ctrmm               | √       |
| Level3 | ztrmm               | √       |
| Level3 | htrmm               | √       |
| Level3 | strsm               | √       |
| Level3 | dtrsm               | √       |
| Level3 | ctrsm               | √       |
| Level3 | ztrsm               | √       |
| Levelx | shad                | √       |
| Levelx | dhad                | √       |
| Levelx | chad                | √       |
| Levelx | zhad                | √       |
| Levelx | somatcopy           | √       |
| Levelx | domatcopy           | √       |
| Levelx | comatcopy           | √       |
| Levelx | zomatcopy           | √       |
| Levelx | homatcopy           | √       |
| Levelx | sim2col             | √       |
| Levelx | dim2col             | √       |
| Levelx | cim2col             | √       |
| Levelx | zim2col             | √       |
| Levelx | him2col             | √       |
| Levelx | scol2im             | √       |
| Levelx | dcol2im             | √       |
| Levelx | ccol2im             | √       |
| Levelx | zcol2im             | √       |
| Levelx | hcol2im             | √       |
| Levelx | sconvgem            | √       |
| Levelx | dconvgemm           | √       |
| Levelx | hconvgemm           | √       |
| Levelx | saxpybatched        | √       |
| Levelx | daxpybatched        | √       |
| Levelx | caxpybatched        | √       |
| Levelx | zaxpybatched        | √       |
| Levelx | haxpybatched        | √       |
| Levelx | sgemmbatched        | √       |
| Levelx | dgemmbatched        | √       |
| Levelx | cgemmbatched        | √       |
| Levelx | zgemmbatched        | √       |
| Levelx | hgemmbatched        | √       |
| Levelx | sgemmstridedbatched | √       |
| Levelx | dgemmstridedbatched | √       |
| Levelx | cgemmstridedbatched | √       |
| Levelx | zgemmstridedbatched | √       |
| Levelx | hgemmstridedbatched | √       |
| Levelx | sgemmwithtempbuffer | √       |
| Levelx | dgemmwithtempbuffer | √       |
| Levelx | cgemmwithtempbuffer | √       |
| Levelx | zgemmwithtempbuffer | √       |
| Levelx | hgemmwithtempbuffer | √       |
| Levelx | sgemmtempbuffersize | √       |
| Levelx | dgemmtempbuffersize | √       |
| Levelx | cgemmtempbuffersize | √       |
| Levelx | zgemmtempbuffersize | √       |
| Levelx | hgemmtempbuffersize | √       |
| Levelx | clearcached         | √       |
| Levelx | fillcache           | √       |
| Levelx | overrideparameters  | √       |

**注释**：

- ×表示未实现
- √表示已实现
- Levelx表示自定义扩展，一般考虑性能提高，比如说支持batched功能等



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

