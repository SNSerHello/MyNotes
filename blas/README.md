# Blas笔记

## Blas基本概念

| **Level 1** | Vector operations, e.g. ![y = \alpha x + y](https://www.gnu.org/software/gsl/doc/html/_images/math/3a4ee7e5f509b73b93a60ae89de7ec45b5153ccb.png) |
| ----------- | ------------------------------------------------------------ |
| **Level 2** | Matrix-vector operations, e.g. ![y = \alpha A x + \beta y](https://www.gnu.org/software/gsl/doc/html/_images/math/49298dd474df9a1f79945837753810102ec29708.png) |
| **Level 3** | Matrix-matrix operations, e.g. ![C = \alpha A B + C](https://www.gnu.org/software/gsl/doc/html/_images/math/4d080773fb8002612a41ccb9d90f682c34cf0255.png) |

| **DOT**  | scalar product, ![x^T y](https://www.gnu.org/software/gsl/doc/html/_images/math/ee24d7173f7f14b9fd8884785facae1b663a4521.png) |
| -------- | ------------------------------------------------------------ |
| **AXPY** | vector sum, ![\alpha x + y](https://www.gnu.org/software/gsl/doc/html/_images/math/ebdecdbe6bf281fe730824d1935dc9bbf77e6a4a.png) |
| **MV**   | matrix-vector product, ![A x](https://www.gnu.org/software/gsl/doc/html/_images/math/e0386fd2d4cc02dc39b150be4ebdaf8bf99e45d4.png) |
| **SV**   | matrix-vector solve, ![inv(A) x](https://www.gnu.org/software/gsl/doc/html/_images/math/1bf4147270885b9a9ffa8551a65ca70a13936808.png) |
| **MM**   | matrix-matrix product, ![A B](https://www.gnu.org/software/gsl/doc/html/_images/math/34ccc48b914229aacaa56bbe50da70740efd6938.png) |
| **SM**   | matrix-matrix solve, ![inv(A) B](https://www.gnu.org/software/gsl/doc/html/_images/math/88b04b9d592846e553d2fc9b269068dd5aa701c5.png) |

| **GE** | general           |
| ------ | ----------------- |
| **GB** | general band      |
| **SY** | symmetric         |
| **SB** | symmetric band    |
| **SP** | symmetric packed  |
| **HE** | hermitian         |
| **HB** | hermitian band    |
| **HP** | hermitian packed  |
| **TR** | triangular        |
| **TB** | triangular band   |
| **TP** | triangular packed |

| **S** | single real    |
| ----- | -------------- |
| **D** | double real    |
| **C** | single complex |
| **Z** | double complex |



## Blas Level1 (vector - vector)

### asum

$$
result = \sum_{i=1}^{n}(|Re(x_i)| + |Im(x_i)|)
$$

### axpy

$$
y \leftarrow alpha * x + y
$$

其中：alpha是标量，x，y是向量

### copy

$$
y \leftarrow  x
$$

### dot

$$
result = \sum_{i=1}^{n}X_iY_i
$$

### sdsdot

$$
result = sb + \sum_{i=1}^{n}X_iY_i
$$

其中: sb是float32，x，y是double（float64）

### dotc

$$
result = \sum_{i=1}^{n}\overline{X_i}Y_i
$$

其中：x，y是复数

### dotu

$$
result = \sum_{i=1}^{n}X_iY_i
$$

其中：x，y是复数

### nrm2

欧几里得距离
$$
result = \| x\|_2
$$

#### 补充：ℓp-norm距离

$$
\|x\|_{p}=\left(\sum_{i=1}^{n}|x_i|^p\right)^{1/p}
$$

### rot

$$
\begin{split}\left[\begin{array}{c}
   x\\
   y
\end{array}\right]
\leftarrow
\left[\begin{array}{c}
   \phantom{-}c*x + s*y\\
   -s*x + c*y
\end{array}\right]\end{split}
$$

### rotg

$$
\begin{split}\begin{bmatrix}
c & s \\
-s & c
\end{bmatrix}.
\begin{bmatrix}
a \\
b
\end{bmatrix}=\begin{bmatrix}
r \\
0
\end{bmatrix}\end{split}
$$

### rotm

$$
\begin{split}\begin{bmatrix}x_i \\ y_i\end{bmatrix}=
H
\begin{bmatrix}x_i \\ y_i\end{bmatrix}\end{split}
$$

### rotmg

$$
\begin{split}\begin{bmatrix}
x1 \\
0
\end{bmatrix}=H\begin{bmatrix}
x1\sqrt{d1} \\
y1\sqrt{d2}
\end{bmatrix}\end{split}
$$

### scal

$$
x \leftarrow alpha*x
$$

### swap

$$
\begin{split}\left[\begin{array}{c}
   y\\
   x
\end{array}\right]
\leftarrow
\left[\begin{array}{c}
   x\\
   y
\end{array}\right]\end{split}
$$

### iamax

返回向量中绝对值最大值的索引

### iamin

返回向量中绝对值最小的索引



## Blas Level2 (matrix - vector)

### gbmv

$$
y \leftarrow alpha*op(A)*x + beta*y
$$

其中：A为带状矩阵

### gemv

$$
y \leftarrow alpha*op(A)*x + beta*y
$$

其中：A为普通矩阵

### ger

仅仅支持实数
$$
A \leftarrow alpha*x*y^T + A
$$

### gerc

仅仅支持复数
$$
A \leftarrow alpha*x*y^H + A
$$
其中：H表示共轭转置

### geru

仅仅支持复数
$$
A \leftarrow alpha*x*y^T + A
$$

### hbmv

仅仅支持复数
$$
y \leftarrow alpha*A*x + beta*y
$$
其中：A是赫米特带状矩阵(Hermitian band matrix)

### hemv

仅仅支持复数
$$
y \leftarrow alpha*A*x + beta*y
$$
其中：A是赫米特矩阵(Hermitian matrix)
$$
{\displaystyle A{\text{ Hermitian}}\quad \iff \quad a_{ij}={\overline {{a}_{ji}}}}
$$


### her

仅仅支持复数
$$
A \leftarrow alpha*x*x^H + A
$$
其中：A是赫米特矩阵(Hermitian matrix)， H表示共轭转置

### her2

仅仅支持复数
$$
A \leftarrow alpha*x*y^H + conjg(alpha)*y*x^H + A
$$
其中：A是赫米特矩阵(Hermitian matrix)， H表示共轭转置

### hpmv

仅仅支持复数
$$
y \leftarrow alpha*A*x + beta*y
$$
其中：A是赫米特包装矩阵（Hermitian packed matrix）

包装格式一般是为了节省内存，比如说：

| UPLO | Triangular matrix ***A\***                                   | Packed storage in array AP                                   |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `U'  | ![$ \left( \begin{array}{cccc} a_{11} & a_{12} & a_{13} & a_{14} \\ & a_{22} & a_{23} & a_{24} \\ & & a_{33} & a_{34} \\ & & & a_{44} \end{array} \right) $](https://www.netlib.org/lapack/lug/img918.gif) | ![$ a_{11} \; \underbrace{a_{12} \: a_{22}} \; \underbrace{a_{13} \: a_{23} \: a_{33}} \; \underbrace{a_{14} \: a_{24} \: a_{34} \: a_{44}} $](https://www.netlib.org/lapack/lug/img926.gif) |
| `L'  | ![$ \left( \begin{array}{cccc} a_{11} & & & \\ a_{21} & a_{22} & & \\ a_{31} & a_{32} & a_{33} & \\ a_{41} & a_{42} & a_{43} & a_{44} \end{array} \right) $](https://www.netlib.org/lapack/lug/img920.gif) | ![$ \underbrace{a_{11} \: a_{21} \: a_{31} \: a_{41}} \; \underbrace{a_{22} \: a_{32} \: a_{42}} \; \underbrace{a_{33} \: a_{43}} \; a_{44} $](https://www.netlib.org/lapack/lug/img927.gif) |

### hpr

仅仅支持复数
$$
A \leftarrow alpha*x*x^H + A
$$
其中：A是赫米特包装矩阵（Hermitian packed matrix）， H表示共轭转置

### hpr2

仅仅支持复数
$$
A \leftarrow alpha*x*y^H + conjg(alpha)*y*x^H + A
$$
其中：A是赫米特包装矩阵（Hermitian packed matrix）， H表示共轭转置

### sbmv

仅仅支持实数
$$
y \leftarrow alpha*A*x + beta*y
$$
其中：A是对称带状矩阵（symmetric band matrix）

### spmv

仅仅支持实数
$$
y \leftarrow alpha*A*x + beta*y
$$
其中：A是对称包装矩阵（symmetric packed matrix）

### spr

仅仅支持实数
$$
A \leftarrow alpha*x*x^T + A
$$
其中：A是对称包装矩阵（symmetric packed matrix）

### spr2

仅仅支持实数
$$
A \leftarrow alpha*x*y^T + alpha*y*x^T + A
$$
其中：A是对称包装矩阵（symmetric packed matrix）

### symv

仅仅支持实数
$$
y \leftarrow alpha*A*x + beta*y
$$
其中：A是对称矩阵（symmetric matrix）

### syr

仅仅支持实数
$$
A \leftarrow alpha*x*x^T + A
$$
其中：A是对称矩阵（symmetric matrix）

### syr2

仅仅支持实数
$$
A \leftarrow alpha*x*y^T + alpha*y*x^T + A
$$
其中：A是对称矩阵（symmetric matrix）

### tbmv

$$
x \leftarrow op(A)*x
$$

其中：A是三角带状矩阵（triangular band matrix）

### tbsv

$$
op(A)*x = b
$$

其中：A是三角带状矩阵（triangular band matrix）

### tpmv

$$
x \leftarrow op(A)*x
$$

其中：A是三角包装矩阵（triangular packed matrix）

### tpsv

$$
op(A)*x = b
$$

其中：A是三角包装矩阵（triangular packedmatrix）

### trmv

$$
x \leftarrow op(A)*x
$$

其中：A为三角矩阵

### trsv

$$
op(A)*x = b
$$

其中：A是三角矩阵



## Blas Level3

### gemm

$$
C \leftarrow alpha*op(A)*op(B) + beta*C
$$

### hemm

仅仅支持复数

1）当left_right == side::left时
$$
C \leftarrow alpha*op(A)*op(B) + beta*C
$$
其中：A是赫米特矩阵（Hermitian matrix），B、C是一般矩阵

2）当left_right == side::right时
$$
C \leftarrow alpha*B*A + beta*C
$$
其中：A是赫米特矩阵（Hermitian matrix），B、C是一般矩阵

### herk

仅仅支持复数
$$
C \leftarrow alpha*op(A)*op(A)^H + beta*C
$$
其中：A是一般矩阵，C是赫米特矩阵（Hermitian matrix）

### her2k

仅仅支持复数

1）当trans == transpose::nontrans时
$$
C \leftarrow alpha*A*B^H + conjg(alpha)*B*A^H + beta*C
$$
其中：A、B是一般矩阵，C是赫米特矩阵（Hermitian matrix）

2）当trans= = transpose::conjtrans时
$$
C \leftarrow alpha*B*A^H + conjg(alpha)*A*B^H + beta*C
$$
其中：A、B是一般矩阵，C是赫米特矩阵（Hermitian matrix）

### symm

1）当left_right == side::left时
$$
C \leftarrow alpha*A*B + beta*C
$$
其中：A是对称矩阵，B、C是一般矩阵

2）当left_right == side::right时
$$
C \leftarrow alpha*B*A + beta*C
$$
其中：A是对称矩阵，B、C是一般矩阵

### syrk

$$
C \leftarrow alpha*op(A)*op(A)^T + beta*C
$$

其中：A是一般矩阵，C是对称矩阵

### syr2k

1）当trans == transpose::nontrans时
$$
C \leftarrow alpha*(A*B^T + B*A^T) + beta*C
$$
其中：C是对称矩阵，A、B是一般矩阵

2）当trans= = transpose::conjtrans时
$$
C \leftarrow alpha*(A^T*B + B^T*A) + beta * C
$$
其中：C是对称矩阵，A、B是一般矩阵

### trmm

1）当left_right == side::left时
$$
B \leftarrow alpha*op(A)*B
$$
其中：A是三角矩阵，B是一般矩阵

2）当left_right == side::right时
$$
B \leftarrow alpha*B*op(A)
$$
其中：A是三角矩阵，B是一般矩阵

### trsm

1）当left_right == side::left时
$$
op(A)*X = alpha*B
$$
其中：A是三角矩阵，B是一般矩阵

2）当left_right == side::right时
$$
X*op(A) = alpha*B
$$
其中：A是三角矩阵，B是一般矩阵



## 参考

- [oneMKL](https://oneapi-src.github.io/oneMKL/domains/blas/blas.html)
- [Wolfram Basic Linear Algebra](https://reference.wolfram.com/language/LowLevelLinearAlgebra/guide/BLASGuide.html)
- [Norm (mathematics)](https://en.wikipedia.org/wiki/Norm_(mathematics))
- [Hermitian matrix](https://en.wikipedia.org/wiki/Hermitian_matrix)
- [Packed Storage](https://www.netlib.org/lapack/lug/node123.html#:~:text=For%20complex%20Hermitian%20matrices%2C%20packing,the%20upper%20triangle%20by%20rows.)
- [Netlib BLAS Manual](http://netlib.org/blas/)
- [Wiki BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms)
- [GSL BLAS](https://www.gnu.org/software/gsl/doc/html/blas.html)

