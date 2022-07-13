# GCC环境搭建

## Ubuntu20.04LTS

### 1. 安装GCC v7版本

```bash
sudo apt install libgcc-7-dev \
	gcc-7 \
	g++-7 \
	gcc-7-locales \
	g++-7-multilib \
	gcc-7-doc \
	gcc-7-multilib \
	libstdc++-7-doc \
	lib32stdc++6-7-dbg \
	libx32stdc++6-7-dbg \
	gfortran-7 \
	libgfortran-7-dev \
	gfortran-7-multilib \
	gfortran-7-doc
```

### 2. update-alternatives不同的GCC版本

```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 940
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 940
sudo update-alternatives --install /usr/bin/cpp cpp /usr/bin/cpp-9 940
sudo update-alternatives --install /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-9 940
sudo update-alternatives --install /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-9 940
sudo update-alternatives --install /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-9 940
sudo update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-9 940
sudo update-alternatives --install /usr/bin/gcov-dump gcov-dump /usr/bin/gcov-dump-9 940
sudo update-alternatives --install /usr/bin/gcov-tool gcov-tool /usr/bin/gcov-tool-9 940
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gfortran x86_64-linux-gnu-gfortran /usr/bin/x86_64-linux-gnu-gfortran-9 940
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcov x86_64-linux-gnu-gcov /usr/bin/x86_64-linux-gnu-gcov-9 940
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcov-tool x86_64-linux-gnu-gcov-tool /usr/bin/x86_64-linux-gnu-gcov-tool-9 940
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcov-dump x86_64-linux-gnu-gcov-dump /usr/bin/x86_64-linux-gnu-gcov-dump-9 940
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc-ranlib x86_64-linux-gnu-gcc-ranlib /usr/bin/x86_64-linux-gnu-gcc-ranlib-9 940
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc-nm x86_64-linux-gnu-gcc-nm /usr/bin/x86_64-linux-gnu-gcc-nm-9 940
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc-ar x86_64-linux-gnu-gcc-ar /usr/bin/x86_64-linux-gnu-gcc-ar-9 940
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc /usr/bin/x86_64-linux-gnu-gcc-9 940
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/x86_64-linux-gnu-g++-9 940
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-cpp x86_64-linux-gnu-cpp /usr/bin/x86_64-linux-gnu-cpp-9 940

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 750
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 750
sudo update-alternatives --install /usr/bin/cpp cpp /usr/bin/cpp-7 750
sudo update-alternatives --install /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-7 750
sudo update-alternatives --install /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-7 750
sudo update-alternatives --install /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-7 750
sudo update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-7 750
sudo update-alternatives --install /usr/bin/gcov-dump gcov-dump /usr/bin/gcov-dump-7 750
sudo update-alternatives --install /usr/bin/gcov-tool gcov-tool /usr/bin/gcov-tool-7 750
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gfortran x86_64-linux-gnu-gfortran /usr/bin/x86_64-linux-gnu-gfortran-7 750
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcov x86_64-linux-gnu-gcov /usr/bin/x86_64-linux-gnu-gcov-7 750
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcov-tool x86_64-linux-gnu-gcov-tool /usr/bin/x86_64-linux-gnu-gcov-tool-7 750
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcov-dump x86_64-linux-gnu-gcov-dump /usr/bin/x86_64-linux-gnu-gcov-dump-7 750
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc-ranlib x86_64-linux-gnu-gcc-ranlib /usr/bin/x86_64-linux-gnu-gcc-ranlib-7 750
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc-ar x86_64-linux-gnu-gcc-ar /usr/bin/x86_64-linux-gnu-gcc-ar-7 750
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc /usr/bin/x86_64-linux-gnu-gcc-7 750
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/x86_64-linux-gnu-g++-7 750
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-cpp x86_64-linux-gnu-cpp /usr/bin/x86_64-linux-gnu-cpp-7 750
```

### 3. 使用制定版本GCC

默认使用GCC V9版本，如果要使用GCC v7版本，则需要运行如下命令

#### GCC v7

```bash
sudo update-alternatives --set gcc /usr/bin/gcc-7
sudo update-alternatives --set g++ /usr/bin/g++-7
sudo update-alternatives --set cpp /usr/bin/cpp-7
sudo update-alternatives --set gcc-ar /usr/bin/gcc-ar-7
sudo update-alternatives --set gcc-nm /usr/bin/gcc-nm-7
sudo update-alternatives --set gcc-ranlib /usr/bin/gcc-ranlib-7
sudo update-alternatives --set gcov /usr/bin/gcov-7
sudo update-alternatives --set gcov-dump /usr/bin/gcov-dump-7
sudo update-alternatives --set gcov-tool /usr/bin/gcov-tool-7
sudo update-alternatives --set x86_64-linux-gnu-gfortran /usr/bin/x86_64-linux-gnu-gfortran-7
sudo update-alternatives --set x86_64-linux-gnu-gcov /usr/bin/x86_64-linux-gnu-gcov-7
sudo update-alternatives --set x86_64-linux-gnu-gcov-tool /usr/bin/x86_64-linux-gnu-gcov-tool-7
sudo update-alternatives --set x86_64-linux-gnu-gcov-dump /usr/bin/x86_64-linux-gnu-gcov-dump-7
sudo update-alternatives --set x86_64-linux-gnu-gcc-ranlib /usr/bin/x86_64-linux-gnu-gcc-ranlib-7
sudo update-alternatives --set x86_64-linux-gnu-gcc-ar /usr/bin/x86_64-linux-gnu-gcc-ar-7
sudo update-alternatives --set x86_64-linux-gnu-gcc /usr/bin/x86_64-linux-gnu-gcc-7
sudo update-alternatives --set x86_64-linux-gnu-g++ /usr/bin/x86_64-linux-gnu-g++-7
sudo update-alternatives --set x86_64-linux-gnu-cpp /usr/bin/x86_64-linux-gnu-cpp-7
```

#### GCC v9

```bash
sudo update-alternatives --set gcc /usr/bin/gcc-9
sudo update-alternatives --set g++ /usr/bin/g++-9
sudo update-alternatives --set cpp /usr/bin/cpp-9
sudo update-alternatives --set gcc-ar /usr/bin/gcc-ar-9
sudo update-alternatives --set gcc-nm /usr/bin/gcc-nm-9
sudo update-alternatives --set gcc-ranlib /usr/bin/gcc-ranlib-9
sudo update-alternatives --set gcov /usr/bin/gcov-9
sudo update-alternatives --set gcov-dump /usr/bin/gcov-dump-9
sudo update-alternatives --set gcov-tool /usr/bin/gcov-tool-9
sudo update-alternatives --set x86_64-linux-gnu-gfortran /usr/bin/x86_64-linux-gnu-gfortran-9
sudo update-alternatives --set x86_64-linux-gnu-gcov /usr/bin/x86_64-linux-gnu-gcov-9
sudo update-alternatives --set x86_64-linux-gnu-gcov-tool /usr/bin/x86_64-linux-gnu-gcov-tool-9
sudo update-alternatives --set x86_64-linux-gnu-gcov-dump /usr/bin/x86_64-linux-gnu-gcov-dump-9
sudo update-alternatives --set x86_64-linux-gnu-gcc-ranlib /usr/bin/x86_64-linux-gnu-gcc-ranlib-9
sudo update-alternatives --set x86_64-linux-gnu-gcc-ar /usr/bin/x86_64-linux-gnu-gcc-ar-9
sudo update-alternatives --set x86_64-linux-gnu-gcc /usr/bin/x86_64-linux-gnu-gcc-9
sudo update-alternatives --set x86_64-linux-gnu-g++ /usr/bin/x86_64-linux-gnu-g++-9
sudo update-alternatives --set x86_64-linux-gnu-cpp /usr/bin/x86_64-linux-gnu-cpp-9
```

#### 设置安装的GCC最高版本

```bash
sudo update-alternatives --auto gcc
sudo update-alternatives --auto g++
sudo update-alternatives --auto cpp
sudo update-alternatives --auto gcc-ar
sudo update-alternatives --auto gcc-nm
sudo update-alternatives --auto gcc-ranlib
sudo update-alternatives --auto gcc-nm
sudo update-alternatives --auto gcov
sudo update-alternatives --auto gcov-dump
sudo update-alternatives --auto gcov-tool
sudo update-alternatives --auto x86_64-linux-gnu-gfortran
sudo update-alternatives --auto x86_64-linux-gnu-gcov
sudo update-alternatives --auto x86_64-linux-gnu-gcov-tool
sudo update-alternatives --auto x86_64-linux-gnu-gcov-dump
sudo update-alternatives --auto x86_64-linux-gnu-gcc-ranlib
sudo update-alternatives --auto x86_64-linux-gnu-gcc-ar
sudo update-alternatives --auto x86_64-linux-gnu-gcc
sudo update-alternatives --auto x86_64-linux-gnu-g++
sudo update-alternatives --auto x86_64-linux-gnu-cpp
```

#### 显示GCC版本

```bash
g++ -v
cpp --version
gcc-ar -v
nm --version
gcc-ranlib -v
gcc-nm --version
gcov -v
gcov-dump -v
gcov-tool -v
x86_64-linux-gnu-gfortran -v
x86_64-linux-gnu-gcov -v
x86_64-linux-gnu-gcov-tool -v
x86_64-linux-gnu-gcov-dump -v
x86_64-linux-gnu-gcc-ranlib -v
x86_64-linux-gnu-gcc-ar -v
x86_64-linux-gnu-gcc -v
x86_64-linux-gnu-g++ -v
x86_64-linux-gnu-cpp --version
```

### 其他

#### 安装GCC v8

```
sudo apt install gcc-8 \
	g++-8 \
	libstdc++-8-dev \
	g++-8-multilib \
	gcc-8-locales \
	gcc-8-doc \
	libstdc++-8-doc \
	libgcc-8-dev \
	gcc-8-multilib \
	lib32gcc-8-dev \
	lib32stdc++-8-dev \
	libx32gcc-8-dev \
	libx32stdc++-8-dev \
	lib32stdc++6-8-dbg \
	libx32stdc++6-8-dbg \
	gfortran-8 \
	libgfortran-8-dev
```

#### update-alternatives for GCC v8

```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 840
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 840
sudo update-alternatives --install /usr/bin/cpp cpp /usr/bin/cpp-8 840
sudo update-alternatives --install /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-8 840
sudo update-alternatives --install /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-8 840
sudo update-alternatives --install /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-8 840
sudo update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-8 840
sudo update-alternatives --install /usr/bin/gcov-dump gcov-dump /usr/bin/gcov-dump-8 840
sudo update-alternatives --install /usr/bin/gcov-tool gcov-tool /usr/bin/gcov-tool-8 840
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gfortran x86_64-linux-gnu-gfortran /usr/bin/x86_64-linux-gnu-gfortran-8 840
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcov x86_64-linux-gnu-gcov /usr/bin/x86_64-linux-gnu-gcov-8 840
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcov-tool x86_64-linux-gnu-gcov-tool /usr/bin/x86_64-linux-gnu-gcov-tool-8 840
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcov-dump x86_64-linux-gnu-gcov-dump /usr/bin/x86_64-linux-gnu-gcov-dump-8 840
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc-ranlib x86_64-linux-gnu-gcc-ranlib /usr/bin/x86_64-linux-gnu-gcc-ranlib-8 840
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc-nm x86_64-linux-gnu-gcc-nm /usr/bin/x86_64-linux-gnu-gcc-nm-8 840
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc-ar x86_64-linux-gnu-gcc-ar /usr/bin/x86_64-linux-gnu-gcc-ar-8 840
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc /usr/bin/x86_64-linux-gnu-gcc-8 840
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/x86_64-linux-gnu-g++-8 840
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-cpp x86_64-linux-gnu-cpp /usr/bin/x86_64-linux-gnu-cpp-8 840
```

### 设置GCC v8

```
sudo update-alternatives --set gcc /usr/bin/gcc-8
sudo update-alternatives --set g++ /usr/bin/g++-8
sudo update-alternatives --set cpp /usr/bin/cpp-8
sudo update-alternatives --set gcc-ar /usr/bin/gcc-ar-8
sudo update-alternatives --set gcc-nm /usr/bin/gcc-nm-8
sudo update-alternatives --set gcc-ranlib /usr/bin/gcc-ranlib-8
sudo update-alternatives --set gcov /usr/bin/gcov-8
sudo update-alternatives --set gcov-dump /usr/bin/gcov-dump-8
sudo update-alternatives --set gcov-tool /usr/bin/gcov-tool-8
sudo update-alternatives --set x86_64-linux-gnu-gfortran /usr/bin/x86_64-linux-gnu-gfortran-8
sudo update-alternatives --set x86_64-linux-gnu-gcov /usr/bin/x86_64-linux-gnu-gcov-8
sudo update-alternatives --set x86_64-linux-gnu-gcov-tool /usr/bin/x86_64-linux-gnu-gcov-tool-8
sudo update-alternatives --set x86_64-linux-gnu-gcov-dump /usr/bin/x86_64-linux-gnu-gcov-dump-8
sudo update-alternatives --set x86_64-linux-gnu-gcc-ranlib /usr/bin/x86_64-linux-gnu-gcc-ranlib-8
sudo update-alternatives --set x86_64-linux-gnu-gcc-ar /usr/bin/x86_64-linux-gnu-gcc-ar-8
sudo update-alternatives --set x86_64-linux-gnu-gcc /usr/bin/x86_64-linux-gnu-gcc-8
sudo update-alternatives --set x86_64-linux-gnu-g++ /usr/bin/x86_64-linux-gnu-g++-8
sudo update-alternatives --set x86_64-linux-gnu-cpp /usr/bin/x86_64-linux-gnu-cpp-8
```



## 参考

- [How to Use update-alternatives Command on Ubuntu](https://linuxhint.com/update_alternatives_ubuntu/)
- [CMake: unsupported GNU version -- gcc versions later than 8 are not supported](https://stackoverflow.com/questions/65605972/cmake-unsupported-gnu-version-gcc-versions-later-than-8-are-not-supported)