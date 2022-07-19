# Jupyterlab环境搭建

下面的操作假设系统中已经安装了nodejs和java，否则在搭建jupytrlab之前，先把它们的环境搭建好。

- Windows
  - [node-v16.16.0-x64.msi](https://nodejs.org/dist/v16.16.0/node-v16.16.0-x64.msi)
  - [java环境](https://www.java.com/zh-CN/download/)
- Linux
  - [node-v16.16.0-linux-x64.tar.xz](https://nodejs.org/dist/v16.16.0/node-v16.16.0-linux-x64.tar.xz)
  - `sudo apt install default-jdk`

```bash
$ wget https://github.com/plantuml/plantuml/releases/download/v1.2021.12/plantuml-1.2021.12.jar
# 在linux系统中，最好把plantuml.jar放在/usr/local/bin目录下，这样它能够被自动发现
$ ln -s plantuml-1.2021.12.jar plantuml.jar
$ conda env create --file jupyterlab.yaml
$ conda activate jupyterlab
(jupyterlab) $ pip3 install --upgrade iplantuml jupyterlab-code-formatter
(jupyterlab) $ jupyter lab build
```

## 检查jupyterlab环境

```python
import iplantuml
```

### plantuml(web版)

```python
%%plantuml -n demo_web

@startuml
Alice -> Bob: Authentication Request
Bob --> Alice: Authentication Response
@enduml
```

运行后，生成`demo_web.svg`文件

![use plantuml web](images/demo_web.svg)

### plantuml(1.2022.6版)

```python
%%plantuml -p ./plantuml-1.2022.6.jar -n demo_1_2022_6

@startuml
Alice -> Bob: Authentication Request
Bob --> Alice: Authentication Response
@enduml
```

运行后，生成`demo_1_2022_6.svg`文件

![use plantuml-1.2022.6.jar](images/demo_1_2022_6.svg)

### plantuml(1.2021.12版)

```python
%%plantuml -p ./plantuml-1.2021.12.jar -n demo_1_2021_12

@startuml
Alice -> Bob: Authentication Request
Bob --> Alice: Authentication Response
@enduml
```

运行后，生成`demo_1_2021_12.svg`文件

![use plantuml-1.2021.12.jar](images/demo_1_2021_12.svg)

[plantuml release](https://github.com/plantuml/plantuml/releases)的最新版本[plantuml-1.2022.6.jar](https://github.com/plantuml/plantuml/releases/download/v1.2022.6/plantuml-1.2022.6.jar)，它与web版本生成的结果都是黑白svg文件，而老版本的[plantuml-1.2021.12.jar](https://github.com/plantuml/plantuml/releases/download/v1.2021.12/plantuml-1.2021.12.jar)显示彩色图片，所以建议采用老版本。[plantuml](https://github.com/plantuml/plantuml)的release老版本会被删除，所以将`plantuml-1.2021.12.jar`放在本地`plantuml`目录中以备不时之需。



## 参考

- [plantuml](https://github.com/plantuml/plantuml)
- [nodejs](https://nodejs.org/zh-cn/download/)