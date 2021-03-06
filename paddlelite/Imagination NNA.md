# Imagination NNA

Imagination NNA的nnadpater实现目录结构如下所示

```bash
Paddle-Lite/lite/backends/nnadapter/nnadapter/src/driver/imagination_nna
│  CMakeLists.txt
│  dependencies.cmake
│  driver.cc
│  engine.cc
│  engine.h
│  imgdnn_manager.cc
│  imgdnn_manager.h
│  utility.cc
│  utility.h
│
└─converter
        all.h
        conv2d.cc
        converter.cc
        converter.h
        elementwise.cc
        mat_mul.cc
        pool2d.cc
        reshape.cc
        softmax.cc
        unary_activations.cc

```

其中

- `all.h`定义了所有的`Imagination NNA`实现算子，因为使用了头文件的形式，所有在converter.cc中多了一些奇怪的`undef`的内容，代码第一眼看上去有些邋遢。如果我们使用另外一种形式来定义，那么将会看上去很整洁，比如说我们定义`all.def`文件，如下

  ```c++
  REGISTER_CONVERTER(AVERAGE_POOL_2D, ConvertPool2D)
  REGISTER_CONVERTER(CONV_2D, ConvertConv2D)
  REGISTER_CONVERTER(MAX_POOL_2D, ConvertPool2D)
  REGISTER_CONVERTER(RELU, ConvertUnaryActivations)
  REGISTER_CONVERTER(RELU6, ConvertUnaryActivations)
  REGISTER_CONVERTER(SOFTMAX, ConvertSoftmax)
  REGISTER_CONVERTER(MAT_MUL, ConvertMatMul)
  REGISTER_CONVERTER(ADD, ConvertElementwise)
  REGISTER_CONVERTER(SUB, ConvertElementwise)
  REGISTER_CONVERTER(MUL, ConvertElementwise)
  REGISTER_CONVERTER(DIV, ConvertElementwise)
  REGISTER_CONVERTER(MAX, ConvertElementwise)
  REGISTER_CONVERTER(MIN, ConvertElementwise)
  REGISTER_CONVERTER(RESHAPE, ConvertReshape)
  
  #undef REGISTER_CONVERTER
  ```

  在宣称引入各个函数的时候如下使用

  ```c++
  #define REGISTER_CONVERTER(__op_type__, __func_name__) \
    extern int __func_name__(Converter* converter, core::Operation* operation);
  #include "driver/imagination_nna/converter/all.def"  // NOLINT
  ```

  在实现Apply函数的时候可以如下使用

  ```c++
  int Converter::Apply(core::Model* model) {
    // Convert the NNAdapter operations to the imgdnn operators
    std::vector&lt;core::Operation*&gt; operations =
        SortOperationsInTopologicalOrder(model);
    for (auto operation : operations) {
      NNADAPTER_VLOG(5) &lt;&lt; &quot;Converting &quot; &lt;&lt; OperationTypeToString(operation-&gt;type)
                        &lt;&lt; &quot; ...&quot;;
      switch (operation-&gt;type) {
  #define REGISTER_CONVERTER(__op_type__, __func_name__) \
    case NNADAPTER_##__op_type__:                        \
      __func_name__(this, operation);                    \
      break;
  #include &quot;driver/imagination_nna/converter/all.def&quot;  // NOLINT
        default:
          NNADAPTER_LOG(FATAL) &lt;&lt; &quot;Unsupported operation(&quot;
                               &lt;&lt; OperationTypeToString(operation-&gt;type)
                               &lt;&lt; &quot;) is found.&quot;;
          break;
      }
    }
    return NNADAPTER_NO_ERROR;
  }
  ```

- `converter.cc`实现`Imagination NNA`模块内部使用的函数，其中最重要的是`Apply`函数，它首先调用`SortOperationsInTopologicalOrder(model)`获得网络拓扑操作顺序，然后一个`for`语句完成网络的输入到输出的全部过程。

- `converter`目录下的其他后缀为`cc`的文件基本上对应着每个算子的实现，参考下**表1 Imagination NNA算子对照表**，它有7个大类型的算子，即对应

  - `ConvertPool2D` → conv2d.cc
  - `ConvertElementwise`  → elementwise.cc
  - `ConvertMatMul` → mat_mul.cc
  - `ConvertPool2D` → pool2d.cc
  - `ConvertReshape` → reshape.cc
  - `ConvertSoftmax` → softmax.cc
  - `ConvertUnaryActivations` → unary_activations.cc

- 



## 实现的算子

​                                                  **表1 Imagination NNA算子对照表**

| 算子类型                  | 算子函数名              |
| ------------------------- | ----------------------- |
| NNADAPTER_AVERAGE_POOL_2D | ConvertPool2D           |
| NNADAPTER_CONV_2D         | ConvertConv2D           |
| NNADAPTER_MAX_POOL_2D     | ConvertPool2D           |
| NNADAPTER_RELU            | ConvertUnaryActivations |
| NNADAPTER_SOFTMAX         | ConvertSoftmax          |
| NNADAPTER_MAT_MUL         | ConvertMatMul           |
| NNADAPTER_ADD             | ConvertElementwise      |
| NNADAPTER_SUB             | ConvertElementwise      |
| NNADAPTER_DIV             | ConvertElementwise      |
| NNADAPTER_MAX             | ConvertElementwise      |
| NNADAPTER_MIN             | ConvertElementwise      |
| NNADAPTER_RESHAPE         | ConvertReshape          |

**注释**：更多算子类型定义，请参考：`Paddle-Lite/lite/backends/nnadapter/nnadapter/include/nnadapter/nnadapter.h`



## Imagination NNA编译

需要增加`NNADAPTER_WITH_IMAGINATION_NNA`和`NNADAPTER_IMAGINATION_NNA_SDK_ROOT`定义，在命令行中需要定义`--nnadapter_with_imagination_nna=ON`和`--nnadapter_imagination_nna_sdk_root=your_imgdnn_root/imagination_nna_sdk`，例如

```bash
$ ./lite/tools/build_linux.sh --with_extra=ON \
	--with_log=ON \
	--with_nnadapter=ON \
	--nnadapter_with_imagination_nna=ON \
	--nnadapter_imagination_nna_sdk_root=your_imgdnn_root/imagination_nna_sdk \
	full_publish
```

Imagination NNA库可以从Paddle-Lite官网下载，如下

```bash
$ wget --no-check-certificate \
	https://paddlelite-demo.bj.bcebos.com/devices/imagination/imagination_nna_sdk.tar.gz
$ tar xvzf imagination_nna_sdk.tar.gz
$ cd imagination_nna_sdk
$ tree
imagination_nna_sdk
├─include
│  └─imgdnn
│          imgdnn.h
│
├─lib
│      libcrypto.so.1.1
│      libimgcustom.so
│      libimgdnn.so
│      libnnasession.so
│      libssl.so
│      libssl.so.1.1
│      libz.so.1
│      libz.so.1.2.11
│
└─nna-tools
    └─config
            mapconfig_q8a.json
            mirage_hw_config06_23_2_6500_301.json
```

在`imagination_nna\CMakeLists.txt`中，定义了它的设备名`set(DEVICE_NAME imagination_nna)`,这个设备名被使用在

- 生成的`Imagination NNA`设备的动态库`libimagination_nna.so`，详细参考：`imagination_nna\CMakeLists.txt`

- 生成的`Imagination NNA`设备的注册信息，通过`libimagination_nna.so`暴露给使用者，代码如下所示

  ```c++
  NNADAPTER_EXPORT nnadapter::driver::Device NNADAPTER_AS_SYM2(DEVICE_NAME) = {
      .name = NNADAPTER_AS_STR2(DEVICE_NAME),
      .vendor = "Imagination",
      .type = NNADAPTER_ACCELERATOR,
      .version = 1,
      .open_device = nnadapter::imagination_nna::OpenDevice,
      .close_device = nnadapter::imagination_nna::CloseDevice,
      .create_context = nnadapter::imagination_nna::CreateContext,
      .destroy_context = nnadapter::imagination_nna::DestroyContext,
      .validate_program = 0,
      .create_program = nnadapter::imagination_nna::CreateProgram,
      .destroy_program = nnadapter::imagination_nna::DestroyProgram,
      .execute_program = nnadapter::imagination_nna::ExecuteProgram,
  };
  ```

  其中：

  - `NNADAPTER_AS_SYM2(DEVICE_NAME)`被解析成`__nnadapter_device__imagination_nna`，更多详细内容请参考：`lite/backends/nnadapter/nnadapter/include/nnadapter/utility/micros.h`
  - `NNADAPTER_AS_STR2(DEVICE_NAME)`被解析成`imagination_nna`
  - NNADAPTER_ACCELERATOR为神经网络加速器，在`lite/backends/nnadapter/nnadapter/include/nnadapter/nnadapter.h`定义了3种设备
    - NNADAPTER_CPU 表示CPU
    - NNADAPTER_GPU 表示GPU
    - NNADAPTER_ACCELERATOR 表示神经网络加速器，用于nnadapter设备

- `libimgdnn.so`是IMGDNN的动态库

- `nnadapter::driver::Device`定义，请参考：`lite/backends/nnadapter/nnadapter/include/nnadapter/driver/device.h`

## Imagination NNA设备挂接

Paddle-Lite对nnadapter设备的注册分为2种

- `builtin_device`，它是默认的CPU设备（实现nnadapter，一般不用），如下

  ```c++
  #define BUILTIN_DEVICE_NAME builtin_device
  nnadapter::driver::Device NNADAPTER_AS_SYM2(BUILTIN_DEVICE_NAME) = {
      .name = NNADAPTER_AS_STR2(BUILTIN_DEVICE_NAME),
      .vendor = "Paddle",
      .type = NNADAPTER_CPU,
      .version = 1,
      .open_device = nnadapter::builtin_device::OpenDevice,
      .close_device = nnadapter::builtin_device::CloseDevice,
      .create_context = nnadapter::builtin_device::CreateContext,
      .destroy_context = nnadapter::builtin_device::DestroyContext,
      .validate_program = nnadapter::builtin_device::ValidateProgram,
      .create_program = nnadapter::builtin_device::CreateProgram,
      .destroy_program = nnadapter::builtin_device::DestroyProgram,
      .execute_program = nnadapter::builtin_device::ExecuteProgram,
  };
  ```

  **source**：`lite/backends/nnadapter/nnadapter/src/runtime/device.cc`

- `imagination_nna`-like设备，通过设备管理器的Find功能来装载。

  ```c++
  std::pair<void*, driver::Device*>* DeviceManager::Find(const char* name) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < devices_.size(); i++) {
      auto device = devices_[i].second;
      if (strcmp(device->second->name, name) == 0) {
        return device;
      }
    }
    void* library = nullptr;
    driver::Device* driver = nullptr;
    if (strcmp(name, NNADAPTER_AS_STR2(BUILTIN_DEVICE_NAME)) == 0) {
      driver = &::NNADAPTER_AS_SYM2(BUILTIN_DEVICE_NAME);
    } else {
      // Load if the driver of target device is not registered.
      std::string symbol = std::string("__nnadapter_device__") + name;
      std::string path = std::string("lib") + name + std::string(".so");
      library = dlopen(path.c_str(), RTLD_NOW);
      if (!library) {
        NNADAPTER_LOG(FATAL)
            << "Failed to load the nnadapter device HAL library for '" << name
            << "' from " << path << ", " << dlerror();
        return nullptr;
      }
      driver = reinterpret_cast<driver::Device*>(dlsym(library, symbol.c_str()));
      if (!driver) {
        dlclose(library);
        NNADAPTER_LOG(ERROR) << "Failed to find the symbol '" << symbol
                             << "' from " << path << ", " << dlerror();
        return nullptr;
      }
    }
    void* handle = nullptr;
    int result = driver->open_device(&handle);
    if (result != NNADAPTER_NO_ERROR) {
      NNADAPTER_LOG(ERROR) << "Failed to open device '" << name
                           << "', result=" << result;
      return nullptr;
    }
    auto device = new std::pair<void*, driver::Device*>(handle, driver);
    NNADAPTER_CHECK(device) << "Failed to allocate for device '" << name
                            << "', out of memory!";
    devices_.emplace_back(library, device);
    return device;
  }
  
  Device::Device(const std::string& name) {
    device_ = DeviceManager::get().Find(name.c_str());
  }
  ```

  **source**：`lite/backends/nnadapter/nnadapter/src/runtime/device.cc`

  吐槽(题外话)

  - 代码写的比较奇怪，因为既然是C++代码，为啥还要不断的用指针和`strcmp`等方式来处理？

  - 内存管理方面有些瑕疵，比如说

    ```c++
    DeviceManager::~DeviceManager() {
      std::lock_guard<std::mutex> lock(mutex_);
      for (size_t i = 0; i < devices_.size(); i++) {
        auto device = devices_[i].second;
        auto handle = device->first;
        auto driver = device->second;
        if (device) { <=== 这里是什么逻辑？为啥不是handle?
          driver->close_device(handle);
        }
        void* library = devices_[i].first;
        if (library) {
          dlclose(library);
        }
        delete device;
      }
      devices_.clear();
    }
    ```

    OK，这个不是重点，重点是代码的上下文实现逻辑。

颖脉NNA使用了Imagination NNA方案，如下图1展示了它的调用关系图，其他方案以此类推。

```mermaid
sequenceDiagram
	participant a as users
	participant b as Paddlelite
	participant c as NNAdapter Runtime
	participant d as NNAdapter HAL
	participant e as Imagination NNA
	
	a ->> b: call APIs of libpaddle_full_api_shared.so/libpaddle_light_api_shared.so
	activate a
	activate b
	b ->> c: call APIs of libnnadapter.so
	activate c
	c ->> d: call APIs of libimagination_nna.so
	activate d
	d ->> e: call APIs of libimgdnn.so
	activate e
	e ->> e: do somethings
	e ->> d: return result1
	deactivate e
	d ->> d: do somethings
	d ->> c: return result2
	deactivate d
	c ->> c: do somethings
	c ->> b: return result3
	deactivate c
	b ->> b: do somethings
	b ->> a: return result4
	deactivate b
	deactivate a
```

​                                                                                                         图1 Imagination NNA调用关系图

![nnadapter调用关系图](images/paddle_lite_and_nnadapter_dynamic_shared_library.jpg)

​                                                                                                          图2 NNAdapter调用关系图



### Imagination NNA Relationships

```mermaid
classDiagram
	%% driver: nnadapter/include/nnadapter/driver/device.h
	class nnadapter_driver_device {
		+name
		+vendor
		+type
		+version
		+open_device()
		+close_device()
		+create_context()
		+destroy_context()
		+validate_program()
		+create_program()
		+destroy_program()
		+execute_program()
	}
	nnadapter_driver_device ..> Model
	nnadapter_driver_device ..> Cache
	nnadapter_driver_device ..> Argument
	nnadapter_driver_device ..> Device
	nnadapter_driver_device ..> Context
	nnadapter_driver_device ..> Program
	%% engine: nnadapter/src/driver/imagination_nna/engine.h
	Program ..> Model
	Program ..> Cache
	Program ..> Argument
	Program --* Operand
	Program --* Context
	Program --* IMG_Classes
	Context --* Device
    Device --* IMGDevice
	%% converter: nnadapter/src/driver/imagination_nna/converter/converter.h
	Converter --* ImgdnnManager
	Converter --* IMG_CLasses
	Converter ..> Model
	Converter ..> Operation
	Converter ..> Operand
```

## Imagination NNA Sequence Diagram

```mermaid
sequenceDiagram
	participant engine as nnadapter engine
	participant runtime
	participant driver
	participant imagination_nna
	
	rect rgb(191, 223, 255)
	Note left of engine: Construct Engine
	engine ->> runtime: call NNAdapterDevice_acquire(name, ...)
	activate engine
	activate runtime
	runtime ->> runtime: Find(name)
	runtime ->> driver: call open_device
	activate driver
	driver ->> imagination_nna: OpenDevice
	imagination_nna ->> imagination_nna: new Device()
	imagination_nna ->> driver: return imagination_device object
	driver ->> runtime: return driver_device object
	deactivate driver
	runtime ->> runtime: check IsValid()
	runtime ->> engine: return nnadatper_device object
	deactivate runtime
	
	engine ->> runtime: call NNAdapterDevice_getName
	activate runtime
	runtime ->> runtime: check IsValid()
	runtime ->> engine: return nnadapter_device_name
	deactivate runtime
	
	engine ->> runtime: call NNAdapterDevice_getVendor
	activate runtime
	runtime ->> runtime: check IsValid()
	runtime ->> engine: return nnadapter_device_vendor
	deactivate runtime

	engine ->> runtime: call NNAdapterDevice_getType
	Note right of runtime: NNADAPTER_ACCELERATOR type for nnadapter
	activate runtime
	runtime ->> runtime: check IsValid()
	runtime ->> engine: return nnadapter_device_type
	deactivate runtime

	engine ->> runtime: call NNAdapterDevice_getVersion
	activate runtime
	runtime ->> runtime: check IsValid()
	runtime ->> engine: return nnadapter_device_version
	deactivate runtime
	
	engine ->> runtime: call NNAdapterContext_create(...)
	activate runtime
	runtime ->> driver: call CreateContext(...)
	activate driver
	driver ->> imagination_nna: new Context(...)
	activate imagination_nna
	imagination_nna ->> driver: return imagination_nna_context object
	deactivate imagination_nna
	driver ->> runtime: return runtime_context object
	deactivate driver
	runtime ->> engine: return nnadapter_context object
	deactivate runtime
	end
	deactivate engine
    
	rect rgb(200, 150, 255)
	Note left of engine: Run Engine
	engine ->> runtime: Run()
	activate engine
	runtime ->> runtime: Execute()
	runtime ->> runtime: NNAdapterExecution_compute(...)
	runtime ->> driver: call ExecuteProgram(...)
	activate driver
	driver ->> imagination_nna: imagination_nna_execute()
	activate imagination_nna
	imagination_nna ->> driver: return NNADAPTER_NO_ERROR
	deactivate imagination_nna
	driver ->> runtime: return NNADAPTER_NO_ERROR
	deactivate driver
	runtime ->> engine: return true
	deactivate engine
	end
```


## 参考

- [Github Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite)
- [Imagination NNA](https://github.com/SNSerHello/Paddle-Lite/blob/develop/docs/demo_guides/imagination_nna.md)
- [NNAdapter：飞桨推理 AI 硬件统一适配框架](https://paddle-lite.readthedocs.io/zh/develop/develop_guides/nnadapter.html)

