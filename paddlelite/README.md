# Paddle-Lite

## Architecture

![Paddle-Lite Architecture](images/paddle_lite_with_nnadapter.jpg)

​                                                   图1 Paddle-Lite框架



## Paddlelite Inference

### LightPredictor类关系图

```mermaid
classDiagram
	LightPredictor --* Scope
	LightPredictor --* RuntimeProgram
    LightPredictor --* ProgramDesc
    LightPredictor --> TensorLite
    LightPredictor --> Variable
    LightPredictor --* PrecisionType
    
    LightPredictorImpl --|> PaddlePredictor
    LightPredictorImpl --* LightPredictor
    LightPredictorImpl --> Tensor
    LightPredictorImpl --> MobileConfig
    
    PaddlePredictor --> Tensor
    PaddlePredictor --> LiteModelType
    PaddlePredictor --* PowerMode
    
    MobileConfig --|> ConfigBase
    
    CxxConfig --|> ConfigBase
    CxxConfig --* Place
    CxxConfig --* CxxModelBuffer
    CxxConfig --* QuantType
    CxxConfig --* DataLayoutType
    
    ConfigBase --* PowerMode
    ConfigBase --* CLTuneMode
    ConfigBase --* CLPrecisionType
    
    Place --* TargetType
    Place --* PrecisionType
    Place --* DataLayoutType
```

**source**: `lite/api/light_api.h`



### Predictor类关系图

```mermaid
classDiagram
    Predictor --* Scope
    Predictor --* ProgramDesc
    Predictor --* RuntimeProgram
    Predictor --* Place
    Predictor --* PrecisionType
    Predictor --> LiteModelType
    Predictor --> CxxConfig
    Predictor --> CxxModelBuffer
    
    CxxPaddleApiImpl --|> PaddlePredictor
    CxxPaddleApiImpl --* Predictor
    CxxPaddleApiImpl --* CxxConfig
    CxxPaddleApiImpl --> Tensor
    CxxPaddleApiImpl --> LiteModelType
    
    PaddlePredictor --> Tensor
    PaddlePredictor --> LiteModelType
    PaddlePredictor --* PowerMode

	CxxConfig --|> ConfigBase
	CxxConfig --* Place
	CxxConfig --* CxxModelBuffer
	CxxConfig --* QuantType
```

**source**: `lite/api/cxx_api.h`



### Python类与C++类映射关系

| Python类        | C++类              |
| --------------- | ------------------ |
| Opt             | OptBase            |
| CxxPredictor    | CxxPaddleApiImpl   |
| LightPredictor  | LightPredictorImpl |
| CxxConfig       | CxxConfig          |
| MobileConfig    | MobileConfig       |
| PowerMode       | PowerMode          |
| CLTuneMode      | CLTuneMode         |
| CLPrecisionType | CLPrecisionType    |
| PrecisionType   | PrecisionType      |
| MLUCoreVersion  | MLUCoreVersion     |
| TargetType      | TargetType         |
| DataLayoutType  | DataLayoutType     |
| Place           | Place              |
| Tensor          | Tensor             |

**source**: `lite/api/python/pybind/pybind.cc`



### Python函数与C++函数映射关系

```c++
#ifndef LITE_ON_TINY_PUBLISH
  m->def("create_paddle_predictor",
         [](const CxxConfig &config) -> std::unique_ptr<CxxPaddleApiImpl> {
           auto x = std::unique_ptr<CxxPaddleApiImpl>(new CxxPaddleApiImpl());
           x->Init(config);
           return std::move(x);
         });
#endif
  m->def("create_paddle_predictor",
         [](const MobileConfig &config) -> std::unique_ptr<LightPredictorImpl> {
           auto x =
               std::unique_ptr<LightPredictorImpl>(new LightPredictorImpl());
           x->Init(config);
           return std::move(x);
         });
```

这个Python的实现与C/C++实现还是有所区别的，因为编写C/C++推断程序的时候，我们是通过以下模板函数来创建预测器的

```c++
template <typename ConfigT>
LITE_API std::shared_ptr<PaddlePredictor> CreatePaddlePredictor(const ConfigT&);
```



**light api中的实现**

```c++
namespace lite_api {

template <>
std::shared_ptr<PaddlePredictor> CreatePaddlePredictor(
    const MobileConfig& config) {
  auto x = std::make_shared<lite::LightPredictorImpl>();
  x->Init(config);
  return x;
}

}  // namespace lite_api
```



**cxx api中的实现**

```c++
namespace lite_api {

template <>
std::shared_ptr<PaddlePredictor> CreatePaddlePredictor(
    const CxxConfig &config) {
  static std::mutex mutex_conf;
  std::unique_lock<std::mutex> lck(mutex_conf);
  auto x = std::make_shared<lite::CxxPaddleApiImpl>();
  x->Init(config);
  return x;
}

}  // namespace lite_api
```

**区别**

- C/C++中使用了锁保护，而Python中没有
- C/C++使用了`std::shared_ptr`而Python中使用了`std::unique_ptr`
- C/C++返回的是`PaddlePredictor`，而Python中返回的是`CxxPaddleApiImpl`或者`LightPredictorImpl`。从类关系图看，`PaddlePredictor`是`CxxPaddleApiImpl`和`LightPredictorImpl`的父类，这个可能是pybind11功能限制导致的差异。



### Pybind11绑定方法

上面的类与函数对应的映射关系，通过`lite/api/python/pybind/pybind.h`中的如下方式建立关系

```
PYBIND11_MODULE(lite, m) {
  m.doc() = "C++ core of Paddle-Lite";

  BindLiteApi(&m);
#ifndef LITE_ON_TINY_PUBLISH
  BindLiteOpt(&m);
#endif
}
```



## Paddle-Lite重要组件

### [OpenCL](https://github.com/SNSerHello/MyNotes/blob/main/paddlelite/OpenCL.md)



### NNAdapter

- #### [Imagination NNA](https://github.com/SNSerHello/MyNotes/blob/main/paddlelite/Imagination%20NNA.md)



## 参考

- [Imagination NNA](https://github.com/SNSerHello/MyNotes/blob/main/paddlelite/Imagination%20NNA.md)
- [OpenCL](https://github.com/SNSerHello/MyNotes/blob/main/paddlelite/OpenCL.md)