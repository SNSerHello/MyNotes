# OpenCL

## 支持的AI网络

当前OpenCL支持22个网络，如下所示：

- MobileNetV1
- MobileNetV2
- MobileNetV3_large
- MobileNetV3_small
- EfficientNetB0
- ResNet18
- ResNet50
- VGG16
- VGG19
- SSD-MobileNetV3-large
- ch_ppocr_mobile_v2.0_cls_infer
- ch_ppocr_mobile_v2.0_det_infer
- ch_ppocr_mobile_v2.0_rec_infer
- DB
- inference_dnn
- Deeplabv3
- UNet
- bisenet
- fastscnn
- bisenet_v2
- FaceBoxes*
- MTCNN



## 支持的AI算子

当前支持86个OpenCl算子，如下所示：

- abs
- acos
- arg_max
- asin
- atan
- batch_norm
- bilinear_interp
- bilinear_interp_v2
- box_coder
- clip
- concat
- conv2d
- conv2d_transpose
- cos
- depthwise_conv2d
- depthwise_conv2d_transpose
- dropout
- elementwise_add
- elementwise_div
- elementwise_floordiv
- elementwise_max
- elementwise_min
- elementwise_mod
- elementwise_mul
- elementwise_pow
- elementwise_sub
- exp
- expand
- fc
- flatten
- flatten2
- fusion_elementwise_add_activation
- fusion_elementwise_div_activation
- fusion_elementwise_mul_activation
- fusion_elementwise_sub_activation
- gather
- gelu
- greater_than
- grid_sampler
- hard_sigmoid
- hard_swish
- instance_norm
- io_copy
- io_copy_once
- layer_norm
- layout
- layout_once
- leaky_relu
- log
- lrn
- matmul
- matmul_v2
- nearest_interp
- nearest_interp_v2
- pad2d
- pixel_shuffle
- pool2d
- prelu
- reduce_max
- reduce_mean
- relu
- relu6
- reshape
- reshape2
- rsqrt
- scale
- shape
- shuffle_channel
- sigmoid
- sin
- slice
- softmax
- split
- sqrt
- square
- squeeze
- squeeze2
- swish
- sync_batch_norm
- tan
- tanh
- transpose
- transpose2
- unsqueeze
- unsqueeze2
- yolo_box



## Paddle-Lite v2.11的算子

合计250个算子（包含融合算子），OpenCL支持86个。

| 算子名                            | 是否支持OpenCL |
| --------------------------------- | -------------- |
| abs                               | Y              |
| acos                              | Y              |
| affine_channel                    | N              |
| affine_grid                       | N              |
| anchor_generator                  | N              |
| arg_max                           | Y              |
| arg_min                           | N              |
| argsort                           | N              |
| asin                              | Y              |
| assign                            | N              |
| assign_value                      | N              |
| atan                              | Y              |
| attention_padding_mask            | N              |
| axpy                              | N              |
| batch_norm                        | Y              |
| beam_search                       | N              |
| beam_search_decode                | N              |
| bilinear_interp                   | Y              |
| bilinear_interp_v2                | Y              |
| box_clip                          | N              |
| box_coder                         | Y              |
| calib                             | N              |
| calib_once                        | N              |
| cast                              | N              |
| clip                              | Y              |
| collect_fpn_proposals             | N              |
| concat                            | Y              |
| conditional_block                 | N              |
| conv2d                            | Y              |
| conv2d_transpose                  | Y              |
| conv3d                            | N              |
| correlation                       | N              |
| cos                               | Y              |
| cos_sim                           | N              |
| crf_decoding                      | N              |
| crop                              | N              |
| crop_tensor                       | N              |
| ctc_align                         | N              |
| cumsum                            | N              |
| decode_bboxes                     | N              |
| deformable_conv                   | N              |
| density_prior_box                 | N              |
| depthwise_conv2d                  | Y              |
| depthwise_conv2d_transpose        | Y              |
| dequantize_linear                 | N              |
| distribute_fpn_proposals          | N              |
| dropout                           | Y              |
| elementwise_add                   | Y              |
| elementwise_div                   | Y              |
| elementwise_floordiv              | Y              |
| elementwise_max                   | Y              |
| elementwise_min                   | Y              |
| elementwise_mod                   | Y              |
| elementwise_mul                   | Y              |
| elementwise_pow                   | Y              |
| elementwise_sub                   | Y              |
| elu                               | N              |
| equal                             | N              |
| erf                               | N              |
| exp                               | Y              |
| expand                            | Y              |
| expand_as                         | N              |
| expand_v2                         | N              |
| fc                                | Y              |
| feed                              | N              |
| fetch                             | N              |
| fill_any_like                     | N              |
| fill_constant                     | N              |
| fill_constant_batch_size_like     | N              |
| fill_zeros_like                   | N              |
| flatten                           | Y              |
| flatten2                          | Y              |
| flatten_contiguous_range          | N              |
| flip                              | N              |
| floor                             | N              |
| fusion_elementwise_add_activation | Y              |
| fusion_elementwise_div_activation | Y              |
| fusion_elementwise_max_activation | N              |
| fusion_elementwise_min_activation | N              |
| fusion_elementwise_mul_activation | Y              |
| fusion_elementwise_pow_activation | N              |
| fusion_elementwise_sub_activation | Y              |
| gather                            | Y              |
| gather_nd                         | N              |
| gather_tree                       | N              |
| gaussian_random                   | N              |
| gelu                              | Y              |
| generate_proposals                | N              |
| generate_proposals_v2             | N              |
| greater_equal                     | N              |
| greater_than                      | Y              |
| grid_sampler                      | Y              |
| group_norm                        | N              |
| gru                               | N              |
| gru_unit                          | N              |
| hard_sigmoid                      | Y              |
| hard_swish                        | Y              |
| im2sequence                       | N              |
| increment                         | N              |
| index_select                      | N              |
| instance_norm                     | Y              |
| inverse                           | N              |
| io_copy                           | Y              |
| io_copy_once                      | Y              |
| is_empty                          | N              |
| layer_norm                        | Y              |
| layout                            | Y              |
| layout_once                       | Y              |
| leaky_relu                        | Y              |
| less_equal                        | N              |
| less_than                         | N              |
| linspace                          | N              |
| lod_array_length                  | N              |
| lod_reset                         | N              |
| log                               | Y              |
| log_softmax                       | N              |
| logical_and                       | N              |
| logical_not                       | N              |
| logical_or                        | N              |
| logical_xor                       | N              |
| lookup_table                      | N              |
| lookup_table_dequant              | N              |
| lookup_table_v2                   | N              |
| lrn                               | Y              |
| lstm                              | N              |
| match_matrix_tensor               | N              |
| matmul                            | Y              |
| matmul_v2                         | Y              |
| matrix_nms                        | N              |
| max_pool2d_with_index             | N              |
| mean                              | N              |
| merge_lod_tensor                  | N              |
| meshgrid                          | N              |
| mish                              | N              |
| mul                               | N              |
| multiclass_nms                    | N              |
| multiclass_nms2                   | N              |
| multiclass_nms3                   | N              |
| nearest_interp                    | Y              |
| nearest_interp_v2                 | Y              |
| negative                          | N              |
| norm                              | N              |
| not_equal                         | N              |
| one_hot                           | N              |
| one_hot_v2                        | N              |
| p_norm                            | N              |
| pad2d                             | Y              |
| pad3d                             | N              |
| pixel_shuffle                     | Y              |
| polygon_box_transform             | N              |
| pool2d                            | Y              |
| pow                               | N              |
| prelu                             | Y              |
| print                             | N              |
| prior_box                         | N              |
| quantize_linear                   | N              |
| range                             | N              |
| read_from_array                   | N              |
| reciprocal                        | N              |
| reduce_all                        | N              |
| reduce_any                        | N              |
| reduce_max                        | Y              |
| reduce_mean                       | Y              |
| reduce_min                        | N              |
| reduce_prod                       | N              |
| reduce_sum                        | N              |
| relu                              | Y              |
| relu6                             | Y              |
| relu_clipped                      | N              |
| reshape                           | Y              |
| reshape2                          | Y              |
| retinanet_detection_output        | N              |
| reverse                           | N              |
| rnn                               | N              |
| roi_align                         | N              |
| roi_perspective_transform         | N              |
| rsqrt                             | Y              |
| sampling_id                       | N              |
| scale                             | Y              |
| scatter                           | N              |
| scatter_nd_add                    | N              |
| search_aligned_mat_mul            | N              |
| search_attention_padding_mask     | N              |
| search_fc                         | N              |
| search_grnn                       | N              |
| search_group_padding              | N              |
| search_seq_arithmetic             | N              |
| search_seq_depadding              | N              |
| search_seq_fc                     | N              |
| search_seq_softmax                | N              |
| select_input                      | N              |
| sequence_arithmetic               | N              |
| sequence_concat                   | N              |
| sequence_conv                     | N              |
| sequence_expand                   | N              |
| sequence_expand_as                | N              |
| sequence_mask                     | N              |
| sequence_pad                      | N              |
| sequence_pool                     | N              |
| sequence_reshape                  | N              |
| sequence_reverse                  | N              |
| sequence_softmax                  | N              |
| sequence_topk_avg_pooling         | N              |
| sequence_unpad                    | N              |
| shape                             | Y              |
| shuffle_channel                   | Y              |
| sigmoid                           | Y              |
| sign                              | N              |
| sin                               | Y              |
| slice                             | Y              |
| softmax                           | Y              |
| softplus                          | N              |
| softsign                          | N              |
| sparse_conv2d                     | N              |
| split                             | Y              |
| split_lod_tensor                  | N              |
| sqrt                              | Y              |
| square                            | Y              |
| squeeze                           | Y              |
| squeeze2                          | Y              |
| stack                             | N              |
| strided_slice                     | N              |
| subgraph                          | N              |
| sum                               | N              |
| swish                             | Y              |
| sync_batch_norm                   | Y              |
| tan                               | Y              |
| tanh                              | Y              |
| tensor_array_to_tensor            | N              |
| thresholded_relu                  | N              |
| tile                              | N              |
| top_k                             | N              |
| top_k_v2                          | N              |
| transpose                         | Y              |
| transpose2                        | Y              |
| tril_triu                         | N              |
| unbind                            | N              |
| unfold                            | N              |
| uniform_random                    | N              |
| unique_with_counts                | N              |
| unsqueeze                         | Y              |
| unsqueeze2                        | Y              |
| unstack                           | N              |
| var_conv_2d                       | N              |
| where                             | N              |
| where_index                       | N              |
| while                             | N              |
| write_back                        | N              |
| write_to_array                    | N              |
| yolo_box                          | Y              |

## OpenCL Kernel组织结构

### 目录结构

`cl_common.h`是一个公用的头文件，每个`*.cl`文件中都会有`#include <cl_common.h>`

- buffer：包含12个Kernels
  - concat
  - depthwise_conv2d
  - elementwise_add
  - fc
  - im2col
  - matmul
  - pool
  - relu
  - sigmoid
  - slice
  - transpose
  - yolo_box
- image：58个Kernels
  - activation
  - argmax
  - batch_norm
  - bilinear_interp
  - box_coder
  - channel_add
  - clip
  - concat_default
  - concat
  - conv2d_1x1_default
  - conv2d_1x1_default_mali
  - conv2d_1x1_opt
  - conv2d_3x3_default
  - conv2d_3x3
  - conv2d_5x5
  - conv2d_5x5_opt
  - conv2d_7x7
  - conv2d_7x7_opt
  - conv2d_common
  - conv2d_transpose
  - conv2d_winograd_3x3s1
  - depthwise_conv2d_basic
  - depthwise_conv2d
  - depthwise_conv2d_transpose
  - dropout
  - elementwise_add
  - elementwise_broadcast
  - elementwise
  - elementwise_mul
  - elementwise_sub
  - expand
  - fc
  - gather
  - greater_than
  - grid_sampler
  - instance_norm
  - layer_norm
  - layout
  - lrn
  - matmul
  - matmul_opt
  - matmul_unpersistable_y
  - matmul_xtranspose
  - max
  - nearest_interp
  - pad2d
  - pixel_shuffle
  - pool_deprecated
  - pool
  - reduce
  - reshape
  - scale
  - shuffle_channel
  - slice
  - softmax
  - split
  - transpose
  - trigonometric

```bash
Paddle-Lite\lite\backends\opencl\cl_kernel
|   cl_common.h
|
+---buffer
|       concat_kernel.cl
|       depthwise_conv2d_kernel.cl
|       elementwise_add_kernel.cl
|       fc_kernel.cl
|       im2col_kernel.cl
|       mat_mul_kernel.cl
|       pool_kernel.cl
|       relu_kernel.cl
|       sigmoid_kernel.cl
|       slice_kernel.cl
|       transpose_kernel.cl
|       yolo_box_kernel.cl
|
\---image
        activation_kernel.cl
        argmax_kernel.cl
        batch_norm_kernel.cl
        bilinear_interp_kernel.cl
        box_coder_kernel.cl
        channel_add_kernel.cl
        clip_kernel.cl
        concat_default_kernel.cl
        concat_kernel.cl
        conv2d_1x1_default_kernel.cl
        conv2d_1x1_default_mali_kernel.cl
        conv2d_1x1_opt_kernel.cl
        conv2d_3x3_default_kernel.cl
        conv2d_3x3_kernel.cl
        conv2d_5x5_kernel.cl
        conv2d_5x5_opt_kernel.cl
        conv2d_7x7_kernel.cl
        conv2d_7x7_opt_kernel.cl
        conv2d_common_kernel.cl
        conv2d_transpose_kernel.cl
        conv2d_winograd_3x3s1_kernel.cl
        depthwise_conv2d_basic_kernel.cl
        depthwise_conv2d_kernel.cl
        depthwise_conv2d_transpose_kernel.cl
        dropout_kernel.cl
        elementwise_add_kernel.cl
        elementwise_broadcast_kernel.cl
        elementwise_kernel.cl
        elementwise_mul_kernel.cl
        elementwise_sub_kernel.cl
        expand_kernel.cl
        fc_kernel.cl
        gather_kernel.cl
        greater_than_kernel.cl
        grid_sampler_kernel.cl
        instance_norm_kernel.cl
        layer_norm_kernel.cl
        layout_kernel.cl
        lrn_kernel.cl
        matmul_kernel.cl
        matmul_opt_kernel.cl
        matmul_unpersistable_y_kernel.cl
        matmul_xtranspose_kernel.cl
        max_kernel.cl
        nearest_interp_kernel.cl
        pad2d_kernel.cl
        pixel_shuffle_kernel.cl
        pool_deprecated_kernel.cl
        pool_kernel.cl
        reduce_kernel.cl
        reshape_kernel.cl
        scale_kernel.cl
        shuffle_channel_kernel.cl
        slice_kernel.cl
        softmax_kernel.cl
        split_kernel.cl
        transpose_kernel.cl
        trigonometric_kernel.cl
```

### opencl_kernels_files查询表

在编译Paddle-Lite时，Paddle-Lite会通过`lite/tools/cmake_tools/gen_opencl_code.py`工具将上述所有支持的Kernels文件制作成一张`opencl_kernels_files`查询表，将它存在在`lite/backends/opencl/opencl_kernels_source.cc`文件中。在`CLRuntime::CreateProgramFromSource(..., file_name, ...)`运行的时候会根据文件名，比如说`file_name="image/argmax_kernel.cl"`从`opencl_kernels_files`中获取它的kernel内容来创建`cl::Program`对象，详情可以参见：`lite/backends/opencl/cl_runtime.cc`第332行`CLRuntime::CreateProgramFromSource()`。

### OpenCL Kernel编译

在编译OpenCL的Kernel时候，为提高性能，会依次使用Cache，二进制Kernel和源代码Kernel进行编译，如下图1所示

```mermaid
flowchart TB
	A[编译Kernel] --> B{Cache中是否已经存在?}
	B -->|Yes| C[编译完成]
	B -->|No| D{是否是预编译的二进制Kernel?}
	D -->|Yes| E[装载预编译二进制Kernel]
	D -->|No| I
	E --> F{检查是否是合法预编译二进制Kernel?}
	F -->|Yes| G[装载Kernel内容并创建Program对象]
	G --> H[编译Program并存放到Cache中]
	H --> C
	F -->|No| I[检查Kernel定义是否已经在opencl_kernels_files表中?]
	I -->|No| J[编译Kernel失败]
	I -->|Yes| G
```

​                                              图1 编译OpenCL Kernel算法



## 参考

- [Github Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite)
- [Paddle-Lite Models Supported](https://paddle-lite.readthedocs.io/zh/latest/quick_start/support_model_list.html)
- [Paddle-Lite Operators Supported](https://paddle-lite.readthedocs.io/zh/latest/quick_start/support_operation_list.html)