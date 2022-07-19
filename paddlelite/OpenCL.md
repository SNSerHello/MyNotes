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



## 参考

- [Github Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite)
- [Paddle-Lite Models Supported](https://paddle-lite.readthedocs.io/zh/latest/quick_start/support_model_list.html)
- [Paddle-Lite Operators Supported](https://paddle-lite.readthedocs.io/zh/latest/quick_start/support_operation_list.html)