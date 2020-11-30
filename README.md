 EfficientDet : Scalable and Efficient Object Detection

0) EfficientDet_Preprocessing
   
   0.1) preprocessing_data
      
       0.1.1) prepare labels in one list per one image
       
       0.1.1.1) initial_j : 0
       
       0.1.1.2) dictionary
   
       0.1.1.3) dataframe

1) data_information

   1.1) images_path : (9947 images)
   
   1.2) image_labels : (row - 193736 labels), (column - image, label, left, width, top, height)
   
       1.2.1) rearange image_labels by column : 'image'
   
   1.3) image_names : list that have image_jpg names(9946 numbers)
   
   1.4) data_frame

2) EfficientDet_Encoder
  
  2.1) using efficientnet_b0 
  
  2.2) stages to make c series(1 ~ 5)
   
       2.2.1) input : (512, 512, 3)
       
       2.2.2) c1 : (256, 256, 16) after stage1
       
       2.2.3) c2 : (128, 128, 24) after stage2
       
       2.2.4) c3 : (64, 64, 40) after stage3
       
       2.2.5) c4 : (32, 32, 112) after stage4
       
       2.2.6) c5 : (16, 16, 320) after stage5
   
   2.3) make p series(1 ~ 7) using convolution
       
       2.3.1) conv_to_p3
       
       2.3.2) conv_to_p4
       
       2.3.3) conv_to_p5
       
       2.3.4) conv_to_p6
       
       2.3.5) conv_to_p7

3) EfficientDet_Anchor
   
   3.1) feature_level : maximum_level ~ minimum_level
   
   3.2) anchor_scale : {32 * 32 / stride, 64 * 64 / stride, ..., 512 * 512 / stride}
  
   3.3) anchor_octave : {2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)}
   
   3.4) anchor_ratio : {1:2, 1:1, 2:1}
   
   3.5) stride : the unit size of each level's feature_map
   
   3.6) anchor_base : anchor_base * stride * anchor_octave
   
   3.7) center_base : get anchor's center
   
   3.8) bbox : boundary_boxes -> (ymin, xmin, ymax, xmax)
   
   3.9) anchor_bboxes 

4) EfficientDet_Dataset
   
   4.1) resized_image : (512, 512, 3)
   
   4.2) feature_map : p3, p4, p5, p6, p7
   
   4.3) helmet_label : helmet, helmet_blurred, helmet_partial, helmet_difficult, helmet_sideline
   
   4.4) gt_box
       
       4.4.1) left
       
       4.4.2) right
       
       4.4.3) top
       
       4.4.4) bottom

   4.5) compute_iou
   
   4.6) labeling using threshold (by compute_iou)
    
   4.7) total_indices : positive_indices + negative_indices + ignored_indices
   
   4.8) positive_case
      
       4.8.1) positive_colume : all cd_boxes' ious over one gt_box
       
       4.8.2.1) there is a case that one box has 2 gt_boxes
       
       4.8.2) positive_indices
       
       4.8.3) total_indices : remove positive_ones
       
       4.8.4) positive_box
       
       4.8.5) positive_cls_label
       
       4.8.6) positive_reg_label
   4.9) negative_case
       
       4.9.1) negative_column : all cd_boxes' ious over one gt_box
       
       4.9.2) negative_indices
       
       4.9.3) total_indices : remove positive_ones
       
       4.9.4) negative_box
       
       4.9.5) negative_cls_label
       
       4.9.6) negative_reg_label
 
   4.10) ignored_case
       
       4.10.1) ignored_indices = total_indices - (positive_indices + negative_indices)
       
       4.10.2) ignored_box
       
       4.10.3) ignored_cls_label
       
       4.10.4) ignored_reg_label
  
   4.11) cd_box
  
   4.12) cls_label
  
   4.13) reg_label
  
   4.14) dataloader

5) EfficientDet_Bifpn : sequential of num_layers * bifpn_layer
   
   5.1) doublewise_convolution : depthwise_separable_convolution
       
       5.1.1) depthwise_convolution
       
       5.1.2) pointwise convolution
       
       5.1.3) batch_normalization
       
       5.1.4) activation_function : relu
 
   
   5.2) bifpn_layer
       
       5.2.1) convolution to make intermediate_features
       
       5.2.2) convolution to make output_features
       
       5.2.3) initial_weights : linear_combination - w1 & w2 
       
       5.2.3.1) make value bigger& equal than zero
       
       5.2.3.2) make |weights| = 1 to do linear_combination 
 
       5.2.4) resizing_features by interpolation
       
       5.2.5) intermediate_features : p3_td, p4_td, p5_td, p6_td, p7_td
       
       5.2.6) output_features : p3_out, p4_out, p5_out, p6_out, p7_out

7) EfficientDet_Model
   
   7.1) classification_network
       
       7.1.1) fcn_layer1 : 4 * convolution to get same feature_size
       
       7.1.2) fcn_layer2 : 1 * convolution to get num_anchors * num_classes

   7.2) regression_network
      
       7.2.1) fcn_layer1 : 4 * convolution to get same feature_size
       
       7.2.2) fcn_layer2 : 1 * convolution to get num_anchors * num_bbox_points

8) EfficientDet_loss
    
   8.1) Efficient_Cls_loss
      
       8.1.1) cls_loss : focal_loss for multi_classification 
       
       8.1.1.1) weight : rescaling_weight given to each class
       
       8.1.1.2) cross_entropy_loss : logp_t, p_t
       
       8.1.1.3) focal_loss : - (1 - p_t)**(gamma) * (logp_t)
   
   8.2) Efficient_Reg_loss
       
       8.2.1) reg_loss : smoothl1_loss

9) EfficientDet_Train

10) EfficientDet_Fit
