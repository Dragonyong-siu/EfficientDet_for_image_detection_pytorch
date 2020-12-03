#3) EfficientDet_Anchor
 #3.1) feature_level : maximum_level ~ minimum_level
 #3.2) anchor_scale : {32 * 32 / stride, 64 * 64 / stride, ..., 512 * 512 / stride}
 #3.3) anchor_octave : {2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)}
 #3.4) anchor_ratio : {1:2, 1:1, 2:1}
 #3.5) stride : the unit size of each level's feature_map
 #3.6) anchor_base : anchor_base * stride * anchor_octave
 #3.7) center_base : get anchor's center
 #3.8) bbox : boundary_boxes -> (ymin, xmin, ymax, xmax)
 #3.9) anchor_bboxes 

import numpy as np
import torch

class anchor():
  def generate_anchors(self, 
                       image_shape,
                       minimum_level,
                       maximum_level,
                       anchor_scales,
                       anchor_ratios, 
                       anchor_octaves):
    
    anchor_bboxes = []

    #3.1) feature_level : maximum_level - minimum_level
    for i in range(minimum_level, maximum_level + 1):
      level_bboxes = []      

      #3.3) anchor_octave : {2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)}
      for j in range(len(anchor_octaves)):
        
        #3.4) anchor_ratio : {1:2, 1:1, 2:1}
        for k in range(len(anchor_ratios)):

          stride = int(2 ** (i))
          anchor_scale = anchor_scales[i - minimum_level]
          anchor_octave = anchor_octaves[j]
          anchor_ratio = anchor_ratios[k]
          
          #3.6) anchor_base : anchor_base * stride * anchor_octave
          anchor_base = anchor_scale * stride * anchor_octave  

          anchor_y_base = anchor_base * anchor_ratio[0]
          anchor_x_base = anchor_base * anchor_ratio[1]

          #3.7) center_base : get anchor's center
          center_y_base = np.arange(stride * 0.5, image_shape[0], stride)
          center_x_base = np.arange(stride * 0.5, image_shape[1], stride)
          
          center_base = np.meshgrid(center_x_base, center_y_base)
          center_x = center_base[0]
          center_x = center_x.reshape(-1)
          center_y = center_base[1]
          center_y = center_y.reshape(-1)
          
          #3.8) bbox : boundary_boxes -> (ymin, xmin, ymax, xmax)
          bbox = np.vstack([center_y - (anchor_y_base * 0.5),
                            center_x - (anchor_x_base * 0.5),
                            center_y + (anchor_y_base * 0.5),
                            center_x + (anchor_x_base * 0.5)])
 
          bbox = np.swapaxes(bbox, axis1 = 0, axis2 = 1) 
          level_bboxes.append(np.expand_dims(bbox, axis = 1))
        
      level_bboxes = np.concatenate(level_bboxes, axis = 1)
      level_bboxes = level_bboxes.reshape([-1, 4])
      level_bboxes = torch.Tensor(level_bboxes)
      anchor_bboxes.append(level_bboxes)
    
    #3.9) anchor_bboxes  
    return anchor_bboxes 

EfficientDet_Anchor = anchor()   
