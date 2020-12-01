#4) EfficientDet_Dataset
 #4.1) resized_image : (512, 512, 3)
 #4.2) feature_map : p3, p4, p5, p6, p7
 #4.3) helmet_label : helmet, helmet_blurred, helmet_partial, helmet_difficult, helmet_sideline
 #4.4) gt_box
  #4.4.1) left
  #4.4.2) right
  #4.4.3) top
  #4.4.4) bottom

 #4.5) compute_iou
 #4.6) labeling using threshold (by compute_iou)
 #4.7) total_indices : positive_indices + negative_indices + ignored_indices
 #4.8) positive_case
  #4.8.1) positive_colume : all cd_boxes' ious over one gt_box
   #4.8.2.1) there is a case that one box has 2 gt_boxes
  #4.8.2) positive_indices
  #4.8.3) total_indices : remove positive_ones
  #4.8.4) positive_box
  #4.8.5) positive_cls_label
  #4.8.6) positive_reg_label
 #4.9) negative_case
  #4.9.1) negative_column : all cd_boxes' ious over one gt_box
  #4.9.2) negative_indices
  #4.9.3) total_indices : remove positive_ones
  #4.9.4) negative_box
  #4.9.5) negative_cls_label
  #4.9.6) negative_reg_label
 #4.10) ignored_case
  #4.10.1) ignored_indices = total_indices - (positive_indices + negative_indices)
  #4.10.2) ignored_box
  #4.10.3) ignored_cls_label
  #4.10.4) ignored_reg_label
 #4.11) cd_box
 #4.12) cls_label
 #4.13) reg_label
 #4.14) remove_element
  
import PIL
import torchvision

class EfficientDet_Dataset(torch.utils.data.Dataset):
  def __init__(self, 
               data, 
               encoder, 
               image_path, 
               image_size, 
               anchor_bbox,
               positive_threshold,
               negative_threshold):
    self.data = data
    self.encoder = encoder
    self.image_path = image_path
    self.image_size = image_size
    self.anchor_bbox = anchor_bbox
    self.positive_threshold = positive_threshold
    self.negative_threshold = negative_threshold

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    
    # pil_image
    # original_size
    pil_image = PIL.Image.open(self.image_path + self.data['image'][index])
    original_size = np.array(pil_image).shape
    original_height = original_size[0]
    original_width = original_size[1]
    
    # transform_resize
    #4.1) resized_image : (512, 512, 3)
    transform_resize = torchvision.transforms.Resize([self.image_size, self.image_size])
    resized_image = transform_resize(pil_image)

    # image_array
    # image_tensor
    image_array = np.array(resized_image)
    image_copy = image_array.copy()
    image_tensor = torch.Tensor(image_copy)
    image_tensor = image_tensor.contiguous().view(-1, 3, self.image_size, self.image_size)
    image_tensor = image_tensor.to(device)

    #4.2) feature_map : p3, p4, p5, p6, p7
    feature_map = self.encoder(image_tensor)
    p3 = feature_map[0].squeeze(0)
    p4 = feature_map[1].squeeze(0)
    p5 = feature_map[2].squeeze(0)
    p6 = feature_map[3].squeeze(0)
    p7 = feature_map[4].squeeze(0)
    p_set = [p3, p4, p5, p6, p7]

    #4.3) helmet_label : helmet, helmet_blurred, helmet_partial, helmet_difficult, helmet_sideline
    helmet_label = self.data['label'][index]
    
    indices_set = [] 
    cd_box_set = []
    cls_label_set = []
    reg_label_set = []
    for k in range(len(p_set)):

      #4.4.1) left
      left = self.data['left'][index]
      left = np.array(left)
      left = left * (self.image_size / original_width)

      #4.4.2) right
      width = self.data['width'][index]
      width = np.array(width)
      width = width * (self.image_size / original_width)
      right = left + width

      #4.4.3) top
      top = self.data['top'][index]
      top = np.array(top)
      top = top * (self.image_size / original_height)

      #4.4.4) bottom
      height = self.data['height'][index]
      height = np.array(height)
      height = height * (self.image_size / original_height)
      bottom = top - height

      #4.4) gt_box
      left = left.reshape(-1, 1)
      right = right.reshape(-1, 1)
      top = top.reshape(-1, 1)
      bottom = bottom.reshape(-1, 1)
      gt_box = np.concatenate([bottom, left, top, right], axis = 1)
      gt_box = torch.Tensor(gt_box)
      gt_box = gt_box.to(device)

      #4.6) labeling using threshold (by compute_iou)
      iou_matrices = []
      gt_base = torch.zeros([self.anchor_bbox[k].shape[0], 4])
      for i in range(gt_box.shape[0]):
        gt_base[:, :] = gt_box[i]
        gt_base = gt_base.to(device)
        iou_matrix = self.compute_ious(self.anchor_bbox[k], gt_base)
        iou_matrix = iou_matrix.unsqueeze(1)
        iou_matrices.append(iou_matrix)
      iou_matrices = torch.cat(iou_matrices, dim = 1)

      #4.7) total_indices : positive_indices + negative_indices + ignored_indices
      total_indices = torch.arange(self.anchor_bbox[k].shape[0])
      total_indices = total_indices.tolist()
   
      #4.8) positive_case
      positive_indices_set = [] 
      positive_boxes = []
      positive_cls_labels = []
      positive_reg_labels = []
      positive_matrix = torch.greater_equal(iou_matrices, self.positive_threshold)
      positive_matrix = positive_matrix.float()
      for j in range(iou_matrices.shape[1]):

        #4.8.1) positive_colume : all cd_boxes' ious over one gt_box
        positive_column = positive_matrix[:, j]
        positive_column = positive_column.tolist()
        
        #4.8.2) positive_indices
        positive_indices = list(filter(lambda index: positive_column[index] >= 1, 
                                       range(len(positive_column))))
        
        #4.8.2.1) there is a case that one box has 2 gt_boxes
        if len(positive_indices_set + positive_indices) != list(set(positive_indices_set + positive_indices)):
          positive_indices_set = list(set(positive_indices_set + positive_indices))
        else: positive_indices_set += (positive_indices) 

        #4.8.3) total_indices : remove positive_ones
        total_indices = self.remove_element(total_indices, positive_indices)

        positive_indices = torch.Tensor(positive_indices)
        positive_indices = positive_indices.long()
        positive_indices = positive_indices.to(device)
        
        #4.8.4) positive_box
        positive_box = torch.index_select(input = self.anchor_bbox[k], 
                                          dim = 0, 
                                          index = positive_indices)
        
        #4.8.5) positive_cls_label
        label = self.helmet_encoding(helmet_label[j])
        positive_cls_label = torch.zeros([positive_box.shape[0]])
        positive_cls_label[:] = label
        positive_cls_label = positive_cls_label.tolist()
        
        #4.8.6) positive_reg_label
        gt_base = torch.zeros([positive_box.shape[0], 4])
        gt_base[:, :] = gt_box[j]
        gt_base = gt_base.to(device)
        positive_reg_label = self.targets_transform(positive_box, gt_base)
        positive_reg_label = positive_reg_label.tolist()

        positive_box = positive_box.tolist()
        positive_boxes += (positive_box)
        positive_cls_labels += (positive_cls_label)
        positive_reg_labels += (positive_reg_label)


      #4.9) negative_case
      negative_indices_set = [] 
      negative_boxes = []
      negative_cls_labels = []
      negative_reg_labels = []
      negative_matrix = torch.less(iou_matrices, self.negative_threshold)
      negative_matrix = negative_matrix.float()
      
      #4.9.1) negative_column : all cd_boxes' ious over one gt_box
      negative_column = negative_matrix.sum(1)
      negative_column = negative_column.tolist()
        
      #4.9.2) negative_indices
      negative_indices = list(filter(lambda index: negative_column[index] == iou_matrices.shape[1], 
                                     range(len(negative_column))))
      negative_indices_set = negative_indices 

      #4.9.3) total_indices : remove positive_ones
      total_indices = self.remove_element(total_indices, negative_indices)

      negative_indices = torch.Tensor(negative_indices) 
      negative_indices = negative_indices.long()
      negative_indices = negative_indices.to(device)
        
      #4.9.4) negative_box
      negative_box = torch.index_select(input = self.anchor_bbox[k], 
                                        dim = 0, 
                                        index = negative_indices)
        
      #4.9.5) negative_cls_label
      label = self.helmet_encoding('background')
      negative_cls_label = torch.zeros([negative_box.shape[0]])
      negative_cls_label[:] = label
      negative_cls_label = negative_cls_label.tolist()
        
      #4.9.6) negative_reg_label
      gt_base = torch.zeros([negative_box.shape[0], 4])
      gt_base[:, :] = gt_box[j]
      gt_base = gt_base.to(device)
      negative_reg_label = self.targets_transform(negative_box, gt_base)
      negative_reg_label = negative_reg_label.tolist()
        
      negative_box = negative_box.tolist()
      negative_boxes += (negative_box)
      negative_cls_labels += (negative_cls_label)
      negative_reg_labels += (negative_reg_label)
 

      #4.10) ignored_case
      ignored_indices_set = [] 
      ignored_boxes = []
      ignored_cls_labels = []
      ignored_reg_labels = []
      
      #4.10.1) ignored_indices = total_indices - (positive_indices + negative_indices)
      ignored_indices = total_indices
      ignored_indices_set = ignored_indices 
      ignored_indices = torch.Tensor(ignored_indices)
      ignored_indices = ignored_indices.long()
      ignored_indices = ignored_indices.to(device)
      
      #4.10.2) ignored_box
      ignored_box = torch.index_select(input = self.anchor_bbox[k], 
                                       dim = 0, 
                                       index = ignored_indices)
      
      #4.10.3) ignored_cls_label
      label = self.helmet_encoding('ignored')
      ignored_cls_label = torch.zeros([ignored_box.shape[0]])
      ignored_cls_label[:] = label
      ignored_cls_label = ignored_cls_label.tolist()
      
      #4.10.4) ignored_reg_label
      gt_base = torch.zeros([ignored_box.shape[0], 4])
      gt_base[:, :] = gt_box[j]
      gt_base = gt_base.to(device)
      ignored_reg_label = self.targets_transform(ignored_box, gt_base)
      ignored_reg_label = ignored_reg_label.tolist()

      ignored_box = ignored_box.tolist()
      ignored_boxes += (ignored_box)
      ignored_cls_labels += (ignored_cls_label)
      ignored_reg_labels += (ignored_reg_label)

      #4.11) cd_box
      cd_box = positive_boxes + negative_boxes + ignored_boxes
      
      #4.12) cls_label
      cls_label = positive_cls_labels + negative_cls_labels + ignored_cls_labels
      
      #4.13) reg_label
      reg_label = positive_reg_labels + negative_reg_labels + ignored_reg_labels

      #4.14) indices
      indices = positive_indices_set + negative_indices_set + ignored_indices_set
      
      sorted_tuple = sorted(list(zip(indices, list(zip(list(zip(cd_box, cls_label)), reg_label)))))
      sorted_tuple = list(zip(*sorted_tuple))[1]
      reg_label = list(zip(*sorted_tuple))[1]
      sorted_tuple = list(zip(*sorted_tuple))[0]
      cd_box = list(zip(*sorted_tuple))[0]
      cls_label = list(zip(*sorted_tuple))[1]

      cd_box = torch.Tensor(cd_box)
      cd_box = cd_box.to(device)
      cd_box_set.append(cd_box)
      
      cls_label = torch.Tensor(cls_label)
      cls_label = cls_label.to(device)
      cls_label_set.append(cls_label)
      
      reg_label = torch.Tensor(reg_label)
      reg_label = reg_label.to(device)
      reg_label_set.append(reg_label)

    dictionary = {}
    dictionary['p_set'] = p_set
    # dictionary['cd_box_set'] = cd_box_set
    dictionary['cls_label_set'] = cls_label_set
    dictionary['reg_label_set'] = reg_label_set

    return dictionary

  def compute_ious(self, cd_boxes, gt_boxes):

    #4.5) compute_ious
    cd_ymins = cd_boxes[:, 0]
    cd_xmins = cd_boxes[:, 1]
    cd_ymaxs = cd_boxes[:, 2]
    cd_xmaxs = cd_boxes[:, 3]

    gt_ymins = gt_boxes[:, 0]
    gt_xmins = gt_boxes[:, 1]
    gt_ymaxs = gt_boxes[:, 2]
    gt_xmaxs = gt_boxes[:, 3]

    x_left = torch.max(cd_xmins, gt_xmins)
    x_right = torch.min(cd_xmaxs, gt_xmaxs)
    y_bottom = torch.max(cd_ymins, gt_ymins)
    y_top = torch.min(cd_ymaxs, gt_ymaxs)

    width = torch.sub(x_right, x_left)
    height = torch.sub(y_top, y_bottom)

    zero = torch.Tensor([0])
    zero = zero.to(device)

    width = torch.max(width, zero)
    height = torch.max(height, zero)
    union = torch.mul(width, height)

    cd_width = torch.sub(cd_xmaxs, cd_xmins)
    cd_height = torch.sub(cd_ymaxs, cd_ymins)
    cd_area = torch.mul(cd_width, cd_height)

    gt_width = torch.sub(gt_xmaxs, gt_xmins)
    gt_height = torch.sub(gt_ymaxs, gt_ymins)
    gt_area = torch.mul(gt_width, gt_height)

    total_area = cd_area + gt_area - union
    ious = union / total_area

    return ious

  def helmet_encoding(self, input):

    if input == 'Helmet':
      encoded_label = torch.Tensor([0])
      encoded_label = encoded_label.long()
      encoded_label = encoded_label.to(device)

    elif input == 'Helmet-Blurred':
      encoded_label = torch.Tensor([1])
      encoded_label = encoded_label.long()
      encoded_label = encoded_label.to(device)
    
    elif input == 'Helmet-Difficult':
      encoded_label = torch.Tensor([2])
      encoded_label = encoded_label.long()
      encoded_label = encoded_label.to(device)
    
    elif input == 'Helmet-Partial':
      encoded_label = torch.Tensor([3])
      encoded_label = encoded_label.long()
      encoded_label = encoded_label.to(device)
    
    elif input == 'Helmet-Sideline':
      encoded_label = torch.Tensor([4])
      encoded_label = encoded_label.long()
      encoded_label = encoded_label.to(device)
    
    elif input == 'background':
      encoded_label = torch.Tensor([5])
      encoded_label = encoded_label.long()
      encoded_label = encoded_label.to(device)

    else: # input == 'ignored':
      encoded_label = torch.Tensor([6])
      encoded_label = encoded_label.long()
      encoded_label = encoded_label.to(device)
      
    return encoded_label

  def targets_transform(self, cd_boxes, gt_boxes):
    cx = (cd_boxes[:, 1] + cd_boxes[:, 3]) / 2
    cy = (cd_boxes[:, 0] + cd_boxes[:, 2]) / 2
    cw = cd_boxes[:, 3] - cd_boxes[:, 1]
    ch = cd_boxes[:, 2] - cd_boxes[:, 0]

    gx = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    gy = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gw = gt_boxes[:, 3] - gt_boxes[:, 1]
    gh = gt_boxes[:, 2] - gt_boxes[:, 0]
  
    tx = (gx - cx) / cw
    ty = (gy - cy) / ch
    tw = torch.log(gw / cw)
    th = torch.log(gh / ch)

    targets = torch.stack([tx, ty, tw, th], dim =  -1)

    return targets

  def remove_element(self, total_list, part_list):

    #4.14) remove_element
    for i in range(len(part_list)):
      if part_list[i] in total_list:
        total_list.remove(part_list[i])

    return total_list  
