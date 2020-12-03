#7) EfficientDet_Model
 #7.1) classification_network
  #7.1.1) fcn_layer1 : 4 * convolution to get same feature_size
  #7.1.2) fcn_layer2 : 1 * convolution to get num_anchors * num_classes

 #7.2) regression_network
  #7.2.1) fcn_layer1 : 4 * convolution to get same feature_size
  #7.2.2) fcn_layer2 : 1 * convolution to get num_anchors * num_bbox_points

from collections import OrderedDict

class bbox_networks(nn.Module):
  def __init__(self):
    super(bbox_networks, self).__init__()
    self.num_classes = 1
    self.num_anchors = 9
    self.num_bbox_points = 4
    self.feature_size = 160
    
    #7.1.1) fcn_layer1 : 4 * convolution to get same feature_size
    #7.1.2) fcn_layer2 : 1 * convolution to get num_anchors * num_classes
    self.cls_layer = nn.Sequential(
        OrderedDict([('cls_conv1', nn.Conv2d(self.feature_size, 
                                             self.feature_size, 
                                             kernel_size = 3,
                                             padding = 1)),
                     ('relu1', nn.ReLU()),
                     ('cls_conv2', nn.Conv2d(self.feature_size, 
                                             self.feature_size, 
                                             kernel_size = 3,
                                             padding = 1)),
                     ('relu2', nn.ReLU()),
                     ('cls_conv3', nn.Conv2d(self.feature_size, 
                                             self.feature_size, 
                                             kernel_size = 3,
                                             padding = 1)),
                     ('relu3', nn.ReLU()),
                     ('cls_conv4', nn.Conv2d(self.feature_size, 
                                             self.feature_size, 
                                             kernel_size = 3,
                                             padding = 1)),
                     ('relu4', nn.ReLU()),
                     ('cls_out', nn.Conv2d(self.feature_size, 
                                           self.num_classes * self.num_anchors, 
                                           kernel_size = 3,
                                           padding = 1)),
                     ('sigmoid', nn.Sigmoid())]))
    
    #7.2.1) fcn_layer1 : 4 * convolution to get same feature_size
    #7.2.2) fcn_layer2 : 1 * convolution to get num_anchors * num_bbox_points
    self.reg_layer = nn.Sequential(
        OrderedDict([('reg_conv1', nn.Conv2d(self.feature_size, 
                                             self.feature_size, 
                                             kernel_size = 3,
                                             padding = 1)),
                     ('relu1', nn.ReLU(inplace = True)),
                     ('reg_conv2', nn.Conv2d(self.feature_size, 
                                             self.feature_size, 
                                             kernel_size = 3,
                                             padding = 1)),
                     ('relu2', nn.ReLU(inplace = True)),
                     ('reg_conv3', nn.Conv2d(self.feature_size, 
                                             self.feature_size, 
                                             kernel_size = 3,
                                             padding = 1)),
                     ('relu3', nn.ReLU(inplace = True)),
                     ('reg_conv4', nn.Conv2d(self.feature_size, 
                                             self.feature_size, 
                                             kernel_size = 3,
                                             padding = 1)),
                     ('relu4', nn.ReLU(inplace = True)),
                     ('reg_out', nn.Conv2d(self.feature_size, 
                                           self.num_bbox_points * self.num_anchors, 
                                           kernel_size = 3,
                                           padding = 1))]))

  def forward(self, feature):
    
    #7.1) classification_network
    cls_logit = self.cls_layer(feature)
    cls_logit = cls_logit.permute(0, 2, 3, 1)
    cls_logit = cls_logit.contiguous().view(feature.shape[0], 
                                            -1, 
                                            self.num_classes)
    
    #7.2) regression_network
    reg_logit = self.reg_layer(feature)
    reg_logit = reg_logit.permute(0, 2, 3, 1)
    reg_logit = reg_logit.contiguous().view(feature.shape[0], 
                                            -1,
                                            self.num_bbox_points)

    return cls_logit, reg_logit

EfficientDet_Model = bbox_networks().to(device)
