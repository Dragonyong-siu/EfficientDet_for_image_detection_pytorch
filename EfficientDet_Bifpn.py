#5) EfficientDet_Bifpn : sequential of num_layers * bifpn_layer
 #5.1) doublewise_convolution : depthwise_separable_convolution
  #5.1.1) depthwise_convolution
  #5.1.2) pointwise convolution
  #5.1.3) batch_normalization
  #5.1.4) activation_function : relu
 
 #5.2) bifpn_layer
  #5.2.1) convolution to make intermediate_features
  #5.2.2) convolution to make output_features
  #5.2.3) initial_weights : linear_combination - w1 & w2 
   #5.2.3.1) make value bigger& equal than zero
   #5.2.3.2) make |weights| = 1 to do linear_combination 
 
  #5.2.4) resizing_features by interpolation
  #5.2.5) intermediate_features : p3_td, p4_td, p5_td, p6_td, p7_td
  #5.2.6) output_features : p3_out, p4_out, p5_out, p6_out, p7_out

import torch.nn.functional as F

#5.1) doublewise_convolution : depthwise_separable_convolution
class doublewise_convolution(nn.Module):
  def __init__(self, in_channel, out_channel):
    super(doublewise_convolution, self).__init__()
    self.c_in = in_channel
    self.c_out = out_channel

    self.depthwise_kernel = 1
    self.depthwise_stride = 1
    self.depthwise_padding = 0
    self.depthwise_dilation = 1
    self.depthwise_groups = 256

    self.pointwise_kernel = 1
    self.pointwise_stride = 1
    self.pointwise_padding = 0
    self.pointwise_dilation = 1
    self.pointwise_groups = 1

    self.momentum = 0.9997
    self.epsilon = 4e-5

    #5.1.1) depthwise_convolution
    self.depthwise_convolution = nn.Conv2d(in_channels = self.c_in,
                                           out_channels = self.c_out,
                                           kernel_size = self.depthwise_kernel,
                                           stride = self.depthwise_stride,
                                           padding = self.depthwise_padding,
                                           dilation = self.depthwise_dilation,
                                           groups = self.depthwise_groups,
                                           bias = False)
    
    #5.1.2) pointwise convolution
    self.pointwise_convolution = nn.Conv2d(in_channels = self.c_in,
                                           out_channels = self.c_out,
                                           kernel_size = self.pointwise_kernel,
                                           stride = self.pointwise_stride,
                                           padding = self.pointwise_padding,
                                           dilation = self.pointwise_dilation,
                                           groups = self.pointwise_groups,
                                           bias = False)
    
    #5.1.3) batch_normalization
    self.batch_normalization = nn.BatchNorm2d(num_features = self.c_out,
                                             eps = self.epsilon,
                                             momentum = self.momentum)
    
    #5.1.4) activation_function : relu
    self.activation_function = nn.ReLU()
      
  def forward(self, input):
    output = self.depthwise_convolution(input)
    output = self.pointwise_convolution(output)
    output = self.batch_normalization(output)
    output = self.activation_function(output)
    
    return output

#5.2) bifpn_layer
class bifpn_layer(nn.Module):
  def __init__(self):
    super(bifpn_layer, self).__init__()
    self.feature_size = 256
    self.epsilon = 1e-5
    
    #5.2.1) convolution to make intermediate_feature
    self.convto_p3_td = doublewise_convolution(self.feature_size, self.feature_size)
    self.convto_p4_td = doublewise_convolution(self.feature_size, self.feature_size)
    self.convto_p5_td = doublewise_convolution(self.feature_size, self.feature_size)
    self.convto_p6_td = doublewise_convolution(self.feature_size, self.feature_size)
    
    #5.2.2) convolution to make output_feature
    self.convto_p4_out = doublewise_convolution(self.feature_size, self.feature_size)
    self.convto_p5_out = doublewise_convolution(self.feature_size, self.feature_size)
    self.convto_p6_out = doublewise_convolution(self.feature_size, self.feature_size)
    self.convto_p7_out = doublewise_convolution(self.feature_size, self.feature_size)

    #5.2.3) initial_weights : linear_combination : w1 & w2
    self.w1 = nn.Parameter(torch.ones([2, 4]))
    self.w2 = nn.Parameter(torch.ones([3, 4]))

    self.activation_function = nn.ReLU()

  def forward(self, inputs):
    p3, p4, p5, p6, p7 = inputs
    
    #5.2.3.1) make value bigger& equal than zero
    w1 = self.activation_function(self.w1)
    w2 = self.activation_function(self.w2)
    
    #5.2.3.2) make |weights| = 1 to do linear_combination 
    w1 = w1 / torch.sum(w1, dim = 0) + self.epsilon
    w2 = w2 / torch.sum(w2, dim = 0) + self.epsilon

    #5.2.5) intermediate_features : p3_td, p4_td, p5_td, p6_td, p7_td
    p7_td = p7
    p6_td = self.convto_p6_td(w1[0, 0] * p6 + w1[1, 0] * self.resize_feature(p7_td, 2))
    p5_td = self.convto_p5_td(w1[0, 1] * p5 + w1[1, 1] * self.resize_feature(p6_td, 2))
    p4_td = self.convto_p4_td(w1[0, 2] * p4 + w1[1, 2] * self.resize_feature(p5_td, 2))
    p3_td = self.convto_p3_td(w1[0, 3] * p3 + w1[1, 3] * self.resize_feature(p4_td, 2))

    #5.2.6) output_features : p3_out, p4_out, p5_out, p6_out, p7_out
    p3_out = p3_td
    p4_out = self.convto_p4_out(w2[0, 0] * p4 + w2[1, 0] * p4_td + w2[2, 0] * self.resize_feature(p3_out, 0.5))
    p5_out = self.convto_p5_out(w2[0, 1] * p5 + w2[1, 1] * p5_td + w2[2, 1] * self.resize_feature(p4_out, 0.5))
    p6_out = self.convto_p6_out(w2[0, 2] * p6 + w2[1, 2] * p6_td + w2[2, 2] * self.resize_feature(p5_out, 0.5))
    p7_out = self.convto_p7_out(w2[0, 3] * p7 + w2[1, 3] * p7_td + w2[2, 3] * self.resize_feature(p6_out, 0.5))

    return [p3_out, p4_out, p5_out, p6_out, p7_out]
  
  #5.2.4) resizing_features by interpolation
  def resize_feature(self, feature, scale_factor):
    resized_feature = F.interpolate(input = feature, 
                                    scale_factor = scale_factor,
                                    recompute_scale_factor=True)
    return resized_feature

#5) EfficientDet_Bifpn : sequential of num_layers * bifpn_layer
class bifpn(nn.Module):
  def __init__(self):
    super(bifpn, self).__init__()
    self.num_layers = 3
    self.bifpn_layer = bifpn_layer()
    bifpn_layers = []
    for _ in range(self.num_layers):
      bifpn_layers.append(self.bifpn_layer)
    self.bifpn = nn.Sequential(*bifpn_layers) 

  def forward(self, inputs):
    features = self.bifpn(inputs)
    return features

EfficientDet_Bifpn = bifpn().to(device)
