#2) EfficientDet_Encoder
 #2.1) using efficientnet_b0 
 #2.2) stages to make c series(1 ~ 5)
  #2.2.1) input : (512, 512, 3)
  #2.2.2) c1 : (256, 256, 16) after stage1
  #2.2.3) c2 : (128, 128, 24) after stage2
  #2.2.4) c3 : (64, 64, 40) after stage3
  #2.2.5) c4 : (32, 32, 112) after stage4
  #2.2.6) c5 : (16, 16, 320) after stage5
 #2.3) make p series(1 ~ 7) using convolution
  #2.3.1) conv_to_p3
  #2.3.2) conv_to_p4
  #2.3.3) conv_to_p5
  #2.3.4) conv_to_p6
  #2.3.5) conv_to_p7

import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class efficientnet(nn.Module):
  def __init__(self):
    super(efficientnet, self).__init__()

    #2.1) using efficientnet_b0 
    self.efficientnet_b = EfficientNet.from_pretrained('efficientnet-b3')

    #2.2) stages to make c series(1 ~ 5)
    self.stage1 = nn.Sequential(self.efficientnet_b._conv_stem,
                                self.efficientnet_b._bn0,
                                self.efficientnet_b._blocks[0],
                                self.efficientnet_b._blocks[1])
    
    self.stage2 = nn.Sequential(self.efficientnet_b._blocks[2],
                                self.efficientnet_b._blocks[3],
                                self.efficientnet_b._blocks[4])
    
    self.stage3 = nn.Sequential(self.efficientnet_b._blocks[5],
                                self.efficientnet_b._blocks[6],
                                self.efficientnet_b._blocks[7])
    
    self.stage4 = nn.Sequential(self.efficientnet_b._blocks[8],
                                self.efficientnet_b._blocks[9],
                                self.efficientnet_b._blocks[10],
                                self.efficientnet_b._blocks[11],
                                self.efficientnet_b._blocks[12],
                                self.efficientnet_b._blocks[13],
                                self.efficientnet_b._blocks[14],
                                self.efficientnet_b._blocks[15],
                                self.efficientnet_b._blocks[16],
                                self.efficientnet_b._blocks[17])
    
    self.stage5 = nn.Sequential(self.efficientnet_b._blocks[18],
                                self.efficientnet_b._blocks[19],
                                self.efficientnet_b._blocks[20],
                                self.efficientnet_b._blocks[21],
                                self.efficientnet_b._blocks[22],
                                self.efficientnet_b._blocks[23],
                                self.efficientnet_b._blocks[24])
    

    #2.3) make p series(3 ~ 7) using convolution
    self.feature_size = 160
    self.c3_size = 48
    self.c4_size = 136
    self.c5_size = 384
    self.p6_size = 160

    self.p3_kernel_size = 1
    self.p3_stride = 1
    self.p3_padding = 0
    self.conv_to_p3 = nn.Conv2d(in_channels = self.c3_size, 
                                out_channels = self.feature_size, 
                                kernel_size = self.p3_kernel_size, 
                                stride = self.p3_stride, 
                                padding = self.p3_padding)

    self.p4_kernel_size = 1
    self.p4_stride = 1
    self.p4_padding = 0
    self.conv_to_p4 = nn.Conv2d(in_channels = self.c4_size, 
                                out_channels = self.feature_size, 
                                kernel_size = self.p4_kernel_size, 
                                stride = self.p4_stride, 
                                padding = self.p4_padding)

    self.p5_kernel_size = 1
    self.p5_stride = 1
    self.p5_padding = 0
    self.conv_to_p5 = nn.Conv2d(in_channels = self.c5_size, 
                                out_channels = self.feature_size, 
                                kernel_size = self.p5_kernel_size, 
                                stride = self.p5_stride, 
                                padding = self.p5_padding)

    self.p6_kernel_size = 3
    self.p6_stride = 2
    self.p6_padding = 1
    self.conv_to_p6 = nn.Conv2d(in_channels = self.c5_size, 
                                out_channels = self.feature_size, 
                                kernel_size = self.p6_kernel_size, 
                                stride = self.p6_stride, 
                                padding = self.p6_padding)

    self.p7_kernel_size = 3
    self.p7_stride = 2
    self.p7_padding = 1
    self.c_out = 160
    self.momentum = 0.9997
    self.epsilon = 4e-5
    self.conv_to_p7 = nn.Conv2d(in_channels = self.p6_size, 
                                out_channels = self.feature_size, 
                                kernel_size = self.p7_kernel_size, 
                                stride = self.p7_stride, 
                                padding = self.p7_padding)
    
    #2.4) activation & batch_normalization
    self.activation = nn.ReLU(inplace = True)
    self.batch_normalization = nn.BatchNorm2d(self.c_out, self.momentum, self.epsilon)

  def forward(self, image):
    
    #2.2.2) c1 : (256, 256, 16) after stage1
    c1 = self.stage1(image)

    #2.2.3) c2 : (128, 128, 24) after stage2
    c2 = self.stage2(c1)

    #2.2.4) c3 : (64, 64, 40) after stage3
    c3 = self.stage3(c2)

    #2.2.5) c4 : (32, 32, 112) after stage4
    c4 = self.stage4(c3)

    #2.2.6) c5 : (16, 16, 192) after stage5
    c5 = self.stage5(c4)

    #2.3.1) conv_to_p3 : (64, 64, 256)
    p3 = self.conv_to_p3(c3)
    
    #2.3.2) conv_to_p4 : (32, 32, 256)
    p4 = self.conv_to_p4(c4)

    #2.3.3) conv_to_p51 : (16, 16, 256)
    p5 = self.conv_to_p5(c5)

    #2.3.4) conv_to_p6 : (8, 8, 256) 
    p6 = self.conv_to_p6(c5)

    #2.3.5) conv_to_p7 : (4, 4, 256)
    p7 = self.activation(p6)
    p7 = self.conv_to_p7(p7)
    p7 = self.batch_normalization(p7)
    p7 = self.activation(p7)

    return p3, p4, p5, p6, p7

EfficientDet_Encoder = efficientnet().to(device)
