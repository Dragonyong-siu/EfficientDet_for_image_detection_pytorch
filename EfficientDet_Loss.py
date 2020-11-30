#8) EfficientDet_loss
 #8.1) Efficient_Cls_loss
  #8.1.1) cls_loss : focal_loss for multi_classification 
   #8.1.1.1) weight : rescaling_weight given to each class
   #8.1.1.2) cross_entropy_loss : logp_t, p_t
   #8.1.1.3) focal_loss : - (1 - p_t)**(gamma) * (logp_t)
 #8.2) Efficient_Reg_loss
  #8.2.1) reg_loss : smoothl1_loss

from torch.autograd import Variable

#8.1) Efficient_Cls_loss
 #8.1.1) cls_loss : focal_loss for multi_classification 
class cls_loss(nn.modules.loss._WeightedLoss):
  def __init__(self, alpha, gamma):
    super(cls_loss, self).__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.weight = torch.Tensor([0, 0.05, 0.19, 0.19, 0.19, 0.19, 0.19]).to(device)

  def forward(self, input, target):
    
    #8.1.1.1) weight : rescaling_weight given to each class
    weight = Variable(self.weight)

    #8.1.1.2) cross_entropy_loss : logp_t, p_t
    ce_loss_function = nn.CrossEntropyLoss(weight = weight)
    logp_t = - ce_loss_function(input, target)
    p_t =  torch.exp(logp_t)
    
    #8.1.1.3) focal_loss : - alpha * (1 - p_t)**(gamma) * (logp_t)
    focal_loss = - (self.alpha) * ((1 - p_t) ** (self.gamma)) * logp_t

    return focal_loss

#8.2) Efficient_Reg_loss
 #8.2.1) reg_loss : smoothl1_loss
class reg_loss():
  def smoothl1_loss(self, input, target):

    loss_function = nn.SmoothL1Loss()
    loss = loss_function(input, target)

    return loss

EfficientDet_Cls_loss = cls_loss(alpha = 0.25, gamma = 5)
EfficientDet_Reg_loss = reg_loss()
