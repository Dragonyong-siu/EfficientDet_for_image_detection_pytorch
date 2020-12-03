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
    ## self.weight = torch.Tensor([])

  def forward(self, input, target):
    
    #8.1.1.1) weight : rescaling_weight given to each class
    ## weight = Variable(self.weight)
    ## weight = weight.to(device)

    #8.1.1.2) cross_entropy_loss : logp_t, p_t
    onehot = torch.less(target, torch.Tensor([2]).to(device)).float()
    target = torch.mul(target, onehot)
    target = target.long()

    ce_loss_function = nn.CrossEntropyLoss(reduction = 'none')
    ce_loss = ce_loss_function(input, target)
    logp_t =  - torch.mul(ce_loss, onehot)
    p_t =  torch.exp(logp_t)
    
    #8.1.1.3) focal_loss : - alpha * (1 - p_t)**(gamma) * (logp_t)
    focal_loss = - (self.alpha) * ((1 - p_t) ** (self.gamma)) * logp_t
    focal_loss = torch.sum(focal_loss)
    focal_loss = focal_loss / onehot.sum()
    return focal_loss

#8.2) Efficient_Reg_loss
 #8.2.1) reg_loss : smoothl1_loss
class reg_loss():
  def smoothl1_loss(self, input, target):
    
    onehots = torch.zeros([target.shape[0], 4])
    onehots = onehots.to(device)

    onehot = torch.greater(torch.abs(target).sum(1),
                           torch.Tensor([0]).to(device))
    onehot = onehot.float()
    onehot = onehot.unsqueeze(1)
    onehots[:, :] = onehot
    
    loss_function = nn.SmoothL1Loss(reduction = 'none')
    loss = loss_function(input, target)
    loss = torch.mul(loss, onehots)

    normalizer = torch.abs(torch.mean(loss, dim = 1))
    normalizer = torch.greater(normalizer, torch.Tensor([0]).to(device))
    normalizer = normalizer.sum()
    
    loss = torch.sum(loss)
    if normalizer != 0:
      loss = loss / normalizer

    return loss

EfficientDet_Cls_loss = cls_loss(alpha = 1, gamma = 2)
EfficientDet_Reg_loss = reg_loss()

#8.2) Efficient_Reg_loss
 #8.2.1) reg_loss : smoothl1_loss
class reg_loss():
  def smoothl1_loss(self, input, target):

    loss_function = nn.SmoothL1Loss()
    loss = loss_function(input, target)

    return loss

EfficientDet_Cls_loss = cls_loss(alpha = 1, gamma = 2)
EfficientDet_Reg_loss = reg_loss()
