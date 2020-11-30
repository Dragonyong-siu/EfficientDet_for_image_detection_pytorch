#9) EfficientDet_Train
 #9.1) p
 #9.2) cls_label
 #9.3) reg_label
 #9.4) cls_logit
 #9.5) reg_logit

from tqdm import tqdm

def EfficientDet_Train(dataloader, model, loss_class1, loss_class2, optimizer):

  model.train()
  book = tqdm(dataloader, total = len(dataloader))

  total_cls_loss = 0.0
  total_reg_loss = 0.0
  for bi, dictionary in enumerate(book):

    p_set = dictionary['p_set']
    cls_label_set = dictionary['cls_label_set']
    reg_label_set = dictionary['reg_label_set']
    for i in range(len(p_set)):

      #9.1) p
      p = p_set[i].to(device)

      #9.2) cls_label
      cls_label = cls_label_set[i]
      cls_label = cls_label.reshape(-1)
      cls_label = cls_label.long()
      cls_label = cls_label.to(device)
      
      #9.3) reg_label
      reg_label = reg_label_set[i]
      reg_label = reg_label.reshape(-1, 4)
      reg_label = reg_label.to(device)

      model.zero_grad()
      
      #9.4) cls_logit
      cls_logit, reg_logit = model(p)
      cls_logit = cls_logit.reshape(-1, 7)

      #9.5) reg_logit
      reg_logit = reg_logit.reshape(-1, 4)

      cls_loss = loss_class1(cls_logit, cls_label)
      reg_loss = loss_class2.smoothl1_loss(reg_logit, reg_label)

      cls_loss.backward(retain_graph = True)
      reg_loss.backward(retain_graph = True)
      
      optimizer.step()
      total_cls_loss += cls_loss.item()
      total_reg_loss += reg_loss.item()

  average_cls_loss = total_cls_loss / len(dataloader)
  average_reg_loss = total_reg_loss / len(dataloader)
  print(' average_cls_loss: {0:.2f}'.format(average_cls_loss))
  print(' average_reg_loss: {0:.2f}'.format(average_reg_loss))
