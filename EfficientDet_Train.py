#9) EfficientDet_Train
 #9.1) p
 #9.2) cls_label
 #9.3) reg_label
 #9.4) cls_logit
 #9.5) reg_logit

from tqdm import tqdm

def EfficientDet_Train(dataloader, 
                       image_size, 
                       encoder, 
                       model, 
                       loss_class1,
                       loss_class2, 
                       optimizer,
                       scheduler):

  model.train()
  book = tqdm(dataloader, total = len(dataloader))

  total_cls_loss = 0.0
  total_reg_loss = 0.0
  for bi, dictionary in enumerate(book):

    image_tensor = dictionary['image_tensor']

    cls_label3 = dictionary['cls_label3']
    cls_label4 = dictionary['cls_label4']
    cls_label5 = dictionary['cls_label5']
    cls_label6 = dictionary['cls_label6']
    cls_label7 = dictionary['cls_label7']
    cls_label_set = [cls_label3, cls_label4, cls_label5, cls_label6, cls_label7]
    
    reg_label3 = dictionary['reg_label3']
    reg_label4 = dictionary['reg_label4']
    reg_label5 = dictionary['reg_label5']
    reg_label6 = dictionary['reg_label6']
    reg_label7 = dictionary['reg_label7']
    reg_label_set = [reg_label3, reg_label4, reg_label5, reg_label6, reg_label7]
    
    image_tensor = image_tensor.view(-1, 3, image_size, image_size)
    image_tensor = image_tensor.to(device)
    p_set = encoder(image_tensor)
    for i in range(5):

      #9.1) p
      p = p_set[i]

      #9.2) cls_label 
      cls_label = cls_label_set[i]
      cls_label = cls_label.view(-1)
      cls_label = cls_label.long()
      cls_label = cls_label.to(device)
      
      #9.3) reg_label 
      reg_label = reg_label_set[i]
      reg_label = reg_label.view(-1 ,4)
      reg_label = reg_label.to(device)

      model.zero_grad()
      
      #9.4) cls_logit
      cls_logit, reg_logit = model(p)
      cls_logit = cls_logit.view(-1, 1)

      #9.5) reg_logit 
      reg_logit = reg_logit.view(-1, 4)

      cls_loss = loss_class1(cls_logit, cls_label)
      reg_loss = loss_class2.smoothl1_loss(reg_logit, reg_label)
      ## loss = cls_loss + reg_loss

      cls_loss.backward(retain_graph = True)
      reg_loss.backward(retain_graph = True)
      
      optimizer.step()
      total_cls_loss += cls_loss.item()
      total_reg_loss += reg_loss.item()

      ## del cls_loss
      ## del reg_loss
  
  ## scheduler.step()

  average_cls_loss = total_cls_loss / len(dataloader)
  average_reg_loss = total_reg_loss / len(dataloader)
  print(' average_cls_loss: {0:.2f}'.format(average_cls_loss))
  print(' average_reg_loss: {0:.2f}'.format(average_reg_loss))
