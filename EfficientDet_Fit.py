#10) EfficientDet_Fit

def EfficientDet_Fit(train_function, 
                     dataloader, 
                     size, 
                     encoder,
                     model, 
                     loss_class1, 
                     loss_class2, 
                     epoches, 
                     learning_rate):

  optimizer = torch.optim.SGD(model.parameters(),
                              lr = learning_rate,
                              momentum = 0.9,
                              weight_decay = 4e-3)
  
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 3, verbose = True)
  for i in range(epoches):
    print(f"epoches:{i+1}")
    print('train')
    train_function(dataloader, size, encoder, model, loss_class1, loss_class2, optimizer, scheduler)
  torch.save(model, '/content/gdrive/My Drive/' + f'EfficientDet_Model:{i+1}')
