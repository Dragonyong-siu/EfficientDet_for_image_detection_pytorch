#10) EfficientDet_Fit

def EfficientDet_Fit(train_function, 
                     dataloader, 
                     model, 
                     loss_class1, 
                     loss_class2, 
                     epoches, 
                     learning_rate):

  optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
  for i in range(epoches):
    print(f"epoches:{i+1}")
    print('train')
    train_function(dataloader, model, loss_class1, loss_class2, optimizer)
    torch.save(model, '/content/gdrive/My Drive/' + f'EfficientDet_Model:{i+1}')
