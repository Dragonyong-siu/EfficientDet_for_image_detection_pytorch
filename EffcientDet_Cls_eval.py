#11) EfficientDet_Cls_eval
 #11.1) validate_dataloader
 #11.2) calculate the accuracy

from sklearn.metrics import accuracy_score

#11.1) validate_dataloader
anchors = EfficientDet_Anchor.generate_anchors(image_shape = [512, 512], 
                                               minimum_level = 3, 
                                               maximum_level = 7, 
                                               anchor_scales = [1.5, 0.8, 0.5, 0.3, 0.1], 
                                               anchor_ratios = [(0.9, 1.1), (1, 1), (1.1, 0.9)], 
                                               anchor_octaves = [2**(0), 2**(1/3), 2**(2/3)])
# dataset
dataset = EfficientDet_Dataset(data = data_frame,
                               image_path = images_path, 
                               image_size = 512, 
                               anchor_bbox = anchors, 
                               positive_threshold = torch.Tensor([0.3]), 
                               negative_threshold = torch.Tensor([0.3]))

validate_dataloader = DataLoader(dataset,
                                 batch_size = 1, 
                                 shuffle = True,
                                 drop_last = True)

#11) EfficientDet_Cls_eval
def evaluate_cls(dataloader, encoder, model):
  
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
      for bi, dictionary in enumerate(dataloader):
        image_tensor = dictionary['image_tensor']
        image_tensor = image_tensor.to(device)

        p_set = encoder(image_tensor)

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

        positive_indices3 = dictionary['positive_indices3']
        positive_indices4 = dictionary['positive_indices4']
        positive_indices5 = dictionary['positive_indices5']
        positive_indices6 = dictionary['positive_indices6']
        positive_indices7 = dictionary['positive_indices7']
        positive_indices_list = [positive_indices3, 
                                 positive_indices4, 
                                 positive_indices5, 
                                 positive_indices6, 
                                 positive_indices7]

        for i in range(len(p_set)):
          logits = model(p_set[i])
          logits = logits[0].cpu()
          logits = torch.greater_equal(logits, torch.Tensor([0.5])).float().view(-1)
          ## logits = torch.argmax(logits.view(-1, 2), dim = 1)
          ## logits = logits.tolist()

          cls_label = cls_label_set[i]
          cls_label = cls_label.view(-1)
          ## cls_label = cls_label.tolist()

          ## ignored_indices = ignored_indices_list[i]
          ## ignored_len = ignored_indices[0]
          ## ignored_indices = ignored_indices[1:(ignored_len + 1)]

          ## logits = np.delete(logits, ignored_indices)
          ## cls_label = np.delete(cls_label, ignored_indices)

          positive_indices = positive_indices_list[i]
          positive_len = positive_indices[0]
          positive_indices = positive_indices[1:(positive_len + 1)]
          positive_indices = torch.Tensor(positive_indices).long()

          logits = torch.index_select(logits, 
                                      0,
                                      positive_indices)
          cls_label = torch.index_select(cls_label,
                                         0,
                                         positive_indices)

          preds.extend(logits.tolist())
          labels.extend(cls_label.tolist())
        
    return preds, labels
    
#11.2) calculate the accuracy
preds, labels = evaluate_cls(validate_dataloader, EfficientDet_Encoder, EfficientDet_Model)
accuracy = accuracy_score(preds, labels)
