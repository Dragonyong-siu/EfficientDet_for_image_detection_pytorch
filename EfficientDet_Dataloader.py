#4) EfficientDet_Dataset
 #4.14) train_dataloader

from torch.utils.data import DataLoader

# anchors
anchors = EfficientDet_Anchor.generate_anchors(image_shape = [512, 512], 
                                               minimum_level = 3, 
                                               maximum_level = 7, 
                                               anchor_scales = [1.5, 1.4, 0.6, 0.5, 0.5], 
                                               anchor_ratios = [(0.7, 1.4), (1, 1), (1.4, 0.7)], 
                                               anchor_octaves = [2**(0), 2**(1/3), 2**(2/3)])
# dataset
dataset = EfficientDet_Dataset(data = data_frame,
                               encoder = EfficientDet_Encoder,
                               image_path = images_path, 
                               image_size = 512, 
                               anchor_bbox = anchors, 
                               positive_threshold = torch.Tensor([0.4]).to(device), 
                               negative_threshold = torch.Tensor([0.3]).to(device))

train_dataloader = DataLoader(dataset,
                              batch_size = 1, 
                              shuffle = True,
                              drop_last = True)
