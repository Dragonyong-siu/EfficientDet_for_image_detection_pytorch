#1) data_information
 #1.1) images_path : (9947 images)
 #1.2) image_labels : (row - 193736 labels), (column - image, label, left, width, top, height)
  #1.2.1) rearange image_labels by column : 'image'
 #1.3) image_names : list that have image_jpg names(9946 numbers)
 #1.4) data_frame

import pandas as pd

#1.1) images_path
images_path = '/content/gdrive/My Drive/Impact_Detection/images/'

#1.2) image_labels
 #1.2.1) rearange image_labels by column : 'image'
image_labels = pd.read_csv('/content/gdrive/My Drive/Impact_Detection/image_labels.csv')
image_labels = image_labels.sort_values(by = 'image')
image_labels = image_labels.reset_index()

#1.3) image_names
image_names = []
for i in range(len(image_labels)):
  image_names.append(image_labels['image'][i])
image_names = set(image_names)
image_names = sorted(image_names)   

#1.4) data_frame
data_frame = EfficientDet_Preprocessing.rearrange_data(data = image_labels, 
                                                       initial_j = 0,
                                                       jpg_name = image_names)
