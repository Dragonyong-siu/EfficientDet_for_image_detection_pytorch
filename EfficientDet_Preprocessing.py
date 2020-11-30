#0) EfficientDet_Preprocessing
 #0.1) preprocessing_data
  #0.1.1) prepare labels in one list per one image
   #0.1.1.1) initial_j : 0
   #0.1.1.2) dictionary
   #0.1.1.3) dataframe

class preprocess_data():
  def rearrange_data(self, data, initial_j, jpg_name):
    
    #0.1.1.1) initial_j : 0
    j = initial_j
    
    images = []
    labels = []
    lefts = []
    widths = []
    tops = []
    heights = []
    for i in range(len(jpg_name)):
      label = []
      left = []
      width = []
      top = []
      height = []
      while jpg_name[i] == data['image'][j]:
        label.append(data['label'][j])
        left.append(data['left'][j])
        width.append(data['width'][j])
        top.append(data['top'][j])
        height.append(data['height'][j])
        j = j + 1
        if j == len(data):
          break
      images.append(jpg_name[i])
      labels.append(label)
      lefts.append(left)
      widths.append(width)
      tops.append(top)
      heights.append(height)

    #0.1.1.2) dictionary
    dictionary = {}
    dictionary['image'] = images
    dictionary['label'] = labels
    dictionary['left'] = lefts
    dictionary['width'] = widths
    dictionary['top'] = tops
    dictionary['height'] = heights
    
    #0.1.1.3) dataframe
    dataframe = pd.DataFrame(dictionary)

    return dataframe

EfficientDet_Preprocessing = preprocess_data()
