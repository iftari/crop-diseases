import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
import os
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay



#loading data


images=[]
labels=[]
files=os.listdir(r'.\processed_data')
for pimg in files:
    if pimg.endswith(('.png', '.jpg', '.jpeg')):
        image=imread(f'processed_data\\{pimg}').ravel()/255
        label=int(pimg.split('.')[0][-1])
        images.append(image)
        labels.append(label) 

print("Images and labels loaded successfully!") 
images_new = np.array([image for image in images])
Labels = np.array(labels)

#Model

from joblib import load
model=load('svmClassifier')

ypred=model.predict(images_new)
cm = confusion_matrix(Labels, ypred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()