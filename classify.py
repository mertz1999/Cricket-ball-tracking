from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import pickle
import cv2
import os


# Read folders and save labels with images
folders=os.listdir('dataset/')

images=[]
labels= []
for folder in folders:
    files=os.listdir('dataset/'+folder)
    for file in files:
        img=cv2.imread('dataset/'+folder+'/'+file,0)
        img=cv2.resize(img,(20,20))
        
        images.append(img)
        labels.append(int(folder))

# extract deatures and split dataset
images = np.array(images)
features = images.reshape(len(images),-1)
x_tr,x_val,y_tr,y_val = train_test_split(features,labels, test_size=0.2, stratify=labels,random_state=32)
print("Dataset has been created!")

# Train Random forest on training data
rfc = RandomForestClassifier(max_depth=3) 
rfc.fit(x_tr,y_tr)

# Chake accuracy and other methods.
y_pred = rfc.predict(x_val)
print(classification_report(y_val,y_pred))

# Save model on .sav file with pickle tool
filename = './model/classification.sav'
pickle.dump(rfc, open(filename, 'wb'))
