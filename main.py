import os
import cv2
from cv2 import imshow
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

fldr = 'D:\\Mood Analysis\\Images'

files=os.listdir(fldr)
# print(files)

Emotions = files
print(Emotions)

i=0
last=[]
images=[]
labels=[]

for fle in files:
    idx = Emotions.index(fle)
    label = idx

    total = fldr + '/' + fle
    files_exp = os.listdir(total)

    for fle_2 in files_exp:
        file_main = total + '/' + fle_2
        print(file_main + "   " + str(label))
        image = cv2.imread(file_main)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (48, 48))
        images.append(image)
        labels.append(label)
        i += 1
    last.append(i)

images_f=np.array(images)
labels_f=np.array(labels)
images_f_2=images_f/255
labels_encoded=tf.keras.utils.to_categorical(labels_f,num_classes=len(Emotions))
X_train, X_test, Y_train, Y_test= train_test_split(images_f_2, labels_encoded,test_size=0.25)

