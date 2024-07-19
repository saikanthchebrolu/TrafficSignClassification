#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
os.chdir('A:/Traffic_Sign_Recognition')
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout


# In[9]:


data = []
labels = []
classes = 45
cur_path = os.getcwd()


# In[10]:


cur_path


# In[11]:


for i in range(classes):
    path = os.path.join(cur_path,'train',str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '\\'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(e)


# In[58]:


data = np.array(data)
labels = np.array(labels)


# In[59]:


print(len(data))


# In[60]:


print(len(labels))


# In[3]:


import numpy as np
np.save('./training/data',data)
np.save('./training/target',labels)


# In[20]:


data=np.load('./training/data.npy')
labels=np.load('./training/target.npy')


# In[21]:


print(data.shape, labels.shape)


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)


# In[23]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[16]:


print(y_test[0])


# In[24]:


y_train = to_categorical(y_train, 45)
y_test = to_categorical(y_test, 45)


# In[25]:


print(X_train[0])


# In[26]:


print(X_train.shape[1:])


# In[69]:


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
# We have 43 classes that's why we have defined 43 in the dense
model.add(Dense(45, activation='softmax'))


# In[70]:


#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[71]:


epochs = 20
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test,y_test))


# In[88]:


X_test.shape


# In[89]:


y_test.shape


# In[76]:


# accuracy 
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# In[77]:


# Loss
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[78]:


def testing(testcsv):
    y_test = pd.read_csv(testcsv)
    label = y_test["ClassId"].values
    imgs = y_test["Path"].values
    data=[]
    for img in imgs:
        image = Image.open(img)
        image = image.resize((30,30))
        data.append(np.array(image))
    X_test=np.array(data)
    return X_test,label


# In[79]:


X_test, label = testing('A:\\Traffic_Sign_Recognition\\Test.csv')


# In[80]:


Y_pred = model.predict(X_test)
Y_pred


# In[ ]:





# In[81]:


model.save("./training/TSR.h5")


# In[46]:


import os
os.chdir(r'A:\Traffic_Sign_Recognition')
from keras.models import load_model
model = load_model('./training/TSR.h5')


# In[47]:


model.save("./training/TSR1.h5")


# In[48]:


# Classes of trafic signs
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons',
            43:'Fences',
            44:'Speed limit (15km/h)'
          }


# In[49]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def test_on_img(img):
    data=[]
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
    X_test=np.array(data)
    Y_pred = model.predict(X_test)
    return image,Y_pred


# In[50]:


import matplotlib.pyplot as plt
plot,prediction = test_on_img(r'A:\Traffic_Sign_Recognition\Test\001_0011.png')
print(prediction)
s = [str(i) for i in prediction] 
print(s)
import numpy as np
# Extract numerical values from the string and convert them to floats
probabilities = [float(value) for value in s[0].strip('[]\n').split()]
# Find the index of the maximum probability
predicted_class_index = np.argmax(probabilities)
print(predicted_class_index)
print("Predicted traffic sign is: ", classes[predicted_class_index])
plt.imshow(plot)
plt.show()


# In[ ]:





# In[ ]:





# In[47]:


X_test.shape


# In[48]:


y_test.shape


# In[97]:


loss,accuracy=model.evaluate(X_test,y_test)
print(accuracy)


# In[44]:


print(Y_pred)


# In[53]:


from sklearn.metrics import confusion_matrix 
Y_pred = np.argmax(Y_pred, axis=1)

c=confusion_matrix(y_test,np.argmax(model.predict(X_test)))
print(c)


# In[ ]:





# In[ ]:




