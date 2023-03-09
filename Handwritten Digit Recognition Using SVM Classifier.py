#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import Libraries

import numpy as np
from sklearn.datasets import load_digits


# In[3]:


#Load Dataset

dataset = load_digits()


# In[4]:


# Summarize Dataset

print(dataset.data)
print(dataset.target)

print(dataset.data.shape)
print(dataset.target.shape)

dataimagelength = len(dataset.images)
print(dataimagelength)


# In[13]:


# Visualize the Dataset

n = 8

import matplotlib.pyplot as plt

plt.gray()
plt.matshow(dataset.images[n])
plt.show()

dataset.images[n]


# In[19]:


# Segregate Dataset into X & Y

X = dataset.images.reshape((dataimagelength,-1))
X


# In[20]:


Y = dataset.target
Y


# In[23]:


# Splitting Dataset into Train & Test

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
print(x_train.shape)
print(x_test.shape)


# In[24]:


# Train the Dataset

from sklearn import svm
model = svm.SVC(kernel='linear')
model.fit(x_train,y_train)


# In[28]:


# Predicting the digit from Test Data

n = int(input("Enter any number:"))

result = model.predict(dataset.images[n].reshape((1,-1)))
plt.imshow(dataset.images[n], cmap=plt.cm.gray_r, interpolation='nearest')
print(result)
print("\n")
plt.axis('off')
plt.title('%i'%result)
plt.show()


# In[29]:


# Prediction for all Test Data

y_pred = model.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# In[30]:


# Accuracy Score

from sklearn.metrics import accuracy_score
print(f"Accuracy :{accuracy_score(y_test,y_pred)*100}")


# In[44]:


# Accuracy Score from different Kernels

from sklearn import svm
model1 = svm.SVC(kernel='linear')
model2 = svm.SVC(kernel='rbf')
model3 = svm.SVC(gamma=0.001)
model4 = svm.SVC(gamma=0.001,C=0.78)

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

y_predModel1 = model1.predict(x_test)
y_predModel2 = model2.predict(x_test)
y_predModel3 = model3.predict(x_test)
y_predModel4 = model4.predict(x_test)

print("Accuracy of the Model 1: {0}%".format(accuracy_score(y_test, y_predModel1)*100))
print("Accuracy of the Model 2: {0}%".format(accuracy_score(y_test, y_predModel2)*100))
print("Accuracy of the Model 3: {0}%".format(accuracy_score(y_test, y_predModel3)*100))
print("Accuracy of the Model 4: {0}%".format(accuracy_score(y_test, y_predModel4)*100))


# In[ ]:




