#!/usr/bin/env python
# coding: utf-8

# # cancer prediction using KNN 

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt #matplotlib is used for plot the graphs,
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv("iris.csv")
data.head()
data.shape
data.info()
sns.lmplot(x ='SL', y ='PL',
          fit_reg = False, hue = 'N',
          data = data)
plt.show()
data.isnull().sum()
corr=data.corr()
corr.nlargest(30,'N')['N']
x=data[['SL','SW','PL','PW','N']]
y=data[['N']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
from sklearn.metrics import accuracy_score
model = KNeighborsClassifier(n_neighbors=8)
model.fit(x_train, y_train)
predict = model.predict(x_test)
accuracy_score(predict,y_test)
print(accuracy_score)
model.score(x_train,y_train)
from sklearn.model_selection import cross_val_score
score=cross_val_score(model,x,y,cv=2)
model.fit(x_train,y_train)
predict=model.predict(x_test)
acc = model.score(x_test,y_test)

