#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


dataset = pd.read_csv("Diabities_210331_154610.csv")
dataset.head(11)


# In[3]:


dataset.info()


# In[9]:


X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]
print(X)


# In[10]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 25 , random_state = 0)


# In[11]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 8 , criterion = 'entropy' , random_state = 0)

classifier.fit(X_train , y_train)
y_pred = classifier.predict(X_test)


# In[13]:


from sklearn.metrics import accuracy_score
acc_logreg2 = round(accuracy_score(y_pred , y_test) , 2)*100
print("Accuracy : ",acc_logreg2)


# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,r2_score,classification_report
logreg = LogisticRegression(solver='lbfgs',max_iter=1000)
logreg.fit(X_train , y_train)
y_pred = logreg.predict(X_test)
acc_logreg1 = round(accuracy_score(y_pred , y_test) , 2)*100
print("Accuray : ",acc_logreg1)


# In[17]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train , y_train)
y_pred = knn.predict(X_test)
acc_knn = round(accuracy_score(y_pred,y_test),2)*100
print("Accuracy : ",acc_knn)


# In[ ]:




