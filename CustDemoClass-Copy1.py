#!/usr/bin/env python
# coding: utf-8

# In[74]:


#DS Basics
import numpy as np
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt

#SKLearn Stuff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

#helpers
get_ipython().run_line_magic('matplotlib', 'inline')


# In[75]:


data = pd.read_csv('/Users/ma25756/Downloads/Demographic_Data.csv')


# In[76]:


data.head


# In[77]:


data.describe()


# In[78]:


data.drop_duplicates()


# In[79]:


data.drop_duplicates().shape


# In[80]:


# Select rows 0, 1, 2 (row 3 is not selected)
data[0:3]


# In[81]:


#features
X = data.iloc[:,0:4]
print('Summary of feature sample')
X.head()


# In[82]:


#dependent variable
y = data['region']


# In[83]:


#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123)


# In[84]:


#Modeling (Classification)
#max depth with 3 was best 
dtc = DecisionTreeClassifier(max_depth=5)
model = dtc.fit(X_train,y_train)


# In[85]:


#Predictions
preds = model.predict(X_test)


# In[86]:


print(classification_report(y_test, preds))


# In[87]:


from sklearn.tree import plot_tree


# In[88]:


fig = plt.figure(figsize=(25,20))
tree = plot_tree(model, feature_names=X.columns,class_names=[ 'Region 1', 'Region 2', 'Region 3', 'Region 4'], filled=True, fontsize=13)


# In[89]:


#definining  the bins - This did not really work for me since precision was 0. 
data ['age bins']= pd.cut(data['age'], bins=[0-10,20,30,40,50,60,70,80,90],labels=False)
print(data)


# In[90]:


data.hist()


# In[91]:


#features
X = data.iloc[:,0:4]
print('Summary of feature sample')
X.head()


# In[92]:


#dependent variable
y = data['age']


# In[93]:


#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123)


# In[94]:


#Modeling (Classification)
#max depth with 3 was best 
dtc = DecisionTreeClassifier(max_depth=3)
model = dtc.fit(X_train,y_train)


# In[95]:


#Predictions
preds = model.predict(X_test)


# In[96]:


print(classification_report(y_test, preds))


# In[97]:


from sklearn.tree import plot_tree


# In[98]:


fig = plt.figure(figsize=(25,20))
tree = plot_tree(model, feature_names=X.columns, filled=True, fontsize=13)


# In[99]:


data['age_bins'] = pd.cut(data['age'], bins=[4],labels=False)
print(data)


# In[100]:


data.hist()


# In[101]:


#dependent variable
y = data['age']


# In[102]:


#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123)


# In[103]:


#Modeling (Classification)
#max depth with 3 was best 
dtc = DecisionTreeClassifier(max_depth=3)
model = dtc.fit(X_train,y_train)


# In[104]:


#Predictions
preds = model.predict(X_test)


# In[105]:


#accuracy is .13 so it is is not helpful to have binned 4 ways 
print(classification_report(y_test, preds))


# In[106]:


data.hist()


# In[107]:


#dependent variable
y = data['amount']


# In[108]:


#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123)


# In[109]:


#Modeling (Classification)
#label type is continuous so model cannot be constructed
dtc = DecisionTreeClassifier(max_depth=5)
model = dtc.fit(X_train,y_train)


# In[ ]:


#Predictions
preds = model.predict(X_test)


# In[ ]:


#Classification metrics can't handle a mix of continuous and multiclass targets
print(classification_report(y_test, preds))


# In[110]:


model = DecisionTreeClassifier()


# In[111]:


print(cross_val_score(model, X, y, cv=3)) # cv = number of folds being held out


# In[ ]:




