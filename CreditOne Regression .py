#!/usr/bin/env python
# coding: utf-8

# ## Regression vs. Classification
# 
# What I will be exploring: Since I noticed a very slight correlation between Pay and BILL AMT, I want to see how to explore these using the algorithms below. Below I have renamed the variables PAY to Status and Bill Amt to BillState since it was much easier for me to understand from a high level how these variables interact with one another. 

# In[1]:


#imports
#nunpy, pandas,scipy, math, matplotlib
import numpy as np
import pandas as pd
import scipy
from math import sqrt
import matplotlib.pyplot as plt

#estimators
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import linear_model

#model metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

#helpers
get_ipython().run_line_magic('matplotlib', 'inline')

#cross validation
#initially did #from sklearn.cross_validation import train_test_split and did not work since it moved to another library
from sklearn.model_selection import train_test_split


# In[2]:


#import clean pre-processed data
#The following data does contain dummies whereas my other did not
#index_col=[0] seeks to skip over the unwanted column that df supplies
rawData = pd.read_csv('creditone_new.csv', index_col=[0])
rawData.head()


# In[3]:


df = rawData


# In[4]:


rawData.info()


# In[5]:


#Since it's so difficult to follow, I renamed columns so that things are easier for me to follow in this way
df.rename(columns={ 'Pay 0':'Status_Sept','Pay 2':'Status_Aug','Pay 3':'Status_Jul','Pay 4':'Status_Jun','Pay 5':'Status_May','Pay 6':'Status_April','BILL_AMT1':'Bill_StateSept','BILL_AMT2':'Bill_StateAug','BILL_AMT3':'Bill_StateJul','BILL_AMT4':'Bill_StateJun','BILL_AMT5':'Bill_StateMay','BILL_AMT6':'Bill_StateApril','PAY_AMT1':'Payment_Sept','PAY_AMT2':'Payment_Aug','PAY_AMT3': 'Payment_July','PAY_AMT4':'Payment_June','PAY_AMT5':'Payment_May','PAY_AMT6':'Payment_April'}, inplace=True)


# In[6]:


rawData.info()


# In[7]:


#confirm rename
rawData.head()


# In[8]:


##Correlation measures the strength of the relationship between each variable
#Correlation coefficients range between -1 and 1, numbers closer to -1 defining a strong negative corr
#Correlation coefficients closer to 1 show a strong positive correlation and numbers closer to 0 mean little to no cor
#Positive correltion between BILL_StateJuly and Bill_StateJun, Bill_StateMay, and Bill_StateApril and 
#Positive correltion between Status_Sept and Status_Aug, Status_Jul, Status_Jun, Status_May, Status_April
corrMat = rawData.corr()
print(corrMat)


# ## Selecting the Data
# 

# In[9]:


#features
#Identify how many columns exist by referencind the info function above. 
#Do not include your dependent (Limit Bal)
X = rawData.iloc[:,1:28]
print('Summary of feature sample')
X.head()


# ## Limit balance as the depedent variable
# Here I am splitting the data into 70% and testing 30% (test size=.30).

# In[10]:


#dependent variable
#will select Default Payment Next Month_default first
y = rawData['Limit Balance']


# In[11]:


#Validate correct column is being used for dependent
#dependent variable
print(y)


# ## Preparing each regression algorithm:

# In[12]:


algosClass = []
algosClass.append(('Random Forest Regressor',RandomForestRegressor()))
algosClass.append(('Linear Regression',LinearRegression()))
algosClass.append(('Support Vector Regression',SVR()))


# In[13]:


#To build and assess models we create an empty list to store the results with another
#Must hold the name of each algorithm so we can print
#regression
results = []
names = []

for name, model in algosClass:
    result = cross_val_score(model, X,y, cv=3, scoring='r2')
    names.append(name)
    results.append(result)


# In[14]:


#Output: 
for i in range(len(names)):
    print(names[i],results[i].mean())


# ## Training for Random Forest Regressor 
# 

# In[15]:


#Use model variable you est. in step 2
#pass training data as we did with test train splot

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123)


# In[16]:


#Choose the algorithm -- random forest regressor
rfr1 = RandomForestRegressor()
rfrFit1 =rfr1.fit(X_train,y_train)


# In[17]:


#Here's the predictions
#The score cannot be right. I am sure of it. Will bin credit balance. 
rfrpreds = rfrFit1.predict(X_test)
predRsquared = r2_score(y_test, rfrpreds)
rmse = sqrt(mean_squared_error(y_test,rfrpreds))
print ('R Squared: %.3f'% predRsquared)
print ('RMSE: %.3f'%rmse)


# In[18]:


plt.scatter(y_test, rfrpreds, alpha= 0.5)
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.show()


# ## Experiment on how binning affects the predictions
# 

# In[19]:


#Binning limit balance 
rawData.describe()


# In[20]:


#Discretizing credit limits
#The label parameter will ensure that the categorical data will return integers
rawData['limit_cut']=pd.qcut(rawData['Limit Balance'],q=4, labels=False)
print(rawData.groupby(['limit_cut'])['limit_cut'].count())


# In[21]:


rawData['limit_cut'].head()


# In[22]:


rawData.info()


# ## Training for Random Forest Regressor after binning
# 

# In[23]:


#dependent variable
#will select Default Payment Next Month_default first
y = rawData['Limit Balance']


# In[24]:


#Use model variable you est. in step 2
#pass training data as we did with test train splot

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123)


# In[25]:


#Choose the algorithm -- random forest regressor
rfr2 = RandomForestRegressor()
rfrFit2 =rfr2.fit(X_train,y_train)


# In[26]:


#Here's the prediction 
#It is the same as above. Why? Must change the variable to limit_cut in order for it to have an effect.
rfrpreds2 = rfrFit2.predict(X_test)
predRsquared2 = r2_score(y_test, rfrpreds2)
rmse2 = sqrt(mean_squared_error(y_test,rfrpreds2))
print ('R Squared: %.3f'% predRsquared2)
print ('RMSE: %.3f'%rmse2)


# In[27]:


plt.scatter(y_test, rfrpreds2)
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.show();


# ## Training for Linear Regression
# 

# In[28]:


#Choose the algorithm
#Linear Regression
lr1 = LinearRegression ()
lrFit1 = lr1.fit(X_train,y_train)


# In[29]:


#Provides the predictions for Linear Regression
lrpreds1 = lrFit1.predict(X_test)
predRsquared3 = r2_score(y_test, lrpreds1)
rmse3 = sqrt(mean_squared_error(y_test,lrpreds1))
print ('R Squared: %.3f'% predRsquared3)
print ('RMSE: %.3f'%rmse3)


# In[30]:


plt.scatter(y_test, lrpreds1)
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.show();


# ## Training for Support Vector Regression 

# In[31]:


#Choose the algorithm
svr1 = SVR ()
svrFit1 = svr1.fit(X_train,y_train)


# In[32]:


#Provides the predictions for SVR
svrpreds1 = svrFit1.predict(X_test)
predRsquared4 = r2_score(y_test, svrpreds1)
rmse4 = sqrt(mean_squared_error(y_test,svrpreds1))
print ('R Squared: %.3f'% predRsquared4)
print ('RMSE: %.3f'%rmse4)


# In[33]:


plt.scatter(y_test, svrpreds1)
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.show();


# ## RMSE and R Squared
# R SQUARED: It is the proportional improvement in prediction from the regression model, compared to the mean model. It indicates the goodness of fit of the model.
#  
# RMSE: Lower values of RMSE indicate better fit. RMSE is a good measure of how accurately the model predicts the response, and it is the most important criterion for fit if the main purpose of the model is prediction.
# 
# Read through this at greater depth and understand it. 
# 

# ## limit_cut as the Dependent 
# I want to use the binned limit balance as a dependent against the other variables. 
# 

# In[34]:


#Select the data
#features
#Identify how many columns exist by referencind the info function above. 
#Do not include your dependent ('limit_cut')
X = rawData.iloc[:,1:29]
print('Summary of feature sample')
X.head()


# In[35]:


#dependent variable
#Selecting the binned data
y = rawData['limit_cut']


# In[36]:


#Validate correct column is being used for dependent
#dependent variable
print(y)


# In[37]:


#Random Forest Regressor
rfr3 = RandomForestRegressor()
rfrFit3 =rfr3.fit(X_train,y_train)


# In[38]:


#Here's the prediction 
#Variable had no bearing in changing the prediction
rfrpreds3 = rfrFit3.predict(X_test)
predRsquared4 = r2_score(y_test, rfrpreds3)
rmse4 = sqrt(mean_squared_error(y_test,rfrpreds3))
print ('R Squared: %.3f'% predRsquared4)
print ('RMSE: %.3f'%rmse4)


# In[39]:


#Linear Regression
lr2 = LinearRegression ()
lrFit2 = lr2.fit(X_train,y_train)


# In[40]:


#Here's the prediction 
#No change to the overall score
lrpreds2 = lrFit2.predict(X_test)
predRsquared5 = r2_score(y_test, lrpreds2)
rmse5 = sqrt(mean_squared_error(y_test,lrpreds2))
print ('R Squared: %.3f'% predRsquared5)
print ('RMSE: %.3f'%rmse5)


# In[41]:


#Support Vector Regression
svr2 = SVR ()
svrFit2 = svr2.fit(X_train,y_train)


# In[42]:


#Provides the predictions for SVR
#No difference from limit balance before binning
svrpreds2 = svrFit2.predict(X_test)
predRsquared6 = r2_score(y_test, svrpreds2)
rmse6 = sqrt(mean_squared_error(y_test,svrpreds2))
print ('R Squared: %.3f'% predRsquared6)
print ('RMSE: %.3f'%rmse6)


# ## Default is the dependent
# 

# In[43]:


#Select the data:
X = rawData.iloc[:,0:27]
print('Summary of feature sample')
X.head()


# In[44]:


#dependent variable
#Selecting the binned data
y = rawData['Default Payment Next Month_default']


# In[45]:


#Random Forest Regressor
rfr4 = RandomForestRegressor()
rfrFit4 =rfr4.fit(X_train,y_train)

# prediction
rfrpreds4 = rfrFit4.predict(X_test)
predRsquared7 = r2_score(y_test, rfrpreds4)
rmse7 = sqrt(mean_squared_error(y_test,rfrpreds4))
print ('R Squared: %.3f'% predRsquared7)
print ('RMSE: %.3f'%rmse7)


# ## Classification Models

# In[46]:


#Importing additional model metrics 
#SKLearn Stuff
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

#helpers
get_ipython().run_line_magic('matplotlib', 'inline')


# In[47]:


algos_Class = []
algos_Class.append(('Random Forest Classifier', RandomForestClassifier()))
algos_Class.append(('Decision Tree Classifier', DecisionTreeClassifier()))


# In[48]:


results = []
names = []
for name, model in algos_Class:
    result = cross_val_score(model, X, y, cv=3, scoring='accuracy')
    names.append(name)
    results.append(result)


# In[49]:


for i in range(len(names)):
    print(names[i],result[i].mean())


# ## Using Random Forest Classifier and Decision Tree Classifier

# In[50]:


#Choose the data
#Select the data:
X = rawData.iloc[:,1:28]
print('Summary of feature sample')
X.head()


# In[51]:


#dependent variable
y = rawData['Limit Balance']


# In[52]:


#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123)


# In[53]:


#Modeling (Classification)
#max depth with 3-5 was best 
dtc1 = DecisionTreeClassifier(max_depth=5)
dtcFit1 = dtc1.fit(X_train,y_train)


# In[54]:


#Predictions
#Accuracy is awful. Will change dependent to binned data.
dtc1pred = dtcFit1.predict(X_test)
print(classification_report(y_test, dtc1pred))


# In[55]:


#Choose the data
X = rawData.iloc[:,1:28]
print('Summary of feature sample')
X.head()


# In[56]:


#dependent variable
#Selecting the binned data
y = rawData['limit_cut']


# In[57]:


#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123)


# In[58]:


#Modeling (Classification)
#max depth with 3-5 was best 
dtc2 = DecisionTreeClassifier(max_depth=5)
dtcFit2 = dtc2.fit(X_train,y_train)


# In[59]:


#Predictions
#Accuracy performed somewhat to 55% and precision at 64%
dtc2pred = dtcFit2.predict(X_test)
print(classification_report(y_test, dtc2pred))


# In[60]:


from sklearn.tree import plot_tree


# In[61]:


fig = plt.figure(figsize=(25,20))
tree = plot_tree(dtc2, feature_names=X.columns, fontsize=13)


# In[62]:


#Random Forest Classifier
#Accuracy was like the flip of a coin. Useless.
rfc1 = RandomForestClassifier(n_estimators=100)
rfcFit1 = rfc1.fit(X_train,y_train)


# In[63]:


#Predictions
#Accuracy is awful 
rfc1preds = rfcFit1.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test,rfc1preds))


# In[64]:


#Gradient Booster Classifier 
gbc1 = GradientBoostingClassifier(max_depth=5,
    n_estimators=3,
    learning_rate=1.0)
gbcFit1 = gbc1.fit(X_train,y_train)


# In[65]:


#Predictions with Gradient Booster 
#Accuracy was not okay -- like the flip of a coin. Useless.
gbc1preds = gbcFit1.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test,gbc1preds))


# ## Changing the dependent 

# In[66]:


#Choose the data
X = rawData.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]]
print('Summary of feature sample')
X.head()


# In[67]:


#dependent variable
y = rawData['Default Payment Next Month_default']


# In[68]:


#Decision Tree Classifier
dtc3 = DecisionTreeClassifier(max_depth=5)
dtcFit3 = dtc3.fit(X_train,y_train)


# In[69]:


#Predictions
#Accuracy performed somewhat to 55% and precision at 64%
dtc3pred = dtcFit3.predict(X_test)
print(classification_report(y_test, dtc3pred))


# In[70]:


#Change features 
X = rawData.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,29]]
print('Summary of feature sample')
X.head()


# In[71]:


#dependent variable
y = rawData['Default Payment Next Month_default']


# In[72]:


#Decision Tree Classifier
dtc4 = DecisionTreeClassifier(max_depth=5)
dtcFit4 = dtc4.fit(X_train,y_train)


# In[73]:


#Predictions
dtc4pred = dtcFit4.predict(X_test)
print(classification_report(y_test, dtc4pred))


# In[74]:


#Random Forest Classifier
#Accuracy was like the flip of a coin. Useless.
rfc2 = RandomForestClassifier(n_estimators=100)
rfcFit2 = rfc2.fit(X_train,y_train)


# In[75]:


#Predictions
rfc2preds = rfcFit2.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test,rfc2preds))


# In[76]:


#Gradient Booster Classifier 
gbc2 = GradientBoostingClassifier(max_depth=5,
    n_estimators=3,
    learning_rate=1.0)
gbcFit2 = gbc2.fit(X_train,y_train)


# In[77]:


#Predictions with Gradient Booster 
#Accuracy was not okay -- like the flip of a coin. Useless.
gbc2preds = gbcFit2.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test,gbc2preds))


# In[78]:


#Changing the features
X = rawData.iloc[:,[3,4,5,6,7,8]]
print('Summary of feature sample')
X.head()


# In[79]:


#dependent variable
y = rawData['Default Payment Next Month_default']


# In[80]:


#Decision Tree Classifier
dtc4 = DecisionTreeClassifier(max_depth=5)
dtcFit4 = dtc4.fit(X_train,y_train)


# In[81]:


#Predictions
#No changes
dtc4pred = dtcFit4.predict(X_test)
print(classification_report(y_test, dtc4pred))


# In[82]:


#Changing the features
X = rawData.iloc[:,[9,10,11,12,13,14]]
print('Summary of feature sample')
X.head()


# In[83]:


#Decision Tree Classifier
dtc5 = DecisionTreeClassifier(max_depth=5)
dtcFit5 = dtc5.fit(X_train,y_train)


# In[84]:


#Predictions
#No changes 
dtc5pred = dtcFit5.predict(X_test)
print(classification_report(y_test, dtc5pred))


# ## Experiment on binning age

# In[85]:


rawData.describe()


# In[86]:


#Discretizing age 
#The label parameter will ensure that the categorical data will return integers
rawData['age_cut']=pd.qcut(rawData['Age'],q=4, labels=False)
print(rawData.groupby(['age_cut'])['age_cut'].count())


# In[87]:


#Confirm it appears in the 
rawData.info()


# In[88]:


#Discretizing Bill Statement
#The label parameter will ensure that the categorical data will return integers
rawData['bill_sept']=pd.qcut(rawData['Bill_StateSept'],q=4, labels=False)
print(rawData.groupby(['bill_sept'])['bill_sept'].count())


# In[89]:


#Discretizing Bill Statement
#The label parameter will ensure that the categorical data will return integers
rawData['bill_aug']=pd.qcut(rawData['Bill_StateAug'],q=4, labels=False)
print(rawData.groupby(['bill_aug'])['bill_aug'].count())


# In[90]:


#Discretizing Bill Statement
#The label parameter will ensure that the categorical data will return integers
rawData['bill_jul']=pd.qcut(rawData['Bill_StateJul'],q=4, labels=False)
print(rawData.groupby(['bill_jul'])['bill_jul'].count())


# In[91]:


#Discretizing Bill Statement
#The label parameter will ensure that the categorical data will return integers
rawData['bill_jun']=pd.qcut(rawData['Bill_StateJun'],q=4, labels=False)
print(rawData.groupby(['bill_jun'])['bill_jun'].count())


# In[92]:


#Discretizing Bill Statement
#The label parameter will ensure that the categorical data will return integers
rawData['bill_may']=pd.qcut(rawData['Bill_StateMay'],q=4, labels=False)
print(rawData.groupby(['bill_may'])['bill_may'].count())


# In[93]:


#Discretizing Bill Statement
#The label parameter will ensure that the categorical data will return integers
rawData['bill_april']=pd.qcut(rawData['Bill_StateApril'],q=4, labels=False)
print(rawData.groupby(['bill_april'])['bill_april'].count())


# In[94]:


#Changing the features
X = rawData.iloc[:,[1,2,3,4,5,6,7,8,15,16,17,18,19,20,21,22,23,24,25,26,29,30,31,32,33,34,36,36]]
print('Summary of feature sample')
X.head()


# In[95]:


#dependent variable
y = rawData['Default Payment Next Month_default']


# In[100]:


#Decision Tree Classifier
dtc6 = DecisionTreeClassifier(max_depth=5)
dtcFit6 = dtc6.fit(X_train,y_train)


# In[101]:


#Predictions
#No changes 
dtc6pred = dtcFit6.predict(X_test)
print(classification_report(y_test, dtc6pred))


# In[138]:


#Changing the features
X = rawData.iloc[:,[3,4,5,26,29,30,31]]
print('Summary of feature sample')
X.head()


# In[139]:


#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123)


# In[140]:


#dependent variable
y = rawData['Default Payment Next Month_default']


# In[178]:


#Decision Tree Classifier
dtc7 = DecisionTreeClassifier(max_depth=3)
dtcFit7 = dtc7.fit(X_train,y_train)


# In[179]:


#Predictions
dtc7pred = dtcFit7.predict(X_test)
print(classification_report(y_test, dtc7pred))


# In[181]:


fig = plt.figure(figsize=(25,20))
tree = plot_tree(dtc7, feature_names=X.columns, fontsize=15)


# In[147]:


#Changing the features
X = rawData.iloc[:,[3,4,5,26,29,30,31,32,33,34,35,36]]
print('Summary of feature sample')
X.head()


# In[148]:


#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123)


# In[149]:


#dependent variable
y = rawData['Default Payment Next Month_default']


# In[182]:


#Decision Tree Classifier
dtc8 = DecisionTreeClassifier(max_depth=3)
dtcFit8 = dtc8.fit(X_train,y_train)


# In[183]:


#Predictions
dtc8pred = dtcFit8.predict(X_test)
print(classification_report(y_test, dtc8pred))


# In[184]:


from sklearn.tree import plot_tree


# In[185]:


fig = plt.figure(figsize=(25,20))
tree = plot_tree(dtc8, feature_names=X.columns, fontsize=13)


# In[ ]:




