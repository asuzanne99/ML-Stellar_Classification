#!/usr/bin/env python
# coding: utf-8

# # Final Project

# ### Part 0: Preparing the Data

# In[2]:


#having issues with numpy (and jupyter notebook) so have to use the following 3 lines for now...
import sys
sys.path.insert(0,'/opt/anaconda3/lib/python3.9/site-packages')
print(sys.path)

#import statements
import numpy as np
import pandas as pd
import math

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier


# In[10]:


#read in the dataset
path = "/Users/nalic/Documents/Georgia Tech/Computational Physics/star_classification.csv"
raw_data = pd.read_csv(path)

#raw_data

raw_data[:12]


# In[5]:


# convert string labels into numbers: galaxy=0, qso=1, star=2
def Encoder(df):
    columnsToEncode = list(df.select_dtypes(include=['category','object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding '+feature)
    return df

num_df = Encoder(raw_data)
num_df[:12]


# In[20]:


# convert into two datasets: features, X (everything excluding "class" column), and labels, y (just the "class" column)
#first convert pd dataframe to np array bc I know how to work with that better
arr = num_df.to_numpy()
type(arr)

#arr[0:3]
labels = arr[:,13:14]

features = np.delete(arr, 13, axis=1)


# ### Part 1: Logistic Regression

# In[5]:


#in this cell are my attempts to build it from scratch; I then realized there are pre-made programs for things 
#like this in libraries such as sklearn

###define the sigmoid function:
#def sigmoid(z):
#    return 1 / (1 + np.exp(-z))

###define the cost function:
#def J(theta, X, y, lamb):
#    summation = sum(((-yp*log(sigmoid(X*theta))) - ((1 .- yp)*log(1 .- sigmoid(X*theta)))))
#    J1 = (1/m) * summation
#    return J1 + (lamb/(2*m)) * (sum(theta(2:rows(theta)).^2))

###define derivative of cost function:
#def grad(theta,X,y,lamb):
#    x = [0]
#    return ((1/m) * Xp*(sigmoid(X*theta) - y)) + [x; (lamb/m)*theta(2:rows(theta))]


# In[ ]:


#define training data
#we'll select 70% of the data for this; we can just use the first .7 * 100,000 = 70,000 since they're not in any order
X,y = features[:70000], labels[:70000]

#fit the model according to the given training data.
clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='multinomial', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None).fit(X, y)

#Predict class labels for samples in X
testdata_features = features[70000:, :]
#I think the below definition is incorrect (trying to get rows 70,000 to 100,000 of the labels column)
testdata_labels = features[:, 70000:]
clf.predict(testdata_features)

#Return the mean accuracy on the given test data and labels.
clf.score(X,y)


# In[ ]:





# ### Part 2: Random Forest

# In[22]:


#fit the model according to the given training data.
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)

#Predict class labels for samples in X
print(clf.predict(testdata_features))

#Return the mean accuracy on the given test data and labels
score(testdata_features, testdata_labels, sample_weight=None)


# ### Part 3: Neural Net

# In[ ]:




