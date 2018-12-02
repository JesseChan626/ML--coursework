#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV


# In[2]:


housingData = pd.read_csv('./housing.csv')
display(housingData)


# In[3]:


sum(housingData.total_bedrooms.isna())#缺少值的表现


# In[4]:


# Get the non-Nan indices
notna = housingData.total_bedrooms.notna()


# In[23]:


#Get the train data of output

outputData = housingData.total_bedrooms.values[notna].reshape(-1,1)
total_rooms = housingData.total_rooms.values[notna].reshape(-1,1) #84.10%
households =housingData.households.values[notna].reshape(-1,1)  #96.05%
population = housingData.population.values[notna].reshape(-1,1) 

#inputData = np.c_[households,total_rooms]#96.92%
#inputData = np.c_[households,total_rooms,population]#95.49%
inputData = np.c_[total_rooms,households]
#inputData = households
#inputData = total_rooms

print(inputData.shape)
print(outputData.shape)

#outputData


# In[24]:


#4:1
X_train, X_test, y_train, y_test = train_test_split(inputData, outputData, test_size=0.2, random_state=50)

# Success
print("训练集与测试集拆分成功，训练集有{}条，测试集有{}条。".format(X_train.shape[0], X_test.shape[0]))


# In[25]:


#Lossfunction

def performance_metric(y_true, y_predict):
    
    score = r2_score(y_true, y_predict)
   
    return score


# In[26]:


def fit_model_k_fold(X, y):
    
    # Create cross-validation sets from the training data
    # cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
    k_fold = KFold(n_splits=10)
    
    # TODO: Create a decision tree regressor object
    regressor = KNeighborsRegressor()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'n_neighbors':range(3,15)}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search object
    grid = GridSearchCV(regressor, param_grid=params,scoring=scoring_fnc,cv=k_fold)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


# In[27]:


# Fit the training data to the model using grid search
reg = fit_model_k_fold(X_train, y_train)

print ("Parameter 'n_neighbors' is {} for the optimal model.".format(reg.get_params()['n_neighbors']))


# In[28]:


print(performance_metric(y_test, reg.predict(X_test)))


# In[29]:


def fit_model_shuffle(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)

    # Create a KNN regressor object
    regressor = KNeighborsRegressor()
    # Create a dictionary for the parameter 'n_neighbors' with a range from 3 to 10
    params = {'n_neighbors':range(3,15)}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object
    grid = GridSearchCV(regressor, param_grid=params,scoring=scoring_fnc,cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


# In[30]:


# Fit the training data to the model using grid search
reg = fit_model_shuffle(X_train, y_train)

print ("Parameter 'n_neighbors' is {} for the optimal model.".format(reg.get_params()['n_neighbors']))


# In[31]:


print(performance_metric(y_test, reg.predict(X_test)))


# In[ ]:




