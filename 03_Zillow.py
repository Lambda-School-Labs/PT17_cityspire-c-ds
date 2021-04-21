#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
rent_price = pd.read_csv('rent_price.csv')
print(rent_price.shape)
rent_price.head()


# # Using Linear regression

# In[5]:


#describing price feature
rent_price['Price'].describe()


# In[6]:


# after looking at the price feature
mean_baseline = 1284.677264


# In[7]:


# calculating error now
errors = mean_baseline - rent_price['Price']
errors


# In[8]:


#calculating mean absolute error
mean_absolute_error = errors.abs().mean()
mean_absolute_error


# In[10]:


#Split train into train and val
from sklearn.model_selection import train_test_split
train,test = train_test_split(rent_price, train_size = 0.80, test_size = 0.20,stratify=rent_price['Bedrooms'],random_state=45)


# In[11]:


train.shape


# In[12]:


test.shape


# In[13]:


# The Price column is the target
target = 'Price'

X_train = train.drop(columns=target)
y_train = train[target]
X_test = test.drop(columns=target)
y_test = test[target]


# In[14]:


# Train Error
from sklearn.metrics import mean_absolute_error
guess = mean_baseline
y_pred = [guess] * len(y_train)
mae = mean_absolute_error(y_train, y_pred)
print(f'Train Error : {mae:.2f} percentage points')


# In[15]:


# Test Error
y_pred = [guess] * len(y_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Test Error : {mae:.2f} percentage points')


# In[16]:


(y_test - y_pred).abs().mean()


# In[ ]:


#Use scikit-learn to fit a multiple regression with three features.


# In[17]:


# 1. Import the appropriate estimator class from Scikit-Learn
from sklearn.linear_model import LinearRegression


# In[18]:


# 2. Instantiate this class
model = LinearRegression()
model


# In[19]:



# Re-arrange X features matrices
features = ['State','CountyName', 
            'Bedrooms']
print(f'Linear Regression, dependent on: {features}')

train = X_train[features]
train


# In[20]:


test = X_test[features]
test


# In[21]:


# Fit the model
model.fit(train, y_train)


# In[22]:


#  Apply the model to new data
y_pred_train = model.predict(train)
mean_absolute_error(y_pred_train, y_train)


# In[23]:


y_pred = model.predict(test)
mean_absolute_error(y_pred, y_test)


# In[24]:


model.intercept_, model.coef_


# In[25]:



# This is easier to read
print('Intercept', model.intercept_)
coefficients = pd.Series(model.coef_, features)
print(coefficients.to_string())


# In[ ]:


# # save the model to disk
# import pickle
# filename = 'rent_recommendation_model.sav'
# pickle.dump(model, open(filename, 'wb'))
            


# # testing the model with a testcase

# In[26]:


from sklearn.preprocessing import OrdinalEncoder
import numpy as np
def predict(State, CountyName, Bedrooms):
    ordinal_encoder = OrdinalEncoder()
    z = ordinal_encoder.fit_transform([['State','CountyName']])
    z = z.flatten()
    z = z.tolist()
    z.append(Bedrooms)
    result = model.predict(np.array([z]))
    return result


# In[27]:


predict('IL','Dupage',2)


# ## testing with pickled model and dataset

# In[28]:



import pandas as pd
import pickle
from pickle import load
filename = 'rent_recommendation_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
#rentdf_pkl = pd.read_pickle("rent_price_dataset.pkl")


# In[29]:


from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pickle
from pickle import load
def predict_rent(State, CountyName, Bedrooms):
    ordinal_encoder = OrdinalEncoder()
    z = ordinal_encoder.fit_transform([['State','CountyName']])
    z = z.flatten()
    z = z.tolist()
    z.append(Bedrooms)
    result = loaded_model.predict(np.array([z]))
    return result


# In[30]:


predict_rent('IL','Dupage',2)


# ## Code with Pydantic

# In[ ]:


import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import pickle
from pickle import load
filename = 'rent_recommendation_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))


# In[ ]:


class Rent_Predict(BaseModel):
    State:str
    CountyName:str
    Bedrooms:int
    
    def dict_df(self):
        """Convert to pandas dataframe with 1 row"""
        y=pd.DataFrame([dict(self)])
        ordinal_encoder = OrdinalEncoder()
        y[['State','CountyName']] = ordinal_encoder.fit_transform(y[['State','CountyName']])
        return y     
        


# In[ ]:


#@router.post("/api/rental_priceprediction")
def rental_price_prediction(rent_predict:Rent_Predict):
   
    
    """ Machine Learning model predicts the rental price of the target city and county
    

    args:
        State: Provide the abbreviation of the target state. For eg NY for New York
        CountyName: Provide the county name.
        Bedrooms: Number of bedrroms required.

    returns:
        Dictionary that contains the requested data, which is converted
        by fastAPI to a json object.
    """
    
        
    result = loaded_model.predict(rent_predict.dict_df()) 
    return {'predicted_price': result[0]}
    
 


# In[11]:





# In[3]:





# In[ ]:




