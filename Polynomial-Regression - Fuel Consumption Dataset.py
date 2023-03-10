#Importing necessary packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[16]:


#Reading the data

df = pd.read_csv("D:\AIML\Dataset\FuelConsumption.csv")

df.head()


# In[17]:


#Creating subset of the master data

cdf = df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]

cdf.head()


# In[18]:


cdf.shape


# In[19]:


#Splitting the data to train & test the model

msk = np.random.rand(len(cdf)) <= 0.80


# In[24]:


train = cdf[msk]
test = cdf[~msk]


# In[25]:


train.shape


# In[26]:


test.shape


# In[27]:


#Plotting the train dataset

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,c='green')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()


# In[28]:


#Import necessary packages

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


# In[29]:


#Creating an array of IV & DV

train_x = np.asanyarray(train[["ENGINESIZE"]]) #IV
train_y = np.asanyarray(train[["CO2EMISSIONS"]]) #DV


# In[30]:


poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)


# In[31]:


train_x_poly


# In[32]:


regr = linear_model.LinearRegression()


# In[33]:


train_yhat = regr.fit(train_x_poly,train_y)


# In[34]:


print(f"The coefficients = {regr.coef_}")


# In[35]:


print(f"The intercept = {regr.intercept_}")


# In[36]:


import numpy as np

xx = np.arange(0,10,0.1)


# In[37]:


yy = regr.coef_[0][1]*xx+regr.coef_[0][2]*np.power(xx,2)+regr.intercept_


# In[38]:


#Plot the graph between IV & DV & use the coef & intercept to plot the best fit line

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,c='red')
xx = np.arange(0,10,0.1)
yy = regr.coef_[0][1]*xx+regr.coef_[0][2]*np.power(xx,2)+regr.intercept_
plt.plot(xx,yy,c='black')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()


# In[39]:


test_x = np.asanyarray(test[["ENGINESIZE"]])
test_y = np.asanyarray(test[["CO2EMISSIONS"]])


# In[40]:


test_x_poly = poly.fit_transform(test_x)


# In[42]:


test_x_poly[0:5]


# In[43]:


#Import necessary packages

from sklearn.metrics import r2_score


# In[44]:


predicted_y = regr.predict(test_x_poly)


# In[45]:


print(f"The mean absolute error = {np.mean(np.absolute(predicted_y-test_y))}")


# In[46]:


print(f"The mean sqaure error = {np.mean((predicted_y-test_y)**2)}" )


# In[47]:


print(f"The R2 score = {r2_score(predicted_y,test_y)*100} %")


# In[48]:


#Transforming IV & storing it in a variable 

trans_data = poly.fit_transform([[3.5]])


# In[49]:


trans_data


# In[50]:


#Using the tranformed variable getting the predicted CO2 Emission for  

regr.predict(trans_data)


# # Polynomian Regression with Multiple IV

# In[51]:


#Import necessary packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[52]:


#Reading the data

cdf1 = df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]

cdf1.head()


# In[53]:


#Split the data to train & test the model

msk1 = np.random.rand(len(cdf1)) <= 0.80


# In[54]:


train = cdf1[msk]
test = cdf1[~msk]


# In[55]:


train.shape


# In[56]:


test.shape


# In[57]:


#Import necessary packages

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


# In[58]:


#Decalring IV & DV

train_x1 = np.asanyarray(train[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]]) #IV
train_y1 = np.asanyarray(train[["CO2EMISSIONS"]]) #DV


# In[123]:


#Transforming the IV

poly = PolynomialFeatures(degree=4)
train_x_poly = poly.fit_transform(train_x1)


# In[124]:


train_x_poly


# In[125]:


regr = linear_model.LinearRegression()


# In[126]:


regr.fit(train_x_poly,train_y1)


# In[127]:


print(f"The coefficients = {regr.coef_}")
print(f"The intercept = {regr.intercept_}")


# In[128]:


#Import necessary package

import numpy as np


# In[129]:


xx = np.arange(0,15,0.1)


# In[130]:


yy = regr.coef_[0][1]*xx+regr.coef_[0][2]*np.power(xx,2)+regr.intercept_


# In[131]:


IV = cdf1[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]]

IV.head()


# In[132]:


#Plotting the train data with the coefficient & intercept to get the best fit line

plt.scatter(train.CYLINDERS,train.CO2EMISSIONS)
xx = np.arange(0,10,0.1)
yy = regr.coef_[0][1]*xx+regr.coef_[0][2]*np.power(xx,2)+regr.intercept_
plt.plot(xx,yy,c='black')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()


# In[95]:


#Creating test data & declaring IV & DV

test_x1 = np.asanyarray(test[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]]) #IV
test_y1 = np.asanyarray(test[["CO2EMISSIONS"]]) #DV


# In[96]:


test_x1_poly = poly.fit_transform(test_x1)


# In[97]:


test_x1_poly


# In[98]:


predicted_y1 = regr.predict(test_x1_poly)


# In[99]:


#Import necessary packges

from sklearn.metrics import r2_score


# In[100]:


print(f"The mean absolute error = {np.mean(np.absolute(predicted_y1-test_y1))}")
print(f"The mean square error = {np.mean((predicted_y1-test_y1)**2)}")
print(f"The R2 Score = {r2_score(predicted_y1,test_y1)*100} %")


# In[101]:


trans_data1 = poly.fit_transform([[2,4,8.5]])


# In[102]:


trans_data1


# In[103]:


regr.predict(trans_data1)


# In[ ]:




