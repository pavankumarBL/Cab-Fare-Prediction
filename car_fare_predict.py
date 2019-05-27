#!/usr/bin/env python
# coding: utf-8

# # Importing Packages and Collecting Data

# In[1]:


#importing the basic Library modules
import pandas as pd
from pandas import Series, DataFrame
import os
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn import linear_model
from fancyimpute import KNN


# In[2]:


os.chdir("C:\\Users\\pavankumar.bl\\Documents\\datascience\\Edwisor\\Project_2\\train_cab")


# In[3]:


train_cab=pd.read_csv("train_cab.csv",parse_dates=['pickup_datetime'])


# In[4]:


test_cab=pd.read_csv("test_cab.csv")


# In[5]:


(train_cab.describe(include='all'))


# # Cleaning

# In[6]:


# Training set
train_cab = train_cab[abs(train_cab["pickup_latitude"]) < 90]
train_cab = train_cab[abs(train_cab["dropoff_latitude"]) < 90]
train_cab = train_cab[abs(train_cab["pickup_longitude"]) < 180]
train_cab = train_cab[abs(train_cab["dropoff_longitude"]) < 180]


# In[10]:


(train_cab.describe(include='all'))


# In[9]:


train_cab['fare_amount'] = pd.to_numeric(train_cab['fare_amount'], errors='coerce')
train_cab = train_cab.dropna(subset=['fare_amount'])


# In[11]:


neg_fare = train_cab.loc[train_cab.fare_amount<0, :].index
train_cab.drop(neg_fare, axis = 0, inplace = True)
# Drop rows greater than 100
fares_to_drop = train_cab.loc[train_cab.fare_amount>100,:].index
train_cab.drop(fares_to_drop, axis = 0, inplace = True)


# In[12]:


(train_cab.describe(include='all'))


# In[13]:


# Drop rows greater than 8 passanger count since only 8 is the max limit in a cab
Passenger_to_drop = train_cab.loc[train_cab.passenger_count>8,:].index
train_cab.drop(Passenger_to_drop, axis = 0, inplace = True)


# In[14]:


train_cab.dtypes


# In[15]:


train_cab['pickup_datetime'] = pd.to_datetime(train_cab['pickup_datetime'], errors='coerce')
train_cab = train_cab.dropna(subset=['pickup_datetime'])


# In[20]:


(train_cab.describe(include='all'))


# In[15]:


train_cab.head(30)


# In[17]:


#replace all Zeroes with NaN to find missing values
train_cab=train_cab.replace(0,np.nan)


# In[18]:


train_cab['year']=train_cab['pickup_datetime'].dt.year
train_cab['month']=train_cab['pickup_datetime'].dt.month
#train_cab['weekdayname']=train_cab['pickup_datetime'].dt.weekday_name
train_cab['weekday']=train_cab['pickup_datetime'].dt.weekday
train_cab['hour']=train_cab['pickup_datetime'].dt.hour


# In[19]:


train_cab=train_cab.drop('pickup_datetime',axis=1)


# # Missing value analysis

# In[21]:


train_cab.shape


# In[22]:


Missing_val=pd.DataFrame(train_cab.isnull().sum()).reset_index()


# In[23]:


#Renaming variables of missing_val dataframe
Missing_val = Missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})

#Calculating percentage missing value
Missing_val['Missing_percentage'] = (Missing_val['Missing_percentage']/len(train_cab))*100

# Sorting missing_val in Descending order
Missing_val = Missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)


# In[24]:


Missing_val


# In[25]:


train_cab.isnull().sum()


# In[26]:


train_cab.dtypes


# In[27]:


train_cab=pd.DataFrame(KNN(k=5).fit_transform(train_cab),columns=train_cab.columns, index=train_cab.index)


# In[28]:


train_cab.head(5)


# In[29]:


train_cab['passenger_count']=train_cab['passenger_count'].astype(int)
train_cab['year']=train_cab['year'].astype(int)
train_cab['weekday']=train_cab['weekday'].astype(int)
train_cab['hour']=train_cab['hour'].astype(int)
train_cab['month']=train_cab['month'].astype(int)


# In[30]:


train_cab.head(30)


# # Feature Engineering

# In[31]:


train_cab['abs_longi']=abs(train_cab['pickup_longitude']-train_cab['dropoff_longitude'])
train_cab['abs_lat']=abs(train_cab['pickup_latitude']-train_cab['dropoff_latitude'])


# In[32]:



# Calculate great circle distance using haversine formula
def great_circle_distance(lon1,lat1,lon2,lat2):
    R = 6371000 # Approximate mean radius of earth (in m)
    
    # Convert decimal degrees to ridians
    lon1,lat1,lon2,lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Distance of lons and lats in radians
    dis_lon = lon2 - lon1
    dis_lat = lat2 - lat1
    
    # Haversine implementation
    a = np.sin(dis_lat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dis_lon/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    dis_m = R*c # Distance in meters
    dis_km = dis_m/1000 # Distance in km
    return dis_km

# Create a column named greate_circle_distance
train_cab['great_circle_distance'] = great_circle_distance(train_cab.pickup_longitude, train_cab.pickup_latitude, train_cab.dropoff_longitude, train_cab.dropoff_latitude)


# In[58]:


train_cab.head(30)


# In[33]:


fig, axarr = plt.subplots(2, 2, figsize=(15, 15))
plt.subplots_adjust(hspace=.3)
sns.barplot(x='year',y='fare_amount',data=train_cab,ax=axarr[0][0])
sns.barplot(x='month',y='fare_amount',data=train_cab,ax=axarr[0][1])
sns.barplot(x='weekday',y='fare_amount',data=train_cab,ax=axarr[1][0])
sns.barplot(x='hour',y='fare_amount',data=train_cab,ax=axarr[1][1])


# # Correlation Analysis

# In[34]:


# Let's see which variables have the strongest and weakest correlation with fare_amount
corr = train_cab.corr().sort_values(by='fare_amount', ascending=False)
fig, ax = plt.subplots(figsize = (20,12))
sns.heatmap(corr, annot = True, cmap ='BrBG', ax = ax, fmt='.2f', linewidths = 0.05, annot_kws = {'size': 17})
ax.tick_params(labelsize = 15)
ax.set_title('Correlation with fare_amount', fontsize = 22)
plt.show()


# In[35]:


# Let's group mean fare_amount by pickup_year to see if there is a pattern.
pivot_year = pd.pivot_table(train_cab, values = 'fare_amount', index = 'year', aggfunc = ['mean'])
print('Mean fare_amount across the classes of year: \n{}'.format(pivot_year))


# In[36]:


# A bar plot would be more helpful to visualize this pattern
fig, ax = plt.subplots(figsize = (15,5))
pivot_year.plot(kind = 'bar', legend = False, color = 'firebrick', ax = ax)
ax.set(title = 'year vs mean fare_amount', ylabel= 'mean fare_amount')
plt.show()


# In[37]:


(train_cab.describe(include='all'))


# In[38]:


#considering taxi ride is limited to certain distance will keep this at 100Km
train_cab = train_cab[train_cab["great_circle_distance"] < 100]
train_cab = train_cab[train_cab["great_circle_distance"] > 0]


# In[39]:


sns.distplot(train_cab.great_circle_distance)


# In[70]:


train_cab.head(5)


# # Model Creation [Linear regression]

# # Simple Linear Regression Model 1

# In[40]:


#Predicting only Distance based 
Column_names=['great_circle_distance']
y=train_cab['fare_amount']
X_train=train_cab[Column_names]
X_test=train_cab[Column_names]


# In[42]:


model=LinearRegression()
model.fit(X_train,y)


# In[43]:


#Score of the model
model.score(X_train,y)


# In[44]:


model.coef_


# In[45]:


model.intercept_


# In[47]:


# calculating coefficients

coeff = DataFrame(X_train.columns)

coeff['Coefficient Estimate'] = Series(model.coef_)

coeff
#Higher Coefficient Estimate more weight is given to that variable


# In[48]:


Estimation_test=model.predict(X_test)


# In[49]:


r2_score(y,Estimation_test)


# In[50]:


train_cab['Predicted_Fare_amount']=Estimation_test


# In[51]:


train_cab[['fare_amount','Predicted_Fare_amount']]


# # Error Metrics

# In[82]:


mean_squared_error(train_cab['fare_amount'],train_cab['Predicted_Fare_amount'])


# In[83]:


np.sqrt(mean_squared_error(train_cab['fare_amount'],train_cab['Predicted_Fare_amount']))


# # Visualization

# In[52]:


#plot for actual vs Predicted values
test = pd.DataFrame({'Predicted':train_cab['Predicted_Fare_amount'],'Actual':y})
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.figure(figsize=(18,12))
plt.plot(test[:200])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind='reg')


# # Linear Regression Model 2

# In[53]:


#Predicting only Distance based and Passenger count
Column_names=['passenger_count','abs_longi','abs_lat']
y=train_cab['fare_amount']
X_train=train_cab[Column_names]
X_test=train_cab[Column_names]


# In[54]:


model=LinearRegression()
model.fit(X_train,y)


# In[55]:


#Let us take a look at the coefficients of this linear regression model.

# calculating coefficients

model.score(X_train,y)


# In[56]:


model.coef_


# In[57]:


model.intercept_


# In[58]:


# calculating coefficients

coeff = DataFrame(X_train.columns)

coeff['Coefficient Estimate'] = Series(model.coef_)

coeff
#Higher Coefficient Estimate more weight is given to that variable


# In[59]:


Estimation_test=model.predict(X_test)


# In[60]:


train_cab['Predicted_Fare_amount_2']=Estimation_test


# In[61]:


train_cab[['fare_amount','Predicted_Fare_amount','Predicted_Fare_amount_2']]


# In[94]:


#Error Metrics


# In[62]:


mean_squared_error(train_cab['fare_amount'],train_cab['Predicted_Fare_amount_2'])


# In[63]:


np.sqrt(mean_squared_error(train_cab['fare_amount'],train_cab['Predicted_Fare_amount_2']))


# In[97]:


#Visualization


# In[98]:


#plot for actual vs Predicted values
test = pd.DataFrame({'Predicted':train_cab['Predicted_Fare_amount_2'],'Actual':y})
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.figure(figsize=(18,12))
plt.plot(test[:200])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind='reg')


# # Linear Regression Model 3

# In[64]:


#Predicting only Distance based and Passenger count and year month and hour
Column_names=['passenger_count','abs_longi','abs_lat','year','month','weekday','hour','great_circle_distance']
y=train_cab['fare_amount']
X_train=train_cab[Column_names]
X_test=train_cab[Column_names]


# In[65]:


model=LinearRegression()
model.fit(X_train,y)


# In[66]:


#Let us take a look at the coefficients of this linear regression model.

# calculating coefficients

model.score(X_train,y)


# In[67]:


model.coef_


# In[68]:


model.intercept_


# In[69]:


# calculating coefficients

coeff = DataFrame(X_train.columns)

coeff['Coefficient Estimate'] = Series(model.coef_)

coeff
#Higher Coefficient Estimate more weight is given to that variable


# In[70]:


Estimation_test=model.predict(X_test)


# In[71]:


train_cab['Predicted_Fare_amount_3']=Estimation_test


# In[72]:


train_cab[['fare_amount','Predicted_Fare_amount','Predicted_Fare_amount_2','Predicted_Fare_amount_3']]


# In[108]:


#error metrics


# In[73]:


mean_squared_error(train_cab['fare_amount'],train_cab['Predicted_Fare_amount_3'])


# In[74]:


np.sqrt(mean_squared_error(train_cab['fare_amount'],train_cab['Predicted_Fare_amount_3']))


# In[111]:


#Visualization


# In[75]:


#plot for actual vs Predicted values
test = pd.DataFrame({'Predicted':train_cab['Predicted_Fare_amount_3'],'Actual':y})
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.figure(figsize=(18,12))
plt.plot(test[:200])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind='reg')


# # Linear Regression Model Using Ridge Method

# In[76]:


#Predicting only Distance based and Passenger count and year month and hour
Column_names=['passenger_count','abs_longi','abs_lat','year','month','weekday','hour','great_circle_distance']
y=train_cab['fare_amount']
X_train=train_cab[Column_names]
X_test=train_cab[Column_names]


# In[77]:


model=linear_model.Ridge(alpha=.5)
model.fit(X_train,y)


# In[78]:


#Let us take a look at the coefficients of this linear regression model.

# calculating coefficients

model.score(X_train,y)


# In[79]:


model.coef_


# In[117]:


# calculating coefficients

coeff = DataFrame(X_train.columns)

coeff['Coefficient Estimate'] = Series(model.coef_)

coeff


# In[118]:


Estimation_test=model.predict(X_test)


# In[119]:


train_cab['Predicted_Fare_amount_ridge']=Estimation_test


# In[ ]:


train_cab[['fare_amount','Predicted_Fare_amount','Predicted_Fare_amount_2','Predicted_Fare_amount_3','Predicted_Fare_amount_ridge']]


# In[121]:


#error metrics


# In[122]:


mean_squared_error(train_cab['fare_amount'],train_cab['Predicted_Fare_amount_ridge'])


# In[123]:


np.sqrt(mean_squared_error(train_cab['fare_amount'],train_cab['Predicted_Fare_amount_ridge']))


# In[124]:


#visualization


# In[125]:


#plot for actual vs Predicted values
test = pd.DataFrame({'Predicted':train_cab['Predicted_Fare_amount_ridge'],'Actual':y})
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.figure(figsize=(18,12))
plt.plot(test[:200])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind='reg')


# # Random forest Model 4

# In[149]:


#Predicting only Distance based and Passenger count and year month and hour
Column_names=['passenger_count','abs_longi','abs_lat','year','month','weekday','hour','great_circle_distance']
y=train_cab['fare_amount']
X_train=train_cab[Column_names]
X_test=train_cab[Column_names]


# In[150]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train,y)
rf_predict = rf.predict(X_test)


# In[151]:


rf.score(X_train,y)


# In[128]:


train_cab['Predicted_Fare_amount_4']=rf_predict


# In[ ]:


train_cab[['fare_amount','Predicted_Fare_amount','Predicted_Fare_amount_2','Predicted_Fare_amount_3','Predicted_Fare_amount_4']]


# In[212]:


#plot for actual vs Predicted values
test = pd.DataFrame({'Predicted':rf_predict,'Actual':y})
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.figure(figsize=(18,12))
plt.plot(test[:200])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind='reg')


# In[130]:


rf.n_features_


# In[131]:


rf.feature_importances_


# In[132]:


Feature_importance = DataFrame(X_train.columns)

Feature_importance['Feature_importance'] = Series(rf.feature_importances_)

Feature_importance


# In[133]:


print(rf.get_params())


# In[135]:


#RF Model Evaluation


# In[ ]:





# In[139]:


#from sklearn.model_selection import GridSearchCV

#param_grid = {"n_estimators": [100, 200],
   # "max_depth": [3, None],
   # "max_features": [1, 3, 5, 8],
  #  "min_samples_split": [2, 5, 10],
   # "min_samples_leaf": [1, 3, 10],
   # "bootstrap": [True, False]}

#model = RandomForestRegressor(random_state=0)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#grid.fit(X_train,y)

#print(grid.best_score_)
#print(grid.best_params_)

#0.8248409265487134
#{'bootstrap': True, 'max_depth': None, 'max_features': 3, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 200}


# In[ ]:


#From this, we can then create the regression using these values


# In[141]:


regressor = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=None, max_features=3, 
                                  min_samples_leaf=3, min_samples_split=2, bootstrap=True)


# In[142]:


regressor.fit(X_train, y)


# In[144]:


regressor.score(X_train, y)


# In[145]:


y_pred = regressor.predict(X_test)


# In[146]:


plt.scatter(x=y, y=y_pred)
plt.xlim([1.5,4])
plt.ylim([1.5,4])
plt.plot([1.5,4],[1.5,4])
plt.show()


# In[ ]:


#The model fits the test set reasonably well with an R2 of 0.74 but as we can see from the graph there are still a
#number of outliers that the model didn’t correctly pick (both on the high and low house prices).


# In[147]:


feature_import = pd.DataFrame(data=regressor.feature_importances_, index=X_train.columns.values, columns=['values'])
feature_import.sort_values(['values'], ascending=False, inplace=True)
feature_import.transpose()


# In[148]:


feature_import.reset_index(level=0, inplace=True)
sns.barplot(x='index', y='values', data=feature_import, palette='deep')
plt.show()


# In[153]:


train_cab['Tuned_model']=y_pred


# In[156]:


r2_score(y, rf_predict)


# # Lets FIT Model for New Test Data 

# In[157]:


test_cab.dtypes


# In[158]:


(test_cab.describe(include='all'))


# In[159]:


test_cab['pickup_datetime'] = pd.to_datetime(test_cab['pickup_datetime'], errors='coerce')
test_cab = test_cab.dropna(subset=['pickup_datetime'])


# In[160]:


(test_cab.describe(include='all'))


# In[162]:


test_cab['year']=test_cab['pickup_datetime'].dt.year
test_cab['month']=test_cab['pickup_datetime'].dt.month
#train_cab['weekdayname']=train_cab['pickup_datetime'].dt.weekday_name
test_cab['weekday']=test_cab['pickup_datetime'].dt.weekday
test_cab['hour']=test_cab['pickup_datetime'].dt.hour


# In[163]:


test_cab['abs_longi']=abs(test_cab['pickup_longitude']-test_cab['dropoff_longitude'])
test_cab['abs_lat']=abs(test_cab['pickup_latitude']-test_cab['dropoff_latitude'])


# In[164]:


test_cab['great_circle_distance'] = great_circle_distance(test_cab.pickup_longitude, test_cab.pickup_latitude, test_cab.dropoff_longitude, test_cab.dropoff_latitude)


# In[167]:


#Predicting only Distance based and Passenger count and year month and hour
Column_names=['passenger_count','abs_longi','abs_lat','year','month','weekday','hour','great_circle_distance']
y=train_cab['fare_amount']
X_train=train_cab[Column_names]
X_test=test_cab[Column_names]


# In[168]:


rf = RandomForestRegressor()
rf.fit(X_train,y)
Fare_amount_test = rf.predict(X_test)


# In[170]:


test_cab['Fare_amount']=Fare_amount_test


# In[ ]:




