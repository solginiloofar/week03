# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 11:41:35 2018

@author: N_Solgi
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline

import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix



dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

data_train=pd.read_csv('C:\Users\N_Solgi\Desktop\Artificial Intelligence\machine learning\week03\wk3_kc_house_train_data.csv')
dat_valid=pd.read_csv('C:\Users\N_Solgi\Desktop\Artificial Intelligence\machine learning\week03\wk3_kc_house_valid_data.csv')
data_test=pd.read_csv('C:\Users\N_Solgi\Desktop\Artificial Intelligence\machine learning\week03\wk3_kc_house_test_data.csv')
data_all=pd.read_csv('C:\Users\N_Solgi\Desktop\Artificial Intelligence\machine learning\week03\kc_house_data.csv')
data_set1=pd.read_csv('C:\Users\N_Solgi\Desktop\Artificial Intelligence\machine learning\week03\wk3_kc_house_set_1_data.csv')
data_set2=pd.read_csv('C:\Users\N_Solgi\Desktop\Artificial Intelligence\machine learning\week03\wk3_kc_house_set_2_data.csv')
data_set3=pd.read_csv('C:\Users\N_Solgi\Desktop\Artificial Intelligence\machine learning\week03\wk3_kc_house_set_3_data.csv')
data_set4=pd.read_csv('C:\Users\N_Solgi\Desktop\Artificial Intelligence\machine learning\week03\wk3_kc_house_set_4_data.csv')

x=data_train.iloc[: , 4:3].values
x2=data_train['sqft_living']* data_train['sqft_living']
x3=data_train['sqft_living']* data_train['sqft_living']*data_train['sqft_living']
y=data_train.iloc[:, 1].values

poly = PolynomialFeatures(degree=3)

poly_x=poly.fit_transform(x)
regressor=LinearRegression()
regressor.fit(poly_x,y)
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(poly.fit_transform(x)),color='blue')
plt.show()

degree_min=2
degree_max=8

def polynomial_dataframe(feature, degree):
    poly_dataframe = pd.DataFrame()
    if degree > 1 :
        for power in range(2,degree+1):
        name='power_'+str(power)
     return poly_dataframe
 



sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
sales = sales.sort(['sqft_living','price'])


poly1_data = polynomial_dataframe(sales['sqft_living'], 1)
poly1_data['price']= sales['price']

model1=DataFrame.LinearRegression.create(poly1_data , target= 'price'  , features = ['power_1'], validation_set=None)
plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
poly1_data['power_1'], model1.predict(poly1_data),'-')












