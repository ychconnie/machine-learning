#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 20:11:26 2025

@author: yichinhuang
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler

movie=pd.read_csv("/Users/yichinhuang/Downloads/movies_dataset.csv")
#dataframe with year and average movie gross 
average=pd.read_csv("/Users/yichinhuang/Downloads/averagegross.csv")

# remove nan rows
movie=movie.dropna(subset=["revenue","genres","release_date"])

#extract variables needed
movie=movie[["revenue","genres","release_date"]]
movie=pd.DataFrame(movie)

#extract year from the dataset for mapping
movie["release_date"]=pd.to_datetime(movie["release_date"])
movie["year"]=movie["release_date"].dt.year

#extract month and classify into seasons
movie["month"]=movie["release_date"].dt.month

#replace the 0 revenue with value of average gross in that year
movie.loc[movie["revenue"]==0,"revenue"]=(movie["year"].map(average.set_index("year")["average gross"]))

#filter rows with revenue> 5 millions
movie=movie[movie["revenue"]>5000000]

#assign the month into 4 seasons
season={1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",6:"summer",
        7:"summer",8:"summer",9:"fall",10:"fall",11:"fall",12:"winter"}
movie["season"]=movie["month"].map(season)
'''spring=[3,4,5]
summer=[6,7,8]
fall=[9,10,11]
winter=[12,1,2]
season=[]
for i in range(len(movie)):
  if movie.iloc[i]["month"] in spring:
    season.append("spring")
  elif movie.iloc[i]["month"] in summer:
    season.append("summer")
  elif movie.iloc[i]["month"] in fall:
    season.append("fall")
  else:
    season.append("winter")
movie["season"]=season'''

#convert date into i-th of day of a year
#movie["day_of_year"]=movie["release_date"].dt.day_of_year

#split the genres into different columns with dummy vars
movie["genres"]=movie["genres"].str.split('-')
genre_dummies = movie['genres'].explode().str.get_dummies().groupby(level=0).max()
movie = pd.concat([movie, genre_dummies], axis=1)

#drop genres that occur less than 5% 
drop_col=[]
for col in range(7,24):
  if col < len(movie.columns):
    drop=movie.columns[col]
    if movie.iloc[:,col].sum()<=0.05*len(movie):
      drop_col.append(drop)

movie=movie.drop(columns=drop_col)

print(movie)
print(movie.columns)


#factorize the season column
movie["season"]=pd.factorize(movie["season"])[0]

movie=movie.apply(pd.to_numeric,errors="coerce")
#set up X and y
X=movie.drop(columns=["revenue","genres","release_date","year","month"])
y=movie["revenue"]
y=y.dropna()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

import statsmodels.api as sm
X_train=sm.add_constant(X_train)
model=sm.OLS(y_train,X_train).fit()
print(model.summary())



#predict mickey
predict_df=pd.read_csv("/Users/yichinhuang/Downloads/Data to predict.xlsx - Sheet1.csv")
predict_df=predict_df[["genres","release_date"]]
predict_df=pd.DataFrame(predict_df)

#extract year from the dataset for mapping
predict_df["release_date"]=pd.to_datetime(predict_df["release_date"])
predict_df["year"]=predict_df["release_date"].dt.year

#extract month and classify into seasons
predict_df["month"]=predict_df["release_date"].dt.month

predict_df["season"]=predict_df["month"].map(season)


#split the genres into different columns with dummy vars
predict_df["genres"]=predict_df["genres"].str.split('-')
genre_dummies = predict_df['genres'].explode().str.get_dummies().groupby(level=0).max()
predict_df = pd.concat([predict_df, genre_dummies], axis=1)

#reformat the prediction df
predict_df=sm.add_constant(predict_df,has_constant="add")
predict_df=predict_df.reindex(columns=X_train.columns,fill_value=0)

#factorize the season column
predict_df["season"]=pd.factorize(predict_df["season"])[0]

predict_df=predict_df.apply(pd.to_numeric,errors="coerce")
#print(predict_df)
#print(predict_df.columns)
#print(X.columns)

#predict_df=sm.add_constant(predict_df)
pred_lr=model.predict(predict_df)
print(pred_lr)

#random forest
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

mse=[]
r2=[]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
for i in range(100,600,100):
    rf=RandomForestRegressor(n_estimators=i)
    rf.fit(X_train,y_train)
    pre_all=rf.predict(X_test)
    
    mse_attempt=mean_squared_error(y_test, pre_all)
    mse.append(mse_attempt)
    r2_attempt=r2_score(y_test, pre_all)
    r2.append(r2_attempt)
print("mse:",mse)
print("r-squared:",r2)
predict_df=predict_df.drop(columns="const",errors="ignore")
pred_rf=rf.predict(predict_df)
print(pred_rf)
n_estimators=[100,200,300,400,500]
plt.plot(n_estimators,mse,marker='o',linestyle='-',color='blue')
plt.xlabel("n_estimators")
plt.ylabel("MSE")
plt.title("n_estimators vs. MSE")
plt.show()

plt.plot(n_estimators,r2,marker='o',linestyle='-',color='blue')
plt.xlabel("n_estimators")
plt.ylabel("R-Squared")
plt.title("n_estimators vs. R^2")
plt.show()

#predict
rf=RandomForestRegressor(n_estimators=500)
rf.fit(X_train,y_train)
#predict_df=predict_df.drop(columns="const",errors="ignore")
pred_rf=rf.predict(predict_df)
print(pred_rf)

print(pred_lr[1])
'''
rf=RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)
pre_all=rf.predict(X_test)
#mse_attempt=mean_squared_error(y_test, pre_all)
#mse.append(mse_attempt)
#r2_attempt=r2_score(y_test, pre_all)
#r2.append(r2_attempt)
print("mse:",mse)
print("r-squared:",r2)
pred_rf=rf.predict(predict_df)
print(pred_rf)
'''
print("Mickey 17 predictions:")
print("The prediction using OLS: "+"$"+str(pred_lr))
print("The prediction using Random Forest: "+"$"+str(pred_rf))
