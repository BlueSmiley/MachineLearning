# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 23:49:49 2019

@author: tonpr
"""

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import RobustScaler
from math import sqrt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from scipy.stats import mstats
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from category_encoders import TargetEncoder,LeaveOneOutEncoder
from sklearn import svm
from shapely.geometry import MultiPoint
from geopy.distance import great_circle,vincenty
from sklearn import ensemble
from sklearn import neighbors
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from catboost import Pool, CatBoostRegressor, cv


def main():
     
    print("running")
    input_file = "tcd ml 2019-20 income prediction training (with labels).csv"
    
    df = pd.read_csv(input_file, header = 0)
    df = process_features(df)
    train, test = train_test_split(df, test_size=0.2) 
    
    x = train.drop('Income in EUR',axis=1)
    y = train["Income in EUR"]
    
    cate_features_index = np.where(x.dtypes != float)[0]
    
    #make the x for train and test (also called validation data) 
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=.85)

#    train_pool = Pool(train, 
#                  train.columns.values, 
#                  cat_features=["Hair Color",
#         "Wears Glasses","University Degree","Gender","Country","Profession"])
#    
#    test_pool = Pool(test, 
#                 cat_features=["Hair Color",
#         "Wears Glasses","University Degree","Gender","Country","Profession"]) 
#    
#    train_y = train.iloc[:,-1]
#    train_X = train.iloc[:,:-1]
#
#    test_y = test.iloc[:,-1]
#    test_X = test.iloc[:,:-1]
    
    
    regr = CatBoostRegressor(task_type="GPU",
                           devices='0:1',eval_metric='RMSE')
    regr.fit(xtrain,ytrain,cat_features=cate_features_index,eval_set=(xtest,ytest))
    #cv_data = cv(regr.get_params(),Pool(x,y,cat_features=cate_features_index),fold_count=3)
   
    #regr.fit(train_pool)
    
    
    
    # Make predictions using the testing set
    pred = regr.predict(test.drop('Income in EUR',axis=1))
    pred = pred.astype(np.int)
    
    # The coefficients
    # print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Root Mean squared error: %.2f"
        %  sqrt(mean_squared_error(test["Income in EUR"], pred)))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(test["Income in EUR"], pred))
    # print('Internal score: %.2f' % regr.score(X, y))
    
    # The rest essentially calculates the answers for the actual thing
    actual_file = "tcd ml 2019-20 income prediction test (without labels).csv"
    finalData = pd.read_csv(actual_file, header = 0)
    finalData = finalData.drop('Income',axis=1)
    finalData = process_features(finalData)
     
     #testing_file2 = "Testing2.csv"
     #finalData.to_csv(testing_file2, index=False)
     
    results = regr.predict(finalData)
    output_file = "tcd ml 2019-20 income prediction submission file.csv"
    output =  pd.read_csv(output_file, header = 0, index_col=False)
     #print(output)
    output["Income"] = results
     #print(output)
    output.to_csv(output_file, index=False)
    
def process_features(data):
    
        # Has no relation to data
    data = data.drop('Instance', 1)
    # data = data.drop('Profession', 1)
#     
    # Constrain Hair color
    mapping = {"Black": 1, "Blond": 2, "Brown": 3}
    data["Hair Color"] = data["Hair Color"].map(mapping)
    # Four buckets. Looking at data most are one of three or basically undefined.
    # To interpolate no correlation between not giving data hence assume mode
    #data["Hair Color"] = data["Hair Color"].fillna(data["Hair Color"].mode()[0])
    # To interpolate as missing values implies new feature
    data["Hair Color"] = data["Hair Color"].fillna(4)
#     
# 
    # Constrain Degree
    mapping = {"Bachelor": 1, "Master": 2, "No": 3,"PhD": 4}
    data["University Degree"] = data["University Degree"].map(mapping)
    # To interpolate people who didn't answer probably have no qualifications
    data["University Degree"] = data["University Degree"].fillna(3)
    # To interpolate as missing values implies new feature
    # data["University Degree"] = data["University Degree"].fillna(5)
    
#     
      # constrain Gender
    mapping = {"male": 1, "female": 2, "other": 3}
    data["Gender"] = data["Gender"].map(mapping)
    # To interpolate as people not answering actually belong to other
    data["Gender"] = data["Gender"].fillna(3)
    # To interpolate as people not answering is new feature
    #data["Gender"] = data["Gender"].fillna(4)
#       
    # interpolate as unknown
    #data["Profession"] = data["Profession"].fillna("Unknown")
    data["Country"] = data["Country"].fillna("Unknown")
    #print(np.shape(data))
    # data["X"] = data["X"].fillna(0) 
    # data["Y"] = data["Y"].fillna(0) 
    #print("After:" + str(np.shape(data)))
    #data = data.fillna(0)
#     
    # Fill all possible columns with column mean
    data = data.fillna(data.mean())
    # Fill all column nans with column mode
    for column in data.columns:
        if len(data[column].mode()) > 0:
            data[column].fillna(data[column].mode()[0], inplace=True)
    data = data.fillna(method="ffill")
    
    # data = data.drop('Country', 1)
    # data = data.drop('Profession', 1)
    # print(data)
    return data
    
    
    
if __name__ == "__main__":
    main()