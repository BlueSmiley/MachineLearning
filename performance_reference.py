# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from sklearn import  cluster
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import RobustScaler
from math import sqrt
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel


def main():
     
    input_file = "tcd ml 2019-20 income prediction training (with labels).csv"   
    df = pd.read_csv(input_file, header = 0)

    (df,ohe) = clean_data(df)
    train, test = train_test_split(df, test_size=0.2) 

    scaler = StandardScaler()
    (train,ohe) = clean_train_data(train)
    
    train_y = train.iloc[:,-1]
    train_X = scaler.fit_transform(train.iloc[:,:-1])
 
    test_y = test.iloc[:,-1]
    test_X = test.iloc[:,:-1]
    test_X = clean_data(test_X, ohe)
    test_X = scaler.transform(test_X)

    regr = linear_model.LassoCV(cv=3, verbose = 2)

    # This works ok but requires LassoCV basically
    selector = SelectFromModel(regr)

    
    train_X = selector.fit_transform(train_X, train_y)
    test_X = selector.transform(test_X)


    # Train the model using the training sets
    regr.fit(train_X, train_y)
    
    
    
    # Make predictions using the testing set
    pred = regr.predict(test_X)
    
    # The coefficients
    # print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Root Mean squared error: %.2f"
        %  sqrt(mean_squared_error(test_y, pred)))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(test_y, pred))
    # print('Internal score: %.2f' % regr.score(X, y))
    print('Internal score: %.2f' % regr.score(test_X, test_y))
    
    # The rest essentially calculates the answers for the actual thing
    actual_file = "tcd ml 2019-20 income prediction test (without labels).csv"
    finalData = pd.read_csv(actual_file, header = 0)
    finalData = finalData.iloc[:,:-1]
    finalData = clean_data(finalData, ohe)
    finalData = scaler.transform(finalData)
    finalData = selector.transform(finalData)
    
    #testing_file2 = "Testing2.csv"
    #finalData.to_csv(testing_file2, index=False)
    
    results = regr.predict(finalData)
    output_file = "tcd ml 2019-20 income prediction submission file.csv"
    output =  pd.read_csv(output_file, header = 0, index_col=False)
    #print(output)
    output["Income"] = results
    #print(output)
    output.to_csv(output_file, index=False)
    

def clean_data(data, ohe):
    """Clean the final data using the given one hot encoder"""
    data = process_features(data)
     
    feature_arr = ohe.transform(data[["University Degree","Country"]]).toarray()
 
    names = ohe.get_feature_names(["University Degree","Country"])
    
    data2 = ohe.transform(data)
         
    df2 = pd.DataFrame(feature_arr, columns = names)
     
     # data2 = df2
     
    data2 = pd.concat([df2.reset_index(drop=True),
    data[["Year of Record","Age","Size of City","Body Height [cm]"]].reset_index(drop=True)], axis=1)
    
    return data2
    
    

def clean_train_data(data):
    """
    Initial training data processing
    """ 
    data = process_features(data)    
    # continous version
    ohe = OneHotEncoder(categories='auto', handle_unknown = 'ignore')
    feature_arr = ohe.fit_transform(data[["University Degree","Country"]]).toarray()
        
    names = ohe.get_feature_names(["University Degree","Country"])
    
    df2 = pd.DataFrame(feature_arr, columns = names)
    data2 = pd.concat([df2.reset_index(drop=True),
    data[["Year of Record","Age","Size of City","Body Height [cm]","Income in EUR"]].reset_index(drop=True)], axis=1)
    
    return (data2,ohe)
    
def process_features(data):
        # Has no relation to data
    data = data.drop('Instance', 1)
    data = data.drop('Profession', 1)
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
    data["Profession"] = data["Profession"].fillna("Unknown")
    data["Country"] = data["Country"].fillna("Unknown")
#     
    # Fill all possible columns with column mean
    data = data.fillna(data.mean())
    # Fill all column nans with column mode
    for column in data.columns:
        if len(data[column].mode()) > 0:
            data[column].fillna(data[column].mode()[0], inplace=True)
    data = data.fillna(method="ffill")
    
    return data