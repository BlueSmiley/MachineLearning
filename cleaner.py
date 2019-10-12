# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
import geopandas as gpd
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


# Great math logic ahead
# (80-15)/5 = 13
ageBinner = KBinsDiscretizer(n_bins=13, encode='ordinal', strategy='quantile')
# (210 - 140)/10 = 7
heightBinner = KBinsDiscretizer(n_bins=7, encode='ordinal', strategy='quantile')
# (3000000- 10000)/ 100000 == 30
cityBinner = KBinsDiscretizer(n_bins=30, encode='ordinal', strategy='quantile')
    
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

# Access built-in Natural Earth data via GeoPandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Get a list (dataframe) of country centroids
gdf = pd.DataFrame(columns=["X","Y","Country"])
gdf["X"] = world.centroid.x
gdf["Y"] = world.centroid.y
gdf["Country"] = world.name


def main():
     
    input_file = "tcd ml 2019-20 income prediction training (with labels).csv"
    # Uncomment to find current path directory, for debuggin reasons
    # print(os.path.isfile(input_file)) 
    
    df = pd.read_csv(input_file, header = 0)

    #  (df,ohe) = clean_data(df)
    
    
    # remove nans in column
    # df = df[np.isfinite(df['Year of Record'])]

    train, test = train_test_split(df, test_size=0.2) 
    train = process_features(train)
    # Output state of df to csv for debugging reasons
    testing_file = "Testing.csv"
    train.to_csv(testing_file, index=False)
#     scaler = StandardScaler()
#     (train,ohe) = clean_train_data(train)
#     print(np.shape(train))
#     print(np.shape(test))
#     #train = reject_all_outliers
#     
#     
#     # To potentially threshold the features based on variance of values in column
#     # sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
#     
#     train_y = train.iloc[:,-1]
#     train_X = scaler.fit_transform(train.iloc[:,:-1])
#     
#     # n_components = np.arange(0, (np.shape(train_X))[1], 100)
#     
#     # pca_scores = compute_scores(train_X,n_components)
#     # n_components_pca = n_components[np.argmax(pca_scores)]
#     pca = PCA(svd_solver='auto', n_components="mle")
#     pca.fit(train_X)
#     # n_components_pca_mle = pca.n_components_
#     # 
#     # # print("best n_components by PCA CV = %d" % n_components_pca)
#     # print("best n_components by PCA MLE = %d" % n_components_pca_mle)
#     
#     # train_X = pca.transform(train_X)
#     
#     
#     # train_X = sel.fit_transform(train_X)
#     test_y = test.iloc[:,-1]
#     test_X = test.iloc[:,:-1]
#     test_X = clean_data(test_X, ohe)
#     test_X = scaler.transform(test_X)
#     # test_X = pca.transform(test_X)
#     # test_X = sel.transform(test_X)
#     
#     print(np.shape(train_X))
#     print(np.shape(test_X))
# 
#     regr = linear_model.LassoCV(cv=3, verbose = 2)
#     # regr = linear_model.SGDRegressor(alpha =0.0001,average=False,early_stopping=False,
#     #     epsilon=0.1,eta0=0.0001,fit_intercept=True,l1_ratio=0.15,learning_rate='invscaling',
#     #     loss='squared_loss',max_iter=1000,n_iter_no_change=5,penalty='l2',power_t=0.25,
#     #     random_state=None,shuffle=True,tol=0.001,validation_fraction=0.1,verbose=2,
#     #     warm_start=False)
#     # regr = linear_model.RidgeCV(cv=5)
#     # regr = svm.SVR(gamma='scale')
#     # Never use this it, takes too long and requires meta transofrmer as well
#     #selector = RFECV(regr, step=1, cv=5)
#     
#     # This works ok but requires LassoCV basically
#     selector = SelectFromModel(regr)
#     
#     # Very bad performance
#     #selector = cluster.FeatureAgglomeration(n_clusters=150)
#     
#     # the worst performance so far :(((
#     #selector = SelectFromModel(ExtraTreesRegressor(n_estimators=100))
#     
#     # Performed ok
#     #selector = SelectKBest(f_regression,k = 150)
#     
#     train_X = selector.fit_transform(train_X, train_y)
#     test_X = selector.transform(test_X)
# 
#     print(np.shape(train_X))
#     print(np.shape(test_X))
# 
#     # Train the model using the training sets
#     regr.fit(train_X, train_y)
#     
#     
#     
#     # Make predictions using the testing set
#     pred = regr.predict(test_X)
#     
#     # The coefficients
#     # print('Coefficients: \n', regr.coef_)
#     # The mean squared error
#     print("Root Mean squared error: %.2f"
#         %  sqrt(mean_squared_error(test_y, pred)))
#     # Explained variance score: 1 is perfect prediction
#     print('Variance score: %.2f' % r2_score(test_y, pred))
#     # print('Internal score: %.2f' % regr.score(X, y))
#     print('Internal score: %.2f' % regr.score(test_X, test_y))
#     # 
#     # # The rest essentially calculates the answers for the actual thing
#     # actual_file = "tcd ml 2019-20 income prediction test (without labels).csv"
#     # finalData = pd.read_csv(actual_file, header = 0)
#     # finalData = finalData.iloc[:,:-1]
#     # finalData = clean_data(finalData, ohe)
#     # finalData = scaler.transform(finalData)
#     # finalData = selector.transform(finalData)
#     # 
#     # #testing_file2 = "Testing2.csv"
#     # #finalData.to_csv(testing_file2, index=False)
#     # 
#     # results = regr.predict(finalData)
#     # output_file = "tcd ml 2019-20 income prediction submission file.csv"
#     # output =  pd.read_csv(output_file, header = 0, index_col=False)
#     # #print(output)
#     # output["Income"] = results
#     # #print(output)
#     # output.to_csv(output_file, index=False)
#     

def clean_data(data, ohe):
    """Clean the final data using the given one hot encoder"""
    data = process_features(data)
    
    # To discretise features
    # data["Age"] = ageBinner.transform(data[["Age"]])
    # data["Body Height [cm]"] = heightBinner.fit_transform(data[["Body Height [cm]"]])
    # data["Size of City"] = cityBinner.fit_transform(data[["Size of City"]])
#     
#     feature_arr = ohe.transform(data[["Hair Color","Size of City","Body Height [cm]",
#         "Wears Glasses","University Degree","Country","Gender","Year of Record","Age"]]).toarray()
# 
#     names = ohe.get_feature_names(["Hair Color","Size of City","Body Height [cm]",
#         "Wears Glasses","University Degree","Country","Gender","Year of Record","Age"])
    
    data2 = ohe.transform(data)
#     feature_arr = ohe.transform(data[["Hair Color",
#         "Wears Glasses","University Degree","Country","Gender"]]).toarray()
# 
#     names = ohe.get_feature_names(["Hair Color",
#         "Wears Glasses","University Degree","Country","Gender"])
#         
#     df2 = pd.DataFrame(feature_arr, columns = names)
#     
#     # data2 = df2
#     
#     data2 = pd.concat([df2.reset_index(drop=True),
#     data[["Year of Record","Age","Size of City","Body Height [cm]"]].reset_index(drop=True)], axis=1)
    
    return data2
    
    

def clean_train_data(data):
    """
    Initial training data processing
    """ 
    data = process_features(data)
    
    # To just remove outliers - Note Terrible performance doing it this way
    # data = reject_outliers(data,"Income in EUR")
    
    # To winsorise incomes
    # data["Income in EUR"] = data["Income in EUR"].apply(using_mstats)
    
    # To winsorise ages
    # data["Age"] = data["Age"].apply(using_mstats)
    
    # To winsorise size of city
    # data["Size of City"] = data["Size of City"].apply(using_mstats)
    
    
    #ohe = OneHotEncoder(categories='auto', handle_unknown = 'ignore')
    
    # data["Age"] = ageBinner.fit_transform(data[["Age"]])
    # 
    # data["Body Height [cm]"] = heightBinner.fit_transform(data[["Body Height [cm]"]])
    # 
    # data["Size of City"] = cityBinner.fit_transform(data[["Size of City"]])
    #     
    # feature_arr = ohe.fit_transform(data[["Hair Color","Size of City","Body Height [cm]",
    #     "Wears Glasses","University Degree","Country","Gender","Year of Record","Age"]]).toarray()
    # 
    # names = ohe.get_feature_names(["Hair Color","Size of City","Body Height [cm]",
    #     "Wears Glasses","University Degree","Country","Gender","Year of Record","Age"])
    
        
    # df2 = pd.DataFrame(feature_arr, columns = names)
    # 
    # data2 = pd.concat([df2.reset_index(drop=True),
    # data[["Income in EUR"]].reset_index(drop=True)], axis=1)
    
    # continous version
    #ohe = OneHotEncoder(categories='auto', handle_unknown = 'ignore')
    ohe = LeaveOneOutEncoder(cols = ["Hair Color",
        "Wears Glasses","University Degree","Country","Gender"])
    # Stop blowing up my processing :(
    #data = data.drop('Profession', 1)
    train_y = data.iloc[:,-1]
    train_X = data.iloc[:,:-1]
    
    ohe.fit(train_X,train_y)
    data2 = pd.concat([ohe.transform(train_X,train_y).reset_index(drop=True), train_y.reset_index(drop=True)],axis = 1)
    # feature_arr = ohe.fit_transform(data[["Hair Color",
    #     "Wears Glasses","University Degree","Country","Gender"]]).toarray()
    #     
    # names = ohe.get_feature_names(["Hair Color",
    #     "Wears Glasses","University Degree","Country","Gender"])
    # 
    # df2 = pd.DataFrame(feature_arr, columns = names)
    # data2 = pd.concat([df2.reset_index(drop=True),
    # data[["Year of Record","Age","Size of City","Body Height [cm]","Income in EUR"]].reset_index(drop=True)], axis=1)
    # 
    # 
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
    #data["Profession"] = data["Profession"].fillna("Unknown")
    data["Country"] = data["Country"].fillna("Unknown")
#     
    # Fill all possible columns with column mean
    data = data.fillna(data.mean())
    # Fill all column nans with column mode
    for column in data.columns:
        if len(data[column].mode()) > 0:
            data[column].fillna(data[column].mode()[0], inplace=True)
    data = data.fillna(method="ffill")
    
    print(np.shape(data))
    data = data.merge(gdf,on="Country",how="left")
    # Just in case
    print(np.shape(data))
    
    print(data)
    return data
    
def compute_scores(X,n_components):
    pca = PCA(svd_solver='auto')

    pca_scores = []
    for n in n_components:
        pca.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X, cv=2)))

    return pca_scores
    
def reject_outliers(data,column):
    u = np.mean(data[column])
    s = np.std(data[column])
    data_filtered = data[(data[column] > u-4*s) & (data[column] < u+4*s)]
    return data_filtered
    
def reject_all_outliers(train):
    """
    Removes all variable outliers based on two standard deviations from mean apparently
    """
    train = reject_outliers(train,"Income in EUR")
    train = reject_outliers(train,"Age")
    train = reject_outliers(train,"Body Height [cm]")
    train = reject_outliers(train,"Size of City")
    train = reject_outliers(train,"Year of Record")
    return train
    
    
#Fitting the PCA algorithm with our Data
    # pca = PCA().fit(train_X)#Plotting the Cumulative Summation of the Explained Variance
    # plt.figure()
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('Number of Components')
    # plt.ylabel('Variance (%)') #for each component
    # plt.title('Pulsar Dataset Explained Variance')
    # plt.show()
    # 
    

# clips top .5% of values and bottom .5% to top and bottom quartiles values respectively
def using_mstats(s):
    return mstats.winsorize(s, limits=[0.05, 0.05])

if __name__ == "__main__":
    main()