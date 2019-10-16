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
#from catboost import Pool, CatBoostRegressor



# Great math logic ahead
# (80-15)/5 = 13
ageBinner = KBinsDiscretizer(n_bins=13, encode='ordinal', strategy='quantile')
# (210 - 140)/10 = 7
heightBinner = KBinsDiscretizer(n_bins=7, encode='ordinal', strategy='quantile')
# (3000000- 10000)/ 100000 == 30
cityBinner = KBinsDiscretizer(n_bins=30, encode='ordinal', strategy='quantile')

incomeBinner = KBinsDiscretizer(n_bins=1000, encode='ordinal', strategy='quantile')
    
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


kms_per_radian = 6371.0088
epsilon = 800 / kms_per_radian
dbscan = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine')
rep_points = []

def main():
     
    input_file = "tcd ml 2019-20 income prediction training (with labels).csv"
    # Uncomment to find current path directory, for debuggin reasons
    # print(os.path.isfile(input_file)) 
    
    df = pd.read_csv(input_file, header = 0)

    #  (df,ohe) = clean_data(df)
    
    # remove nans in column
    # df = df[np.isfinite(df['Year of Record'])]

    train, test = train_test_split(df, test_size=0.2) 
    # train = process_features(train)
    # #Output state of df to csv for debugging reasons
    # testing_file = "Testing.csv"
    # train.to_csv(testing_file, index=False)
    scaler = StandardScaler()
    (train,ohe,rep_points) = clean_train_data(train)
    # testing_file = "Testing.csv"
    # train.to_csv(testing_file, index=False)
    print(np.shape(train))
    print(np.shape(test))
    #train = reject_all_outliers
    
    # To potentially threshold the features based on variance of values in column
    # sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    
    
    train_y = train.iloc[:,-1]
    # print(train_y.columns)
    # print(train_y["Income in EUR"])
    
    # To discretize incomes
    # train_y = pd.DataFrame(train_y,columns = ["Income in EUR"])
    # train_y["Income in EUR"] = incomeBinner.fit_transform(train_y[["Income in EUR"]])
    # train_y["Income in EUR"] = incomeBinner.inverse_transform(train_y[["Income in EUR"]])
    # train_y = train_y.iloc[:,-1]
    
    train_X = scaler.fit_transform(train.iloc[:,:-1])
    
    # pca = PCA(svd_solver='auto', n_components="mle")
    # pca.fit(train_X)
    # n_components_pca_mle = pca.n_components_
    # print("best n_components by PCA MLE = %d" % n_components_pca_mle)
    
    # train_X = pca.transform(train_X)
    
    
    # train_X = sel.fit_transform(train_X)
    test_y = test.iloc[:,-1]
    
    test_X = test.iloc[:,:-1]
    test_X = clean_data(test_X, ohe,rep_points)
    test_X = scaler.transform(test_X)
    # test_X = pca.transform(test_X)
    # test_X = sel.transform(test_X)
    
    print(np.shape(train_X))
    print(np.shape(test_X))
    # 
    # catboost = CatBoostRegressor(
    #                       task_type="GPU",
    #                        devices='0:1')

    regr = neighbors.KNeighborsRegressor(n_neighbors=10, weights= 'distance')
    lasso = linear_model.LassoCV(cv=5, verbose = 0)
    
    # model = linear_model.SGDRegressor()
    # # Grid search - this will take about 1 minute.
    # param_grid = {
    #      'alpha': 10.0 ** -np.arange(1, 7),
    #      'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
    #      'penalty': ['l2', 'l1', 'elasticnet'],
    #      'learning_rate': ['constant', 'optimal', 'invscaling'],
    # }
    # sgd = GridSearchCV(model, param_grid)
    # regr = DecisionTreeRegressor()
    
    # Good performance
    # trees = ExtraTreesRegressor()
    
    # regr = linear_model.MultiTaskElasticNetCV(cv=5)
    #lasso = linear_model.LassoLarsCV(cv=5)
    # regr = linear_model.ElasticNetCV(cv=5)
    #regr = ensemble.RandomForestRegressor(n_estimators=1000)
    # regr = ensemble.GradientBoostingRegressor(n_estimators=1000, subsample=0.5)
    # regr = ensemble.VotingRegressor(estimators=[('knn', knn), ('lr', sgd)])
    # regr = linear_model.SGDRegressor(alpha =0.0001,average=False,early_stopping=False,
    #     epsilon=0.1,eta0=0.0001,fit_intercept=True,l1_ratio=0.15,learning_rate='invscaling',
    #     loss='squared_loss',max_iter=1000,n_iter_no_change=5,penalty='l2',power_t=0.25,
    #     random_state=None,shuffle=True,tol=0.001,validation_fraction=0.1,verbose=2,
    #     warm_start=False)
    #regr = linear_model.RidgeCV(cv=5)
    # regr = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
    #             max_depth = 5, alpha = 10, n_estimators = 10)
    # regr = svm.SVR(gamma='scale')
    # Never use this it, takes too long and requires meta transofrmer as well
    selector = RFECV(lasso, step=1, cv=3)
    
    # This works ok but requires LassoCV basically
    #selector = SelectFromModel(regr)
    
    # Very bad performance
    #selector = cluster.FeatureAgglomeration(n_clusters=150)
    
    # the worst performance so far :(((
    #selector = SelectFromModel(ExtraTreesRegressor(n_estimators=100))
    
    # Performed ok
    #selector = SelectKBest(f_regression,k = 150)
    
    train_X = selector.fit_transform(train_X, train_y)
    test_X = selector.transform(test_X)

    print(np.shape(train_X))
    print(np.shape(test_X))

    # Train the model using the training sets
    regr.fit(train_X, train_y)
    
    
    
    # Make predictions using the testing set
    pred = regr.predict(test_X)
    X = test["Country"]
    le = LabelEncoder()
    X = le.fit_transform(X) 
    plt.scatter(X, test_y,  color='red', alpha = 0.005)
    plt.scatter(X, pred, color='blue', alpha= 0.005)
    plt.ylabel('Income')
    plt.xlabel('Country')
    plt.title('Predicting income')
    plt.show()
    # catboost.fit(train_X, train_y)
    # catpreds = catboost.predict(test_X)
    
    # The coefficients
    # print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Root Mean squared error: %.2f"
        %  sqrt(mean_squared_error(test_y, pred)))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(test_y, pred))
    # print('Internal score: %.2f' % regr.score(X, y))
    #print('Internal score: %.2f' % regr.score(test_X, test_y))
    
    # The rest essentially calculates the answers for the actual thing
    # actual_file = "tcd ml 2019-20 income prediction test (without labels).csv"
    # finalData = pd.read_csv(actual_file, header = 0)
    # finalData = finalData.iloc[:,:-1]
    # finalData = clean_data(finalData, ohe, rep_points)
    # finalData = scaler.transform(finalData)
    # finalData = selector.transform(finalData)
    # 
    # #testing_file2 = "Testing2.csv"
    # #finalData.to_csv(testing_file2, index=False)
    # 
    # results = regr.predict(finalData)
    # output_file = "tcd ml 2019-20 income prediction submission file.csv"
    # output =  pd.read_csv(output_file, header = 0, index_col=False)
    # #print(output)
    # output["Income"] = results
    # #print(output)
    # output.to_csv(output_file, index=False)
    

def clean_data(data, ohe,rep_points):
    """Clean the final data using the given one hot encoder"""
    data = process_features(data)
    # 
    # coords = data.as_matrix(columns=['X', 'Y'])
    # data = data.drop('Y', 1)
    # #data["X"] = dbscan.predict(np.radians(coords))
    # data = data.drop('X', 1)
    #cluster_labels = dbscan.labels_
    #dist_from_clusters = pd.DataFrame(createDistColumns(data,rep_points))
    
    #data = data.join(dist_from_clusters)
    
    # data = dist_from_origin(data)
    # data = data[["X","Y","Z"]]
    # data = data.reset_index(drop=True)
    
    # new_column = dbscan_predict(dbscan,data[["X","Y"]])
    # new_column = pd.DataFrame(new_column, columns = ["Cluster"])
    # data = data.join(new_column)
    
    #data = new_column
    # data = data.join(dist_from_clusters)
    # 
    # dists = dist_from_origin(data)
    data = data.drop('Y',1)
    data = data.drop('X',1)
    # data = data.join(dists)
    #data2 = dist_from_clusters
    
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
    
    
    train_y = data.iloc[:,-1]
    train_y = train_y.reset_index(drop=True)
    train_X = data.iloc[:,:-1]
    
    train_X = process_features(train_X)
    
    rep_points = train_X[["X","Y","Country"]].drop_duplicates()
    # coords = train_X[["X","Y"]].drop_duplicates().as_matrix(columns=['Y', 'X'])
    # dbscan.fit_predict(np.radians(coords))
    # 
    # cluster_labels = dbscan.labels_
    # num_clusters = len(set(cluster_labels))
    # print('Number of clusters: {}'.format(num_clusters))
    # clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
    # centermost_points = clusters.map(get_centermost_point)
    # lats, lons = zip(*centermost_points)
    # rep_points = pd.DataFrame({'lon':lons, 'lat':lats})
    # # dist_from_clusters = pd.DataFrame(createDistColumns(train_X,rep_points))
    # # print("Extra columns:" + str(np.shape(dist_from_clusters)))
    # 
    # 
    # #train_X = train_X.join(new_column)
    # 
    # dists = dist_from_origin(train_X)
    # # 
    # # #print(train_X)
    # # # train_X = train_X[["X","Y","Z"]]
    # # # train_X = train_X.reset_index(drop=True)
    # # 
    # # new_column = dbscan_predict(dbscan,train_X[["X","Y"]])
    # # new_column = pd.DataFrame(new_column, columns = ["Cluster"])
    train_X = train_X.drop("X",1)
    train_X = train_X.drop("Y",1)
    # # train_X = train_X.join(new_column)
    # #train_X = new_column
    # #train_X = train_X.join(dist_from_clusters)
    # train_X = train_X.join(dists)
    #print(train_X)
    #train_X = dist_from_clusters
    
    
    # testing_file = "Testing.csv"
    # dist_from_clusters.to_csv(testing_file, index=False)
    
    # data = data.drop('X', 1)
    
    
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
         "Wears Glasses","University Degree","Gender","Country","Profession"])
    #ohe = LeaveOneOutEncoder(cols = ["Cluster"])
    
    # Stop blowing up my processing :(
    #data = data.drop('Profession', 1)

    ohe.fit(train_X,train_y)
    #data2 = pd.concat([ohe.transform(train_X,train_y).reset_index(drop=True), train_y.reset_index(drop=True)],axis = 1)
    data2 =  pd.concat([ohe.transform(train_X,train_y).reset_index(drop=True),train_y.reset_index(drop=True)],axis=1)
    
    
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
    # # 
    # testing_file = "Testing2.csv"
    # data2.to_csv(testing_file, index=False)
    # testing_file = "Testing2.csv"
    # train_X.to_csv(testing_file, index=False)
    return (data2,ohe,rep_points)
    
    
def process_features(data):
    
    # double sort to preserve order
    data = data.merge(data.merge(gdf,on="Country",how="left",sort=False))
    # testing_file = "Testing2.csv"
    # data.to_csv(testing_file, index=False)  
    mappingX = {"Eswatini":31.4659,"Central African Republic":20.9394, 
        "Czechia": 15.4730, "Laos":102.4955, "Singapore":103.8198, "Equatorial Guinea":10.2679,
        "State of Palestine":35.2332, "South Sudan":31.3070, "North Korea":127.5101,
        "Dominican Republic":-70.1627,"Bosnia and Herzegovina":17.6791,"North Macedonia":21.7453,
        "Malta":14.3754,"Mauritius":57.5522,"Comoros":43.3333,"Bahrain":50.5577,"Maldives":73.2207
        ,"Solomon Islands":160.1562,"Cabo Verde":-23.0418,"DR Congo":21.7587,"Micronesia":150.5508,
        "Grenada":-61.6790,"South Korea":127.7669,"Saint Lucia":-60.9789,"Barbados":-59.5432,
        "Kiribati":-168.7340,"Tonga":-175.1982,"Seychelles":55.4920, "Sao Tome & Principe":6.6131,"Samoa":-172.1046}
        
    mappingY = {"Eswatini":-26.5225,"Central African Republic":6.6111, 
        "Czechia": 49.8175, "Laos":19.8563, "Singapore":1.3521, "Equatorial Guinea":1.6508,
        "State of Palestine":31.9522, "South Sudan":6.8770, "North Korea":40.3399,
        "Dominican Republic":18.7357, "Bosnia and Herzegovina":43.9159,"North Macedonia":41.6086,
        "Malta":35.9375,"Mauritius":-20.3484,"Comoros":-11.6455,"Bahrain":26.0667,"Maldives":3.2028,
        "Solomon Islands":-9.6457,"Cabo Verde":16.5388,"DR Congo":-4.0383,"Micronesia":7.4256,
        "Grenada":12.1165,"South Korea":35.9078,"Saint Lucia":13.9094,"Barbados":13.1939,
        "Kiribati":-3.3704,"Tonga":-21.1790,"Seychelles":-4.6796, "Sao Tome & Principe":0.1864,"Samoa":-13.7590}
        
    data["X"] = data.apply(
        lambda row: mappingX[row["Country"]] if(row["Country"] in mappingX) else row["X"], axis=1)
    data["Y"] = data["Y"] = data.apply(
        lambda row: mappingY[row["Country"]] if(row["Country"] in mappingY) else row["Y"], axis=1)
        
    print(data[data["X"].isna()]["Country"])
    data["X"] = data["X"].fillna(0)
    data["Y"] = data["Y"].fillna(0)
    # data = data.drop('X', 1)
    # data = data.drop('Y', 1)
    
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
    
def compute_scores(X,n_components):
    pca = PCA(svd_solver='auto')

    pca_scores = []
    for n in n_components:
        pca.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X, cv=2)))

    return pca_scores
    
def dist_from_origin(data):
    x = np.cos(data["Y"]) * np.cos(data["X"])
    y = np.cos(data["Y"]) * np.sin(data["X"]) 
    z = np.sin(data["Y"]) 
    
    df = pd.DataFrame(columns=['X', 'Y', 'Z'])
    # 
    # print(np.shape(x))
    # print(np.shape(y))
    # print(np.shape(z))
    df["X"] = x
    df["Y"] = y
    df["Z"] = z
    # print(df)
    # print(z)
    return df


def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)

    
def reject_outliers(data,column):
    u = np.mean(data[column])
    s = np.std(data[column])
    data_filtered = data[(data[column] > u-4*s) & (data[column] < u+4*s)]
    return data_filtered
    
    
def createDistColumns(df,rep_points):
    # print("repPoints:" +str(np.shape(rep_points)))
    columns = df.apply(lambda row: rep_points.apply(lambda comparison: clusterDistances(row,comparison),axis=1).T,axis=1)
    return columns
    
def clusterDistances(row,comparison):
    return great_circle(row.as_matrix(columns=['Y', 'X']), comparison).km
    
def dbscan_predict(dbscan_model, X_new): 
    # X_new.apply(lambda row:print(row.as_matrix(columns=['Y', 'X'])))
    y_new = X_new.apply(lambda row: dbscan_helper(dbscan_model,row.as_matrix(columns=['Y', 'X'])),axis=1)
    return y_new
    
def dbscan_helper(dbscan_model, coord):
    #print(coord)
     # Find a core sample closer than EPS
    cur_lowest = - 1
    curLabel = None
    for i, x_core in enumerate(dbscan_model.components_):
        dist = great_circle(coord, x_core)
        if dist < cur_lowest or cur_lowest == -1:
            # Assign label of x_core to x_new
            cur_lowest = dist 
            curLabel = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
    return curLabel

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
    
    
def plotCountry(df):
    y = df["Income in EUR"]
    X = df["Country"]
    
    le = LabelEncoder()
    X = le.fit_transform(X) 

    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X.reshape(-1, 1), y)
    # Make predictions using the testing set
    pred = regr.predict(X.reshape(-1, 1))

    # Plot outputs
    plt.scatter(X, y,  color=[[0,0,0,0.005]])
    plt.plot(X, pred, color='blue', linewidth=3)
    plt.ylabel('Income')
    plt.xlabel('CitySize')
    plt.title('Predicting income')
    plt.show()
    
def plotProfession(df):
    #df = df[np.isfinite(df['Profession'])]
    
    df = df.dropna()
    y = df["Income in EUR"]
    X = df["Profession"]
    
    le = LabelEncoder()
    X = le.fit_transform(X) 

    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X.reshape(-1, 1), y)
    # Make predictions using the testing set
    pred = regr.predict(X.reshape(-1, 1))

    # Plot outputs
    plt.scatter(X, y,  color=[[0,0,0,0.005]])
    plt.plot(X, pred, color='blue', linewidth=3)
    plt.ylabel('Income')
    plt.xlabel('Profession')
    plt.title('Predicting income')
    plt.show()
    
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