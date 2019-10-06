import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from math import sqrt

def main():
    input_file = "tcd ml 2019-20 income prediction training (with labels).csv"
    # print(os.path.isfile(input_file)) 
    
    df = pd.read_csv(input_file, header = 0)
    # If we only interest in numeric data
    #df = df._get_numeric_data()
    # create a numpy array with the numeric values for input into scikit-learn
    #data = df.as_matrix()

    df = clean_data(df)
    
    
    # remove nans in column
    # df = df[np.isfinite(df['Year of Record'])]
    
    
    #msk = np.random.rand(len(df)) < 0.8
     
    train, test = train_test_split(df, test_size=0.2)
    
    train_y = train.iloc[:,-1]
    train_X = train.iloc[:,:-1]
    test_y = test.iloc[:,-1]
    test_X = test.iloc[:,:-1]
    #print(np.shape(X))
    #print(np.shape(y))

    regr = linear_model.RidgeCV(alphas=[1e-10,0.1, 1.0, 10.0,20], cv=5,scoring='neg_mean_squared_error',normalize = True)

    # Train the model using the training sets
    regr.fit(train_X, train_y)

    
    
    
    # Make predictions using the testing set
    pred = regr.predict(test_X)
    
    actual_file = "tcd ml 2019-20 income prediction test (without labels).csv"
    finalData = pd.read_csv(actual_file, header = 0)
    finalData = cleanFinalData(finalData)
    
    results = regr.predict(finalData)
    output_file = "tcd ml 2019-20 income prediction submission file.csv"
    output =  pd.read_csv(output_file, header = 0, index_col=False)
    #print(output)
    output["Income"] = results
    #print(output)
    output.to_csv(output_file, index=False)

    #plotGender(df)
    
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Root Mean squared error: %.2f"
        %  sqrt(mean_squared_error(test_y, pred)))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(test_y, pred))
    
    # Plot outputs
    # plt.scatter(X.iloc[:,-1], y,  color='black')
    # plt.plot(X.iloc[:,-1], pred, color='blue', linewidth=3)
    
   # plt.xticks(())
    #plt.yticks(())
    
    # plt.show()
    
    
    
def cleanFinalData(data):
    
    # too messy and unrestrained for me to work with rn
    data = data.drop('Profession', 1)
    # Has no relation to data
    data = data.drop('Instance', 1)
    
    # Constrain Hair color
    mapping = {"Black": 1, "Blond": 2, "Brown": 3}
    data["Hair Color"] = data["Hair Color"].map(mapping)
    # Four buckets
    data["Hair Color"] = data["Hair Color"].fillna(4)
    
    # Constrain Degree
    mapping = {"Bachelor": 1, "Master": 2, "No": 3}
    data["University Degree"] = data["University Degree"].map(mapping)
    # Four buckets
    data["University Degree"] = data["University Degree"].fillna(4)
    
    # constrain Gender
    mapping = {"male": 1, "female": 2}
    data["Gender"] = data["Gender"].map(mapping)
    # Four buckets
    data["Gender"] = data["Gender"].fillna(3)
    
    # drop nans
    #data = data.dropna()
    
    # put error value for any remaining nas
    data = data.fillna(-1)
    
    data["Country"] = pd.get_dummies(data["Country"])
    ohe = OneHotEncoder(categories='auto')
    feature_arr = ohe.fit_transform(data[["Hair Color",
        "Wears Glasses","University Degree","Country","Gender"]]).toarray()

    data = np.append(feature_arr, 
        data[["Year of Record","Age","Size of City","Body Height [cm]"]],axis = 1)
    data = pd.DataFrame(data)
    return data



# In future I think plotting all the data might be unnescary but for a restrained problem such as this it's nice to visualise data
   
def plotGender(df):
    mapping = {"male": 1, "female": 2}
    df["Gender"] = df["Gender"].map(mapping)
    # Four buckets
    df = df.fillna(3)
    H1 = df.loc[df["Gender"] == 1]["Income in EUR"]
    H2 = df.loc[df["Gender"] == 2]["Income in EUR"]
    H3 = df.loc[df["Gender"] == 3]["Income in EUR"]
    bins = 50
    # Enable this for overlapping
    # plt.hist(black,bins,label = "black",color = "black", alpha = 0.2)
    # plt.hist(brown,bins,label = "brown",color = "brown", alpha = 0.2)
    # plt.hist(blond,bins,label = "blond",color = "yellow", alpha = 0.2)
    
    # Enable this for non overlapping
    plt.hist([H1,H2,H3],
       label= ["Male","Female","other"],
       bins=50)
    plt.legend()
    plt.axvline(np.mean(H1), color='red', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(H2), color='green', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(H3), color='blue', linestyle='dashed', linewidth=1)
    plt.xlabel('Gender')
    plt.ylabel('count')
    plt.show()

def plotCity(df):
    # remove nans in column
    df = df[np.isfinite(df['Size of City'])]
    
    y = df["Income in EUR"]
    X = df["Size of City"]
    

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
       
def plotHeightHist(df):
    H1= df.loc[df["Body Height [cm]"] < 100]["Income in EUR"]
    H2 = df.loc[(df["Body Height [cm]"] >= 100) & (df["Body Height [cm]"] < 120)]["Income in EUR"]
    H3 = df.loc[(df["Body Height [cm]"] >= 120) & (df["Body Height [cm]"] < 140)]["Income in EUR"]
    H4 = df.loc[(df["Body Height [cm]"] >= 140) & (df["Body Height [cm]"] < 160)]["Income in EUR"]
    H5 = df.loc[(df["Body Height [cm]"] >= 160) & (df["Body Height [cm]"] < 180)]["Income in EUR"]
    H6 = df.loc[df["Body Height [cm]"] > 180]["Income in EUR"]
    bins = 50
    
    # Enable for overlapping
    #plt.hist(youngster,bins,label = "<20", alpha = 0.2)
    #plt.hist(twenties,bins,label = "20-30", alpha = 0.2)
    #plt.hist(thirties,bins,label = "30-40", alpha = 0.2)
    #plt.hist(fourties,bins,label = "40-50", alpha = 0.2)
    #plt.hist(fifties,bins,label = "50-60", alpha = 0.2)
    #plt.hist(other,bins,label = "60+", alpha = 0.2)
    
    #Enable for non-overlapping
    plt.hist([H1,H2,H3,H4,H5,H6],
         label= ["<100","100-120","120-140","140-160","160-180","180+"],
         bins=bins)
    plt.legend()
    plt.axvline(np.mean(H1), color='red', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(H2), color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(H3), color='green', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(H4), color='yellow', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(H5), color='brown', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(H6), color='pink', linestyle='dashed', linewidth=1)
    plt.ylabel('count')
    plt.show()
    
def plotAgeHist(df):
    youngster = df.loc[df["Age"] < 20]["Income in EUR"]
    twenties = df.loc[(df["Age"] >= 20) & (df["Age"] < 30)]["Income in EUR"]
    thirties = df.loc[(df["Age"] >= 30) & (df["Age"] < 40)]["Income in EUR"]
    fourties = df.loc[(df["Age"] >= 40) & (df["Age"] < 50)]["Income in EUR"]
    fifties = df.loc[(df["Age"] >= 50) & (df["Age"] < 60)]["Income in EUR"]
    other = df.loc[df["Age"] > 60]["Income in EUR"]
    bins = 50
    
    # Enable for overlapping
    #plt.hist(youngster,bins,label = "<20", alpha = 0.2)
    #plt.hist(twenties,bins,label = "20-30", alpha = 0.2)
    #plt.hist(thirties,bins,label = "30-40", alpha = 0.2)
    #plt.hist(fourties,bins,label = "40-50", alpha = 0.2)
    #plt.hist(fifties,bins,label = "50-60", alpha = 0.2)
    #plt.hist(other,bins,label = "60+", alpha = 0.2)
    
    #Enable for non-overlapping
    plt.hist([youngster,twenties,thirties,fourties,fifties,other],
         label= ["<20","20-30","30-40","40-50","50-60","60+"],
         bins=bins)
    plt.legend()
    plt.axvline(np.mean(youngster), color='red', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(twenties), color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(thirties), color='green', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(fourties), color='yellow', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(fifties), color='brown', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(other), color='pink', linestyle='dashed', linewidth=1)
    plt.ylabel('count')
    plt.show()
    
def plotHeight(df):
    
    y = df["Income in EUR"]
    X = df["Body Height [cm]"]
    

    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X.reshape(-1, 1), y)
    # Make predictions using the testing set
    pred = regr.predict(X.reshape(-1, 1))

    # Plot outputs
    plt.scatter(X, y,  color=[[0,0,0,0.005]])
    plt.plot(X, pred, color='blue', linewidth=3)
    plt.ylabel('Income')
    plt.xlabel('Height')
    plt.title('Predicting income')
    plt.show()
    
def plotDegree(df):
    mapping = {"Bachelor": 1, "Master": 2, "No": 3}
    df["University Degree"] = df["University Degree"].map(mapping)
    # Four buckets
    df = df.fillna(4)
    bachelors = df.loc[df["University Degree"] == 1]["Income in EUR"]
    masters = df.loc[df["University Degree"] == 2]["Income in EUR"]
    none = df.loc[df["University Degree"] == 3]["Income in EUR"]
    other = df.loc[df["University Degree"] == 4]["Income in EUR"]
    bins = 50
    
    # Enable for overlapping
    #plt.hist(bachelors,bins,label = "bachelors",color = "red", alpha = 0.2)
    #plt.hist(masters,bins,label = "masters",color = "blue", alpha = 0.2)
    #plt.hist(none,bins,label = "none",color = "green", alpha = 0.2)
    #plt.hist(other,bins,label = "other",color = "yellow", alpha = 0.2)
    
    #Enable for non-overlapping
    plt.hist([bachelors,masters,none,other],
         color=["red", "blue","green","yellow"],
         label= ["bachelors","masters","none","other"],
         bins=bins)
    plt.legend()
    plt.axvline(np.mean(bachelors), color='red', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(masters), color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(none), color='green', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(other), color='yellow', linestyle='dashed', linewidth=1)
    plt.ylabel('count')
    plt.show()
    
def plotHair(df):
    mapping = {"Black": 1, "Blond": 2, "Brown": 3}
    df["Hair Color"] = df["Hair Color"].map(mapping)
    # Four buckets
    df = df.fillna(4)
    black = df.loc[df["Hair Color"] == 1]["Income in EUR"]
    blond = df.loc[df["Hair Color"] == 2]["Income in EUR"]
    brown = df.loc[df["Hair Color"] == 3]["Income in EUR"]
    other = df.loc[df["Hair Color"] == 4]["Income in EUR"]
    bins = 50
    plt.hist(black,bins,label = "black",color = "black", alpha = 0.2)
    plt.hist(brown,bins,label = "brown",color = "brown", alpha = 0.2)
    plt.hist(blond,bins,label = "blond",color = "yellow", alpha = 0.2)
    plt.hist(other,bins,label = "other",color = "green", alpha = 0.2)
    # Enable this for non overlapping
    #plt.hist([black,blond,brown,other],
    #    color=["black", "yellow","brown","green"],
    #   label= ["black","blond","brown","other"],
    #   bins=50)
    plt.legend()
    plt.axvline(np.mean(black), color='black', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(brown), color='brown', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(blond), color='yellow', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(other), color='green', linestyle='dashed', linewidth=1)
    plt.ylabel('count')
    plt.show()

def plotGlasses(df):
    glass = df.loc[df["Wears Glasses"] == 1]["Income in EUR"]
    noGlass = df.loc[df["Wears Glasses"] == 0]["Income in EUR"]
    plt.figure()
    plt.hist([glass,noGlass],
         color=["red", "blue"],
         label=["glass","no glass"],
         bins=30)
    plt.axvline(np.mean(glass), color='red', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(noGlass), color='blue', linestyle='dashed', linewidth=1)
    plt.legend()
    plt.ylabel('Count');
    plt.show()

def plotRecord(df):
    # remove nans in column
    df = df[np.isfinite(df['Year of Record'])]
    df = df[np.isfinite(df['Body Height [cm]'])]
    
    y = df["Income in EUR"]
    X = df["Year of Record"]
    

    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X.reshape(-1, 1), y)
    # Make predictions using the testing set
    pred = regr.predict(X.reshape(-1, 1))

    # Plot outputs
    plt.scatter(X, y,  color=[[0,0,0,0.005]])
    plt.plot(X, pred, color='blue', linewidth=3)
    plt.ylabel('Income')
    plt.xlabel('Year')
    plt.title('Predicting income')
    plt.show()

def plotAge(df):
    
    y = df["Income in EUR"]
    X = df["Age"]
    

    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X.reshape(-1, 1), y)
    # Make predictions using the testing set
    pred = regr.predict(X.reshape(-1, 1))

    # Plot outputs
    plt.scatter(X, y,  color=[[0,0,0,0.005]])
    plt.plot(X, pred, color='blue', linewidth=3)
    plt.ylabel('Income')
    plt.xlabel('Age')
    plt.title('Predicting income')
    plt.show()

def clean_data(data):
    data = reject_outliers(data,"Income in EUR")
    data = reject_outliers(data,"Age")
    data = reject_outliers(data,"Body Height [cm]")
    data = reject_outliers(data,"Size of City")
    data = reject_outliers(data,"Year of Record")
    
    # too messy and unrestrained for me to work with rn
    data = data.drop('Profession', 1)
    # Has no relation to data
    data = data.drop('Instance', 1)
    
    # Constrain Hair color
    mapping = {"Black": 1, "Blond": 2, "Brown": 3}
    data["Hair Color"] = data["Hair Color"].map(mapping)
    # Four buckets
    data["Hair Color"] = data["Hair Color"].fillna(4)
    
    # Constrain Degree
    mapping = {"Bachelor": 1, "Master": 2, "No": 3}
    data["University Degree"] = data["University Degree"].map(mapping)
    # Four buckets
    data["University Degree"] = data["University Degree"].fillna(4)
    
    # constrain Gender
    mapping = {"male": 1, "female": 2}
    data["Gender"] = data["Gender"].map(mapping)
    # Four buckets
    data["Gender"] = data["Gender"].fillna(3)
    
    # drop nans
    #data = data.dropna()
    
    # put error value for any remaining nas
    data = data.fillna(-1)
    
    data["Country"] = pd.get_dummies(data["Country"])
    ohe = OneHotEncoder(categories='auto')
    feature_arr = ohe.fit_transform(data[["Hair Color",
        "Wears Glasses","University Degree","Country","Gender"]]).toarray()

    data = np.append(feature_arr, 
        data[["Year of Record","Age","Size of City","Body Height [cm]","Income in EUR"]],axis = 1)
    data = pd.DataFrame(data)
    return data

def reject_outliers(data,column):
    u = np.mean(data[column])
    s = np.std(data[column])
    data_filtered = data[(data[column] > u-2*s) & (data[column] < u+2*s)]
    return data_filtered
    
if __name__ == "__main__":
    main()