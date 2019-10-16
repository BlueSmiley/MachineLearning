import pandas as pd
from sklearn import  linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.feature_selection import RFECV
from category_encoders import LeaveOneOutEncoder
from sklearn import neighbors


def main():
     
    input_file = "tcd ml 2019-20 income prediction training (with labels).csv"
    df = pd.read_csv(input_file, header = 0)

    # Split into train and test
    train, test = train_test_split(df, test_size=0.2)
    # use a standard scaler to scale data 
    scaler = StandardScaler()
    # Process the features and trained encoder
    (train,encoder) = clean_train_data(train)
    
    train_y = train.iloc[:,-1]
    train_X = scaler.fit_transform(train.iloc[:,:-1])
    test_y = test.iloc[:,-1]
    test_X = test.iloc[:,:-1]
    # Preprocess the test data using trained encoder
    test_X = clean_data(test_X, encoder)
    test_X = scaler.transform(test_X)

    regr = neighbors.KNeighborsRegressor(n_neighbors=10, weights= 'distance')
    lasso = linear_model.LassoCV(cv=5, verbose = 0)
    
    # Use recursive feature selection with l2 norm of Lasso
    selector = RFECV(lasso, step=1, cv=3)
    
    # Select features from training and test set
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
    
    # The rest essentially calculates the answers for the actual thing
    actual_file = "tcd ml 2019-20 income prediction test (without labels).csv"
    finalData = pd.read_csv(actual_file, header = 0)
    finalData = finalData.iloc[:,:-1]
    finalData = clean_data(finalData, encoder)
    finalData = scaler.transform(finalData)
    finalData = selector.transform(finalData)
    
    results = regr.predict(finalData)
    output_file = "tcd ml 2019-20 income prediction submission file.csv"
    output =  pd.read_csv(output_file, header = 0, index_col=False)
    output["Income"] = results
    output.to_csv(output_file, index=False)
    

def clean_data(data, encoder):
    """Clean the final data using the given one hot encoder"""
    data = data.reset_index(drop=True)
    data = process_features(data) 
    data2 = encoder.transform(data)   
    return data2
    
    

def clean_train_data(data):
    """
    Initial training data processing
    """ 
    
    data = data.reset_index(drop=True)
    train_y = data.iloc[:,-1]
    train_y = train_y.reset_index(drop=True)
    train_X = data.iloc[:,:-1]
    
    train_X = process_features(train_X)
    
   
    
    encoder = LeaveOneOutEncoder(cols = ["Hair Color",
         "Wears Glasses","University Degree","Gender","Country","Profession"])

    encoder.fit(train_X,train_y)
    data2 =  pd.concat([encoder.transform(train_X,train_y).reset_index(drop=True),train_y.reset_index(drop=True)],axis=1)
    
    
    return (data2,encoder)
    
    
def process_features(data):
    
    # Has no relation to data
    data = data.drop('Instance', 1)
    # data = data.drop('Profession', 1)
     
    # Constrain Hair color
    mapping = {"Black": 1, "Blond": 2, "Brown": 3}
    data["Hair Color"] = data["Hair Color"].map(mapping)
    # Four buckets. Looking at data most are one of three or basically undefined.
    # To interpolate no correlation between not giving data hence assume mode
    #data["Hair Color"] = data["Hair Color"].fillna(data["Hair Color"].mode()[0])
    # To interpolate as missing values implies new feature
    data["Hair Color"] = data["Hair Color"].fillna(4)
    
 
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
    
    return data
    
if __name__ == "__main__":
    main()