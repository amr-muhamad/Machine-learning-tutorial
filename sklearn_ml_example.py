import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib


##################################################
# Load and read red wine data
##################################################
dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url) # Read data
data = pd.read_csv(dataset_url, sep=';') # Read with semi-colon separation
#print(data.head()) # Data read check
#print(data.shape) # Check data shape
#print(data.describe()) # Data summary statistics


##################################################
# Split data into training and test sets
##################################################
Y = data.quality
X = data.drop('quality', axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123, stratify=Y)


##################################################
# Declare data preprocessing steps
##################################################
# Lazy way of scalling data
#X_train_scaled = preprocessing.scale(X_train)  # This way won't be used
#print(X_train_scaled)

scaler = preprocessing.StandardScaler().fit(X_train) # Fitting the transformer API

X_train_scaled = scaler.transform(X_train) # Applying transformer to training data
#print(X_train_scaled.mean(axis=0))
#print(X_train_scaled.std(axis=0))

X_test_scaled = scaler.transform(X_test) # Applying transformer to test data
#print(X_test_scaled.mean(axis=0))
#print(X_test_scaled.std(axis=0))

# Pipeline with preprocessing and model
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))


##################################################
# Declare hyperparameters to tune
##################################################
#print(pipeline.get_params()) # List tunable hyper-parameters
# Declaring hyper-parameters to tune
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'], 'randomforestregressor__max_depth': [None, 5, 3, 1]}


##################################################
# Tune model using a cross-validation pipeline
##################################################
# Sklearn cross-validation with pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, Y_train) # Fit and tune model
#print(clf.best_params_) # Check the best set of parameters found using CV


##################################################
# Refit on the entire training set
##################################################
#print(clf.refit) # Confirm model will be retained


##################################################
# Evaluate model pipeline on test data
##################################################
Y_pred = clf.predict(X_test) # Predict a new sit of data

# Evaluate model performance
print(r2_score(Y_test, Y_pred))
print(mean_squared_error(Y_test, Y_pred))


##################################################
# Save model for future use
##################################################
#joblib.dump(clf, 'rf_regressor.pkl')