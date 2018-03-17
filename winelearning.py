# Provides support for more efficient numerical computation
import numpy as np
# Makes it easier to support dataframes
import pandas as pd

# Function that helps choose between models
from sklearn.model_selection import train_test_split
# Utilities for wrangling data
from sklearn import preprocessing
# Import the random forest family, broad types of models
from sklearn.ensemble import RandomForestRegressor
# Help perform cross-validation
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
# Evaluate model performance
from sklearn.metrics import mean_squared_error, r2_score
# Persist model for future use
from sklearn.externals import joblib

# Step 3 Load red wine data

# read CSV (comma separated values)
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data=pd.read_csv(dataset_url, sep=';')
#print(data.head()) // Shows the first few things

#print (data.shape)
#(1599,12) (samples, features)

# Print summary statistics
#print(data.describe())

# Step 4 Split data into training and test sets

# Separate target features (y) from input features (X)
y = data.quality
X = data.drop('quality', axis=1)

# Use Scikit-Learn's function train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,
	test_size=0.2,
	random_state=123,
	stratify=y)

# Step 5 Declare data preprocessing steps

#scaler = preprocessing.StandardScaler().fit(X_train)

#X_train_scaled = scaler.transform(X_train)
#print (X_train_scaled.mean(axis=0))
# [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
#print (X_train_scaled.std(axis=0))
# [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]


#X_test_scaled = scaler.transform(X_test)
 
#print (X_test_scaled.mean(axis=0))
# [ 0.02776704  0.02592492 -0.03078587 -0.03137977 -0.00471876 -0.04413827
#  -0.02414174 -0.00293273 -0.00467444 -0.10894663  0.01043391]
#print (X_test_scaled.std(axis=0))
# [ 1.02160495  1.00135689  0.97456598  0.91099054  0.86716698  0.94193125
#  1.03673213  1.03145119  0.95734849  0.83829505  1.0286218 ]

# Set up cross-validation pipeline
pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=100))

# Step 6 Declare hyperparameters to tune
# Model parameters can be learned from the data, while hyperparameters cannot

# List the tunable hyperparameters
#print (pipeline.get_params())
# ...
# 'randomforestregressor__criterion': 'mse',
# 'randomforestregressor__max_depth': None,
# 'randomforestregressor__max_features': 'auto',
# 'randomforestregressor__max_leaf_nodes': None,
# ...

# Declare the hyperparameters we want to tune through cross-validation
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

# Step 7 Tune model using a cross-validation pipeline
clf = GridSearchCV(pipeline,hyperparameters,cv=10)
clf.fit(X_train,y_train)
#print (clf.best_params_)

# Step 8 Refit on the entire training set
#print (clf.refit)

# Step 9 Evaluate Model Pipeline on Test Data
pred = clf.predict(X_test)

print (r2_score(y_test, pred))
# 0.45044082571584243
 
print (mean_squared_error(y_test, pred))
# 0.35461593750000003

joblib.dump(clf, 'rf_regressor.pkl')

# Load model again
#clf2 = joblib.load('rf_regressor.pkl')
 
# Predict data set using loaded model
#lf2.predict(X_test)
