#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

# Load and prepare Titanic data
titanic_train = pd.read_csv("./Datasets/train.csv")    # Read the data

# Impute median Age for NA Age values
new_age_var = np.where(titanic_train["Age"].isnull(), # Logical check
                       28,                       # Value if check is true
                       titanic_train["Age"])     # Value if check is false

titanic_train["Age"] = new_age_var

# Set the seed
np.random.seed(12)

# Initialize label encoder
label_encoder = preprocessing.LabelEncoder()

# Convert some variables to numeric
titanic_train["Sex"] = label_encoder.fit_transform(titanic_train["Sex"])

# Initialize the model
rf_model = RandomForestClassifier(n_estimators=1000, # Number of trees
                                  max_features=2,    # Num features considered
                                  oob_score=True)    # Use OOB scoring*

features = ["Sex","Pclass","SibSp","Age","Fare"]

# Train the model
rf_model.fit(X=titanic_train[features],
             y=titanic_train["Survived"])

print("OOB accuracy: ")
print(rf_model.oob_score_)
