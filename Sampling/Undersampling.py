# import random undersampling and other necessary libraries
import numpy as np
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
# import SVM libraries
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# import data
df = pd.read_csv('../Feature-Selection/Assignment3-TrainingData.csv')

# encode data to string https://stackoverflow.com/questions/46500357/valueerror-could-not-convert-string-to-float-med
le = preprocessing.LabelEncoder()
balance_data = df.apply(le.fit_transform)

# Separating the independent variables from dependent variables
x = balance_data.iloc[:, 0:26].values  # independent columns
y = balance_data.iloc[:, -1].values    # target column

# Split train-test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

# summarize class distribution
print("Before undersampling: ", Counter(y_train))

# define undersampling strategy
undersample = RandomUnderSampler(sampling_strategy='majority')

# fit and apply the transform
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)

# summarize class distribution
print("After undersampling: ", Counter(y_train_under))

# SVM
model = SVC()
clf_under = model.fit(X_train_under, y_train_under)
pred_under = clf_under.predict(X_test)

print("ROC AUC score for undersampled data: ", roc_auc_score(y_test, pred_under))
