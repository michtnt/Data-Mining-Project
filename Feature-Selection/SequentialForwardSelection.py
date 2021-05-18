# import libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

# read data
df = pd.read_csv('Assignment3-TrainingData.csv')

# encode data to string https://stackoverflow.com/questions/46500357/valueerror-could-not-convert-string-to-float-med
le = preprocessing.LabelEncoder()
balance_data = df.apply(le.fit_transform)

# split to independent columns (use to predict target) and target column (want to predict)
x = balance_data.iloc[:, 0:26].values  # independent columns
y = balance_data.iloc[:, -1].values    # target column

# train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# outline the Quote_Flag train and test data
y_train = y_train.ravel()
y_test = y_test.ravel()

print('Training dataset shape:', x_train.shape, y_train.shape)
print('Testing dataset shape:', x_test.shape, y_test.shape)

# build RF classifier to use in feature selection
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# Iterate on 10 - 26 features
for i in range(10, 26):
    # Build step forward feature selection
    sfs1 = sfs(clf,
               k_features=i,
               forward=True,
               floating=False,
               verbose=2,
               scoring='accuracy',
               cv=5)

    # Perform SFS
    sfs1 = sfs1.fit(x_train, y_train)
    feat_cols = list(sfs1.k_feature_idx_)
    # print number of features, accuracy score and features subsets
    print(feat_cols)
