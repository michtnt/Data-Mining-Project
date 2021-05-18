import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

data = pd.read_csv("Assignment3-TrainingData.csv")

# encode data to string https://stackoverflow.com/questions/46500357/valueerror-could-not-convert-string-to-float-med
le = preprocessing.LabelEncoder()
balance_data = data.apply(le.fit_transform)

x = balance_data.iloc[:, 0:26]  # independent columns
y = balance_data.iloc[:, -1]    # target column

model = ExtraTreesClassifier()
model.fit(x, y)

# use inbuilt class feature importance of tree based classifiers
print(model.feature_importances_)

# plot graph of feature importance for better visualization
feat_importance = pd.Series(model.feature_importances_, index=x.columns)
feat_importance.nlargest(15).plot(kind="barh")
plt.show()
