import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing

data = pd.read_csv("Assignment3-TrainingData.csv")
# encode data to string https://stackoverflow.com/questions/46500357/valueerror-could-not-convert-string-to-float-med
le = preprocessing.LabelEncoder()
balance_data = data.apply(le.fit_transform)

x = balance_data.iloc[:, 0:26]  # independent columns
y = balance_data.iloc[:, -1:]  # target column

# apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(x, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)

# concat two data frames for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)

featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
print(featureScores.nlargest(15, 'Score'))  # print 15 best features
