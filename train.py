import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv('sample_data/california_housing_train.csv')
test = pd.read_csv('sample_data/california_housing_test.csv')

X_train = data.drop('median_house_value', axis=1)
y_train = data['median_house_value']

X_test = test.drop('median_house_value', axis=1)
y_test = test['median_house_value']

model = LinearRegression()
model.fit(X_train,y_train)

print(f'model score: {model.score(X,y)}')
